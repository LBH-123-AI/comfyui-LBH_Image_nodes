import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageChops
import numpy as np
import cv2
from nodes import MAX_RESOLUTION

# ==================== 通用工具函数 ====================
def tensor2pil(image_tensor):
    if isinstance(image_tensor, torch.Tensor):
        if image_tensor.dim() == 4 and image_tensor.shape[0] == 1:
            image_tensor = image_tensor.squeeze(0)
        if image_tensor.dim() == 3 and image_tensor.shape[0] in [1, 3, 4]:
            to_pil = T.ToPILImage()
            return to_pil(image_tensor.cpu())
        elif image_tensor.dim() == 3 and image_tensor.shape[2] in [1, 3, 4]:
            image_tensor = image_tensor.permute(2, 0, 1)
            to_pil = T.ToPILImage()
            return to_pil(image_tensor.cpu())
        else:
            raise ValueError(f"Unsupported tensor shape: {image_tensor.shape}")
    else:
        raise ValueError("Input must be a torch.Tensor")

def pil2tensor(image_pil):
    image_np = np.array(image_pil).astype(np.float32) / 255.0
    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3, axis=-1)
    image_np = image_np[:, :, :3]
    image_np = image_np.transpose(2, 0, 1)
    return torch.from_numpy(np.expand_dims(image_np, 0))

# ==================== 节点1：裁剪数据生成 ====================
class WAS_Image_Crop_Data:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "crop_x": ("INT", {"default": 0, "min": 0}),
                "crop_y": ("INT", {"default": 0, "min": 0}),
                "crop_width": ("INT", {"default": 512, "min": 1}),
                "crop_height": ("INT", {"default": 512, "min": 1}),
            }
        }

    RETURN_TYPES = ("CROP_DATA",)
    FUNCTION = "generate_crop_data"
    CATEGORY = "image/processing"
    DESCRIPTION = "生成裁剪区域数据，用于后续裁剪操作"

    def generate_crop_data(self, image, crop_x, crop_y, crop_width, crop_height):
        img = tensor2pil(image)
        original_size = img.size
        crop_right = crop_x + crop_width
        crop_bottom = crop_y + crop_height
        crop_data = (original_size, (crop_x, crop_y, crop_right, crop_bottom))
        return (crop_data,)

# ==================== 节点2：从人脸图像计算裁剪数据 ====================
class CropDataFromFace:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "face_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("CROP_DATA",)
    FUNCTION = "calculate_crop"
    CATEGORY = "image/processing"

    def calculate_crop(self, original_image, face_image):
        orig_img = tensor2pil(original_image)
        face_img = tensor2pil(face_image)

        orig_np = np.array(orig_img.convert('L'))
        face_np = np.array(face_img.convert('L'))

        res = cv2.matchTemplate(orig_np, face_np, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)

        top_left = max_loc
        left, top = top_left
        face_w, face_h = face_img.size
        right = left + face_w
        bottom = top + face_h

        crop_data = ((face_w, face_h), (left, top, right, bottom))
        return (crop_data,)

# ==================== 节点3：人脸检测+裁剪 ====================
class FaceDetectorWithCrop:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "image": ("IMAGE", ),
                     "bbox_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "bbox_dilation": ("INT", {"default": 10, "min": -512, "max": 512, "step": 1}),
                     "bbox_crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10, "step": 0.1}),
                     "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),
                     "bbox_detector": ("BBOX_DETECTOR", ),
                     },
                "optional": {
                    "sam_model_opt": ("SAM_MODEL", ),
                    "segm_detector_opt": ("SEGM_DETECTOR", ),
                }}

    RETURN_TYPES = ("SEGS", "MASK", "IMAGE", "CROP_DATA")
    RETURN_NAMES = ("segs", "mask", "cropped_faces", "crop_data")
    FUNCTION = "detect_faces"
    OUTPUT_IS_LIST = (False, False, True, True)

    CATEGORY = "CustomNodes/FaceDetection"

    def detect_faces(self, image, bbox_threshold, bbox_dilation, bbox_crop_factor, drop_size,
                    bbox_detector, sam_model_opt=None, segm_detector_opt=None):
        
        # 设置默认提示词为'face'
        if hasattr(bbox_detector, 'setAux'):
            bbox_detector.setAux('face')
        
        # 使用检测器检测人脸
        segs = bbox_detector.detect(image, bbox_threshold, bbox_dilation, bbox_crop_factor, drop_size)
        
        if hasattr(bbox_detector, 'setAux'):
            bbox_detector.setAux(None)

        # 准备输出列表
        cropped_faces = []
        crop_data_list = []
        
        # 处理每个检测到的人脸
        for seg in segs[1]:
            x1, y1, x2, y2 = seg.crop_region
            # 从原图中裁剪人脸区域
            face_crop = image[:, y1:y2, x1:x2, :]
            cropped_faces.append(face_crop)
            
            # 添加裁剪数据元信息
            original_size = (x2 - x1, y2 - y1)
            crop_coords = (x1, y1, x2, y2)
            crop_data_list.append((original_size, crop_coords))

        # 生成合并的mask
        mask = self.segs_to_combined_mask(segs)

        return (segs, mask, cropped_faces, crop_data_list)

    @staticmethod
    def segs_to_combined_mask(segs):
        if len(segs[1]) == 0:
            return torch.zeros((segs[0][0], segs[0][1]), dtype=torch.float32)
        
        combined_mask = torch.zeros((segs[0][0], segs[0][1]), dtype=torch.float32)
        for seg in segs[1]:
            # 确保mask是torch.Tensor类型
            mask = seg.cropped_mask
            if isinstance(mask, np.ndarray):  # 如果是numpy数组，转换为tensor
                mask = torch.from_numpy(mask).float()
            
            x1, y1, x2, y2 = seg.crop_region
            # 确保区域大小匹配
            h, w = y2-y1, x2-x1
            if mask.shape != (h, w):
                mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear')[0,0]
            
            combined_mask[y1:y2, x1:x2] = torch.maximum(combined_mask[y1:y2, x1:x2], mask)
        
        return combined_mask

# ==================== 注册所有节点 ====================
NODE_CLASS_MAPPINGS = {
    "ImageCropData": WAS_Image_Crop_Data,
    "CropDataFromFace": CropDataFromFace,
    "FaceDetectorWithCrop": FaceDetectorWithCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCropData": "裁剪数据生成",
    "CropDataFromFace": "人脸裁剪数据提取",
    "FaceDetectorWithCrop": "人脸检测+裁剪",
}