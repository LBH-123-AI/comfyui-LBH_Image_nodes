import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageChops
import numpy as np
import cv2

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

# ==================== 节点3：将裁剪图粘贴回原图 ====================
class Image_Stitch_Crop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "crop_image": ("IMAGE",),
                "crop_data": ("CROP_DATA",),
                "crop_blending": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "crop_sharpening": ("INT", {"default": 0, "min": 0, "max": 3, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK_IMAGE")
    FUNCTION = "stitch_image"
    CATEGORY = "image/processing"

    def stitch_image(self, image, crop_image, crop_data=None, crop_blending=0.25, crop_sharpening=0):
        if not crop_data:
            raise ValueError("No crop data provided")

        result_image, result_mask = self.paste_image(
            tensor2pil(image),
            tensor2pil(crop_image),
            crop_data,
            crop_blending,
            crop_sharpening
        )
        return (result_image, result_mask)

    def paste_image(self, image, crop_image, crop_data, blend_amount=0.25, sharpen_amount=1):
        def lingrad(size, direction, white_ratio):
            image = Image.new('RGB', size)
            draw = ImageDraw.Draw(image)
            if direction == 'vertical':
                black_end = int(size[1] * (1 - white_ratio))
                for y in range(size[1]):
                    color = (0, 0, 0) if y <= black_end else (int(((y - black_end) / (size[1] - black_end)) * 255),) * 3
                    draw.line([(0, y), (size[0], y)], fill=color)
            elif direction == 'horizontal':
                black_end = int(size[0] * (1 - white_ratio))
                for x in range(size[0]):
                    color = (0, 0, 0) if x <= black_end else (int(((x - black_end) / (size[0] - black_end)) * 255),) * 3
                    draw.line([(x, 0), (x, size[1])], fill=color)
            return image.convert("L")

        crop_size, (left, top, right, bottom) = crop_data
        crop_image = crop_image.resize(crop_size)

        for _ in range(int(sharpen_amount)):
            crop_image = crop_image.filter(ImageFilter.SHARPEN)

        blended_image = Image.new('RGBA', image.size, (0, 0, 0, 255))
        blended_mask = Image.new('L', image.size, 0)
        crop_padded = Image.new('RGBA', image.size, (0, 0, 0, 0))

        blended_image.paste(image, (0, 0))
        crop_padded.paste(crop_image, (left, top))

        crop_mask = Image.new('L', crop_image.size, 0)
        if top > 0:
            crop_mask = ImageChops.screen(crop_mask, ImageOps.flip(lingrad(crop_image.size, 'vertical', blend_amount)))
        if left > 0:
            crop_mask = ImageChops.screen(crop_mask, ImageOps.mirror(lingrad(crop_image.size, 'horizontal', blend_amount)))
        if right < image.width:
            crop_mask = ImageChops.screen(crop_mask, lingrad(crop_image.size, 'horizontal', blend_amount))
        if bottom < image.height:
            crop_mask = ImageChops.screen(crop_mask, lingrad(crop_image.size, 'vertical', blend_amount))

        crop_mask = ImageOps.invert(crop_mask)
        blended_mask.paste(crop_mask, (left, top))
        blended_image.paste(crop_padded, (0, 0), blended_mask)

        return (pil2tensor(blended_image.convert("RGB")), pil2tensor(blended_mask.convert("RGB")))

# ==================== 注册所有节点 ====================
NODE_CLASS_MAPPINGS = {
    "ImageCropData": WAS_Image_Crop_Data,
    "CropDataFromFace": CropDataFromFace,
    "Image_Stitch_Crop": Image_Stitch_Crop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCropData": "裁剪数据生成",
    "CropDataFromFace": "人脸裁剪数据提取",
    "Image_Stitch_Crop": "图像粘贴融合",
}
