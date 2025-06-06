import torch
import numpy as np
import cv2
import math
from PIL import Image
import nodes
import folder_paths

class ImageTileSplitter:
    """
    图像/视频分块节点 - 支持处理视频序列
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_width": ("INT", {"default": 512, "min": 128, "max": 4096, "step": 8}),
                "tile_height": ("INT", {"default": 512, "min": 128, "max": 4096, "step": 8}),
                "overlap_ratio": ("FLOAT", {"default": 0.15, "min": 0.05, "max": 0.3, "step": 0.01}),
                "process_mode": (["single_frame", "video_sequence"], {"default": "single_frame"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "TILE_INFO")
    RETURN_NAMES = ("tiles", "tile_info")
    FUNCTION = "split_image"
    CATEGORY = "image/processing"

    def split_image(self, image, tile_width, tile_height, overlap_ratio, process_mode):
        batch_size = image.shape[0]
        
        # 单帧处理模式
        if process_mode == "single_frame" or batch_size == 1:
            return self._split_single_frame(image[0], tile_width, tile_height, overlap_ratio)
        # 视频序列处理模式
        else:
            return self._split_video_sequence(image, tile_width, tile_height, overlap_ratio)

    def _split_single_frame(self, image, tile_width, tile_height, overlap_ratio):
        # 转换为NumPy数组并确保在0-1范围
        img = np.clip(image.cpu().numpy(), 0, 1)
        h, w = img.shape[:2]
        
        # 计算重叠像素
        overlap_x = int(tile_width * overlap_ratio)
        overlap_y = int(tile_height * overlap_ratio)
        
        # 计算网格尺寸
        cols = math.ceil((w - overlap_x) / (tile_width - overlap_x))
        rows = math.ceil((h - overlap_y) / (tile_height - overlap_y))
        
        tiles = []
        tile_info = {
            "original_size": (w, h),
            "tile_size": (tile_width, tile_height),
            "grid_size": (cols, rows),
            "overlap": overlap_ratio,
            "positions": [],
            "process_mode": "single_frame"
        }
        
        # 生成分块
        for y in range(rows):
            for x in range(cols):
                x1 = x * (tile_width - overlap_x)
                y1 = y * (tile_height - overlap_y)
                x2 = min(x1 + tile_width, w)
                y2 = min(y1 + tile_height, h)
                
                tile = img[y1:y2, x1:x2]
                
                # 自动填充不足尺寸
                if tile.shape[0] < tile_height or tile.shape[1] < tile_width:
                    padded = np.zeros((tile_height, tile_width, 3), dtype=np.float32)
                    padded[:tile.shape[0], :tile.shape[1]] = tile
                    tile = padded
                
                tiles.append(tile)
                tile_info["positions"].append((x1, y1, x2, y2))
        
        return (torch.from_numpy(np.stack(tiles)), tile_info)

    def _split_video_sequence(self, video, tile_width, tile_height, overlap_ratio):
        # 获取第一帧信息用于初始化
        first_frame = video[0].cpu().numpy()
        h, w = first_frame.shape[:2]
        
        # 计算重叠和网格
        overlap_x = int(tile_width * overlap_ratio)
        overlap_y = int(tile_height * overlap_ratio)
        cols = math.ceil((w - overlap_x) / (tile_width - overlap_x))
        rows = math.ceil((h - overlap_y) / (tile_height - overlap_y))
        
        # 初始化tile_info
        tile_info = {
            "original_size": (w, h),
            "tile_size": (tile_width, tile_height),
            "grid_size": (cols, rows),
            "overlap": overlap_ratio,
            "positions": [],
            "process_mode": "video_sequence",
            "frame_count": video.shape[0]
        }
        
        # 计算所有瓦片位置
        positions = []
        for y in range(rows):
            for x in range(cols):
                x1 = x * (tile_width - overlap_x)
                y1 = y * (tile_height - overlap_y)
                x2 = min(x1 + tile_width, w)
                y2 = min(y1 + tile_height, h)
                positions.append((x1, y1, x2, y2))
        tile_info["positions"] = positions
        
        # 处理所有帧
        all_tiles = []
        for frame_idx in range(video.shape[0]):
            frame = np.clip(video[frame_idx].cpu().numpy(), 0, 1)
            
            for (x1, y1, x2, y2) in positions:
                tile = frame[y1:y2, x1:x2]
                
                # 自动填充不足尺寸
                if tile.shape[0] < tile_height or tile.shape[1] < tile_width:
                    padded = np.zeros((tile_height, tile_width, 3), dtype=np.float32)
                    padded[:tile.shape[0], :tile.shape[1]] = tile
                    tile = padded
                
                all_tiles.append(tile)
        
        return (torch.from_numpy(np.stack(all_tiles)), tile_info)

class NaturalTileMerger:
    """
    自然拼接节点 - 支持视频序列和动态尺寸瓦片的无缝混合
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "tile_info": ("TILE_INFO",),
                "blend_mode": (["gaussian", "linear"], {"default": "gaussian"}),
                "blend_strength": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
            },
            "optional": {
                "color_reference": ("IMAGE", {"default": None}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "merge_tiles"
    CATEGORY = "image/processing"

    def merge_tiles(self, tiles, tile_info, blend_mode, blend_strength, color_reference=None):
        process_mode = tile_info.get("process_mode", "single_frame")
        
        if process_mode == "single_frame":
            return self._merge_single_frame(tiles, tile_info, blend_mode, blend_strength, color_reference)
        else:
            return self._merge_video_sequence(tiles, tile_info, blend_mode, blend_strength, color_reference)

    def _merge_single_frame(self, tiles, tile_info, blend_mode, blend_strength, color_reference):
        w, h = tile_info["original_size"]
        tile_w, tile_h = tile_info["tile_size"]
        cols, rows = tile_info["grid_size"]
        overlap = tile_info["overlap"]
        positions = tile_info["positions"]
        
        # 初始化输出
        merged = np.zeros((h, w, 3), dtype=np.float32)
        weight = np.zeros((h, w), dtype=np.float32)
        
        # 处理参考图像
        ref_img = None
        if color_reference is not None:
            ref_img = color_reference[0].cpu().numpy()
            if ref_img.shape[:2] != (h, w):
                ref_img = cv2.resize(ref_img, (w, h))
            ref_img = np.clip(ref_img, 0, 1)

        # 计算混合权重
        overlap_x = int(tile_w * overlap)
        overlap_y = int(tile_h * overlap)
        
        if blend_mode == "gaussian":
            blend_x = self._gaussian_weights(overlap_x*2, sigma=0.3*blend_strength)
            blend_y = self._gaussian_weights(overlap_y*2, sigma=0.3*blend_strength)
        else:
            blend_x = np.linspace(0, 1, overlap_x*2) * blend_strength
            blend_y = np.linspace(0, 1, overlap_y*2) * blend_strength

        # 处理每个瓦片
        for idx, (x1, y1, x2, y2) in enumerate(positions):
            tile = tiles[idx].cpu().numpy()
            tile = np.clip(tile, 0, 1)[:y2-y1, :x2-x1]
            
            # 颜色调整
            if ref_img is not None:
                ref_patch = ref_img[y1:y2, x1:x2]
                if ref_patch.shape == tile.shape:
                    tile = self._adjust_exposure(tile, ref_patch, strength=0.2)
            
            # 创建权重图
            tile_weight = np.ones(tile.shape[:2], dtype=np.float32)
            
            # 应用混合权重
            if x1 > 0:  # 左边缘
                tile_weight[:, :overlap_x] *= blend_x[:overlap_x]
            if x2 < w:  # 右边缘
                tile_weight[:, -overlap_x:] *= blend_x[-overlap_x:]
            if y1 > 0:  # 上边缘
                tile_weight[:overlap_y, :] *= blend_y[:overlap_y][:, np.newaxis]
            if y2 < h:  # 下边缘
                tile_weight[-overlap_y:, :] *= blend_y[-overlap_y:][:, np.newaxis]
            
            # 合并到输出
            merged[y1:y2, x1:x2] += tile * tile_weight[..., np.newaxis]
            weight[y1:y2, x1:x2] += tile_weight
        
        # 归一化处理
        weight[weight == 0] = 1
        result = np.clip(merged / weight[..., np.newaxis], 0, 1)
        
        return (torch.from_numpy(result).unsqueeze(0),)

    def _merge_video_sequence(self, tiles, tile_info, blend_mode, blend_strength, color_reference):
        w, h = tile_info["original_size"]
        tile_w, tile_h = tile_info["tile_size"]
        cols, rows = tile_info["grid_size"]
        overlap = tile_info["overlap"]
        positions = tile_info["positions"]
        frame_count = tile_info["frame_count"]
        
        # 计算每帧的瓦片数
        tiles_per_frame = len(positions)
        
        # 初始化输出视频
        video_frames = []
        
        # 计算混合权重
        overlap_x = int(tile_w * overlap)
        overlap_y = int(tile_h * overlap)
        
        if blend_mode == "gaussian":
            blend_x = self._gaussian_weights(overlap_x*2, sigma=0.3*blend_strength)
            blend_y = self._gaussian_weights(overlap_y*2, sigma=0.3*blend_strength)
        else:
            blend_x = np.linspace(0, 1, overlap_x*2) * blend_strength
            blend_y = np.linspace(0, 1, overlap_y*2) * blend_strength

        # 处理每一帧
        for frame_idx in range(frame_count):
            # 初始化当前帧
            merged = np.zeros((h, w, 3), dtype=np.float32)
            weight = np.zeros((h, w), dtype=np.float32)
            
            # 处理当前帧的所有瓦片
            for tile_idx, (x1, y1, x2, y2) in enumerate(positions):
                global_tile_idx = frame_idx * tiles_per_frame + tile_idx
                tile = tiles[global_tile_idx].cpu().numpy()
                tile = np.clip(tile, 0, 1)[:y2-y1, :x2-x1]
                
                # 创建权重图
                tile_weight = np.ones(tile.shape[:2], dtype=np.float32)
                
                # 应用混合权重
                if x1 > 0:
                    tile_weight[:, :overlap_x] *= blend_x[:overlap_x]
                if x2 < w:
                    tile_weight[:, -overlap_x:] *= blend_x[-overlap_x:]
                if y1 > 0:
                    tile_weight[:overlap_y, :] *= blend_y[:overlap_y][:, np.newaxis]
                if y2 < h:
                    tile_weight[-overlap_y:, :] *= blend_y[-overlap_y:][:, np.newaxis]
                
                # 合并到当前帧
                merged[y1:y2, x1:x2] += tile * tile_weight[..., np.newaxis]
                weight[y1:y2, x1:x2] += tile_weight
            
            # 归一化当前帧
            weight[weight == 0] = 1
            frame = np.clip(merged / weight[..., np.newaxis], 0, 1)
            video_frames.append(frame)
        
        # 将所有帧堆叠为视频
        video_tensor = torch.from_numpy(np.stack(video_frames))
        return (video_tensor,)

    def _adjust_exposure(self, target, reference, strength=0.2):
        """调整曝光"""
        target_gray = np.mean(target, axis=2)
        ref_gray = np.mean(reference, axis=2)
        scale = (ref_gray.mean() + 1e-6) / (target_gray.mean() + 1e-6)
        scale = 1 + (scale - 1) * strength
        adjusted = target * np.clip(scale, 0.8, 1.2)
        return np.clip(adjusted, 0, 1)

    def _gaussian_weights(self, size, sigma=0.3):
        """生成高斯混合权重"""
        x = np.linspace(-1, 1, size)
        weights = np.exp(-x**2/(2*sigma**2))
        return weights / weights.max()

# 节点注册
NODE_CLASS_MAPPINGS = {
    "ImageTileSplitter": ImageTileSplitter,
    "NaturalTileMerger": NaturalTileMerger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageTileSplitter": "图像/视频分块器",
    "NaturalTileMerger": "图像/视频拼接器",
}