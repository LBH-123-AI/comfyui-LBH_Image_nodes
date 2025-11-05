import torch
import numpy as np
import cv2
import math
from PIL import Image
import nodes
import folder_paths

class ImageTileSplitter:
    """
    图像/视频分块节点 - 支持处理视频序列，并可选择按 tile 分组输出（便于导出独立子视频）
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
                "group_by_tile": ("BOOLEAN", {"default": False}),  # 新增：视频模式下按 tile 分组帧
            },
        }

    RETURN_TYPES = ("IMAGE", "TILE_INFO")
    RETURN_NAMES = ("tiles", "tile_info")
    FUNCTION = "split_image"
    CATEGORY = "image/processing"

    def split_image(self, image, tile_width, tile_height, overlap_ratio, process_mode, group_by_tile=False):
        batch_size = image.shape[0]
        
        if process_mode == "single_frame" or batch_size == 1:
            return self._split_single_frame(image[0], tile_width, tile_height, overlap_ratio)
        else:
            return self._split_video_sequence(image, tile_width, tile_height, overlap_ratio, group_by_tile)

    def _split_single_frame(self, image, tile_width, tile_height, overlap_ratio):
        img = np.clip(image.cpu().numpy(), 0, 1)
        h, w = img.shape[:2]
        
        overlap_x = int(tile_width * overlap_ratio)
        overlap_y = int(tile_height * overlap_ratio)
        
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
        
        for y in range(rows):
            for x in range(cols):
                x1 = x * (tile_width - overlap_x)
                y1 = y * (tile_height - overlap_y)
                x2 = min(x1 + tile_width, w)
                y2 = min(y1 + tile_height, h)
                
                tile = img[y1:y2, x1:x2]
                if tile.shape[0] < tile_height or tile.shape[1] < tile_width:
                    padded = np.zeros((tile_height, tile_width, 3), dtype=np.float32)
                    padded[:tile.shape[0], :tile.shape[1]] = tile
                    tile = padded
                
                tiles.append(tile)
                tile_info["positions"].append((x1, y1, x2, y2))
        
        return (torch.from_numpy(np.stack(tiles)), tile_info)

    def _split_video_sequence(self, video, tile_width, tile_height, overlap_ratio, group_by_tile=False):
        first_frame = video[0].cpu().numpy()
        h, w = first_frame.shape[:2]
        
        overlap_x = int(tile_width * overlap_ratio)
        overlap_y = int(tile_height * overlap_ratio)
        cols = math.ceil((w - overlap_x) / (tile_width - overlap_x))
        rows = math.ceil((h - overlap_y) / (tile_height - overlap_y))
        
        tile_info = {
            "original_size": (w, h),
            "tile_size": (tile_width, tile_height),
            "grid_size": (cols, rows),
            "overlap": overlap_ratio,
            "positions": [],
            "process_mode": "video_sequence",
            "frame_count": video.shape[0],
            "group_by_tile": group_by_tile,  # 标记是否按 tile 分组
        }
        
        positions = []
        for y in range(rows):
            for x in range(cols):
                x1 = x * (tile_width - overlap_x)
                y1 = y * (tile_height - overlap_y)
                x2 = min(x1 + tile_width, w)
                y2 = min(y1 + tile_height, h)
                positions.append((x1, y1, x2, y2))
        tile_info["positions"] = positions
        
        T = video.shape[0]  # 总帧数
        N = len(positions)  # tile 数量

        all_tiles = []

        if group_by_tile:
            # 按 tile 分组：[tile0_frame0, tile0_frame1, ..., tile0_frame{T-1}, tile1_frame0, ...]
            for pos_idx in range(N):
                x1, y1, x2, y2 = positions[pos_idx]
                for t in range(T):
                    frame = np.clip(video[t].cpu().numpy(), 0, 1)
                    tile = frame[y1:y2, x1:x2]
                    if tile.shape[0] < tile_height or tile.shape[1] < tile_width:
                        padded = np.zeros((tile_height, tile_width, 3), dtype=np.float32)
                        padded[:tile.shape[0], :tile.shape[1]] = tile
                        tile = padded
                    all_tiles.append(tile)
        else:
            # 原始顺序：[frame0_tile0, frame0_tile1, ..., frame1_tile0, ...]
            for t in range(T):
                frame = np.clip(video[t].cpu().numpy(), 0, 1)
                for (x1, y1, x2, y2) in positions:
                    tile = frame[y1:y2, x1:x2]
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
                "upscale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),  # 新增：支持放大后拼接
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "merge_tiles"
    CATEGORY = "image/processing"

    def merge_tiles(self, tiles, tile_info, blend_mode, blend_strength, color_reference=None, upscale_factor=1.0):
        process_mode = tile_info.get("process_mode", "single_frame")
        group_by_tile = tile_info.get("group_by_tile", False)
        T = tile_info.get("frame_count", 1)

        if process_mode == "single_frame":
            return self._merge_single_frame(tiles, tile_info, blend_mode, blend_strength, color_reference, upscale_factor)
        else:
            if group_by_tile:
                # 需要将 tiles 从 [N*T, H, W, C] 转回 [T*N, H, W, C] 的原始顺序
                N = len(tile_info["positions"])
                total = tiles.shape[0]
                assert total == N * T, f"Expected {N*T} tiles, got {total}"
                # 重排：从 [tile0_f0, tile0_f1, ..., tile1_f0, ...] → [f0_t0, f0_t1, ..., f1_t0, ...]
                reordered = []
                for t in range(T):
                    for n in range(N):
                        idx = n * T + t
                        reordered.append(tiles[idx])
                tiles = torch.stack(reordered)
            return self._merge_video_sequence(tiles, tile_info, blend_mode, blend_strength, color_reference, upscale_factor)

    def _merge_single_frame(self, tiles, tile_info, blend_mode, blend_strength, color_reference, upscale_factor=1.0):
        w, h = tile_info["original_size"]
        tile_w, tile_h = tile_info["tile_size"]
        cols, rows = tile_info["grid_size"]
        overlap = tile_info["overlap"]
        positions = tile_info["positions"]
        
        w_out = int(w * upscale_factor)
        h_out = int(h * upscale_factor)
        
        merged = np.zeros((h_out, w_out, 3), dtype=np.float32)
        weight = np.zeros((h_out, w_out), dtype=np.float32)
        
        ref_img = None
        if color_reference is not None:
            ref_img = color_reference[0].cpu().numpy()
            if ref_img.shape[:2] != (h, w):
                ref_img = cv2.resize(ref_img, (w, h))
            ref_img = np.clip(ref_img, 0, 1)

        overlap_x = int(tile_w * overlap)
        overlap_y = int(tile_h * overlap)
        overlap_x_out = int(overlap_x * upscale_factor)
        overlap_y_out = int(overlap_y * upscale_factor)
        
        if blend_mode == "gaussian":
            blend_x = self._gaussian_weights(overlap_x_out * 2, sigma=0.3 * blend_strength)
            blend_y = self._gaussian_weights(overlap_y_out * 2, sigma=0.3 * blend_strength)
        else:
            blend_x = np.linspace(0, 1, max(1, overlap_x_out * 2)) * blend_strength
            blend_y = np.linspace(0, 1, max(1, overlap_y_out * 2)) * blend_strength

        for idx, (x1, y1, x2, y2) in enumerate(positions):
            tile = tiles[idx].cpu().numpy()
            tile = np.clip(tile, 0, 1)[:y2-y1, :x2-x1]
            
            if ref_img is not None:
                ref_patch = ref_img[y1:y2, x1:x2]
                if ref_patch.shape == tile.shape:
                    tile = self._adjust_exposure(tile, ref_patch, strength=0.2)
            
            x1_o = int(x1 * upscale_factor)
            y1_o = int(y1 * upscale_factor)
            x2_o = int(x2 * upscale_factor)
            y2_o = int(y2 * upscale_factor)

            expected_h, expected_w = y2_o - y1_o, x2_o - x1_o
            if tile.shape[0] != expected_h or tile.shape[1] != expected_w:
                tile = cv2.resize(tile, (expected_w, expected_h), interpolation=cv2.INTER_LINEAR)

            tile_weight = np.ones(tile.shape[:2], dtype=np.float32)
            
            if x1_o > 0 and overlap_x_out > 0:
                tile_weight[:, :overlap_x_out] *= blend_x[:overlap_x_out]
            if x2_o < w_out and overlap_x_out > 0:
                tile_weight[:, -overlap_x_out:] *= blend_x[-overlap_x_out:]
            if y1_o > 0 and overlap_y_out > 0:
                tile_weight[:overlap_y_out, :] *= blend_y[:overlap_y_out][:, np.newaxis]
            if y2_o < h_out and overlap_y_out > 0:
                tile_weight[-overlap_y_out:, :] *= blend_y[-overlap_y_out:][:, np.newaxis]
            
            merged[y1_o:y2_o, x1_o:x2_o] += tile * tile_weight[..., np.newaxis]
            weight[y1_o:y2_o, x1_o:x2_o] += tile_weight
        
        weight[weight == 0] = 1
        result = np.clip(merged / weight[..., np.newaxis], 0, 1)
        return (torch.from_numpy(result).unsqueeze(0),)

    def _merge_video_sequence(self, tiles, tile_info, blend_mode, blend_strength, color_reference, upscale_factor=1.0):
        w, h = tile_info["original_size"]
        tile_w, tile_h = tile_info["tile_size"]
        cols, rows = tile_info["grid_size"]
        overlap = tile_info["overlap"]
        positions = tile_info["positions"]
        T = tile_info["frame_count"]
        
        tiles_per_frame = len(positions)
        assert tiles.shape[0] == T * tiles_per_frame, "Tile count mismatch"

        w_out = int(w * upscale_factor)
        h_out = int(h * upscale_factor)
        
        video_frames = []
        overlap_x = int(tile_w * overlap)
        overlap_y = int(tile_h * overlap)
        overlap_x_out = int(overlap_x * upscale_factor)
        overlap_y_out = int(overlap_y * upscale_factor)
        
        if blend_mode == "gaussian":
            blend_x = self._gaussian_weights(overlap_x_out * 2, sigma=0.3 * blend_strength)
            blend_y = self._gaussian_weights(overlap_y_out * 2, sigma=0.3 * blend_strength)
        else:
            blend_x = np.linspace(0, 1, max(1, overlap_x_out * 2)) * blend_strength
            blend_y = np.linspace(0, 1, max(1, overlap_y_out * 2)) * blend_strength

        for frame_idx in range(T):
            merged = np.zeros((h_out, w_out, 3), dtype=np.float32)
            weight = np.zeros((h_out, w_out), dtype=np.float32)
            
            for tile_idx, (x1, y1, x2, y2) in enumerate(positions):
                global_idx = frame_idx * tiles_per_frame + tile_idx
                tile = tiles[global_idx].cpu().numpy()
                tile = np.clip(tile, 0, 1)[:y2-y1, :x2-x1]
                
                x1_o = int(x1 * upscale_factor)
                y1_o = int(y1 * upscale_factor)
                x2_o = int(x2 * upscale_factor)
                y2_o = int(y2 * upscale_factor)

                expected_h, expected_w = y2_o - y1_o, x2_o - x1_o
                if tile.shape[0] != expected_h or tile.shape[1] != expected_w:
                    tile = cv2.resize(tile, (expected_w, expected_h), interpolation=cv2.INTER_LINEAR)

                tile_weight = np.ones(tile.shape[:2], dtype=np.float32)
                
                if x1_o > 0 and overlap_x_out > 0:
                    tile_weight[:, :overlap_x_out] *= blend_x[:overlap_x_out]
                if x2_o < w_out and overlap_x_out > 0:
                    tile_weight[:, -overlap_x_out:] *= blend_x[-overlap_x_out:]
                if y1_o > 0 and overlap_y_out > 0:
                    tile_weight[:overlap_y_out, :] *= blend_y[:overlap_y_out][:, np.newaxis]
                if y2_o < h_out and overlap_y_out > 0:
                    tile_weight[-overlap_y_out:, :] *= blend_y[-overlap_y_out:][:, np.newaxis]
                
                merged[y1_o:y2_o, x1_o:x2_o] += tile * tile_weight[..., np.newaxis]
                weight[y1_o:y2_o, x1_o:x2_o] += tile_weight
            
            weight[weight == 0] = 1
            frame = np.clip(merged / weight[..., np.newaxis], 0, 1)
            video_frames.append(frame)
        
        video_tensor = torch.from_numpy(np.stack(video_frames))
        return (video_tensor,)

    def _adjust_exposure(self, target, reference, strength=0.2):
        target_gray = np.mean(target, axis=2)
        ref_gray = np.mean(reference, axis=2)
        scale = (ref_gray.mean() + 1e-6) / (target_gray.mean() + 1e-6)
        scale = 1 + (scale - 1) * strength
        adjusted = target * np.clip(scale, 0.8, 1.2)
        return np.clip(adjusted, 0, 1)

    def _gaussian_weights(self, size, sigma=0.3):
        if size <= 1:
            return np.array([1.0])
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