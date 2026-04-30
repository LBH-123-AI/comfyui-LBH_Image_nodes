import torch

# ==================== 分段器（10段 + 翻页 + real_frames） ====================
class UltraStitchTiler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "segment_length": ("INT", {"default": 81, "min": 1, "max": 512}),
                "overlap_frames": ("INT", {"default": 8, "min": 0, "max": 64}),
                "start_segment": ("INT", {"default": 0, "min": 0, "max": 90, "step": 10}),
            }
        }

    RETURN_TYPES = ("IMAGE",) * 10 + ("INT", "INT", "INT")
    RETURN_NAMES = tuple(f"seg{i}" for i in range(1, 11)) + ("current_page", "total_pages", "real_frames")
    FUNCTION = "tile"
    CATEGORY = "UltraStitch"
    OUTPUT_NODE = True

    def tile(self, images: torch.Tensor, segment_length: int, overlap_frames: int, start_segment: int):
        # === 统一为 BHWC 格式 (B, H, W, 3) ===
        if images.dim() == 4 and images.shape[1] == 3:  # BCHW → BHWC
            images = images.permute(0, 2, 3, 1)
            was_bchw = True
        else:
            was_bchw = False

        total_frames = images.shape[0]
        step = max(1, segment_length - overlap_frames)
        total_segments = max(1, (total_frames + step - 1) // step)
        total_pages = max(1, (total_segments + 9) // 10)
        page = min(start_segment // 10, total_pages - 1)
        start_frame = page * 10 * step
        h, w = images.shape[1], images.shape[2]
        placeholder = torch.zeros((1, h, w, 3), dtype=images.dtype, device=images.device)

        segs = []
        real_frame_indices = set()  # 去重统计真实帧

        for i in range(10):
            s = start_frame + i * step
            e = min(s + segment_length, total_frames)

            if s >= total_frames:
                seg = placeholder
            else:
                seg = images[s:e]
                if seg.shape[0] == 0:
                    seg = placeholder
                else:
                    # 记录真实帧索引（去重）
                    real_frame_indices.update(range(s, e))
                    # 不补帧！保持原始长度
            segs.append(seg)

        # === 恢复原始格式 ===
        if was_bchw:
            segs = [seg.permute(0, 3, 1, 2) for seg in segs]

        current = page + 1
        real_frames_in_page = len(real_frame_indices)

        print(f"UltraStitch 分段器 [第 {current}/{total_pages} 页] "
              f"| 尺寸 {h}x{w} | 格式 {'BCHW' if was_bchw else 'BHWC'} "
              f"| 本页真实帧 {real_frames_in_page}")

        return tuple(segs) + (current, total_pages, real_frames_in_page)


# ==================== 拼接器（调色+叠化） ====================
class UltraStitchMerger:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "overlap_frames": ("INT", {"default": 8, "min": 0, "max": 64}),
                "curve": (["none", "cosine", "sine", "cubic", "linear"], {"default": "cosine"}),
                "color_match": ("BOOLEAN", {"default": True}),
            },
            "optional": {f"seg{i}": ("IMAGE",) for i in range(1, 11)}
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "merge"
    CATEGORY = "UltraStitch"
    OUTPUT_NODE = True

    def merge(self, overlap_frames: int, curve: str, color_match: bool, **segs):
        # 过滤空段 + 跳过 placeholder（全零 1 帧）
        valid = []
        for s in segs.values():
            if s is None or s.shape[0] == 0:
                continue
            # 跳过 1 帧全零（placeholder）
            if s.shape[0] == 1 and torch.all(s == 0):
                continue
            valid.append(s)

        if not valid:
            h, w = 512, 512
            return (torch.zeros((1, h, w, 3), dtype=torch.float32, device="cpu"),)

        device = valid[0].device
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        result = valid[0].to(dtype)

        for curr in valid[1:]:
            curr = curr.to(dtype)
            overlap = min(overlap_frames, result.shape[0], curr.shape[0])

            # 调色（只在 overlap 区域）
            if color_match and overlap > 0:
                matched = self._match_color(result[-overlap:], curr[:overlap])
            else:
                matched = curr[:overlap]

            # 叠化曲线
            if overlap == 0 or curve == "none":
                result = torch.cat([result, curr], dim=0)
                continue

            x = torch.linspace(0, 1, overlap, device=device)
            if curve == "cosine":
                alpha = (1 - torch.cos(x * torch.pi)) / 2
            elif curve == "sine":
                alpha = torch.sin(x * torch.pi / 2)
            elif curve == "cubic":
                alpha = x * x * (3 - 2 * x)
            else:  # linear
                alpha = x
            alpha = alpha.view(-1, 1, 1, 1)
            blended = result[-overlap:] * (1 - alpha) + matched * alpha
            result = torch.cat([result[:-overlap], blended, curr[overlap:]], dim=0)

        return (result.float(),)

    def _match_color(self, a, b):
        a32, b32 = a.float(), b.float()
        ma, sa = a32.mean([1, 2], keepdim=True), a32.std([1, 2], keepdim=True) + 1e-6
        mb, context = b32.mean([1, 2], keepdim=True), b32.std([1, 2], keepdim=True) + 1e-6
        return ((b32 - mb) * (sa / context) + ma).clamp(0, 1).to(b.dtype)


# ==================== 注册 ====================
NODE_CLASS_MAPPINGS = {
    "UltraStitchTiler": UltraStitchTiler,
    "UltraStitchMerger": UltraStitchMerger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UltraStitchTiler": "UltraStitch 视频分段（叠帧+翻页）",
    "UltraStitchMerger": "UltraStitch 视频拼接（调色+叠化）",
}