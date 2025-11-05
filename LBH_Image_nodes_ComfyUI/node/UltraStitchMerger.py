import torch

# ==================== 分段器（10段 + 翻页） ====================
class UltraStitchTiler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "segment_length": ("INT", {"default": 81, "min": 1, "max": 512}),   # 0 会炸
                "overlap_frames": ("INT", {"default": 8, "min": 0, "max": 64}),
                "start_segment": ("INT", {"default": 0, "min": 0, "max": 90, "step": 10}),
            }
        }

    RETURN_TYPES = ("IMAGE",) * 10 + ("INT", "INT")
    RETURN_NAMES = tuple(f"seg{i}" for i in range(1, 11)) + ("current_page", "total_pages")
    FUNCTION = "tile"
    CATEGORY = "UltraStitch"
    OUTPUT_NODE = True

    def tile(self, images: torch.Tensor, segment_length: int, overlap_frames: int, start_segment: int):
        total_frames = images.shape[0]
        step = max(1, segment_length - overlap_frames)  # 强制 step >= 1
        total_segments = (total_frames + step - 1) // step
        total_pages = max(1, (total_segments + 9) // 10)
        page = min(start_segment // 10, total_pages - 1)
        start_frame = page * 10 * step

        segs = []
        for i in range(10):
            s = start_frame + i * step
            e = s + segment_length
            seg = images[s:e] if s < total_frames else images[:0]  # 空 tensor
            segs.append(seg)

        real_segs = sum(1 for s in segs if s.shape[0] > 0)
        current = page + 1
        print(f"UltraStitch 分段器 [第 {current}/{total_pages} 页] "
              f"seg{page*10+1}~seg{page*10+10} | {real_segs} 段有效 | {total_frames} 帧")
        return tuple(segs) + (current, total_pages)


# ==================== 拼接器（调色+叠化） ====================
class UltraStitchMerger:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "overlap_frames": ("INT", {"default": 8, "min": 0, "max": 64}),
                # 新增：none 什么都不做！
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
        # 过滤空段
        valid = [s for s in segs.values() if s is not None and s.shape[0] > 0]
        if not valid:
            return (torch.zeros(1, 512, 512, 3, device="cpu"),)

        device = valid[0].device
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        result = valid[0].to(dtype)

        for curr in valid[1:]:
            curr = curr.to(dtype)
            overlap = min(overlap_frames, result.shape[0], curr.shape[0])

            # 1. 调色（只在 overlap 区域）
            if color_match and overlap > 0:
                matched = self._match_color(result[-overlap:], curr[:overlap])
            else:
                matched = curr[:overlap]

            # 2. 叠化曲线
            if overlap == 0 or curve == "none":
                # 直接拼接，0 帧重叠也安全
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
        mb, sb = b32.mean([1, 2], keepdim=True), b32.std([1, 2], keepdim=True) + 1e-6
        return ((b32 - mb) * (sa / sb) + ma).clamp(0, 1).to(b.dtype)


# ==================== 注册 ====================
NODE_CLASS_MAPPINGS = {
    "UltraStitchTiler": UltraStitchTiler,
    "UltraStitchMerger": UltraStitchMerger,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "UltraStitchTiler": "UltraStitch 视频分段（叠帧+翻页）",
    "UltraStitchMerger": "UltraStitch 视频拼接（调色+叠化）",
}