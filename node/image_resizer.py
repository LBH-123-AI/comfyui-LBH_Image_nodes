import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from nodes import common_ksampler

class ImageResizer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "longest_side": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "interpolation": (["nearest", "bilinear", "bicubic", "lanczos"], {"default": "lanczos"}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resize_image"
    CATEGORY = "image/processing"
    DESCRIPTION = "Resize image uniformly by longest side while maintaining aspect ratio"

    def resize_image(self, image, longest_side, interpolation="lanczos"):
        # Convert tensor to PIL Image
        img = to_pil_image(image[0].permute(2, 0, 1))
        
        original_width, original_height = img.size
        
        # Calculate new dimensions while maintaining aspect ratio
        if original_width > original_height:
            new_width = longest_side
            new_height = int(original_height * (longest_side / original_width))
        else:
            new_height = longest_side
            new_width = int(original_width * (longest_side / original_height))
        
        # Choose interpolation method
        interp_methods = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS
        }
        interp = interp_methods.get(interpolation, Image.LANCZOS)
        
        # Resize the image
        resized_img = img.resize((new_width, new_height), interp)
        
        # Convert back to tensor
        resized_tensor = to_tensor(resized_img).permute(1, 2, 0).unsqueeze(0)
        
        return (resized_tensor,)

NODE_CLASS_MAPPINGS = {
    "ImageResizer": ImageResizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageResizer": "等边缩放",
}