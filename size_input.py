import torch
from comfy.sd import VAE
from comfy.utils import common_upscale
from nodes import common_ksampler
from comfy_extras.nodes_freelunch import FreeU_V2

class SizeInput:
    """
    自定义节点：允许用户输入宽度和高度数值
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {
                    "default": 512, 
                    "min": 64,
                    "max": 8192,
                    "step": 8
                }),
                "height": ("INT", {
                    "default": 512, 
                    "min": 64,
                    "max": 8192,
                    "step": 8
                }),
            },
        }
    
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_size"
    CATEGORY = "custom"

    def get_size(self, width, height):
        return (width, height)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "SizeInput": SizeInput
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SizeInput": "尺寸输入"
}