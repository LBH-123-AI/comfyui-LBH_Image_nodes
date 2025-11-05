import random
import torch

class RandomSeedNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "generate_seed"
    CATEGORY = "utils"

    def generate_seed(self, seed):
        if seed == 0:  # 0表示自动生成随机种子
            seed = random.randint(0, 0xffffffffffffffff)
        torch.manual_seed(seed)
        return (seed,)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "RandomSeedNode": RandomSeedNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RandomSeedNode": "随机种"
}