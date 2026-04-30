import datetime
import re

# ==========================================
# 节点类定义
# ==========================================

class GetCurrentTimeString:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "format": ("STRING", {
                    "default": "yyyy-MM-dd_HH-mm-ss_%seed%",
                    "multiline": False
                }),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("time_string",)
    FUNCTION = "get_time_string"
    CATEGORY = "utils/time"
    OUTPUT_NODE = True

    def get_time_string(self, format, seed=0):
        try:
            # 1. 处理 %seed% 占位符
            # 将字符串中的 %seed% 替换为实际的种子数字
            format_str = format.replace("%seed%", str(seed))
            
            # 2. 将友好格式转换为 Python strftime 格式
            # 支持 yyyy-MM-dd HH:mm:ss 风格
            format_str = self._convert_friendly_format(format_str)
            
            # 3. 获取当前时间并格式化
            now = datetime.datetime.now()
            result = now.strftime(format_str)
            
            return (result,)
        except Exception as e:
            # 如果出错，返回错误信息字符串，防止工作流崩溃
            return (f"Error: {str(e)}",)

    def _convert_friendly_format(self, fmt):
        """
        将类似 yyyy-MM-dd 的格式转换为 Python %Y-%m-%d 格式
        注意替换顺序，避免冲突 (如 yyyy 包含 yy)
        """
        replacements = {
            'yyyy': '%Y',  # 年
            'yy': '%y',    # 年后两位
            'MM': '%m',    # 月
            'dd': '%d',    # 日
            'HH': '%H',    # 时 (24 小时)
            'hh': '%I',    # 时 (12 小时)
            'mm': '%M',    # 分
            'ss': '%S',    # 秒
        }
        
        for friendly, python_fmt in replacements.items():
            # 使用 replace 直接替换
            fmt = fmt.replace(friendly, python_fmt)
            
        return fmt

# ==========================================
# 节点注册映射 (不要分开)
# ==========================================

NODE_CLASS_MAPPINGS = {
    "GetCurrentTimeString": GetCurrentTimeString,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GetCurrentTimeString": "Get Current Time String",
}