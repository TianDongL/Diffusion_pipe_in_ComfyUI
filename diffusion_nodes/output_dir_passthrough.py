import os

class OutputDirPassthrough:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_dir": ("STRING", {
                    "forceInput": True,
                    "tooltip": "训练输出目录（来自通用训练设置）"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_dir",)
    FUNCTION = "passthrough"
    CATEGORY = "Diffusion-Pipe/Utils"
    
    def passthrough(self, output_dir):
        print(f"[输出目录传递] 传递路径: {output_dir}")
        return (output_dir,) 