import os
import json
import logging
from typing import Dict, Any, Tuple

class ModelConfig:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("model_path", {
                    "tooltip": "模型路径，根据不同的模型，选择不同的模型路径，具体查看注释"
                }),
                "model_type": (["diffusion", "transformers"], {
                    "default": "diffusion",
                    "tooltip": "选择模型格式类型 (diffusion / transformers)"
                }),
                "dtype": (["bfloat16", "float16", "float32"], {
                    "default": "bfloat16",
                    "tooltip": "基础数据类型"
                }),
                "diffusion_transformer_dtype": (["auto", "bfloat16", "float8","float8_e5m2"], {
                    "default": "bfloat16",
                    "tooltip": "Transformer特定数据类型（支持float8用于LoRA训练）"
                }),
                "timestep_sample_method": (["logit_normal", "uniform"], {
                    "default": "logit_normal",
                    "tooltip": "时间步采样方法，通常为logit_normal"
                }),
            }
        }
    
    RETURN_TYPES = ("model_config",)
    RETURN_NAMES = ("model_config",)
    FUNCTION = "generate_model_config"
    CATEGORY = "Diffusion-Pipe/Config"

    def generate_model_config(self, model_path, dtype: str, diffusion_transformer_dtype: str, timestep_sample_method: str, model_type: str) -> Tuple[dict]:
        try:
            model_config = {
                "dtype": dtype,
                "timestep_sample_method": timestep_sample_method,
            }
            
            if isinstance(model_path, dict):
                if "error" in model_path:
                    logging.error(f"模型路径配置错误: {model_path['error']}")
                    return ({"error": model_path["error"]},)
                
                final_config = model_path.copy()
                final_config.update(model_config)
                model_config = final_config
                
                path_model_type = model_path.get("type", "未知")
                path_info = f"模型类型: {path_model_type}, 配置项: {len(model_path)}"
            else:
                model_config["checkpoint_path"] = model_path
                path_info = f"模型路径: {model_path}"
            
            if diffusion_transformer_dtype != "auto":
                if model_type == "diffusion":
                    model_config["diffusion_model_dtype"] = diffusion_transformer_dtype
                else:
                    model_config["transformer_dtype"] = diffusion_transformer_dtype
            
            logging.info(f"成功生成模型配置，{path_info}")
            
            return (model_config,)
            
        except Exception as e:
            logging.error(f"模型配置生成失败: {str(e)}")
            return ({"error": str(e)},) 