import os
import json
import logging
from typing import Dict, Any, Tuple

class ModelConfig:
    """模型配置节点 - 配置训练时的模型参数"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("model_path", {
                    "tooltip": "模型路径，根据不同的模型，选择不同的模型路径，具体查看注释"
                }),
                "dtype": (["bfloat16", "float16", "float32"], {
                    "default": "bfloat16",
                    "tooltip": "基础数据类型"
                }),
                "transformer_dtype": (["auto", "bfloat16", "float8","float8_e5m2"], {
                    "default": "auto",
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

    def generate_model_config(self, model_path, dtype: str, transformer_dtype: str, timestep_sample_method: str) -> Tuple[dict]:
        """生成模型配置"""
        try:
            # 构建完整的模型配置字典
            model_config = {
                "dtype": dtype,
                "timestep_sample_method": timestep_sample_method,
            }
            
            # 处理不同类型的model_path输入
            if isinstance(model_path, dict):
                # 检查是否有错误
                if "error" in model_path:
                    logging.error(f"模型路径配置错误: {model_path['error']}")
                    return ({"error": model_path["error"]},)
                
                # 如果是字典（来自模型节点），则直接合并所有配置（包括type）
                # 先合并model_path，再用当前配置覆盖
                final_config = model_path.copy()
                final_config.update(model_config)
                model_config = final_config
                
                # 获取模型类型信息用于显示
                model_type = model_path.get("type", "未知")
                path_info = f"模型类型: {model_type}, 配置项: {len(model_path)}"
            else:
                # 如果是字符串（来自CheckpointPathNode或DiffusersPathNode），则设置为checkpoint_path
                model_config["checkpoint_path"] = model_path
                path_info = f"模型路径: {model_path}"
            
            # 添加transformer_dtype（仅当非auto时）
            if transformer_dtype != "auto":
                model_config["transformer_dtype"] = transformer_dtype
            
            logging.info(f"成功生成模型配置，{path_info}")
            
            return (model_config,)
            
        except Exception as e:
            logging.error(f"模型配置生成失败: {str(e)}")
            return ({"error": str(e)},) 