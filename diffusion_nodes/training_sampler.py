import json
import logging
from typing import Tuple


class TrainingSamplerConfig:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            },
            "optional": {
                "sample_every_n_steps": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "tooltip": "每N步采样一次图片，0表示禁用按步数采样"
                }),
                "sample_every_n_epochs": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 100,
                    "tooltip": "每N个epoch采样一次图片，0表示禁用按epoch采样"
                }),
                "num_inference_steps": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 100,
                    "tooltip": "推理采样步数"
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "CFG引导强度 (Classifier-Free Guidance Scale)"
                }),
                "guidance_value": ("FLOAT", {
                    "default": 4.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "模型内部的guidance参数值"
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "采样图片高度"
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "采样图片宽度"
                }),
                "sample_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "采样时使用的提示词（留空则使用训练数据的caption）"
                }),
            }
        }
    
    RETURN_TYPES = ("SAMPLER_CONFIG",)
    RETURN_NAMES = ("sampler_config",)
    FUNCTION = "generate_sampler_config"
    CATEGORY = "Diffusion-Pipe/Config"

    def generate_sampler_config(self, **kwargs) -> Tuple[str]:
        try:
            sampler_config = {}
            
            # Only add to config if at least one sampling condition is enabled
            sample_every_n_steps = kwargs.get('sample_every_n_steps', 0)
            sample_every_n_epochs = kwargs.get('sample_every_n_epochs', 0)
            
            if sample_every_n_steps > 0:
                sampler_config['sample_every_n_steps'] = sample_every_n_steps
            
            if sample_every_n_epochs > 0:
                sampler_config['sample_every_n_epochs'] = sample_every_n_epochs
            
            sampler_config['height'] = kwargs.get('height', 1024)
            sampler_config['width'] = kwargs.get('width', 1024)
            
            # Add inference parameters
            num_inference_steps = kwargs.get('num_inference_steps', 20)
            guidance_scale = kwargs.get('guidance_scale', 5.0)
            guidance_value = kwargs.get('guidance_value', 4.0)
            sample_prompt = kwargs.get('sample_prompt', '').strip()
            
            sampler_config['num_inference_steps'] = num_inference_steps
            sampler_config['guidance_scale'] = guidance_scale
            sampler_config['guidance_value'] = guidance_value
            
            if sample_prompt:
                sampler_config['sample_prompt'] = sample_prompt
            
            if sampler_config:
                logging.info(f"训练采样器配置已生成: {sampler_config}")
            else:
                logging.info("训练采样器未启用 (sample_every_n_steps和sample_every_n_epochs都为0)")
            
            config_json = json.dumps(sampler_config, indent=2, ensure_ascii=False)
            
            return (config_json,)
            
        except Exception as e:
            logging.error(f"训练采样器配置生成失败: {str(e)}")
            return ("{}",)
