import os
import glob
import logging
from typing import List, Tuple

def normalize_wsl_path(path):
    """
    规范化WSL2环境下的路径
    将Windows路径转换为WSL路径，或保持Linux路径不变
    """
    if not path:
        return path
        
    # 如果是Windows驱动器路径格式 (如 Z:\path 或 C:\path)
    if len(path) >= 3 and path[1] == ':' and path[2] in ['\\', '/']:
        drive_letter = path[0].lower()
        rest_path = path[3:].replace('\\', '/')
        
        # 特殊处理Z:驱动器（通常映射到WSL根目录）
        if drive_letter == 'z':
            # 如果rest_path以home开头，直接映射到根目录
            if rest_path.startswith('home/'):
                return f'/{rest_path}'
            else:
                return f'/mnt/{drive_letter}/{rest_path}'
        else:
            return f'/mnt/{drive_letter}/{rest_path}'
    
    # 如果已经是Linux路径格式，保持正斜杠
    elif path.startswith('/'):
        return path.replace('\\', '/')
    
    # 相对路径处理 - 保持原样，不进行路径规范化（避免在Windows下转换为反斜杠）
    else:
        return path     

# 尝试导入toml库
try:
    import toml
except ImportError:
    toml = None

class SDXLModelNode:
    """SDXL模型加载节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint_path": ("STRING", {
                    "default": "",
                    "tooltip": "SDXL checkpoint文件的完整路径"
                }),
                "dtype": (["bfloat16", "float16", "float32"], {
                    "default": "bfloat16",
                    "tooltip": "模型数据类型"
                }),
            },
            "optional": {
                "v_pred": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "启用v-prediction模式（如NoobAI vpred模型）"
                }),
                "min_snr_gamma": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "最小信噪比gamma值（0为禁用）"
                }),
                "debiased_estimation_loss": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "启用去偏估计损失"
                }),
                "unet_lr": ("FLOAT", {
                    "default": 4e-5,
                    "min": 1e-8,
                    "max": 1e-2,
                    "step": 1e-6,
                    "tooltip": "UNet学习率"
                }),
                "text_encoder_1_lr": ("FLOAT", {
                    "default": 2e-5,
                    "min": 1e-8,
                    "max": 1e-2,
                    "step": 1e-6,
                    "tooltip": "Text Encoder 1学习率"
                }),
                "text_encoder_2_lr": ("FLOAT", {
                    "default": 2e-5,
                    "min": 1e-8,
                    "max": 1e-2,
                    "step": 1e-6,
                    "tooltip": "Text Encoder 2学习率"
                }),
            }
        }
    
    RETURN_TYPES = ("model_path",)
    RETURN_NAMES = ("model_path",)
    FUNCTION = "get_sdxl_config"
    CATEGORY = "Diffusion-Pipe/Model"

    def get_sdxl_config(self, checkpoint_path: str, dtype: str, v_pred: bool = False, 
                       min_snr_gamma: float = 0.0, debiased_estimation_loss: bool = False,
                       unet_lr: float = 4e-5, text_encoder_1_lr: float = 2e-5, 
                       text_encoder_2_lr: float = 2e-5) -> Tuple[dict]:
        """获取SDXL模型配置"""
        try:
            if not checkpoint_path.strip():
                return ({"error": "checkpoint_path不能为空"},)
            
            # WSL2环境路径处理
            normalized_path = normalize_wsl_path(checkpoint_path.strip())
            
            # 构建SDXL模型配置
            config = {
                "type": "sdxl",
                "checkpoint_path": normalized_path,
                "dtype": dtype,
                "unet_lr": unet_lr,
                "text_encoder_1_lr": text_encoder_1_lr,
                "text_encoder_2_lr": text_encoder_2_lr,
            }
            
            # 添加可选参数（只有在非默认值时才添加）
            if v_pred:
                config["v_pred"] = True
            
            if min_snr_gamma > 0:
                config["min_snr_gamma"] = min_snr_gamma
            
            if debiased_estimation_loss:
                config["debiased_estimation_loss"] = True
            
            return (config,)
            
        except Exception as e:
            return ({"error": str(e)},)


class FluxModelNode:
    """Flux模型加载节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "diffusers_path": ("STRING", {
                    "default": "",
                    "tooltip": "Flux diffusers模型文件夹的完整路径"
                }),
                "transformer_path": ("STRING", {
                    "default": "",
                    "tooltip": "transformer文件的完整路径（当启用单独的transformer文件时需要如/data2/imagegen_models/flux-dev-single-files/consolidated_s6700-schnell.safetensors）"
                }),
                "bypass_guidance_embedding": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "绕过guidance embedding（FLEX.1-alpha启用）"
                }),
                "flux_shift": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "启用flux_shift"
                }),
            }
        }
    
    RETURN_TYPES = ("model_path",)
    RETURN_NAMES = ("model_path",)
    FUNCTION = "get_flux_config"
    CATEGORY = "Diffusion-Pipe/Model"

    def get_flux_config(self, flux_shift: bool, diffusers_path: str = "", transformer_path: str = "", bypass_guidance_embedding: bool = False) -> Tuple[dict]:
        """获取Flux模型配置"""
        try:
            # 构建Flux模型配置
            config = {
                "type": "flux",
                "flux_shift": flux_shift,
            }
            
            # 添加diffusers_path（仅当提供时）
            if diffusers_path and diffusers_path.strip():
                # WSL2环境路径处理
                normalized_diffusers_path = normalize_wsl_path(diffusers_path.strip())
                config["diffusers_path"] = normalized_diffusers_path
                        
            # 处理可选的transformer文件
            if transformer_path and transformer_path.strip():
                # WSL2环境路径处理：将Windows路径转换为WSL路径
                abs_transformer_path = normalize_wsl_path(transformer_path.strip())
                config["transformer_path"] = abs_transformer_path
            
            # 添加可选参数（只有在非默认值时才添加）
            if bypass_guidance_embedding:
                config["bypass_guidance_embedding"] = True
            
            return (config,)
            
        except Exception as e:
            return ({"error": str(e)},)
    



class LTXVideoModelNode:
    """LTX-Video模型加载节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "diffusers_path": ("STRING", {
                    "default": "",
                    "tooltip": "LTX-Video diffusers模型文件夹的完整路径（如：/data/models/LTX-Video）"
                }),
            },
            "optional": {
                "use_single_file": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "是否使用单个模型文件（.safetensors）"
                }),
                "single_file_path": ("STRING", {
                    "default": "",
                    "tooltip": "单个模型文件的完整路径（如：/data2/imagegen_models/LTX-Video/ltx-video-2b-v0.9.1.safetensors）"
                }),
                "first_frame_conditioning_p": ("FLOAT", {
                    "default": 1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "使用第一帧作为条件的概率（i2v训练）"
                }),
            }
        }
    
    RETURN_TYPES = ("model_path",)
    RETURN_NAMES = ("model_path",)
    FUNCTION = "get_ltx_video_config"
    CATEGORY = "Diffusion-Pipe/Model"

    def get_ltx_video_config(self, diffusers_path: str, use_single_file: bool = False, 
                           single_file_path: str = "", first_frame_conditioning_p: float = 0.0) -> Tuple[dict]:
        """获取LTX-Video模型配置"""
        try:
            if not diffusers_path.strip():
                return ({"error": "diffusers_path不能为空"},)
            
            # WSL2环境路径处理
            normalized_diffusers_path = normalize_wsl_path(diffusers_path.strip())
            
            # 构建LTX-Video模型配置
            config = {
                "type": "ltx-video",
                "diffusers_path": normalized_diffusers_path,
            }
            
            # 处理单个文件路径
            if use_single_file and single_file_path.strip():
                # WSL2环境路径处理
                normalized_single_file_path = normalize_wsl_path(single_file_path.strip())
                config["single_file_path"] = normalized_single_file_path
            
            # 添加可选参数（只有在非默认值时才添加）
            if first_frame_conditioning_p > 0:
                config["first_frame_conditioning_p"] = first_frame_conditioning_p
            
            return (config,)
            
        except Exception as e:
            return ({"error": str(e)},)


class HunyuanVideoModelNode:
    """HunyuanVideo模型加载节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "ckpt_path": ("STRING", {
                    "default": "",
                    "tooltip": "HunyuanVideo官方推理脚本的ckpt路径（如：/home/anon/HunyuanVideo/ckpts）"
                }),
                "transformer_path": ("STRING", {
                    "default": "",
                    "tooltip": "Transformer模型文件的完整路径（如：/data2/imagegen_models/hunyuan_video_comfyui/hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors），不填则加载官方推理脚本ckpt"
                }),
                "vae_path": ("STRING", {
                    "default": "",
                    "tooltip": "VAE文件或文件夹的完整路径，不填则加载官方推理脚本ckpt"
                }),
                "llm_path": ("STRING", {
                    "default": "",
                    "tooltip": "LLM文件夹的完整路径，不填则加载官方推理脚本ckpt"
                }),
                "clip_path": ("STRING", {
                    "default": "",
                    "tooltip": "CLIP文件或文件夹的完整路径，不填则加载官方推理脚本ckpt"
                }),
            }
        }
    
    RETURN_TYPES = ("model_path",)
    RETURN_NAMES = ("model_path",)
    FUNCTION = "get_hunyuan_video_config"
    CATEGORY = "Diffusion-Pipe/Model"

    def get_hunyuan_video_config(self, ckpt_path: str = "", transformer_path: str = "",
                               vae_path: str = "", llm_path: str = "", 
                               clip_path: str = "") -> Tuple[dict]:
        """获取HunyuanVideo模型配置"""
        try:
            # 构建HunyuanVideo模型配置
            config = {
                "type": "hunyuan-video",
            }
            
            # 处理ckpt_path
            if ckpt_path.strip():
                config["ckpt_path"] = normalize_wsl_path(ckpt_path.strip())
            
            # 处理transformer_path
            if transformer_path.strip():
                config["transformer_path"] = normalize_wsl_path(transformer_path.strip())
            
            # 处理vae_path
            if vae_path.strip():
                config["vae_path"] = normalize_wsl_path(vae_path.strip())
            
            # 处理llm_path
            if llm_path.strip():
                config["llm_path"] = normalize_wsl_path(llm_path.strip())
            
            # 处理clip_path
            if clip_path.strip():
                config["clip_path"] = normalize_wsl_path(clip_path.strip())
            
            return (config,)
            
        except Exception as e:
            return ({"error": str(e)},)


class CosmosModelNode:
    """Cosmos模型加载节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "transformer_path": ("STRING", {
                    "default": "",
                    "tooltip": "Transformer模型文件的完整路径（如：/data2/imagegen_models/cosmos/cosmos-1.0-diffusion-7b-text2world.pt）"
                }),
                "vae_path": ("STRING", {
                    "default": "",
                    "tooltip": "VAE文件的完整路径"
                }),
                "text_encoder_path": ("STRING", {
                    "default": "",
                    "tooltip": "Text Encoder文件的完整路径"
                }),
            }
        }
    
    RETURN_TYPES = ("model_path",)
    RETURN_NAMES = ("model_path",)
    FUNCTION = "get_cosmos_config"
    CATEGORY = "Diffusion-Pipe/Model"

    def get_cosmos_config(self, transformer_path: str, vae_path: str, text_encoder_path: str) -> Tuple[dict]:
        """获取Cosmos模型配置"""
        try:
            # 构建Cosmos模型配置
            config = {
                "type": "cosmos",
            }
            
            # 处理transformer_path
            if transformer_path.strip():
                config["transformer_path"] = normalize_wsl_path(transformer_path.strip())
            else:
                return ({"error": "Transformer路径不能为空"},)
            
            # 处理vae_path
            if vae_path.strip():
                config["vae_path"] = normalize_wsl_path(vae_path.strip())
            else:
                return ({"error": "VAE路径不能为空"},)
            
            # 处理text_encoder_path
            if text_encoder_path.strip():
                config["text_encoder_path"] = normalize_wsl_path(text_encoder_path.strip())
            else:
                return ({"error": "Text Encoder路径不能为空"},)
            
            return (config,)
            
        except Exception as e:
            return ({"error": str(e)},)


class Lumina2ModelNode:
    """Lumina 2模型加载节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "transformer_path": ("STRING", {
                    "default": "",
                    "tooltip": "Transformer模型文件的完整路径（如：/data2/imagegen_models/lumina-2-single-files/lumina_2_model_bf16.safetensors）"
                }),
                "llm_path": ("STRING", {
                    "default": "",
                    "tooltip": "LLM文件或文件夹的完整路径"
                }),
                "vae_path": ("STRING", {
                    "default": "",
                    "tooltip": "VAE文件的完整路径"
                }),
            },
            "optional": {
                "lumina_shift": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "启用Lumina shift（分辨率相关的时间步偏移）"
                }),
            }
        }
    
    RETURN_TYPES = ("model_path",)
    RETURN_NAMES = ("model_path",)
    FUNCTION = "get_lumina2_config"
    CATEGORY = "Diffusion-Pipe/Model"

    def get_lumina2_config(self, transformer_path: str, llm_path: str, vae_path: str, 
                          lumina_shift: bool = True) -> Tuple[dict]:
        """获取Lumina 2模型配置"""
        try:
            # 构建Lumina 2模型配置
            config = {
                "type": "lumina_2",
            }
            
            # 处理transformer_path
            if transformer_path.strip():
                config["transformer_path"] = normalize_wsl_path(transformer_path.strip())
            else:
                return ({"error": "Transformer路径不能为空"},)
            
            # 处理llm_path
            if llm_path.strip():
                config["llm_path"] = normalize_wsl_path(llm_path.strip())
            else:
                return ({"error": "LLM路径不能为空"},)
            
            # 处理vae_path
            if vae_path.strip():
                config["vae_path"] = normalize_wsl_path(vae_path.strip())
            else:
                return ({"error": "VAE路径不能为空"},)
            
            # 添加可选参数（只有在非默认值时才添加）
            if lumina_shift:
                config["lumina_shift"] = True
            
            return (config,)
            
        except Exception as e:
            return ({"error": str(e)},)


class Wan21ModelNode:
    """Wan2.1模型加载节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_path": ("STRING", {
                    "default": "",
                    "tooltip": "Wan2.1模型checkpoint目录的完整路径，必填，至少需要包含模型所需的config文件（/data2/imagegen_models/Wan2.1-T2V-1.3B）"
                }),
            },
            "optional": {
                "transformer_path": ("STRING", {
                    "default": "",
                    "tooltip": "你也可以选用safetensors格式的Transformer的模型文件（如：/data2/imagegen_models/wan_comfyui/wan2.1_t2v_1.3B_bf16.safetensors）"
                }),
                "llm_path": ("STRING", {
                    "default": "",
                    "tooltip": "可选：LLM文件路径（如：/data2/imagegen_models/wan_comfyui/wrapper/umt5-xxl-enc-bf16.safetensors）"
                }),

            }
        }
    
    RETURN_TYPES = ("model_path",)
    RETURN_NAMES = ("model_path",)
    FUNCTION = "get_wan21_config"
    CATEGORY = "Diffusion-Pipe/Model"

    def get_wan21_config(self, ckpt_path: str, transformer_path: str = "", llm_path: str = "") -> Tuple[dict]:
        """获取Wan2.1模型配置"""
        try:
            # 构建Wan2.1模型配置
            config = {
                "type": "wan",
            }
            
            # 处理ckpt_path（必需参数）
            if ckpt_path.strip():
                config["ckpt_path"] = normalize_wsl_path(ckpt_path.strip())
            else:
                return ({"error": "ckpt_path不能为空"},)
            
            # 处理可选的transformer_path
            if transformer_path.strip():
                config["transformer_path"] = normalize_wsl_path(transformer_path.strip())
            
            # 处理可选的llm_path
            if llm_path.strip():
                config["llm_path"] = normalize_wsl_path(llm_path.strip())
            
            return (config,)
            
        except Exception as e:
            return ({"error": str(e)},)


class ChromaModelNode:
    """Chroma模型加载节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "diffusers_path": ("STRING", {
                    "default": "",
                    "tooltip": "Flux diffusers模型文件夹的完整路径（用于加载VAE和text encoder，如：/data/models/FLUX.1-dev）"
                }),
                "transformer_path": ("STRING", {
                    "default": "",
                    "tooltip": "Chroma单模型文件的完整路径（如：/data2/imagegen_models/chroma/chroma-unlocked-v10.safetensors）"
                }),
            },
            "optional": {
                "flux_shift": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "分辨率相关的时间步偏移，向更多噪声偏移"
                }),
            }
        }
    
    RETURN_TYPES = ("model_path",)
    RETURN_NAMES = ("model_path",)
    FUNCTION = "get_chroma_config"
    CATEGORY = "Diffusion-Pipe/Model"

    def get_chroma_config(self, diffusers_path: str, transformer_path: str, flux_shift: bool = True) -> Tuple[dict]:
        """获取Chroma模型配置"""
        try:
            if not diffusers_path.strip():
                return ({"error": "diffusers_path不能为空"},)
            
            if not transformer_path.strip():
                return ({"error": "transformer_path不能为空"},)
            
            # WSL2环境路径处理
            normalized_diffusers_path = normalize_wsl_path(diffusers_path.strip())
            normalized_transformer_path = normalize_wsl_path(transformer_path.strip())
            
            # 构建Chroma模型配置
            config = {
                "type": "chroma",
                "diffusers_path": normalized_diffusers_path,
                "transformer_path": normalized_transformer_path,
            }
            
            # 添加可选参数（只有在非默认值时才添加）
            if flux_shift:
                config["flux_shift"] = True
            
            return (config,)
            
        except Exception as e:
            return ({"error": str(e)},)


class HiDreamModelNode:
    """HiDream模型加载节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "diffusers_path": ("STRING", {
                    "default": "",
                    "tooltip": "HiDream diffusers模型文件夹的完整路径（如：/data/models/HiDream-I1-Full）"
                }),
                "llama3_path": ("STRING", {
                    "default": "",
                    "tooltip": "Llama3模型文件夹的完整路径（如：/data/models/llama3）"
                }),
            },
            "optional": {
                "llama3_4bit": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "启用Llama3 4bit量化"
                }),
                "max_llama3_sequence_length": ("INT", {
                    "default": 128,
                    "min": 32,
                    "max": 2048,
                    "step": 1,
                    "tooltip": "Llama3最大序列长度"
                }),
                "flux_shift": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "使用分辨率相关的时间步偏移（类似Flux）"
                }),
            }
        }
    
    RETURN_TYPES = ("model_path",)
    RETURN_NAMES = ("model_path",)
    FUNCTION = "get_hidream_config"
    CATEGORY = "Diffusion-Pipe/Model"

    def get_hidream_config(self, diffusers_path: str, llama3_path: str, llama3_4bit: bool = True,
                          max_llama3_sequence_length: int = 128, flux_shift: bool = False) -> Tuple[dict]:
        """获取HiDream模型配置"""
        try:
            if not diffusers_path.strip():
                return ({"error": "diffusers_path不能为空"},)
            
            if not llama3_path.strip():
                return ({"error": "llama3_path不能为空"},)
            
            # WSL2环境路径处理
            normalized_diffusers_path = normalize_wsl_path(diffusers_path.strip())
            normalized_llama3_path = normalize_wsl_path(llama3_path.strip())
            
            # 构建HiDream模型配置
            config = {
                "type": "hidream",
                "diffusers_path": normalized_diffusers_path,
                "llama3_path": normalized_llama3_path,
            }
            
            # 添加可选参数（只有在非默认值时才添加）
            if llama3_4bit:
                config["llama3_4bit"] = True
            
            if max_llama3_sequence_length != 128:
                config["max_llama3_sequence_length"] = max_llama3_sequence_length
            
            if flux_shift:
                config["flux_shift"] = True
            
            return (config,)
            
        except Exception as e:
            return ({"error": str(e)},)


class SD3ModelNode:
    """Stable Diffusion 3模型加载节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "diffusers_path": ("STRING", {
                    "default": "",
                    "tooltip": "SD3 diffusers模型文件夹的完整路径（需要完整的Diffusers文件夹，如：/data/models/stable-diffusion-3-medium-diffusers）"
                }),
            },
            "optional": {
                "flux_shift": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "使用分辨率相关的时间步偏移（类似Flux）"
                }),
            }
        }
    
    RETURN_TYPES = ("model_path",)
    RETURN_NAMES = ("model_path",)
    FUNCTION = "get_sd3_config"
    CATEGORY = "Diffusion-Pipe/Model"

    def get_sd3_config(self, diffusers_path: str, flux_shift: bool = False) -> Tuple[dict]:
        """获取SD3模型配置"""
        try:
            if not diffusers_path.strip():
                return ({"error": "diffusers_path不能为空"},)
            
            # WSL2环境路径处理
            normalized_diffusers_path = normalize_wsl_path(diffusers_path.strip())
            
            # 构建SD3模型配置
            config = {
                "type": "sd3",
                "diffusers_path": normalized_diffusers_path,
            }
            
            # 添加可选参数（只有在非默认值时才添加）
            if flux_shift:
                config["flux_shift"] = True
            
            return (config,)
            
        except Exception as e:
            return ({"error": str(e)},)


class CosmosPredict2ModelNode:
    """Cosmos Predict2模型加载节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "transformer_path": ("STRING", {
                    "default": "",
                    "tooltip": "Transformer模型文件路径（如：/data2/imagegen_models/Cosmos-Predict2-2B-Text2Image/model.pt）"
                }),
                "vae_path": ("STRING", {
                    "default": "",
                    "tooltip": "VAE文件的完整路径（建议使用万相的vae，如：/data/models/wan_2.1_vae.safetensors）"
                }),
                "t5_path": ("STRING", {
                    "default": "",
                    "tooltip": "T5模型文件的完整路径（注意！使用旧版T5模型文件，如：/data2/imagegen_models/comfyui-models/oldt5_xxl_fp16.safetensors）"
                }),
            }
        }
    
    RETURN_TYPES = ("model_path",)
    RETURN_NAMES = ("model_path",)
    FUNCTION = "get_cosmos_predict2_config"
    CATEGORY = "Diffusion-Pipe/Model"

    def get_cosmos_predict2_config(self, transformer_path: str, vae_path: str, t5_path: str) -> Tuple[dict]:
        """获取Cosmos Predict2模型配置"""
        try:
            # 构建Cosmos Predict2模型配置
            config = {
                "type": "cosmos_predict2",
            }
            
            # 处理transformer_path
            if transformer_path.strip():
                config["transformer_path"] = normalize_wsl_path(transformer_path.strip())
            else:
                return ({"error": "Transformer路径不能为空"},)
            
            # 处理vae_path
            if vae_path.strip():
                config["vae_path"] = normalize_wsl_path(vae_path.strip())
            else:
                return ({"error": "VAE路径不能为空"},)
            
            # 处理t5_path
            if t5_path.strip():
                config["t5_path"] = normalize_wsl_path(t5_path.strip())
            else:
                return ({"error": "T5路径不能为空"},)
            
            return (config,)
            
        except Exception as e:
            return ({"error": str(e)},)


class OmniGen2ModelNode:
    """OmniGen2模型加载节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "diffusers_path": ("STRING", {
                    "default": "",
                    "tooltip": "OmniGen2 diffusers模型文件夹的完整路径（需要完整的官方checkpoint目录，如：/data/models/OmniGen-v1）"
                }),
            },
            "optional": {
                "flux_shift": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "使用分辨率相关的时间步偏移（类似Flux），默认禁用"
                }),
            }
        }
    
    RETURN_TYPES = ("model_path",)
    RETURN_NAMES = ("model_path",)
    FUNCTION = "get_omnigen2_config"
    CATEGORY = "Diffusion-Pipe/Model"

    def get_omnigen2_config(self, diffusers_path: str, flux_shift: bool = False) -> Tuple[dict]:
        """获取OmniGen2模型配置"""
        try:
            if not diffusers_path.strip():
                return ({"error": "diffusers_path不能为空"},)
            
            # WSL2环境路径处理
            normalized_diffusers_path = normalize_wsl_path(diffusers_path.strip())
            
            # 构建OmniGen2模型配置
            config = {
                "type": "omnigen2",
                "diffusers_path": normalized_diffusers_path,
            }
            
            # 添加可选参数（只有在非默认值时才添加）
            if flux_shift:
                config["flux_shift"] = True
            
            return (config,)
            
        except Exception as e:
            return ({"error": str(e)},)


class FluxKontextModelNode:
    """Flux Kontext模型加载节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "diffusers_path": ("STRING", {
                    "default": "",
                    "tooltip": "Flux Dev diffusers模型文件夹的完整路径（如：/data/models/FLUX.1-dev，用于加载VAE和text encoder）"
                }),
            },
            "optional": {
                "transformer_path": ("STRING", {
                    "default": "",
                    "tooltip": "Flux Kontext单模型文件的完整路径（如：/data2/imagegen_models/flux-dev-single-files/flux1-kontext-dev.safetensors），可选填写以节省空间"
                }),
                "flux_shift": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "分辨率相关的时间步偏移，向更多噪声偏移"
                }),
            }
        }
    
    RETURN_TYPES = ("model_path",)
    RETURN_NAMES = ("model_path",)
    FUNCTION = "get_flux_kontext_config"
    CATEGORY = "Diffusion-Pipe/Model"

    def get_flux_kontext_config(self, diffusers_path: str, transformer_path: str = "", 
                               flux_shift: bool = True) -> Tuple[dict]:
        """获取Flux Kontext模型配置"""
        try:
            if not diffusers_path.strip():
                return ({"error": "diffusers_path不能为空"},)
            
            # WSL2环境路径处理
            normalized_diffusers_path = normalize_wsl_path(diffusers_path.strip())
            
            # 构建Flux Kontext模型配置
            config = {
                "type": "flux",
                "diffusers_path": normalized_diffusers_path,
                "flux_shift": flux_shift,
            }
            
            # 处理可选的transformer文件
            if transformer_path.strip():
                config["transformer_path"] = normalize_wsl_path(transformer_path.strip())
            
            return (config,)
            
        except Exception as e:
            return ({"error": str(e)},)


class Wan22ModelNode:
    """Wan2.2模型加载节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_path": ("STRING", {
                    "default": "",
                    "tooltip": "Wan2.2模型checkpoint目录的完整路径，必填，至少需要包含VAE和config文件（如：/data/imagegen_models/Wan2.2-T2V-A14B）"
                }),
                "min_t": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "tooltip": "最小时间步范围，用于控制训练的噪声范围（低噪声模型：0，高噪声模型：0.875）"
                }),
                "max_t": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "tooltip": "最大时间步范围，用于控制训练的噪声范围（低噪声模型：0.875，高噪声模型：1）"
                }),
            },
            "optional": {
                "transformer_path": ("STRING", {
                    "default": "",
                    "tooltip": "Transformer模型路径，可指向子文件夹（如：/data/imagegen_models/Wan2.2-T2V-A14B/low_noise_model）或你的ComfyUI文件下的模型（如：/data/imagegen_models/comfyui-models/wan2.2_t2v_low_noise_14B_fp16.safetensors）"
                }),
                "llm_path": ("STRING", {
                    "default": "",
                    "tooltip": "可选：LLM文件路径，用于你的ComfyUI文件下的模型加载（如：/data2/imagegen_models/comfyui-models/umt5_xxl_fp16.safetensors）"
                }),
            }
        }
    
    RETURN_TYPES = ("model_path",)
    RETURN_NAMES = ("model_path",)
    FUNCTION = "get_wan22_config"
    CATEGORY = "Diffusion-Pipe/Model"

    def get_wan22_config(self, ckpt_path: str, transformer_path: str = "", llm_path: str = "", 
                        min_t: float = 0.0, max_t: float = 1.0) -> Tuple[dict]:
        """获取Wan2.2模型配置"""
        try:
            # 构建Wan2.2模型配置
            config = {
                "type": "wan",
            }
            
            # 处理ckpt_path（必需参数）
            if ckpt_path.strip():
                config["ckpt_path"] = normalize_wsl_path(ckpt_path.strip())
            else:
                return ({"error": "ckpt_path不能为空"},)
            
            # 处理可选的transformer_path
            if transformer_path.strip():
                config["transformer_path"] = normalize_wsl_path(transformer_path.strip())
            
            # 处理可选的llm_path
            if llm_path.strip():
                config["llm_path"] = normalize_wsl_path(llm_path.strip())
            
            # 添加时间步范围参数（必填参数）
            config["min_t"] = min_t
            config["max_t"] = max_t
            
            return (config,)
            
        except Exception as e:
            return ({"error": str(e)},)


class QwenImageModelNode:
    """Qwen-Image模型加载节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "diffusers_path": ("STRING", {
                    "default": "",
                    "tooltip": "Qwen-Image diffusers模型文件夹的完整路径（如：/data/models/Qwen-Image）"
                }),
                "transformer_path": ("STRING", {
                    "default": "",
                    "tooltip": "Transformer模型文件的完整路径（如：/data/imagegen_models/comfyui-models/qwen_image_bf16.safetensors）"
                }),
                "text_encoder_path": ("STRING", {
                    "default": "",
                    "tooltip": "Text Encoder文件的完整路径，如'/data/imagegen_models/comfyui-models/qwen_2.5_vl_7b.safetensors"
                }),
                "tokenizer_path": ("STRING", {
                    "default": "",
                    "tooltip": "Tokenizer文件夹的完整路径"
                }),
                "vae_path": ("STRING", {
                    "default": "",
                    "tooltip": "VAE文件的完整路径（如：/data/imagegen_models/Qwen-Image/vae/diffusion_pytorch_model.safetensors）"
                }),
            }
        }
    
    RETURN_TYPES = ("model_path",)
    RETURN_NAMES = ("model_path",)
    FUNCTION = "get_qwen_image_config"
    CATEGORY = "Diffusion-Pipe/Model"

    def get_qwen_image_config(self, transformer_path: str = "", text_encoder_path: str = "", 
                             tokenizer_path: str = "", vae_path: str = "", diffusers_path: str = "") -> Tuple[dict]:
        """获取Qwen-Image模型配置"""
        try:
            # 构建Qwen-Image模型配置
            config = {
                "type": "qwen_image",
            }
            
            # 处理diffusers_path
            if diffusers_path.strip():
                config["diffusers_path"] = normalize_wsl_path(diffusers_path.strip())
            
            # 处理transformer_path
            if transformer_path.strip():
                config["transformer_path"] = normalize_wsl_path(transformer_path.strip())
            
            # 处理text_encoder_path
            if text_encoder_path.strip():
                config["text_encoder_path"] = normalize_wsl_path(text_encoder_path.strip())
            
            # 处理tokenizer_path
            if tokenizer_path.strip():
                config["tokenizer_path"] = normalize_wsl_path(tokenizer_path.strip())
            
            # 处理vae_path
            if vae_path.strip():
                config["vae_path"] = normalize_wsl_path(vae_path.strip())
            
            return (config,)
            
        except Exception as e:
            return ({"error": str(e)},)


class QwenImageEditModelNode:
    """Qwen-Image-Edit模型加载节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "diffusers_path": ("STRING", {
                    "default": "",
                    "tooltip": "Qwen-Image或Qwen-Image-Edit diffusers模型文件夹的完整路径（如：/data/models/Qwen-Image-Edit）"
                }),
            },
            "optional": {
                "transformer_path": ("STRING", {
                    "default": "",
                    "tooltip": "Transformer模型文件的完整路径（如：/data/imagegen_models/comfyui-models/qwen_image_edit_bf16.safetensors），仅在使用Qwen-Image diffusers文件夹时需要填写"
                }),
            }
        }
    
    RETURN_TYPES = ("model_path",)
    RETURN_NAMES = ("model_path",)
    FUNCTION = "get_qwen_image_edit_config"
    CATEGORY = "Diffusion-Pipe/Model"

    def get_qwen_image_edit_config(self, diffusers_path: str, transformer_path: str = "") -> Tuple[dict]:
        """获取Qwen-Image-Edit模型配置"""
        try:
            if not diffusers_path.strip():
                return ({"error": "diffusers_path不能为空"},)
            
            # WSL2环境路径处理
            normalized_diffusers_path = normalize_wsl_path(diffusers_path.strip())
            
            # 构建Qwen-Image-Edit模型配置
            config = {
                "type": "qwen_image",
                "diffusers_path": normalized_diffusers_path,
            }
            
            # 处理可选的transformer_path
            if transformer_path.strip():
                config["transformer_path"] = normalize_wsl_path(transformer_path.strip())
            
            return (config,)
            
        except Exception as e:
            return ({"error": str(e)},)


class AdapterConfigNode:
    """适配器配置节点 - 配置LoRA等适配器参数"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "adapter_type": (["lora", "none"], {
                    "default": "lora",
                    "tooltip": "适配器类型，选择lora启用LoRA训练，选择none进行全量微调"
                }),
            },
            "optional": {
                "rank": ("INT", {
                    "default": 32,
                    "min": 16,
                    "max": 1024,
                    "step": 16,
                    "tooltip": "LoRA的秩（rank），控制LoRA的参数量和表达能力，必须是16的倍数"
                }),
                "dtype": (["bfloat16", "float16", "float32"], {
                    "default": "bfloat16",
                    "tooltip": "LoRA权重的数据类型"
                }),
                "init_from_existing": ("STRING", {
                    "default": "",
                    "tooltip": "从已有的LoRA权重初始化（可选，填写完整路径如：/data/diffusion_pipe_training_runs/something/epoch50）"
                }),
            }
        }
    
    RETURN_TYPES = ("ADAPTER_CONFIG",)
    RETURN_NAMES = ("adapter_config",)
    FUNCTION = "generate_adapter_config"
    CATEGORY = "Diffusion-Pipe/Config"

    def generate_adapter_config(self, adapter_type: str, rank: int = 32, dtype: str = "bfloat16",
                               init_from_existing: str = "") -> Tuple[dict]:
        """
        生成适配器配置
        
        Args:
            adapter_type: 适配器类型 ("lora" 或 "none")
            rank: LoRA的rank参数
            dtype: 数据类型
            init_from_existing: 从现有LoRA初始化的路径（可选）
            
        Returns:
            tuple: (adapter_config_dict,)
        """
        try:
            logging.info(f"开始生成适配器配置: type={adapter_type}, rank={rank}, dtype={dtype}")
            
            if adapter_type == "none":
                # 全量微调模式，不使用适配器
                return ({},)
            
            # 构建适配器配置
            config = {
                "type": adapter_type,
                "rank": rank,
                "dtype": dtype
            }
            
            # 处理初始化路径（如果提供）
            if init_from_existing and init_from_existing.strip():
                abs_init_path = normalize_wsl_path(init_from_existing.strip())
                config["init_from_existing"] = abs_init_path
                logging.info(f"添加初始化路径: {abs_init_path}")
            
            # 包装在adapter section中
            adapter_config = {"adapter": config}
            
            logging.info(f"成功生成适配器配置，类型: {adapter_type}，包含 {len(config)} 个参数")
            
            return (adapter_config,)
            
        except Exception as e:
            logging.error(f"适配器配置生成失败: {str(e)}", exc_info=True)
            return ({},)


class OptimizerConfigNode:
    """
    优化器配置节点
    支持多种优化器类型：AdamW, AdamW8bitKahan, Automagic, Prodigy等
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "optimizer_type": (["adamw_optimi", "AdamW8bitKahan", "automagic", "Prodigy"], {
                    "default": "AdamW8bitKahan",
                    "tooltip": "优化器类型选择"
                }),
            },
            "optional": {
                "lr": ("FLOAT", {
                    "default": 2e-5,
                    "min": 1e-8,
                    "max": 1.0,
                    "step": 1e-8,
                    "tooltip": "学习率"
                }),
                "beta1": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Adam优化器的beta1参数"
                }),
                "beta2": ("FLOAT", {
                    "default": 0.99,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Adam优化器的beta2参数"
                }),
                "weight_decay": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "tooltip": "权重衰减系数"
                }),
                "eps": ("FLOAT", {
                    "default": 1e-8,
                    "min": 1e-12,
                    "max": 1e-4,
                    "step": 1e-9,
                    "tooltip": "Adam 的数值稳定性参数"
                }),

            }
        }
    
    RETURN_TYPES = ("OPTIMIZER_CONFIG",)
    RETURN_NAMES = ("optimizer_config",)
    FUNCTION = "generate_optimizer_config"
    CATEGORY = "Diffusion-Pipe/Config"

    def generate_optimizer_config(self, optimizer_type: str, lr: float = 2e-5, 
                                beta1: float = 0.9, beta2: float = 0.99,
                                weight_decay: float = 0.01, eps: float = 1e-8):
        """
        生成优化器配置
        """
        try:
            config = {
                "type": optimizer_type
            }
            
            # 根据不同优化器类型设置参数
            if optimizer_type == "adamw_optimi":
                config.update({
                    "lr": lr,
                    "betas": [beta1, beta2],
                    "weight_decay": weight_decay,
                    "eps": eps
                })
                
            elif optimizer_type == "AdamW8bitKahan":
                config.update({
                    "lr": lr,
                    "betas": [beta1,beta2],
                    "weight_decay": weight_decay
                })
                
            elif optimizer_type == "automagic":
                config.update({
                    "weight_decay": weight_decay
                })
                
            elif optimizer_type == "Prodigy":
                config.update({
                    "lr": lr,
                    "betas": [beta1, beta2],
                    "weight_decay": weight_decay
                })
            
            return (config,)
            
        except Exception as e:
            return ({"error": str(e)},)


class HunyuanImage21ModelNode:
    """HunyuanImage-2.1模型加载节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "transformer_path": ("STRING", {
                    "default": "",
                    "tooltip": "Transformer模型文件的完整路径（如：/data/imagegen_models/comfyui-models/hunyuanimage2.1.safetensors）"
                }),
                "vae_path": ("STRING", {
                    "default": "",
                    "tooltip": "VAE文件的完整路径（如：/data/imagegen_models/comfyui-models/hunyuan_image_2.1_vae_fp16.safetensors）"
                }),
                "text_encoder_path": ("STRING", {
                    "default": "",
                    "tooltip": "Text Encoder文件的完整路径（如：/data/imagegen_models/comfyui-models/qwen_2.5_vl_7b.safetensors）"
                }),
                "byt5_path": ("STRING", {
                    "default": "",
                    "tooltip": "ByT5文件的完整路径（如：/data/imagegen_models/comfyui-models/byt5_small_glyphxl_fp16.safetensors）"
                }),
            }
        }
    
    RETURN_TYPES = ("model_path",)
    RETURN_NAMES = ("model_path",)
    FUNCTION = "get_hunyuan_image21_config"
    CATEGORY = "Diffusion-Pipe/Model"

    def get_hunyuan_image21_config(self, transformer_path: str, vae_path: str, 
                                   text_encoder_path: str, byt5_path: str) -> Tuple[dict]:
        """获取HunyuanImage-2.1模型配置"""
        try:
            # 构建HunyuanImage-2.1模型配置
            config = {
                "type": "hunyuan_image",
            }
            
            # 处理transformer_path
            if transformer_path.strip():
                config["transformer_path"] = normalize_wsl_path(transformer_path.strip())
            else:
                return ({"error": "Transformer路径不能为空"},)
            
            # 处理vae_path
            if vae_path.strip():
                config["vae_path"] = normalize_wsl_path(vae_path.strip())
            else:
                return ({"error": "VAE路径不能为空"},)
            
            # 处理text_encoder_path
            if text_encoder_path.strip():
                config["text_encoder_path"] = normalize_wsl_path(text_encoder_path.strip())
            else:
                return ({"error": "Text Encoder路径不能为空"},)
            
            # 处理byt5_path
            if byt5_path.strip():
                config["byt5_path"] = normalize_wsl_path(byt5_path.strip())
            else:
                return ({"error": "ByT5路径不能为空"},)
            
            return (config,)
            
        except Exception as e:
            return ({"error": str(e)},)

