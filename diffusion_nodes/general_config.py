import os
import json
import logging
from typing import Dict, Any, Tuple

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

try:
    import toml
except ImportError:
    toml = None

class GeneralConfig:
    """通用训练设置节点 - 配置训练过程中的通用参数"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "optimizer_config": ("OPTIMIZER_CONFIG", {
                    "tooltip": "优化器配置"
                }),
                "model_config": ("model_config", {
                    "tooltip": "模型配置（来自模型配置节点）"
                }),
                "dataset_config": ("DATASET_CONFIG", {
                    "tooltip": "数据集配置（来自数据集配置节点）"
                }),
                "epochs": ("INT", {
                    "default": 50, 
                    "min": 1, 
                    "max": 1000,
                    "tooltip": "训练轮数"
                }),
                "micro_batch_size_per_gpu": ("INT", {
                    "default": 2, 
                    "min": 1, 
                    "max": 32,
                    "tooltip": "每个GPU的微批次大小"
                }),
                "number_of_gpus": ("INT", {
                    "default": 1, 
                    "min": 1, 
                    "max": 8,
                    "tooltip": "GPU 数量"
                }),
                "pipeline_stages": ("INT", {
                    "default": 1, 
                    "min": 1, 
                    "max": 8,
                    "tooltip": "管道并行阶段数，将模型拆分到的 GPU 数量，应与 GPU 数量匹配"
                }),
                "gradient_accumulation_steps": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 64,
                    "tooltip": "梯度累积步数，0表示自动计算"
                }),
                "gradient_clipping": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 10, 
                    "step": 0,
                    "tooltip": "梯度裁剪阈值，0表示不裁剪"
                }),
                "warmup_steps": ("INT", {
                    "default": 500, 
                    "min": 0, 
                    "max": 5000,
                    "tooltip": "学习率预热步数"
                }),
                "blocks_to_swap": ("INT", {
                    "default": 20, 
                    "min": 0, 
                    "max": 100,
                    "tooltip": "要交换的块数量"
                }),
                "activation_checkpointing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "激活检查点，节省显存，通常启用"
                }),
                "save_dtype": (["bfloat16", "float16", "float32"], {
                    "default": "bfloat16",
                    "tooltip": "保存模型时的数据类型"
                }),
                "partition_method": (["parameters", "uniform", "memory"], {
                    "default": "parameters",
                    "tooltip": "分区方法"
                }),
                "output_dir": ("STRING", {
                    "default": "\ComfyUI\custom_nodes\Diffusion_pipe_in_ComfyUI\output",
                    "tooltip": "训练输出目录路径，指向你的~\ComfyUI\custom_nodes\Diffusion_pipe_in_ComfyUI\output"
                }),
                "config_save_path": ("STRING", {
                    "default": "/ComfyUI/custom_nodes/Diffusion_pipe_in_ComfyUI/train_config/trainconfig.toml",
                    "multiline": False,
                    "tooltip": "训练配置文件保存路径"
                }),
            },
            "optional": {
                "adapter_config": ("ADAPTER_CONFIG", {
                    "tooltip": "适配器配置（可选，用于LoRA等适配器训练）"
                }),
                "eval_every_n_epochs": ("INT", {
                    "default": 1, 
                    "min": 0, 
                    "max": 100,
                    "tooltip": "每N个epoch评估一次，0表示不评估"
                }),
                "eval_before_first_step": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否在第一步之前评估"
                }),
                "eval_micro_batch_size_per_gpu": ("INT", {
                    "default": 1, 
                    "min": 1, 
                    "max": 32,
                    "tooltip": "评估时每个GPU的微批次大小"
                }),
                "eval_gradient_accumulation_steps": ("INT", {
                    "default": 1, 
                    "min": 1, 
                    "max": 64,
                    "tooltip": "评估时的梯度累积步数"
                }),
                "save_every_n_epochs": ("INT", {
                    "default": 1, 
                    "min": 0, 
                    "max": 100,
                    "tooltip": "每N个epoch保存一次，0表示禁用"
                }),
                "checkpoint_every_n_minutes": ("INT", {
                    "default": 120, 
                    "min": 0, 
                    "max": 1440,
                    "tooltip": "每N分钟保存检查点，0表示禁用"
                }),
                "caching_batch_size": ("INT", {
                    "default": 1, 
                    "min": 1, 
                    "max": 32,
                    "tooltip": "预缓存时的批次大小，影响内存使用"
                }),
                "disable_block_swap_for_eval": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "评估时是否禁用块交换"
                }),
                "video_clip_mode": (["none", "single_beginning", "single_middle", "multiple_overlapping"], {
                    "default": "none",
                    "tooltip": "仅适用于视频模型训练。视频帧提取模式 - none:不使用视频模式, single_beginning:从视频开头提取一个片段, single_middle:从视频中间提取一个片段, multiple_overlapping:提取多个可能重叠的片段覆盖整个视频"
                }),
                "eval_datasets": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "评估数据集列表，每行一个数据集名称或路径。留空则使用默认的空列表。支持相对路径和绝对路径。"
                }),
            }
        }
    
    RETURN_TYPES = ("TRAIN_CONFIG", "STRING", "config_path")
    RETURN_NAMES = ("train_config", "output_dir", "config_path")
    FUNCTION = "generate_settings"
    CATEGORY = "Diffusion-Pipe/Config"

    def generate_settings(self, optimizer_config, model_config, dataset_config, epochs: int, micro_batch_size_per_gpu: int, number_of_gpus: int, 
                         pipeline_stages: int, gradient_accumulation_steps: int, gradient_clipping: int, 
                         warmup_steps: int, blocks_to_swap: int, activation_checkpointing: bool, save_dtype: str,
                         partition_method: str, output_dir: str, config_save_path: str, eval_every_n_epochs: int = 1, 
                         eval_before_first_step: bool = True, eval_micro_batch_size_per_gpu: int = 1,
                         eval_gradient_accumulation_steps: int = 1, save_every_n_epochs: int = 1,
                         checkpoint_every_n_minutes: int = 120, caching_batch_size: int = 1,
                         disable_block_swap_for_eval: bool = False, video_clip_mode: str = "none",
                         eval_datasets: str = "", adapter_config=None) -> Tuple[str, str, str]:
        """生成通用训练设置"""
        try:
            # 处理输出目录路径 - 先使用WSL路径规范化
            normalized_output_dir = normalize_wsl_path(output_dir)
            
            if normalized_output_dir == "/output" or normalized_output_dir.startswith("/output/"):
                # 默认输出路径：转换为相对于ComfyUI根目录的绝对路径
                comfyui_root = os.path.join(os.path.dirname(__file__), "..", "..", "..")
                comfyui_root = os.path.abspath(comfyui_root)
                abs_output_dir = os.path.join(comfyui_root, normalized_output_dir.lstrip("/"))
                abs_output_dir = os.path.normpath(abs_output_dir)
            else:
                # 用户提供的完整路径，直接使用已规范化的路径
                if os.path.isabs(normalized_output_dir):
                    abs_output_dir = os.path.normpath(os.path.expanduser(normalized_output_dir))
                else:
                    # 相对路径，相对于当前工作目录
                    abs_output_dir = os.path.normpath(os.path.abspath(os.path.expanduser(normalized_output_dir)))
            
            # 将输出目录路径转换为正斜杠格式（用于配置文件）
            config_output_dir = abs_output_dir.replace('\\', '/')
            
            # 构建设置字典
            settings = {
                "epochs": epochs,
                "micro_batch_size_per_gpu": micro_batch_size_per_gpu,
                "number_of_gpus": number_of_gpus,
                "pipeline_stages": pipeline_stages,
                "warmup_steps": warmup_steps,
                "blocks_to_swap": blocks_to_swap,
                "activation_checkpointing": activation_checkpointing,
                "save_dtype": save_dtype,
                "caching_batch_size": caching_batch_size,
                "partition_method": partition_method,
                "output_dir": config_output_dir,
                "disable_block_swap_for_eval": disable_block_swap_for_eval,
            }
            
            # 处理视频剪辑模式 - 仅在非none时添加到配置中
            if video_clip_mode != "none":
                settings["video_clip_mode"] = video_clip_mode
            
            # 处理评估数据集 - 解析用户输入的多行文本
            eval_datasets_list = []
            if eval_datasets and eval_datasets.strip():
                # 按行分割，去除空行和前后空格
                lines = [line.strip() for line in eval_datasets.strip().split('\n')]
                # 将路径中的反斜杠转换为正斜杠
                eval_datasets_list = [line.replace('\\', '/') for line in lines if line]
            settings["eval_datasets"] = eval_datasets_list
            
            # 处理梯度累积步数 - 0表示不设置此参数（让系统自动计算）
            if gradient_accumulation_steps > 0:
                settings["gradient_accumulation_steps"] = gradient_accumulation_steps
            
            # 处理梯度裁剪 - 0表示不设置此参数（不进行梯度裁剪）
            if gradient_clipping > 0:
                settings["gradient_clipping"] = gradient_clipping
            
            # 处理评估相关参数 - 0表示不评估
            if eval_every_n_epochs > 0:
                settings["eval_every_n_epochs"] = eval_every_n_epochs
                settings["eval_before_first_step"] = eval_before_first_step
                settings["eval_micro_batch_size_per_gpu"] = eval_micro_batch_size_per_gpu
                settings["eval_gradient_accumulation_steps"] = eval_gradient_accumulation_steps
            
            # 处理保存相关参数 - 0表示不定期保存
            if save_every_n_epochs > 0:
                settings["save_every_n_epochs"] = save_every_n_epochs
            
            if checkpoint_every_n_minutes > 0:
                settings["checkpoint_every_n_minutes"] = checkpoint_every_n_minutes
            
            # 合并优化器配置（必需）
            if optimizer_config:
                try:
                    # 如果optimizer_config是字符串，尝试解析为JSON
                    if isinstance(optimizer_config, str):
                        optimizer_dict = json.loads(optimizer_config)
                    else:
                        optimizer_dict = optimizer_config
                    
                    # 将优化器配置合并到设置中
                    if isinstance(optimizer_dict, dict):
                        # 为某些优化器添加默认参数
                        if optimizer_dict.get("type") == "AdamW8bitKahan":
                            # 为AdamW8bitKahan添加gradient_release参数
                            if "gradient_release" not in optimizer_dict:
                                optimizer_dict["gradient_release"] = False
                        
                        # 添加optimizer section以匹配TOML配置格式
                        settings["optimizer"] = optimizer_dict
                        logging.info(f"成功合并优化器配置，类型: {optimizer_dict.get('type', 'unknown')}")
                    else:
                        logging.warning("优化器配置不是有效的字典格式")
                except (json.JSONDecodeError, TypeError) as e:
                    logging.warning(f"无法解析优化器配置: {str(e)}")
            else:
                logging.warning("未提供优化器配置，这可能导致训练失败")
            
            # 合并模型配置（必需）
            if model_config:
                try:
                    # 如果model_config是字符串，尝试解析为JSON
                    if isinstance(model_config, str):
                        model_dict = json.loads(model_config)
                    else:
                        model_dict = model_config
                    
                    # 将模型配置添加到设置中
                    if isinstance(model_dict, dict):
                        # 规范化模型配置中的路径
                        normalized_model_dict = self._normalize_paths_in_dict(model_dict)
                        settings["model"] = normalized_model_dict
                        logging.info(f"成功合并模型配置，类型: {normalized_model_dict.get('type', 'unknown')}")
                    else:
                        logging.warning("模型配置不是有效的字典格式")
                except (json.JSONDecodeError, TypeError) as e:
                    logging.warning(f"无法解析模型配置: {str(e)}")
            else:
                logging.error("未提供模型配置，这是必需的参数")
                raise ValueError("model_config是必需参数，必须连接模型配置节点")
            
            # 处理数据集配置（必需）
            if dataset_config:
                try:
                    # 数据集配置应该包含保存路径信息，我们需要从中提取数据集文件路径
                    # dataset_config 是配置内容字符串，我们需要找到对应的输出路径
                    # 从 GeneralDatasetConfig 节点的 output_path 参数推导数据集文件路径
                    
                    # 简单的方法：通过解析配置内容查找数据集路径模式
                    # 或者直接使用固定的数据集文件路径（基于约定）
                    dataset_path = None
                    
                    # 方法1：尝试从当前工作目录找到最近保存的数据集配置文件
                    dataset_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")
                    dataset_dir = os.path.abspath(dataset_dir)  # 标准化为绝对路径
                    
                    if os.path.exists(dataset_dir):
                        # 查找最新的 .toml 数据集文件
                        toml_files = [f for f in os.listdir(dataset_dir) if f.endswith('.toml')]
                        if toml_files:
                            # 使用最新修改的文件
                            latest_file = max(toml_files, key=lambda f: os.path.getmtime(os.path.join(dataset_dir, f)))
                            dataset_path = os.path.abspath(os.path.join(dataset_dir, latest_file)).replace('\\', '/')
                    
                    # 方法2：如果没有找到文件，使用默认路径
                    if not dataset_path:
                        # 计算相对于ComfyUI根目录的标准化路径
                        comfyui_root = os.path.join(os.path.dirname(__file__), "..", "..", "..")
                        comfyui_root = os.path.abspath(comfyui_root)
                        default_dataset_path = os.path.join(comfyui_root, "custom_nodes", "Diffusion_pipe_in_ComfyUI", "dataset", "dataset.toml")
                        dataset_path = os.path.normpath(os.path.abspath(default_dataset_path)).replace('\\', '/')
                    
                    # 添加数据集引用到配置中
                    settings["dataset"] = dataset_path
                    logging.info(f"数据集配置路径: {dataset_path}")
                    
                except Exception as e:
                    logging.warning(f"处理数据集配置时出错: {str(e)}")
                    # 使用默认数据集路径作为后备方案
                    comfyui_root = os.path.join(os.path.dirname(__file__), "..", "..", "..")
                    comfyui_root = os.path.abspath(comfyui_root)
                    fallback_path = os.path.join(comfyui_root, "custom_nodes", "Diffusion_pipe_in_ComfyUI", "dataset", "dataset.toml")
                    settings["dataset"] = os.path.normpath(os.path.abspath(fallback_path)).replace('\\', '/')
            else:
                logging.error("未提供数据集配置，这是必需的参数")
                raise ValueError("dataset_config是必需参数，必须连接数据集配置节点")
            
            # 合并适配器配置（如果提供）
            if adapter_config:
                try:
                    # 如果adapter_config是字符串，尝试解析为JSON
                    if isinstance(adapter_config, str):
                        adapter_dict = json.loads(adapter_config)
                    else:
                        adapter_dict = adapter_config
                    
                    # 将适配器配置合并到设置中
                    if isinstance(adapter_dict, dict):
                        # 规范化适配器配置中的路径
                        normalized_adapter_dict = self._normalize_paths_in_dict(adapter_dict)
                        settings.update(normalized_adapter_dict)
                        logging.info(f"成功合并适配器配置，包含 {len(normalized_adapter_dict)} 个参数")
                    else:
                        logging.warning("适配器配置不是有效的字典格式")
                except (json.JSONDecodeError, TypeError) as e:
                    logging.warning(f"无法解析适配器配置: {str(e)}")
            else:
                logging.info("未提供适配器配置，将进行全量微调")
            
            # 最终规范化所有路径为正斜杠格式
            settings = self._normalize_paths_in_dict(settings)
            
            # 输出为TOML格式
            if toml:
                try:
                    train_config = toml.dumps(settings)
                    # 将双引号替换为单引号
                    train_config = self._replace_quotes_in_toml(train_config)
                except Exception as e:                   
                    train_config = json.dumps(settings, indent=2, ensure_ascii=False)
                    # JSON格式也替换为单引号
                    train_config = train_config.replace('"', "'")
            else:
                # 如果没有toml库，使用自定义TOML格式化
                train_config = self._format_as_toml(settings)
            
            if config_save_path:
                try:
                    normalized_save_path = normalize_wsl_path(config_save_path)
                    
                    save_dir = os.path.dirname(normalized_save_path)
                    if save_dir:
                        os.makedirs(save_dir, exist_ok=True)
                    
                    import datetime
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    current_cwd = os.getcwd()
                    # 输出配置信息到控制台
                    print(f"[通用训练配置] 配置文件已保存到: {normalized_save_path}")
                    print(f"[通用训练配置] 生成时间: {current_time}")
                    print(f"[通用训练配置] 当前工作目录: {current_cwd}")
                    
                    config_with_path = train_config
                    
                    with open(normalized_save_path, 'w', encoding='utf-8') as f:
                        f.write(config_with_path)
                    
                    display_path = normalized_save_path.replace('\\', '/')
                    logging.info(f"训练配置文件已保存: {display_path}")
                    
                    return (config_with_path, abs_output_dir, normalized_save_path)
                except Exception as e:
                    error_msg = f"保存训练配置文件失败: {str(e)}"
                    print(error_msg)
                    logging.error(error_msg)
            else:
                print("未指定配置保存路径，仅生成配置内容")
            
            return (train_config, abs_output_dir, config_save_path or "")
            
        except Exception as e:
            logging.error(f"通用设置生成失败: {str(e)}")
            return ("{}", "", "")
    
    def _format_as_toml(self, settings: dict) -> str:
        """自定义TOML格式化方法"""
        toml_lines = []
        
        for key, value in settings.items():
            if not isinstance(value, dict):
                toml_lines.append(self._format_toml_value(key, value))
        
        for key, value in settings.items():
            if isinstance(value, dict):
                toml_lines.append(f"\n[{key}]")
                for sub_key, sub_value in value.items():
                    toml_lines.append(self._format_toml_value(sub_key, sub_value))
        
        return '\n'.join(toml_lines)
    
    def _format_toml_value(self, key: str, value) -> str:
        """格式化TOML键值对"""
        if isinstance(value, bool):
            return f"{key} = {str(value).lower()}"
        elif isinstance(value, str):
            return f"{key} = '{value}'"
        elif isinstance(value, list):
            if all(isinstance(x, str) for x in value):
                formatted_list = ', '.join([f"'{x}'" for x in value])
            else:
                formatted_list = ', '.join([str(x) for x in value])
            return f"{key} = [ {formatted_list},]"
        else:
            return f"{key} = {value}"
    
    def _replace_quotes_in_toml(self, toml_text: str) -> str:
        """将TOML文本中的双引号替换为单引号"""
        return toml_text.replace('"', "'")
    
    def _normalize_paths_in_dict(self, data: Any) -> Any:
        """递归地将字典中所有路径字符串的反斜杠转换为正斜杠"""
        if isinstance(data, dict):
            return {key: self._normalize_paths_in_dict(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._normalize_paths_in_dict(item) for item in data]
        elif isinstance(data, str):
            if ('\\' in data or '/' in data) and ('.' in data or data.startswith('/') or (len(data) > 1 and data[1] == ':')):
                return data.replace('\\', '/')
            return data
        else:
            return data 
    
