import os
import json
import logging
from typing import Dict, Any, Tuple

def normalize_wsl_path(path):
    if not path:
        return path
        
    if len(path) >= 3 and path[1] == ':' and path[2] in ['\\', '/']:
        drive_letter = path[0].lower()
        rest_path = path[3:].replace('\\', '/')
        
        if drive_letter == 'z':
            if rest_path.startswith('home/'):
                return f'/{rest_path}'
            else:
                return f'/mnt/{drive_letter}/{rest_path}'
        else:
            return f'/mnt/{drive_letter}/{rest_path}'
    
    elif path.startswith('/'):
        return path.replace('\\', '/')
    
    else:
        return path

try:
    import toml
except ImportError:
    toml = None

class GeneralConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_folder_name": ("STRING", {
                    "default": "training_output",
                    "tooltip": "输出文件夹名称，将自动创建在 @output/ 目录下"
                }),
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
                    "max": 40,
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
            },
            "optional": {
                "adapter_config": ("ADAPTER_CONFIG", {
                    "tooltip": "适配器配置（可选，用于LoRA等适配器训练）"
                }),
                "advanced_config": ("ADVANCED_TRAIN_CONFIG", {
                    "tooltip": "高级训练配置（可选，来自AdvancedTrainConfig节点）"
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
                "eval_dataset_config": ("EVAL_DATASET_CONFIG", {
                    "tooltip": "评估数据集配置（可选，来自EvalDatasetConfig节点）"
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
                         partition_method: str, output_folder_name: str, eval_every_n_epochs: int = 1, 
                         eval_before_first_step: bool = True, eval_micro_batch_size_per_gpu: int = 1,
                         eval_gradient_accumulation_steps: int = 1, save_every_n_epochs: int = 1,
                         checkpoint_every_n_minutes: int = 120, caching_batch_size: int = 1,
                         disable_block_swap_for_eval: bool = False, video_clip_mode: str = "none",
                         adapter_config=None, advanced_config=None, eval_dataset_config=None) -> Tuple[str, str, str]:
        try:
            comfyui_root = os.path.join(os.path.dirname(__file__), "..", "..", "..")
            comfyui_root = os.path.abspath(comfyui_root)
            
            abs_output_dir = os.path.join(comfyui_root, "output", output_folder_name)
            abs_output_dir = os.path.normpath(abs_output_dir)
            
            os.makedirs(abs_output_dir, exist_ok=True)
            
            config_output_dir = abs_output_dir.replace('\\', '/')
            
            config_save_path = os.path.join(os.path.dirname(__file__), "..", "train_config", "trainconfig.toml")
            config_save_path = os.path.normpath(config_save_path)
            
            os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
            
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
            
            if video_clip_mode != "none":
                settings["video_clip_mode"] = video_clip_mode
            
            if gradient_accumulation_steps > 0:
                settings["gradient_accumulation_steps"] = gradient_accumulation_steps
            
            if gradient_clipping > 0:
                settings["gradient_clipping"] = gradient_clipping
            
            if eval_every_n_epochs > 0:
                settings["eval_every_n_epochs"] = eval_every_n_epochs
                settings["eval_before_first_step"] = eval_before_first_step
                settings["eval_micro_batch_size_per_gpu"] = eval_micro_batch_size_per_gpu
                settings["eval_gradient_accumulation_steps"] = eval_gradient_accumulation_steps
            
            if save_every_n_epochs > 0:
                settings["save_every_n_epochs"] = save_every_n_epochs
            
            if checkpoint_every_n_minutes > 0:
                settings["checkpoint_every_n_minutes"] = checkpoint_every_n_minutes
            
            if optimizer_config:
                try:
                    if isinstance(optimizer_config, str):
                        optimizer_dict = json.loads(optimizer_config)
                    else:
                        optimizer_dict = optimizer_config
                    
                    if isinstance(optimizer_dict, dict):
                        if optimizer_dict.get("type") == "AdamW8bitKahan":
                            if "gradient_release" not in optimizer_dict:
                                optimizer_dict["gradient_release"] = False
                        
                        settings["optimizer"] = optimizer_dict
                        logging.info(f"成功合并优化器配置，类型: {optimizer_dict.get('type', 'unknown')}")
                    else:
                        logging.warning("优化器配置不是有效的字典格式")
                except (json.JSONDecodeError, TypeError) as e:
                    logging.warning(f"无法解析优化器配置: {str(e)}")
            else:
                logging.warning("未提供优化器配置，这可能导致训练失败")
            
            if model_config:
                try:
                    if isinstance(model_config, str):
                        model_dict = json.loads(model_config)
                    else:
                        model_dict = model_config
                    
                    if isinstance(model_dict, dict):
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
            
            if dataset_config:
                try:
                    dataset_path = None
                    
                    dataset_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")
                    dataset_dir = os.path.abspath(dataset_dir)  
                    
                    if os.path.exists(dataset_dir):
                        toml_files = [f for f in os.listdir(dataset_dir) if f.endswith('.toml')]
                        if toml_files:
                            latest_file = max(toml_files, key=lambda f: os.path.getmtime(os.path.join(dataset_dir, f)))
                            dataset_path = os.path.abspath(os.path.join(dataset_dir, latest_file)).replace('\\', '/')
                    
                    if not dataset_path:
                        comfyui_root = os.path.join(os.path.dirname(__file__), "..", "..", "..")
                        comfyui_root = os.path.abspath(comfyui_root)
                        default_dataset_path = os.path.join(comfyui_root, "custom_nodes", "Diffusion_pipe_in_ComfyUI", "dataset", "dataset.toml")
                        dataset_path = os.path.normpath(os.path.abspath(default_dataset_path)).replace('\\', '/')
                    
                    settings["dataset"] = dataset_path
                    logging.info(f"数据集配置路径: {dataset_path}")
                    
                except Exception as e:
                    logging.warning(f"处理数据集配置时出错: {str(e)}")
                    comfyui_root = os.path.join(os.path.dirname(__file__), "..", "..", "..")
                    comfyui_root = os.path.abspath(comfyui_root)
                    fallback_path = os.path.join(comfyui_root, "custom_nodes", "Diffusion_pipe_in_ComfyUI", "dataset", "dataset.toml")
                    settings["dataset"] = os.path.normpath(os.path.abspath(fallback_path)).replace('\\', '/')
            else:
                logging.error("未提供数据集配置，这是必需的参数")
                raise ValueError("dataset_config是必需参数，必须连接数据集配置节点")
            
            eval_datasets_list = []
            
            if eval_dataset_config:
                try:
                    eval_dataset_path = None
                    
                    eval_dataset_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")
                    eval_dataset_dir = os.path.abspath(eval_dataset_dir)
                    
                    if os.path.exists(eval_dataset_dir):
                        toml_files = [f for f in os.listdir(eval_dataset_dir) if f.endswith('.toml')]
                        if toml_files:
                            latest_file = max(toml_files, key=lambda f: os.path.getmtime(os.path.join(eval_dataset_dir, f)))
                            eval_dataset_path = os.path.abspath(os.path.join(eval_dataset_dir, latest_file)).replace('\\', '/')
                    
                    if not eval_dataset_path:
                        comfyui_root = os.path.join(os.path.dirname(__file__), "..", "..", "..")
                        comfyui_root = os.path.abspath(comfyui_root)
                        default_eval_dataset_path = os.path.join(comfyui_root, "custom_nodes", "Diffusion_pipe_in_ComfyUI", "dataset", "evaldataset.toml")
                        eval_dataset_path = os.path.normpath(os.path.abspath(default_eval_dataset_path)).replace('\\', '/')
                    
                    if eval_dataset_path:
                        eval_datasets_list.append({
                            'name': 'validation_set',
                            'config': eval_dataset_path
                        })
                        logging.info(f"使用评估数据集配置: {eval_dataset_path}")
                        
                except Exception as e:
                    logging.warning(f"处理评估数据集配置时出错: {str(e)}")
                    comfyui_root = os.path.join(os.path.dirname(__file__), "..", "..", "..")
                    comfyui_root = os.path.abspath(comfyui_root)
                    fallback_path = os.path.join(comfyui_root, "custom_nodes", "Diffusion_pipe_in_ComfyUI", "dataset", "evaldataset.toml")
                    fallback_eval_path = os.path.normpath(os.path.abspath(fallback_path)).replace('\\', '/')
                    eval_datasets_list.append({
                        'name': 'validation_set',
                        'config': fallback_eval_path
                    })
            
            settings["eval_datasets"] = eval_datasets_list
            
            if adapter_config:
                try:
                    if isinstance(adapter_config, str):
                        adapter_dict = json.loads(adapter_config)
                    else:
                        adapter_dict = adapter_config
                    
                    if isinstance(adapter_dict, dict):
                        normalized_adapter_dict = self._normalize_paths_in_dict(adapter_dict)
                        settings.update(normalized_adapter_dict)
                        logging.info(f"成功合并适配器配置，包含 {len(normalized_adapter_dict)} 个参数")
                    else:
                        logging.warning("适配器配置不是有效的字典格式")
                except (json.JSONDecodeError, TypeError) as e:
                    logging.warning(f"无法解析适配器配置: {str(e)}")
            else:
                logging.info("未提供适配器配置，将进行全量微调")
            
            if advanced_config:
                try:
                    if isinstance(advanced_config, str):
                        advanced_dict = json.loads(advanced_config)
                    else:
                        advanced_dict = advanced_config
                    
                    if isinstance(advanced_dict, dict):
                        normalized_advanced_dict = self._normalize_paths_in_dict(advanced_dict)
                        settings.update(normalized_advanced_dict)
                        logging.info(f"成功合并高级配置，包含 {len(normalized_advanced_dict)} 个参数")
                    else:
                        logging.warning("高级配置不是有效的字典格式")
                except (json.JSONDecodeError, TypeError) as e:
                    logging.warning(f"无法解析高级配置: {str(e)}")
            else:
                logging.info("未提供高级配置，使用默认设置")
            
            settings = self._normalize_paths_in_dict(settings)
            
            if toml:
                try:
                    eval_datasets_value = settings.pop('eval_datasets', [])
                    train_config = toml.dumps(settings)
                    train_config = self._replace_quotes_in_toml(train_config)
                    eval_datasets_line = self._format_toml_value('eval_datasets', eval_datasets_value)
                    train_config = eval_datasets_line + '\n' + train_config
                    
                except Exception as e:                   
                    train_config = json.dumps(settings, indent=2, ensure_ascii=False)
                    train_config = train_config.replace('"', "'")
            else:
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
        if isinstance(value, bool):
            return f"{key} = {str(value).lower()}"
        elif isinstance(value, str):
            return f"{key} = '{value}'"
        elif isinstance(value, list):
            if key == 'eval_datasets':
                formatted_items = []
                for item in value:
                    if isinstance(item, dict):
                        dict_parts = []
                        for k, v in item.items():
                            if isinstance(v, str):
                                dict_parts.append(f"{k} = '{v}'")
                            elif isinstance(v, bool):
                                dict_parts.append(f"{k} = {str(v).lower()}")
                            else:
                                dict_parts.append(f"{k} = {v}")
                        formatted_items.append('{' + ', '.join(dict_parts) + '}')
                    elif isinstance(item, str):
                        formatted_items.append(f"'{item}'")
                    else:
                        formatted_items.append(str(item))
                return f"{key} = [{', '.join(formatted_items)}]"
            else:
                if all(isinstance(x, str) for x in value):
                    formatted_list = ', '.join([f"'{x}'" for x in value])
                else:
                    formatted_list = ', '.join([str(x) for x in value])
                return f"{key} = [ {formatted_list},]"
        else:
            return f"{key} = {value}"
    
    def _replace_quotes_in_toml(self, toml_text: str) -> str:
        return toml_text.replace('"', "'")
    
    def _normalize_paths_in_dict(self, data: Any) -> Any:
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
    
