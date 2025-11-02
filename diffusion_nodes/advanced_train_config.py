import json
import logging
from typing import Dict, Any, Tuple

class AdvancedTrainConfig:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            },
            "optional": {
                "max_steps": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100000,
                    "tooltip": "最大训练步数，0表示不限制（使用epochs）"
                }),
                "force_constant_lr": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.00000001,
                    "tooltip": "强制使用恒定学习率，0.0表示不使用"
                }),
                "lr_scheduler": (["constant", "linear"], {
                    "default": "constant", 
                    "tooltip": "学习率调度器类型"
                }),
                
                "pseudo_huber_c": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "伪Huber损失常数c，0.0表示不使用，仅适用于默认损失函数的模型"
                }),
                
                "map_num_proc": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 64,
                    "tooltip": "缓存数据集时的并行进程数，0表示使用默认值，如果你有很多内核和多个GPU，提高这一点可以提高吞吐量"
                }),
                
                "compile": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "使用torch.compile编译模型以加速训练，没有在所有模型上测试过"
                }),
                
                "steps_per_print": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 1000,
                    "tooltip": "每N步打印一次日志"
                }),
                "x_axis_examples": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "在TensorBoard/WandB中使用样本数作为X轴而非步数"
                }),
                
                "save_every_n_steps": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "tooltip": "每N步保存一次模型，0表示禁用，不同于save_every_n_epochs，这个是基于步数保存"
                }),
                "eval_every_n_steps": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "tooltip": "每N步评估一次，0表示禁用，不同于eval_every_n_epochs，这个是基于步数评估"
                }),
                "checkpoint_every_n_epochs": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "tooltip": "每N个epoch保存检查点，0表示禁用,建议启用，否则可能丢失部分训练进度"
                }),
                
                
                "partition_split": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "手动分区分割点，如'10,20'表示层0-9在GPU0，10-19在GPU1，其余在GPU2"
                }),
                
                "reentrant_activation_checkpointing": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "使用可重入激活检查点方法,当使用分布式训练时启用（pipelinestage>1）"
                }),
                
            }
        }
    
    RETURN_TYPES = ("ADVANCED_TRAIN_CONFIG",)
    RETURN_NAMES = ("advanced_config",)
    FUNCTION = "generate_advanced_config"
    CATEGORY = "Diffusion-Pipe/Config"

    def generate_advanced_config(self, **kwargs) -> Tuple[str]:
        try:
            advanced_config = {}
            
            if kwargs.get('max_steps', 0) > 0:
                advanced_config['max_steps'] = kwargs['max_steps']
            
            if kwargs.get('force_constant_lr', 0.0) > 0.0:
                advanced_config['force_constant_lr'] = kwargs['force_constant_lr']
            
            if kwargs.get('lr_scheduler', 'constant') != 'constant':
                advanced_config['lr_scheduler'] = kwargs['lr_scheduler']
            
            if kwargs.get('pseudo_huber_c', 0.0) > 0.0:
                advanced_config['pseudo_huber_c'] = kwargs['pseudo_huber_c']
            
            if kwargs.get('map_num_proc', 0) > 0:
                advanced_config['map_num_proc'] = kwargs['map_num_proc']
            
            if kwargs.get('compile', False):
                advanced_config['compile'] = True
            
            if kwargs.get('steps_per_print', 1) != 1:
                advanced_config['steps_per_print'] = kwargs['steps_per_print']
            
            if kwargs.get('x_axis_examples', False):
                advanced_config['x_axis_examples'] = True
            
            if kwargs.get('save_every_n_steps', 0) > 0:
                advanced_config['save_every_n_steps'] = kwargs['save_every_n_steps']
            
            if kwargs.get('eval_every_n_steps', 0) > 0:
                advanced_config['eval_every_n_steps'] = kwargs['eval_every_n_steps']
            
            if kwargs.get('checkpoint_every_n_epochs', 0) > 0:
                advanced_config['checkpoint_every_n_epochs'] = kwargs['checkpoint_every_n_epochs']
            
            partition_split = kwargs.get('partition_split', '').strip()
            if partition_split:
                try:
                    split_points = [int(x.strip()) for x in partition_split.split(',') if x.strip()]
                    if split_points:
                        advanced_config['partition_method'] = 'manual'
                        advanced_config['partition_split'] = split_points
                except ValueError as e:
                    logging.warning(f"无法解析partition_split参数 '{partition_split}': {e}")
            
            if kwargs.get('reentrant_activation_checkpointing', False):
                advanced_config['reentrant_activation_checkpointing'] = True
            
            config_json = json.dumps(advanced_config, indent=2, ensure_ascii=False)
            logging.info(f"生成高级训练配置，包含 {len(advanced_config)} 个参数")
            
            return (config_json,)
            
        except Exception as e:
            logging.error(f"高级训练配置生成失败: {str(e)}")
            return ("{}",) 