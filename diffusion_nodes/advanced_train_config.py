import json
import logging
from typing import Dict, Any, Tuple

class AdvancedTrainConfig:
    """高级训练配置节点 - 包含main_example.toml中未在GeneralConfig映射的参数"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            },
            "optional": {
                # 训练控制参数
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
                
                # 损失函数参数
                "pseudo_huber_c": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "伪Huber损失常数c，0.0表示不使用，仅适用于默认损失函数的模型"
                }),
                
                # 数据处理参数
                "map_num_proc": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 64,
                    "tooltip": "缓存数据集时的并行进程数，0表示使用默认值，如果你有很多内核和多个GPU，提高这一点可以提高吞吐量"
                }),
                
                # 模型优化
                "compile": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "使用torch.compile编译模型以加速训练，没有在所有模型上测试过"
                }),
                
                # 日志和显示
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
                
                # 保存频率（基于步数）
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
                
                
                # 分区控制（手动分区）
                "partition_split": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "手动分区分割点，如'10,20'表示层0-9在GPU0，10-19在GPU1，其余在GPU2"
                }),
                
                # 高级激活检查点
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
        """生成高级训练配置"""
        try:
            advanced_config = {}
            
            # 训练控制参数
            if kwargs.get('max_steps', 0) > 0:
                advanced_config['max_steps'] = kwargs['max_steps']
            
            if kwargs.get('force_constant_lr', 0.0) > 0.0:
                advanced_config['force_constant_lr'] = kwargs['force_constant_lr']
            
            if kwargs.get('lr_scheduler', 'constant') != 'constant':
                advanced_config['lr_scheduler'] = kwargs['lr_scheduler']
            
            # 损失函数参数
            if kwargs.get('pseudo_huber_c', 0.0) > 0.0:
                advanced_config['pseudo_huber_c'] = kwargs['pseudo_huber_c']
            
            # 数据处理参数
            if kwargs.get('map_num_proc', 0) > 0:
                advanced_config['map_num_proc'] = kwargs['map_num_proc']
            
            # 模型优化
            if kwargs.get('compile', False):
                advanced_config['compile'] = True
            
            # 日志和显示
            if kwargs.get('steps_per_print', 1) != 1:
                advanced_config['steps_per_print'] = kwargs['steps_per_print']
            
            if kwargs.get('x_axis_examples', False):
                advanced_config['x_axis_examples'] = True
            
            # 保存频率（基于步数）
            if kwargs.get('save_every_n_steps', 0) > 0:
                advanced_config['save_every_n_steps'] = kwargs['save_every_n_steps']
            
            if kwargs.get('eval_every_n_steps', 0) > 0:
                advanced_config['eval_every_n_steps'] = kwargs['eval_every_n_steps']
            
            if kwargs.get('checkpoint_every_n_epochs', 0) > 0:
                advanced_config['checkpoint_every_n_epochs'] = kwargs['checkpoint_every_n_epochs']
            
            # 分区控制
            partition_split = kwargs.get('partition_split', '').strip()
            if partition_split:
                try:
                    # 解析分割点字符串，如 "10,20" -> [10, 20]
                    split_points = [int(x.strip()) for x in partition_split.split(',') if x.strip()]
                    if split_points:
                        advanced_config['partition_method'] = 'manual'
                        advanced_config['partition_split'] = split_points
                except ValueError as e:
                    logging.warning(f"无法解析partition_split参数 '{partition_split}': {e}")
            
            # 高级激活检查点
            if kwargs.get('reentrant_activation_checkpointing', False):
                advanced_config['reentrant_activation_checkpointing'] = True
            
            # 输出为JSON字符串
            config_json = json.dumps(advanced_config, indent=2, ensure_ascii=False)
            logging.info(f"生成高级训练配置，包含 {len(advanced_config)} 个参数")
            
            return (config_json,)
            
        except Exception as e:
            logging.error(f"高级训练配置生成失败: {str(e)}")
            return ("{}",) 