# 导入所有节点类
try:
    # 尝试相对导入 (在 ComfyUI 中作为包导入时)
    from .diffusion_nodes.GeneralDatasetConfig import GeneralDatasetConfig
    from .diffusion_nodes.dataset_tools import GeneralDatasetPathNode, EditModelDatasetPathNode, FrameBucketsNode
    from .diffusion_nodes.general_config import GeneralConfig
    from .diffusion_nodes.model_config import ModelConfig
    from .diffusion_nodes.dataset_tools import ArBucketsNode
    from .diffusion_nodes.model_tools import SDXLModelNode,Wan22ModelNode,QwenImageEditModelNode,FluxKontextModelNode,QwenImageModelNode,CosmosPredict2ModelNode,OmniGen2ModelNode, FluxModelNode,SD3ModelNode, LTXVideoModelNode, HunyuanVideoModelNode,HiDreamModelNode,ChromaModelNode,CosmosModelNode,Lumina2ModelNode,Wan21ModelNode
    from .diffusion_nodes.model_tools import AdapterConfigNode, OptimizerConfigNode
    from .diffusion_nodes.start import Train
    from .diffusion_nodes.train_monitor import TensorBoardMonitor
    from .diffusion_nodes.output_dir_passthrough import OutputDirPassthrough
except ImportError:
    # 如果相对导入失败，尝试绝对导入 (直接运行时)
    import os
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    
    from diffusion_nodes.GeneralDatasetConfig import GeneralDatasetConfig
    from diffusion_nodes.dataset_tools import GeneralDatasetPathNode, EditModelDatasetPathNode, FrameBucketsNode
    from diffusion_nodes.general_config import GeneralConfig
    from diffusion_nodes.model_config import ModelConfig
    from diffusion_nodes.dataset_tools import ArBucketsNode
    from diffusion_nodes.model_tools import SDXLModelNode,Wan22ModelNode,QwenImageEditModelNode,FluxKontextModelNode,QwenImageModelNode,CosmosPredict2ModelNode,OmniGen2ModelNode, FluxModelNode,SD3ModelNode, LTXVideoModelNode, HunyuanVideoModelNode,HiDreamModelNode,ChromaModelNode,CosmosModelNode,Lumina2ModelNode,Wan21ModelNode
    from diffusion_nodes.model_tools import AdapterConfigNode, OptimizerConfigNode
    from diffusion_nodes.start import Train
    from diffusion_nodes.train_monitor import TensorBoardMonitor
    from diffusion_nodes.output_dir_passthrough import OutputDirPassthrough

NODE_CLASS_MAPPINGS = {
   
    
    # 数据集配置节点
    "GeneralDatasetConfig": GeneralDatasetConfig,
    "GeneralDatasetPathNode": GeneralDatasetPathNode,
    "EditModelDatasetPathNode": EditModelDatasetPathNode,
    "FrameBucketsNode": FrameBucketsNode,
    "ArBucketsNode": ArBucketsNode,
    
    # 模型工具节点
    "SDXLModelNode": SDXLModelNode,
    "Wan22ModelNode": Wan22ModelNode,
    "QwenImageEditModelNode": QwenImageEditModelNode,
    "FluxModelNode": FluxModelNode,
    "SD3ModelNode": SD3ModelNode,
    "CosmosPredict2ModelNode": CosmosPredict2ModelNode,
    "OmniGen2ModelNode": OmniGen2ModelNode,
    "QwenImageModelNode": QwenImageModelNode,
    "FluxKontextModelNode": FluxKontextModelNode,
    "LTXVideoModelNode": LTXVideoModelNode,
    "HunyuanVideoModelNode": HunyuanVideoModelNode,
    "CosmosModelNode": CosmosModelNode,
    "Lumina2ModelNode": Lumina2ModelNode,
    "Wan21ModelNode": Wan21ModelNode,
    "ChromaModelNode": ChromaModelNode,
    "HiDreamModelNode": HiDreamModelNode,
    
    # 配置生成节点
    "GeneralConfig": GeneralConfig,
    "ModelConfig": ModelConfig,
    "AdapterConfigNode": AdapterConfigNode,
    "OptimizerConfigNode": OptimizerConfigNode,
    # 训练控制节点
    "Train": Train,
    
    # 监控节点
    "TensorBoardMonitor": TensorBoardMonitor,
    
    # 工具节点
    "OutputDirPassthrough": OutputDirPassthrough,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    
    # 数据集配置节点
    "GeneralDatasetConfig": "通用数据集配置",
    "GeneralDatasetPathNode": "通用数据集路径",
    "EditModelDatasetPathNode": "编辑模型数据集路径",
    "FrameBucketsNode": "帧数分桶配置",
    "ArBucketsNode": "宽高比分桶配置",
    
    # 模型工具节点
    "SDXLModelNode": "SDXL模型配置器",
    "Wan22ModelNode": "Wan2.2模型配置器",
    "FluxModelNode": "Flux模型配置器",
    "SD3ModelNode": "SD3模型配置器",
    "CosmosPredict2ModelNode": "Cosmos Predict2模型配置器",
    "OmniGen2ModelNode": "OmniGen2模型配置器",
    "QwenImageModelNode": "Qwen-Image模型配置器",
    "QwenImageEditModelNode": "Qwen-Image-Edit模型配置器",
    "FluxKontextModelNode": "Flux Kontext模型配置器",
    "LTXVideoModelNode": "LTX-Video模型配置器",
    "HunyuanVideoModelNode": "HunyuanVideo模型配置器",
    "HiDreamModelNode": "HiDream模型配置器",
    "CosmosModelNode": "Cosmos模型配置器",
    "Lumina2ModelNode": "Lumina2模型配置器",
    "Wan21ModelNode": "Wan2.1模型配置器",
    "ChromaModelNode": "Chroma模型配置器",
    # 配置生成节点
    "GeneralConfig": "通用训练设置",
    "ModelConfig": "模型配置",
    "AdapterConfigNode": "适配器配置",
    "OptimizerConfigNode": "优化器配置",
    # 训练控制节点
    "Train": "训练启动器",
    
    # 监控节点
    "TensorBoardMonitor": "TensorBoard监控器",
    
    # 工具节点
    "OutputDirPassthrough": "输出目录传递",
}

# 导出节点
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"] 