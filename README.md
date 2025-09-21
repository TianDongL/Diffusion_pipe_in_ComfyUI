 # Diffusion-Pipe ComfyUI 自定义节点

## 项目简介

Diffusion-Pipe ComfyUI 自定义节点是一个强大的扩展插件，为 ComfyUI 提供了完整的 Diffusion 模型训练和微调功能。这个项目允许用户在 ComfyUI 的图形界面中配置和启动各种先进 AI 模型的训练，支持 LoRA 和全量微调，涵盖了当前最热门的图像生成和视频生成模型。

### 核心特性

- 🎯 **可视化训练配置**: 通过 ComfyUI 节点图形化配置训练参数
- 🚀 **多模型支持**: 支持 20+ 种最新的 Diffusion 模型
- 💾 **灵活训练方式**: 支持 LoRA 训练和全量微调
- ⚡ **高性能训练**: 基于 DeepSpeed 的分布式训练支持
- 📊 **实时监控**: 集成 TensorBoard 监控训练过程
- 🔧 **WSL2 优化**: 专门优化的 Windows WSL2 环境支持
- 🎥 **视频训练**: 支持视频生成模型的训练
- 🖼️ **图像编辑**: 支持图像编辑模型的训练

## 系统要求

### 硬件要求
- **GPU**: NVIDIA RTX 3090/4090 或更高（24GB+ VRAM 推荐）
- **内存**: 32GB+ RAM 推荐
- **存储**: 100GB+ 可用空间（用于数据集和模型存储）

### 软件要求
- **操作系统**: Linux / Windows 10/11 + WSL2
- **Python**: 3.8+
- **CUDA**: 11.8+ 或 12.0+
- **ComfyUI**: 最新版本

## 安装指南

### 1. 安装 ComfyUI
确保你在Linux或者WSL2系统上拥有ComfyUI，参考https://docs.comfy.org/installation/manual_install

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# 安装依赖
pip install -r requirements.txt
```

### 2. 安装 Diffusion-Pipe 节点
```bash
# 进入 ComfyUI 自定义节点目录
cd custom_nodes

# 克隆本项目
git clone https://github.com/TianDongL/Diffusion_pipe_in_ComfyUI.git

# 安装依赖
cd Diffusion_pipe_in_ComfyUI
pip install -r requirements.txt
```

### 3. WSL2 环境配置（Windows 用户）
确保在 WSL2 环境中正确配置 CUDA 和 GPU 驱动：

```bash
# 检查 CUDA 版本
nvcc --version

# 检查 GPU 状态
nvidia-smi
```

## 支持的模型

本插件支持超过 20 种最新的 Diffusion 模型，包括：

### 🖼️ 图像生成模型
| 模型 | LoRA | 全量微调 | FP8/量化 | 描述 |
|------|------|----------|----------|------|
| **SDXL** | ✅ | ✅ | ❌ | Stable Diffusion XL |
| **Flux** | ✅ | ✅ | ✅ | Black Forest Labs Flux |
| **SD3** | ✅ | ❌ | ✅ | Stable Diffusion 3 |
| **Lumina2** | ✅ | ✅ | ❌ | 高质量图像生成 |
| **Qwen-Image** | ✅ | ✅ | ✅ | 阿里通义千问图像模型 |
| **HiDream** | ✅ | ❌ | ✅ | 高分辨率图像生成 |
| **Chroma** | ✅ | ✅ | ✅ | 基于 Flux 的改进模型 |
| **Cosmos** | ✅ | ❌ | ❌ | NVIDIA Cosmos 模型 |
| **Cosmos-Predict2** | ✅ | ✅ | ✅ | Cosmos 第二代 |
| **OmniGen2** | ✅ | ❌ | ❌ | 多功能生成模型 |

### 🎬 视频生成模型
| 模型 | LoRA | 全量微调 | FP8/量化 | 描述 |
|------|------|----------|----------|------|
| **LTX-Video** | ✅ | ❌ | ❌ | 文本到视频生成 |
| **HunyuanVideo** | ✅ | ❌ | ✅ | 腾讯混元视频模型 |
| **Wan2.1** | ✅ | ✅ | ✅ | 视频生成模型 |
| **Wan2.2** | ✅ | ✅ | ✅ | 改进版视频生成 |

### 🎨 图像编辑模型
| 模型 | LoRA | 全量微调 | FP8/量化 | 描述 |
|------|------|----------|----------|------|
| **Flux Kontext** | ✅ | ✅ | ✅ | 基于 Flux 的编辑模型 |
| **Qwen-Image-Edit** | ✅ | ✅ | ✅ | 通义千问图像编辑 |

## 节点系统详解

### 🗂️ 数据集配置节点

#### GeneralDatasetConfig（通用数据集配置）
配置训练数据集的核心参数：
- **输入路径**: 数据集目录路径
- **分辨率设置**: 训练分辨率配置 `[512]` 或 `[1280, 720]`
- **宽高比分桶**: 自动处理不同比例的图像
- **数据集重复**: 控制数据使用频率
- **缓存设置**: 优化数据加载性能

#### GeneralDatasetPathNode（通用数据集路径）
处理标准图像-文本对数据集：
```
dataset/
├── image1.jpg
├── image1.txt
├── image2.png
└── image2.txt
```

#### EditModelDatasetPathNode（编辑模型数据集路径）
处理图像编辑数据集：
```
dataset/
├── edit_instructions.jsonl
├── source_images/
└── target_images/
```

#### FrameBucketsNode（帧数分桶配置）
视频训练的帧数配置：
- 支持多种帧长度训练
- 自动批次组织

#### ArBucketsNode（宽高比分桶配置）
自定义宽高比分桶策略：
- 精确控制分桶数量
- 优化 VRAM 使用

### 🤖 模型配置节点

#### 图像模型节点
- **SDXLModelNode**: SDXL 模型配置
- **FluxModelNode**: Flux 模型配置
- **SD3ModelNode**: SD3 模型配置
- **QwenImageModelNode**: 通义千问图像模型
- **HiDreamModelNode**: HiDream 模型配置
- **ChromaModelNode**: Chroma 模型配置
- **Lumina2ModelNode**: Lumina2 模型配置

#### 视频模型节点
- **LTXVideoModelNode**: LTX-Video 配置
- **HunyuanVideoModelNode**: 混元视频配置
- **Wan21ModelNode**: Wan2.1 配置
- **Wan22ModelNode**: Wan2.2 配置

#### 编辑模型节点
- **FluxKontextModelNode**: Flux Kontext 配置
- **QwenImageEditModelNode**: 通义千问编辑配置

### ⚙️ 训练配置节点

#### GeneralConfig（通用训练设置）
核心训练参数配置：
- **训练轮数**: 控制训练持续时间
- **批次大小**: GPU 内存优化
- **学习率调度**: 预热和衰减策略
- **梯度配置**: 累积和裁剪设置
- **优化器设置**: AdamW、AdamW8bit 等
- **内存优化**: 块交换、激活检查点

#### ModelConfig（模型配置）
模型特定配置：
- **数据类型**: bfloat16、float16、float8
- **LoRA 设置**: rank、alpha、dropout
- **量化选项**: FP8、4bit 量化

#### AdapterConfigNode（适配器配置）
LoRA 适配器详细配置：
- **目标模块**: 选择训练的模型部分
- **LoRA 参数**: rank、alpha、目标维度
- **训练策略**: 部分冻结、学习率分层

#### OptimizerConfigNode（优化器配置）
优化器详细设置：
- **优化器类型**: AdamW、Lion、Adafactor
- **学习率**: 基础学习率和调度
- **正则化**: 权重衰减、梯度裁剪

### 🚀 训练控制节点

#### Train（训练启动器）
启动和控制训练过程：
- **配置合并**: 自动合并数据集和训练配置
- **进程管理**: 启动、停止、监控训练
- **错误处理**: 异常捕获和恢复
- **日志输出**: 实时训练状态

#### TensorBoardMonitor（TensorBoard监控器）
实时训练监控：
- **损失曲线**: 训练和验证损失
- **学习率追踪**: 学习率变化曲线
- **GPU 利用率**: 硬件使用情况
- **样本预览**: 生成样本质量监控

#### OutputDirPassthrough（输出目录传递）
简化路径传递的工具节点。

## 快速开始

### 1. 基础 LoRA 训练工作流

1. **准备数据集**
   ```
   my_dataset/
   ├── cute_cat_001.jpg
   ├── cute_cat_001.txt  # "a cute orange cat sitting on grass"
   ├── cute_cat_002.jpg
   └── cute_cat_002.txt
   ```

2. **构建节点工作流**
   ```
   GeneralDatasetPathNode → GeneralDatasetConfig
                         ↓
   FluxModelNode → ModelConfig → GeneralConfig → Train
                                       ↓
                              TensorBoardMonitor
   ```

3. **配置参数**
   - 数据集路径: `/path/to/my_dataset`
   - 模型检查点: `/path/to/flux-dev`
   - 训练轮数: 100
   - 学习率: 1e-4

4. **启动训练**
   点击 `Train` 节点的执行按钮开始训练。

### 2. 高级多 GPU 训练

对于大型模型或数据集：

```python
# 训练配置
epochs = 500
micro_batch_size_per_gpu = 1
number_of_gpus = 2
pipeline_stages = 2
gradient_accumulation_steps = 8
```

## 配置文件详解

### 数据集配置 (dataset.toml)
```toml
[[image_text]]
dataset_paths = ["/path/to/dataset"]
num_repeats = 1
default_caption_prefix = ""
resolutions = [[1024, 1024]]
enable_ar_bucket = true
min_ar = 0.5
max_ar = 2.0
num_ar_buckets = 7
```

### 训练配置 (train.toml)
```toml
[model]
type = "flux"
diffusers_path = "/path/to/flux-dev"
dtype = "bfloat16"
transformer_dtype = "float8"

[adapter]
type = "lora"
rank = 16
alpha = 16
target_modules = ["to_q", "to_k", "to_v", "to_out.0"]

[optimizer]
type = "adamw"
lr = 1e-4
betas = [0.9, 0.999]
weight_decay = 0.01

# 训练设置
epochs = 100
micro_batch_size_per_gpu = 2
gradient_accumulation_steps = 4
save_every_n_epochs = 10
```

## 高级功能

### 内存优化策略

#### 块交换 (Block Swapping)
对于显存不足的情况：
```python
blocks_to_swap = 12  # 交换的 transformer 块数量
```

#### 激活检查点
减少训练时的内存使用：
```python
activation_checkpointing = True
```

#### 梯度释放
实验性内存优化：
```python
gradient_release = True  # 仅适用于特定优化器
```

### 数据集优化

#### 缓存策略
```python
# 强制重新生成缓存
regenerate_cache = True

# 信任现有缓存
trust_cache = True

# 仅生成缓存后退出
cache_only = True
```

#### 分辨率分桶
```python
# 单一分辨率
resolutions = [[1024, 1024]]

# 多分辨率训练
resolutions = [
    [512, 512],
    [768, 768], 
    [1024, 1024],
    [1280, 720]
]
```

### 视频训练特殊配置

#### 帧长度设置
```python
# 视频帧数配置
video_clip_mode = "single_beginning"
frame_buckets = [16, 24, 32]
```

#### 条件设置
```python
# 图像到视频训练
first_frame_conditioning_p = 1.0
```

## 故障排除

### 常见问题

#### 1. WSL2 路径问题
**问题**: Windows 路径无法在 WSL2 中识别
**解决**: 使用路径规范化功能，将 `C:\path` 自动转换为 `/mnt/c/path`

#### 2. CUDA 内存不足
**问题**: `CUDA out of memory`
**解决方案**:
- 减小 `micro_batch_size_per_gpu`
- 启用 `blocks_to_swap`
- 使用 `float8` 量化
- 启用 `activation_checkpointing`

#### 3. 数据集加载失败
**问题**: 找不到图像或文本文件
**解决方案**:
- 检查文件路径和权限
- 确保图像和文本文件配对
- 使用 `regenerate_cache=True` 重新生成缓存

#### 4. 训练过程中断
**问题**: 训练突然停止
**解决方案**:
- 使用 `resume_from_checkpoint` 恢复训练
- 检查磁盘空间
- 监控 GPU 温度

### 调试技巧

#### 启用详细日志
```bash
export PYTHONPATH=/path/to/ComfyUI/custom_nodes/Diffusion_pipe_in_ComfyUI
export CUDA_LAUNCH_BLOCKING=1
```

#### 内存监控
```bash
# 监控 GPU 内存使用
watch -n 1 nvidia-smi

# 监控系统内存
htop
```

#### 验证配置
使用 `cache_only=True` 验证数据集配置是否正确。

## 最佳实践

### 训练策略

#### 学习率调优
1. **起始值**: 从较小的学习率开始 (1e-5)
2. **预热**: 使用 warmup_steps 平缓启动
3. **监控**: 观察损失曲线调整学习率

#### LoRA 参数选择
- **rank**: 16-32 适合大多数情况
- **alpha**: 通常等于 rank
- **dropout**: 0.1 可防止过拟合

#### 数据集准备
1. **质量**: 高质量图像和准确标注
2. **多样性**: 涵盖各种场景和角度
3. **数量**: 100-1000 张图像适合 LoRA 训练

### 性能优化

#### 硬件配置
- **24GB GPU**: 适合大多数 LoRA 训练
- **48GB GPU**: 支持全量微调
- **多 GPU**: 使用 pipeline_stages 并行

#### 软件配置
```bash
# 优化 CUDA 内存分配
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 禁用 P2P (如果有网络问题)
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
```

## 扩展开发

### 添加新模型支持

1. **创建模型文件**
   ```python
   # models/my_model.py
   class MyModel:
       def __init__(self, config):
           pass
   ```

2. **注册节点**
   ```python
   # diffusion_nodes/model_tools.py
   class MyModelNode:
       @classmethod
       def INPUT_TYPES(cls):
           return {...}
   ```

3. **更新映射**
   ```python
   # nodes.py
   NODE_CLASS_MAPPINGS["MyModelNode"] = MyModelNode
   ```

### 自定义数据集格式

继承基础数据集类并实现自定义逻辑：
```python
class CustomDatasetNode:
    def create_config(self, **kwargs):
        # 自定义配置逻辑
        return config
```

## 更新日志

### 版本 2.0
- 新增 Qwen-Image 系列模型支持
- 改进 WSL2 路径处理
- 优化内存使用策略
- 新增视频训练功能

### 版本 1.5
- 新增 Flux 系列模型支持
- 改进训练监控
- 优化配置文件生成

### 版本 1.0
- 基础 SDXL 训练支持
- ComfyUI 节点集成
- 基础 LoRA 训练功能

## 许可证

本项目基于 MIT 许可证开源。

## 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 发起 Pull Request

## 致谢

感谢以下项目和团队：
- ComfyUI 团队
- Hugging Face Diffusers
- DeepSpeed 团队
- 各模型原始作者

## 联系方式

- GitHub Issues: [项目地址]
- 讨论区: [Discord/论坛链接]

---

**注意**: 这是一个功能强大的训练工具，需要深入了解机器学习和 Diffusion 模型。建议在使用前充分了解相关概念和风险。