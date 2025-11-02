![](./img/img.png)


# Diffusion-Pipe In ComfyUI Custom Nodes

<div align="center">

  [![Windows Version](https://img.shields.io/badge/Windows%20Version-Visit%20Repo-blue?style=rounded-pill&logo=github&logoColor=white)](https://github.com/TianDongL/Diffusion_pipe_in_ComfyUI_Win.git)
  
  [![Original Project](https://img.shields.io/badge/Original%20Project-tdrussell's%20diffusion--pipe-purple?labelColor=6c5ce7&color=a29bfe&style=rounded-pill&logo=github&logoColor=white)](https://github.com/tdrussell/diffusion-pipe.git)

  
</div>


# 点击查看 [中文文档](./READMEChinese.md)


## Project Overview

Diffusion-Pipe In ComfyUI custom nodes is a powerful extension plugin that provides complete Diffusion model training and fine-tuning functionality for ComfyUI. This project allows users to configure and launch training for various advanced AI models through ComfyUI's graphical interface, supporting both LoRA and full fine-tuning, covering the most popular image generation and video generation models.

***Video Demo: https://www.bilibili.com/video/BV1DAnKzTEup/?share_source=copy_web&vd_source=5a2c3d8b60d05e98a2e7f4f58f77eba5***

***[📋 View Supported Models](./docs/supported_models.md)***

# Quick Start

## update

* 20251026:support eval 

* 20251030: Supports training Aura models

* 20251103:support MultiImage Edit (qwen2509)



## Installation Guide

### Installation 
Make sure you have ComfyUI on Linux or WSL2 system, refer to https://docs.comfy.org/installation/manual_install

ps: ComfyUI on WSL2 works so well that I even want to delete my ComfyUI on Windows


```bash
conda create -n comfyui_DP python=3.12
```
```bash
conda activate comfyui_DP
```

```bash
cd ~/comfy/ComfyUI/custom_nodes/
```

```bash
git clone --recurse-submodules https://github.com/TianDongL/Diffusion_pipe_in_ComfyUI.git
```

* If you haven't installed submodules, follow these steps 

* If you don't do this step, training will not work

```bash
git submodule init
git submodule update
```

# Install Dependencies
```bash
conda activate comfyui_DP
```
Here are the necessary dependencies for deepspeed, first install PyTorch. It is not listed in the requirements document because some GPUs sometimes require different versions of PyTorch or CUDA, and you may have to find a combination that suits your hardware.
```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```
```bash
cd ~/comfy/ComfyUI/custom_nodes/Diffusion_pipe_in_ComfyUI
```
```bash
pip install -r requirements.txt
```

## 🚀 One-Click Import Workflow

To get you started quickly, we provide pre-configured ComfyUI workflow files:

***[📋 Click to Import Complete Workflow](./example_workflows/DiffusionPipeInComfyUI.json)***

Drag this file into the ComfyUI interface to import the complete training workflow, including all necessary node configurations.

## Please read the prompts in the workflow carefully, this can help you build your dataset

# 📷 Workflow Interface Preview

<div align="center">

![Model Loading Nodes](./img/11.png)
Models can be stored in the ComfyUI model directory

![Start Training ](./img/22.png)
*Disable Train node when debugging*

![Model Configuration](./img/33.png)
Models can be stored in the ComfyUI model directory

![Dataset Configuration](./img/44.png)
Recommend storing datasets in paths like Z:\ComfyUI\custom_nodes\Diffusion_pipe_in_ComfyUI\input\test_vid
Recommend dataset configuration paths like Z:\ComfyUI\custom_nodes\Diffusion_pipe_in_ComfyUI\dataset\testdataset.toml

![Workflow Overview](./img/55.png)

![Monitoring ](./img/66.png)
*kill port will stop all monitoring processes on the current port*

</div>

### Core Features

- 🎯 **Visual Training Configuration**: Graphically configure training parameters through ComfyUI nodes
- 🚀 **Multi-Model Support**: Support for 20+ latest Diffusion models
- 💾 **Flexible Training Methods**: Support for LoRA training and full fine-tuning
- ⚡ **High-Performance Training**: DeepSpeed-based distributed training support
- 📊 **Real-time Monitoring**: Integrated TensorBoard training process monitoring
- 🔧 **WSL2 Optimization**: Specially optimized Windows WSL2 environment support
- 🎥 **Video Training**: Support for video generation model training
- 🖼️ **Image Editing**: Support for image editing model training

## System Requirements

### Hardware Requirements
- * I don't know, you can try :-P	

### Software Requirements
- **Operating System**: Linux / Windows 10/11 + WSL2
- **ComfyUI**: Latest version

## Supported Models

This plugin supports over 20 latest Diffusion models, including:

| Model          | LoRA | Full Fine Tune | fp8/quantization |
|----------------|------|----------------|------------------|
|SDXL            |✅    |✅              |❌                |
|Flux            |✅    |✅              |✅                |
|LTX-Video       |✅    |❌              |❌                |
|HunyuanVideo    |✅    |❌              |✅                |
|Cosmos          |✅    |❌              |❌                |
|Lumina Image 2.0|✅    |✅              |❌                |
|Wan2.1          |✅    |✅              |✅                |
|Chroma          |✅    |✅              |✅                |
|HiDream         |✅    |❌              |✅                |
|SD3             |✅    |❌              |✅                |
|Cosmos-Predict2 |✅    |✅              |✅                |
|OmniGen2        |✅    |❌              |❌                |
|Flux Kontext    |✅    |✅              |✅                |
|Wan2.2          |✅    |✅              |✅                |
|Qwen-Image      |✅    |✅              |✅                |
|Qwen-Image-Edit |✅    |✅              |✅                |
|HunyuanImage-2.1|✅    |✅              |✅                |
|AuraFlow        |✅    |❌              |✅                |


## License

This project is open source under the Apache License 2.0.

## Contributing Guide

Issues and Pull Requests are welcome!

1. Fork the project
2. Create a feature branch
3. Submit changes
4. Create a Pull Request

## Acknowledgments

Thanks to the following projects and teams:
- ComfyUI team
- [@tdrussell](https://github.com/tdrussell/diffusion-pipe.git), the original author of Diffusion_Pipe
- Hugging Face Diffusers
- DeepSpeed team
- Original authors of various models 