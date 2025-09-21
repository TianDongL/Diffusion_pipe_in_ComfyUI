# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
from easydict import EasyDict

#------------------------ Wan shared config ------------------------#
wan_shared_cfg = EasyDict()

# t5
wan_shared_cfg.t5_model = 'umt5_xxl'
wan_shared_cfg.t5_dtype = torch.bfloat16
wan_shared_cfg.text_len = 512

# transformer
wan_shared_cfg.param_dtype = torch.bfloat16

# inference
wan_shared_cfg.num_train_timesteps = 1000
wan_shared_cfg.sample_fps = 16
wan_shared_cfg.sample_neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'

#------------------------ Wan I2V 14B ------------------------#

i2v_14B = EasyDict(__name__='Config: Wan I2V 14B')
i2v_14B.update(wan_shared_cfg)
i2v_14B.sample_neg_prompt = "镜头晃动，" + i2v_14B.sample_neg_prompt

i2v_14B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
i2v_14B.t5_tokenizer = 'google/umt5-xxl'

# clip
i2v_14B.clip_model = 'clip_xlm_roberta_vit_h_14'
i2v_14B.clip_dtype = torch.float16
i2v_14B.clip_checkpoint = 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'
i2v_14B.clip_tokenizer = 'xlm-roberta-large'

# vae
i2v_14B.vae_checkpoint = 'Wan2.1_VAE.pth'
i2v_14B.vae_stride = (4, 8, 8)

# transformer
i2v_14B.patch_size = (1, 2, 2)
i2v_14B.dim = 5120
i2v_14B.ffn_dim = 13824
i2v_14B.freq_dim = 256
i2v_14B.num_heads = 40
i2v_14B.num_layers = 40
i2v_14B.window_size = (-1, -1)
i2v_14B.qk_norm = True
i2v_14B.cross_attn_norm = True
i2v_14B.eps = 1e-6

#------------------------ Wan T2V 14B ------------------------#

t2v_14B = EasyDict(__name__='Config: Wan T2V 14B')
t2v_14B.update(wan_shared_cfg)

# t5
t2v_14B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
t2v_14B.t5_tokenizer = 'google/umt5-xxl'

# vae
t2v_14B.vae_checkpoint = 'Wan2.1_VAE.pth'
t2v_14B.vae_stride = (4, 8, 8)

# transformer
t2v_14B.patch_size = (1, 2, 2)
t2v_14B.dim = 5120
t2v_14B.ffn_dim = 13824
t2v_14B.freq_dim = 256
t2v_14B.num_heads = 40
t2v_14B.num_layers = 40
t2v_14B.window_size = (-1, -1)
t2v_14B.qk_norm = True
t2v_14B.cross_attn_norm = True
t2v_14B.eps = 1e-6

#------------------------ Wan T2V 1.3B ------------------------#

t2v_1_3B = EasyDict(__name__='Config: Wan T2V 1.3B')
t2v_1_3B.update(wan_shared_cfg)

# t5
t2v_1_3B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
t2v_1_3B.t5_tokenizer = 'google/umt5-xxl'

# vae
t2v_1_3B.vae_checkpoint = 'Wan2.1_VAE.pth'
t2v_1_3B.vae_stride = (4, 8, 8)

# transformer
t2v_1_3B.patch_size = (1, 2, 2)
t2v_1_3B.dim = 1536
t2v_1_3B.ffn_dim = 8960
t2v_1_3B.freq_dim = 256
t2v_1_3B.num_heads = 12
t2v_1_3B.num_layers = 30
t2v_1_3B.window_size = (-1, -1)
t2v_1_3B.qk_norm = True
t2v_1_3B.cross_attn_norm = True
t2v_1_3B.eps = 1e-6

#------------------------ Wan I2V A14B ------------------------#

i2v_A14B = EasyDict(__name__='Config: Wan I2V A14B')
i2v_A14B.update(wan_shared_cfg)

i2v_A14B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
i2v_A14B.t5_tokenizer = 'google/umt5-xxl'

# vae
i2v_A14B.vae_checkpoint = 'Wan2.1_VAE.pth'
i2v_A14B.vae_stride = (4, 8, 8)

# transformer
i2v_A14B.patch_size = (1, 2, 2)
i2v_A14B.dim = 5120
i2v_A14B.ffn_dim = 13824
i2v_A14B.freq_dim = 256
i2v_A14B.num_heads = 40
i2v_A14B.num_layers = 40
i2v_A14B.window_size = (-1, -1)
i2v_A14B.qk_norm = True
i2v_A14B.cross_attn_norm = True
i2v_A14B.eps = 1e-6
i2v_A14B.low_noise_checkpoint = 'low_noise_model'
i2v_A14B.high_noise_checkpoint = 'high_noise_model'

# inference
i2v_A14B.sample_shift = 5.0
i2v_A14B.sample_steps = 40
i2v_A14B.boundary = 0.900
i2v_A14B.sample_guide_scale = (3.5, 3.5)  # low noise, high noise

#------------------------ Wan T2V A14B ------------------------#

t2v_A14B = EasyDict(__name__='Config: Wan T2V A14B')
t2v_A14B.update(wan_shared_cfg)

# t5
t2v_A14B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
t2v_A14B.t5_tokenizer = 'google/umt5-xxl'

# vae
t2v_A14B.vae_checkpoint = 'Wan2.1_VAE.pth'
t2v_A14B.vae_stride = (4, 8, 8)

# transformer
t2v_A14B.patch_size = (1, 2, 2)
t2v_A14B.dim = 5120
t2v_A14B.ffn_dim = 13824
t2v_A14B.freq_dim = 256
t2v_A14B.num_heads = 40
t2v_A14B.num_layers = 40
t2v_A14B.window_size = (-1, -1)
t2v_A14B.qk_norm = True
t2v_A14B.cross_attn_norm = True
t2v_A14B.eps = 1e-6
t2v_A14B.low_noise_checkpoint = 'low_noise_model'
t2v_A14B.high_noise_checkpoint = 'high_noise_model'

# inference
t2v_A14B.sample_shift = 12.0
t2v_A14B.sample_steps = 40
t2v_A14B.boundary = 0.875
t2v_A14B.sample_guide_scale = (3.0, 4.0)  # low noise, high noise

#------------------------ Wan TI2V 5B ------------------------#

ti2v_5B = EasyDict(__name__='Config: Wan TI2V 5B')
ti2v_5B.update(wan_shared_cfg)

# t5
ti2v_5B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
ti2v_5B.t5_tokenizer = 'google/umt5-xxl'

# vae
ti2v_5B.vae_checkpoint = 'Wan2.2_VAE.pth'
ti2v_5B.vae_stride = (4, 16, 16)

# transformer
ti2v_5B.patch_size = (1, 2, 2)
ti2v_5B.dim = 3072
ti2v_5B.ffn_dim = 14336
ti2v_5B.freq_dim = 256
ti2v_5B.num_heads = 24
ti2v_5B.num_layers = 30
ti2v_5B.window_size = (-1, -1)
ti2v_5B.qk_norm = True
ti2v_5B.cross_attn_norm = True
ti2v_5B.eps = 1e-6

# inference
ti2v_5B.sample_fps = 24
ti2v_5B.sample_shift = 5.0
ti2v_5B.sample_steps = 50
ti2v_5B.sample_guide_scale = 5.0
ti2v_5B.frame_num = 121