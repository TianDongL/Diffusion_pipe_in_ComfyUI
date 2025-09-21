import math

import torch
from torch import nn
import torch.nn.functional as F
import safetensors
import transformers
from transformers import T5TokenizerFast, T5EncoderModel
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

from models.base import BasePipeline, PreprocessMediaFile, make_contiguous
from models.cosmos_predict2_modeling import MiniTrainDIT
from utils.common import load_state_dict, AUTOCAST_DTYPE
from utils.offloading import ModelOffloader
from models.wan.vae2_1 import WanVAE_


KEEP_IN_HIGH_PRECISION = ['x_embedder', 't_embedder', 't_embedding_norm', 'final_layer']


def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def _video_vae(pretrained_path=None, z_dim=None, device='cpu', **kwargs):
    """
    Autoencoder3d adapted from Stable Diffusion 1.x, 2.x and XL.
    """
    # params
    cfg = dict(
        dim=96,
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0)
    cfg.update(**kwargs)

    # init model
    with torch.device('meta'):
        model = WanVAE_(**cfg)

    # load checkpoint
    model.load_state_dict(
        load_state_dict(pretrained_path), assign=True)

    return model


class WanVAE:
    def __init__(self,
                 z_dim=16,
                 vae_pth=None,
                 dtype=torch.float,
                 device="cpu"):
        self.dtype = dtype
        self.device = device

        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=dtype, device=device)
        self.std = torch.tensor(std, dtype=dtype, device=device)
        self.scale = [self.mean, 1.0 / self.std]

        # init model
        self.model = _video_vae(
            pretrained_path=vae_pth,
            z_dim=z_dim,
        ).eval().requires_grad_(False).to(device)


def vae_encode(tensor, vae):
    return vae.model.encode(tensor, vae.scale)


def get_dit_config(state_dict, key_prefix=''):
    dit_config = {}
    dit_config["max_img_h"] = 240
    dit_config["max_img_w"] = 240
    dit_config["max_frames"] = 128
    concat_padding_mask = True
    dit_config["in_channels"] = (state_dict['{}x_embedder.proj.1.weight'.format(key_prefix)].shape[1] // 4) - int(concat_padding_mask)
    dit_config["out_channels"] = 16
    dit_config["patch_spatial"] = 2
    dit_config["patch_temporal"] = 1
    dit_config["model_channels"] = state_dict['{}x_embedder.proj.1.weight'.format(key_prefix)].shape[0]
    dit_config["concat_padding_mask"] = concat_padding_mask
    dit_config["crossattn_emb_channels"] = 1024
    dit_config["pos_emb_cls"] = "rope3d"
    dit_config["pos_emb_learnable"] = True
    dit_config["pos_emb_interpolation"] = "crop"
    dit_config["min_fps"] = 1
    dit_config["max_fps"] = 30

    dit_config["use_adaln_lora"] = True
    dit_config["adaln_lora_dim"] = 256
    if dit_config["model_channels"] == 2048:
        dit_config["num_blocks"] = 28
        dit_config["num_heads"] = 16
    elif dit_config["model_channels"] == 5120:
        dit_config["num_blocks"] = 36
        dit_config["num_heads"] = 40

    if dit_config["in_channels"] == 16:
        dit_config["extra_per_block_abs_pos_emb"] = False
        dit_config["rope_h_extrapolation_ratio"] = 4.0
        dit_config["rope_w_extrapolation_ratio"] = 4.0
        dit_config["rope_t_extrapolation_ratio"] = 1.0
    elif dit_config["in_channels"] == 17:
        dit_config["extra_per_block_abs_pos_emb"] = False
        dit_config["rope_h_extrapolation_ratio"] = 3.0
        dit_config["rope_w_extrapolation_ratio"] = 3.0
        dit_config["rope_t_extrapolation_ratio"] = 1.0

    dit_config["extra_h_extrapolation_ratio"] = 1.0
    dit_config["extra_w_extrapolation_ratio"] = 1.0
    dit_config["extra_t_extrapolation_ratio"] = 1.0
    dit_config["rope_enable_fps_modulation"] = False

    return dit_config


def _tokenize(tokenizer, prompts):
    return tokenizer.batch_encode_plus(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512,
        return_length=True,
        return_offsets_mapping=False,
    )

def _compute_text_embeddings(text_encoder, input_ids, attn_mask):
    input_ids = input_ids.to(text_encoder.device)
    attn_mask = attn_mask.to(text_encoder.device)

    outputs = text_encoder(input_ids=input_ids, attention_mask=attn_mask)

    encoded_text = outputs.last_hidden_state
    lengths = attn_mask.sum(dim=1).cpu()

    for batch_id in range(encoded_text.shape[0]):
        length = lengths[batch_id]
        # This is tricky. Based on Nvidia's official code, when the prompt is '' or when dropping out text embeddings,
        # the embeddings are zeroed out completely. But the attention mask for an empty string looks like
        # [1, 0, 0, ...] because of the BOS token. So it's not zeroed out unless we explicitly set length to 0 here.
        # If you don't do this, training on unconditional text embeddings will NaN the loss pretty quickly.
        if length == 1:
            length = 0
        encoded_text[batch_id][length:] = 0

    return encoded_text


class CosmosPredict2Pipeline(BasePipeline):
    name = 'cosmos_predict2'
    framerate = 16
    checkpointable_layers = ['TransformerLayer']
    adapter_target_modules = ['Block']

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']
        self.offloader = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)
        dtype = self.model_config['dtype']
        self.cache_text_embeddings = self.model_config.get('cache_text_embeddings', True)

        # This isn't a nn.Module.
        self.vae = WanVAE(
            vae_pth=self.model_config['vae_path'],
            device='cpu',
            dtype=dtype,
        )
        # These need to be on the device the VAE will be moved to during caching.
        self.vae.mean = self.vae.mean.to('cuda')
        self.vae.std = self.vae.std.to('cuda')
        self.vae.scale = [self.vae.mean, 1.0 / self.vae.std]

        self.tokenizer = T5TokenizerFast(
            vocab_file='configs/t5_old/spiece.model',
            tokenizer_file='configs/t5_old/tokenizer.json',
        )
        t5_state_dict = load_state_dict(self.model_config['t5_path'])
        if self.model_config.get('text_encoder_nf4', False):
            quantization_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=dtype,
            )
        else:
            quantization_config = None
        self.text_encoder = T5EncoderModel.from_pretrained(
            None,
            config='configs/t5_old/config.json',
            state_dict=t5_state_dict,
            torch_dtype='auto',
            local_files_only=True,
            quantization_config=quantization_config,
        )
        if quantization_config is None and self.model_config.get('text_encoder_fp8', False):
            for name, p in self.text_encoder.named_parameters():
                if p.ndim == 2 and not ('shared' in name or 'relative_attention_bias' in name):
                    p.data = p.data.to(torch.float8_e4m3fn)
        self.text_encoder.requires_grad_(False)

    def load_diffusion_model(self):
        dtype = self.model_config['dtype']
        transformer_dtype = self.model_config.get('transformer_dtype', dtype)

        state_dict = load_state_dict(self.model_config['transformer_path'])
        # Remove 'net.' prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('net.'):
                k = k[len('net.'):]
            new_state_dict[k] = v
        state_dict = new_state_dict

        dit_config = get_dit_config(state_dict)
        with init_empty_weights():
            transformer = MiniTrainDIT(**dit_config)
        for name, p in transformer.named_parameters():
            dtype_to_use = dtype if (any(keyword in name for keyword in KEEP_IN_HIGH_PRECISION) or p.ndim == 1) else transformer_dtype
            set_module_tensor_to_device(transformer, name, device='cpu', dtype=dtype_to_use, value=state_dict[name])
        self.transformer = transformer
        self.transformer.train()
        for name, p in self.transformer.named_parameters():
            p.original_name = name

    def get_vae(self):
        return self.vae.model

    def get_text_encoders(self):
        if self.cache_text_embeddings:
            return [self.text_encoder]
        else:
            return []

    def save_adapter(self, save_dir, peft_state_dict):
        self.peft_config.save_pretrained(save_dir)
        # ComfyUI format.
        peft_state_dict = {'diffusion_model.'+k: v for k, v in peft_state_dict.items()}
        safetensors.torch.save_file(peft_state_dict, save_dir / 'adapter_model.safetensors', metadata={'format': 'pt'})

    def save_model(self, save_dir, state_dict):
        state_dict = {'net.'+k: v for k, v in state_dict.items()}
        safetensors.torch.save_file(state_dict, save_dir / 'model.safetensors', metadata={'format': 'pt'})

    def get_preprocess_media_file_fn(self):
        return PreprocessMediaFile(
            self.config,
            support_video=True,
            framerate=self.framerate,
        )

    def get_call_vae_fn(self, vae):
        def fn(tensor):
            p = next(vae.parameters())
            tensor = tensor.to(p.device, p.dtype)
            latents = vae_encode(tensor, self.vae)
            return {'latents': latents}
        return fn

    def get_call_text_encoder_fn(self, text_encoder):
        def fn(captions, is_video):
            # args are lists
            batch_encoding = _tokenize(self.tokenizer, captions)
            encoded_text = _compute_text_embeddings(self.text_encoder, batch_encoding.input_ids, batch_encoding.attention_mask)
            return {'prompt_embeds': encoded_text}
        return fn

    # Note to myself / future readers:
    # The timestep sampling, input construction, and loss function have a different formulation here than how Nvidia does it
    # in the official code. It wasn't obvious at first, but if you work through the math you will see the this model is just
    # a standard rectified flow model, the same as Flux, SD3, Lumina 2, etc. The ONLY difference is that in the way Nvidia
    # formulated it, you end up with an effective loss weighting of t**2 + (1-t)**2. This is a quadratic that is 1 at the endpoints
    # t=0 and t=1, and 0.5 at t=0.5. So, the middle timesteps are downweighted slightly. I left out this weighting because I don't
    # see any justification or point to doing it. As such, everything here aligns with the other rectified flow models.
    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        mask = inputs['mask']

        if self.cache_text_embeddings:
            prompt_embeds_or_batch_encoding = (inputs['prompt_embeds'],)
        else:
            batch_encoding = _tokenize(self.tokenizer, inputs['caption'])
            prompt_embeds_or_batch_encoding = (batch_encoding.input_ids, batch_encoding.attention_mask)

        bs, channels, num_frames, h, w = latents.shape

        if mask is not None:
            mask = mask.unsqueeze(1)  # make mask (bs, 1, img_h, img_w)
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')  # resize to latent spatial dimension
            mask = mask.unsqueeze(2)  # make mask same number of dims as target

        timestep_sample_method = self.model_config.get('timestep_sample_method', 'logit_normal')

        if timestep_sample_method == 'logit_normal':
            dist = torch.distributions.normal.Normal(0, 1)
        elif timestep_sample_method == 'uniform':
            dist = torch.distributions.uniform.Uniform(0, 1)
        else:
            raise NotImplementedError()

        if timestep_quantile is not None:
            t = dist.icdf(torch.full((bs,), timestep_quantile, device=latents.device))
        else:
            t = dist.sample((bs,)).to(latents.device)

        if timestep_sample_method == 'logit_normal':
            sigmoid_scale = self.model_config.get('sigmoid_scale', 1.0)
            t = t * sigmoid_scale
            t = torch.sigmoid(t)

        if shift := self.model_config.get('shift', None):
            t = (t * shift) / (1 + (shift - 1) * t)
        elif self.model_config.get('flux_shift', False):
            mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
            t = time_shift(mu, 1.0, t)

        noise = torch.randn_like(latents)
        t_expanded = t.view(-1, 1, 1, 1, 1)
        noisy_latents = (1 - t_expanded)*latents + t_expanded*noise
        target = noise - latents

        return (noisy_latents, t.view(-1, 1), *prompt_embeds_or_batch_encoding), (target, mask)

    def to_layers(self):
        transformer = self.transformer
        text_encoder = None if self.cache_text_embeddings else self.text_encoder
        layers = [InitialLayer(transformer, text_encoder)]
        for i, block in enumerate(transformer.blocks):
            layers.append(TransformerLayer(block, i, self.offloader))
        layers.append(FinalLayer(transformer))
        return layers

    def enable_block_swap(self, blocks_to_swap):
        transformer = self.transformer
        blocks = transformer.blocks
        num_blocks = len(blocks)
        assert (
            blocks_to_swap <= num_blocks - 2
        ), f'Cannot swap more than {num_blocks - 2} blocks. Requested {blocks_to_swap} blocks to swap.'
        self.offloader = ModelOffloader(
            'TransformerBlock', blocks, num_blocks, blocks_to_swap, True, torch.device('cuda'), self.config['reentrant_activation_checkpointing']
        )
        transformer.blocks = None
        transformer.to('cuda')
        transformer.blocks = blocks
        self.prepare_block_swap_training()
        print(f'Block swap enabled. Swapping {blocks_to_swap} blocks out of {num_blocks} blocks.')

    def prepare_block_swap_training(self):
        self.offloader.enable_block_swap()
        self.offloader.set_forward_only(False)
        self.offloader.prepare_block_devices_before_forward()

    def prepare_block_swap_inference(self, disable_block_swap=False):
        if disable_block_swap:
            self.offloader.disable_block_swap()
        self.offloader.set_forward_only(True)
        self.offloader.prepare_block_devices_before_forward()


class InitialLayer(nn.Module):
    def __init__(self, model, text_encoder):
        super().__init__()
        self.x_embedder = model.x_embedder
        self.pos_embedder = model.pos_embedder
        if model.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder = model.extra_pos_embedder
        self.t_embedder = model.t_embedder
        self.t_embedding_norm = model.t_embedding_norm
        self.text_encoder = text_encoder
        self.model = [model]

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x_B_C_T_H_W, timesteps_B_T, *prompt_embeds_or_batch_encoding = inputs

        if len(prompt_embeds_or_batch_encoding) == 1:
            crossattn_emb = prompt_embeds_or_batch_encoding[0]
        else:
            with torch.no_grad():
                crossattn_emb = _compute_text_embeddings(self.text_encoder, *prompt_embeds_or_batch_encoding)

        padding_mask = torch.zeros(x_B_C_T_H_W.shape[0], 1, x_B_C_T_H_W.shape[3], x_B_C_T_H_W.shape[4], dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.model[0].prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=None,
            padding_mask=padding_mask,
        )
        assert extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is None
        assert rope_emb_L_1_1_D is not None

        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        outputs =  make_contiguous(x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, rope_emb_L_1_1_D, adaln_lora_B_T_3D, timesteps_B_T)
        for item in outputs:
            item.requires_grad_(True)
        return outputs


class TransformerLayer(nn.Module):
    def __init__(self, block, block_idx, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, rope_emb_L_1_1_D, adaln_lora_B_T_3D, timesteps_B_T = inputs

        self.offloader.wait_for_block(self.block_idx)
        x_B_T_H_W_D = self.block(x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, rope_emb_L_1_1_D=rope_emb_L_1_1_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, rope_emb_L_1_1_D, adaln_lora_B_T_3D, timesteps_B_T)


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.final_layer = model.final_layer
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    def get_per_sigma_loss_weights(self, sigma: torch.Tensor) -> torch.Tensor:
        return (sigma**2 + self.pipe.sigma_data**2) / (sigma * self.pipe.sigma_data) ** 2

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, rope_emb_L_1_1_D, adaln_lora_B_T_3D, timesteps_B_T = inputs
        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        net_output_B_C_T_H_W = self.unpatchify(x_B_T_H_W_O)
        return net_output_B_C_T_H_W
