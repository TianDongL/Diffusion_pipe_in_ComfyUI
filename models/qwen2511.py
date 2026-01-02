import json
from pathlib import Path
from typing import Union, Tuple, Optional
import math

import torch
from torch import nn
import torch.nn.functional as F
import diffusers
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.attention_processor import Attention
import safetensors
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
import transformers
from PIL import Image, ImageOps

from models.base import BasePipeline, PreprocessMediaFile, make_contiguous
from utils.common import AUTOCAST_DTYPE, get_lin_function, time_shift, iterate_safetensors
from utils.offloading import ModelOffloader

from typing import Any, Dict, List, Optional, Tuple, Union
from math import prod
import numpy as np

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import AttentionMixin, FeedForward
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, RMSNorm
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph

logger = logging.get_logger(__name__)


KEEP_IN_HIGH_PRECISION = ['time_text_embed', 'img_in', 'txt_in', 'norm_out', 'proj_out']

def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent).to(timesteps.dtype)
    emb = timesteps[:, None].float() * emb[None, :]

    emb = scale * emb

    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if use_real:
        cos, sin = freqs_cis 

        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
        return out
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(1)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
        return x_out.type_as(x)



from diffusers.models.embeddings import TimestepEmbedding, Timesteps
import functools
from typing import List, Tuple, Union

class QwenTimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep, hidden_states):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))  # (N, D)
        return timesteps_emb



# Copied from transformer_qwenimage.py
class QwenEmbedRope(nn.Module):
    def __init__(self, theta: int, axes_dim: List[int], scale_rope=False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = torch.cat(
            [
                self.rope_params(pos_index, self.axes_dim[0], self.theta),
                self.rope_params(pos_index, self.axes_dim[1], self.theta),
                self.rope_params(pos_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.neg_freqs = torch.cat(
            [
                self.rope_params(neg_index, self.axes_dim[0], self.theta),
                self.rope_params(neg_index, self.axes_dim[1], self.theta),
                self.rope_params(neg_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )

        # DO NOT USING REGISTER BUFFER HERE, IT WILL CAUSE COMPLEX NUMBERS LOSE ITS IMAGINARY PART
        self.scale_rope = scale_rope

    def rope_params(self, index, dim, theta=10000):
        """
        Args:
            index: [0, 1, 2, 3] 1D Tensor representing the position index of the token
        """
        assert dim % 2 == 0
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    def forward(
        self,
        video_fhw: Union[Tuple[int, int, int], List[Tuple[int, int, int]]],
        txt_seq_lens: List[int],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            # RoPE frequencies are cached via a lru_cache decorator on _compute_video_freqs
            video_freq = self._compute_video_freqs(frame, height, width, idx)
            video_freq = video_freq.to(device)
            vid_freqs.append(video_freq)

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len, ...]
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs

    @functools.lru_cache(maxsize=128)
    def _compute_video_freqs(self, frame: int, height: int, width: int, idx: int = 0) -> torch.Tensor:
        seq_lens = frame * height * width
        freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat([freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0)
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
        return freqs.clone().contiguous()


class QwenDoubleStreamAttnProcessor2_0:
    """
    Attention processor for Qwen double-stream architecture, matching DoubleStreamLayerMegatron logic. This processor
    implements joint attention computation where text and image streams are processed together.
    """

    _attention_backend = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "QwenDoubleStreamAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        if encoder_hidden_states is None:
            raise ValueError("QwenDoubleStreamAttnProcessor2_0 requires encoder_hidden_states (text stream)")

        seq_txt = encoder_hidden_states.shape[1]

        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        joint_hidden_states = dispatch_attention_fn(
            joint_query,
            joint_key,
            joint_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )

        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        txt_attn_output = joint_hidden_states[:, :seq_txt, :]  # Text part
        img_attn_output = joint_hidden_states[:, seq_txt:, :]  # Image part

        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)  # dropout

        txt_attn_output = attn.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


@maybe_allow_in_graph
class QwenImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
        zero_cond_t: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        # Image processing modules
        self.img_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),  # For scale, shift, gate for norm1 and norm2
        )
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,  # Enable cross attention for joint computation
            added_kv_proj_dim=dim,  # Enable added KV projections for text stream
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=QwenDoubleStreamAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=eps,
        )
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        # Text processing modules
        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),  # For scale, shift, gate for norm1 and norm2
        )
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        # Text doesn't need separate attention - it's handled by img_attn joint computation
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.zero_cond_t = zero_cond_t

    def _modulate(self, x, mod_params, index=None):
        """Apply modulation to input tensor"""
        # x: b l d, shift: b d, scale: b d, gate: b d
        shift, scale, gate = mod_params.chunk(3, dim=-1)

        if index is not None:
            # Assuming mod_params batch dim is 2*actual_batch (chunked into 2 parts)
            # So shift, scale, gate have shape [2*actual_batch, d]
            actual_batch = shift.size(0) // 2
            shift_0, shift_1 = shift[:actual_batch], shift[actual_batch:]  # each: [actual_batch, d]
            scale_0, scale_1 = scale[:actual_batch], scale[actual_batch:]
            gate_0, gate_1 = gate[:actual_batch], gate[actual_batch:]

            # index: [b, l] where b is actual batch size
            # Expand to [b, l, 1] to match feature dimension
            index_expanded = index.unsqueeze(-1)  # [b, l, 1]

            # Expand chunks to [b, 1, d] then broadcast to [b, l, d]
            shift_0_exp = shift_0.unsqueeze(1)  # [b, 1, d]
            shift_1_exp = shift_1.unsqueeze(1)  # [b, 1, d]
            scale_0_exp = scale_0.unsqueeze(1)
            scale_1_exp = scale_1.unsqueeze(1)
            gate_0_exp = gate_0.unsqueeze(1)
            gate_1_exp = gate_1.unsqueeze(1)

            # Use torch.where to select based on index
            shift_result = torch.where(index_expanded == 0, shift_0_exp, shift_1_exp)
            scale_result = torch.where(index_expanded == 0, scale_0_exp, scale_1_exp)
            gate_result = torch.where(index_expanded == 0, gate_0_exp, gate_1_exp)
        else:
            shift_result = shift.unsqueeze(1)
            scale_result = scale.unsqueeze(1)
            gate_result = gate.unsqueeze(1)

        return x * (1 + scale_result) + shift_result, gate_result

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        modulate_index: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get modulation parameters for both streams
        img_mod_params = self.img_mod(temb)  # [B, 6*dim]

        if self.zero_cond_t:
            temb = torch.chunk(temb, 2, dim=0)[0]
        txt_mod_params = self.txt_mod(temb)  # [B, 6*dim]

        # Split modulation parameters for norm1 and norm2
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]

        # Process image stream - norm1 + modulation
        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1, modulate_index)

        # Process text stream - norm1 + modulation
        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        # Use QwenAttnProcessor2_0 for joint attention computation
        # This directly implements the DoubleStreamLayerMegatron logic:
        # 1. Computes QKV for both streams
        # 2. Applies QK normalization and RoPE
        # 3. Concatenates and runs joint attention
        # 4. Splits results back to separate streams
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=img_modulated,  # Image stream (will be processed as "sample")
            encoder_hidden_states=txt_modulated,  # Text stream (will be processed as "context")
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        # QwenAttnProcessor2_0 returns (img_output, txt_output) when encoder_hidden_states is provided
        img_attn_output, txt_attn_output = attn_output

        # Apply attention gates and add residual (like in Megatron)
        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        # Process image stream - norm2 + MLP
        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2, modulate_index)
        img_mlp_output = self.img_mlp(img_modulated2)
        hidden_states = hidden_states + img_gate2 * img_mlp_output

        # Process text stream - norm2 + MLP
        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        txt_mlp_output = self.txt_mlp(txt_modulated2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

        # Clip to prevent overflow for fp16
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states

class LocalQwenImageTransformer2DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin
):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["QwenImageTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _repeated_blocks = ["QwenImageTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: Optional[int] = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
        zero_cond_t: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)

        self.time_text_embed = QwenTimestepProjEmbeddings(embedding_dim=self.inner_dim)

        self.txt_norm = RMSNorm(joint_attention_dim, eps=1e-6)

        self.img_in = nn.Linear(in_channels, self.inner_dim)
        self.txt_in = nn.Linear(joint_attention_dim, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    zero_cond_t=zero_cond_t,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False
        self.zero_cond_t = zero_cond_t
        self.register_buffer("__index_timestep_zero__", torch.tensor([0]))

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens: Optional[List[int]] = None,
        guidance: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states = self.img_in(hidden_states)

        timestep = timestep.to(hidden_states.dtype)

        if self.zero_cond_t:
            timestep = torch.cat([timestep, timestep * 0], dim=0)
            modulate_index = torch.tensor(
                [[0] * prod(sample[0]) + [1] * sum([prod(s) for s in sample[1:]]) for sample in img_shapes],
                device=timestep.device,
                dtype=torch.int,
            )
        else:
            modulate_index = None

        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, hidden_states)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states)
        )

        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    encoder_hidden_states_mask,
                    temb,
                    image_rotary_emb,
                    attention_kwargs,
                    modulate_index,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=attention_kwargs,
                    modulate_index=modulate_index,
                )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        if self.zero_cond_t:
            temb = temb.chunk(2, dim=0)[0]
        # Use only the image part (hidden_states) from the dual-stream blocks
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

class Qwen2511Pipeline(BasePipeline):
    name = 'qwen2511'
    checkpointable_layers = ['TransformerLayer']
    adapter_target_modules = ['QwenImageTransformerBlock']

    prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    prompt_template_encode_start_idx = 34
    prompt_template_encode_edit = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
    prompt_template_encode_start_idx_edit = 64

    vlm_image_size = 1024

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']
        self.offloader = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)
        dtype = self.model_config['dtype']

        self.preprocess_media_file_fn = PreprocessMediaFile(self.config, support_video=True, framerate=1)

        current_dir = Path(__file__).parent.parent
        configs_dir = current_dir / 'configs' / 'qwen_2511'
        
        tokenizer = transformers.Qwen2Tokenizer.from_pretrained(str(configs_dir / 'tokenizer'), local_files_only=True)
        processor = transformers.Qwen2VLProcessor.from_pretrained(str(configs_dir / 'processor'), local_files_only=True)

        if 'text_encoder_path' in self.model_config:
            text_encoder_path = self.model_config['text_encoder_path']
        else:
            text_encoder_path = Path(self.model_config['diffusers_path']) / 'text_encoder'
        text_encoder_config = transformers.Qwen2_5_VLConfig.from_pretrained(str(configs_dir / 'text_encoder'), local_files_only=True)
        with init_empty_weights():
            text_encoder = transformers.Qwen2_5_VLForConditionalGeneration(text_encoder_config)
        for key, tensor in iterate_safetensors(text_encoder_path):
            key = key.replace('model.', 'language_model.')
            key = 'model.' + key
            if 'lm_head' in key:
                key = 'lm_head.weight'
            set_module_tensor_to_device(text_encoder, key, device='cpu', dtype=dtype, value=tensor)

        if 'vae_path' in self.model_config:
            vae_path = self.model_config['vae_path']
        else:
            vae_path = Path(self.model_config['diffusers_path']) / 'vae'
        with open(configs_dir / 'vae' / 'config.json') as f:
            vae_config = json.load(f)
        with init_empty_weights():
            vae = diffusers.AutoencoderKLQwenImage.from_config(vae_config)
        for key, tensor in iterate_safetensors(vae_path):
            set_module_tensor_to_device(vae, key, device='cpu', dtype=dtype, value=tensor)

        self.diffusers_pipeline = diffusers.QwenImagePipeline(
            scheduler=None,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=None,
        )

        self.diffusers_pipeline.processor = processor

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(dtype)
        )
        latents_std = torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(dtype)
        self.vae.register_buffer('latents_mean_tensor', latents_mean)
        self.vae.register_buffer('latents_std_tensor', latents_std)

    def load_diffusion_model(self):
        dtype = self.model_config['dtype']
        transformer_dtype = self.model_config.get('transformer_dtype', dtype)

        if 'transformer_path' in self.model_config:
            transformer_path = self.model_config['transformer_path']
        else:
            transformer_path = Path(self.model_config['diffusers_path']) / 'transformer'
        
        current_dir = Path(__file__).parent.parent
        configs_dir = current_dir / 'configs' / 'qwen_2511'
        
        with open(configs_dir / 'transformer' / 'config.json') as f:
            json_config = json.load(f)

        with init_empty_weights():
            transformer = LocalQwenImageTransformer2DModel.from_config(json_config)
            
            if not hasattr(transformer, 'img_in'):
                transformer.img_in = nn.Linear(64, 3072, bias=True)
            
            if not hasattr(transformer, 'time_text_embed'):
                transformer.time_text_embed = QwenTimestepProjEmbeddings(3072)
            
            if not hasattr(transformer, 'guidance_embed') and json_config.get('guidance_embeds', False):
                 transformer.guidance_embed = TimestepEmbedding(in_channels=256, time_embed_dim=3072, act_fn="silu")
        for key, tensor in iterate_safetensors(transformer_path):
            dtype_to_use = dtype if any(keyword in key for keyword in KEEP_IN_HIGH_PRECISION) or tensor.ndim == 1 else transformer_dtype
            set_module_tensor_to_device(transformer, key, device='cpu', dtype=dtype_to_use, value=tensor)

        attn_processor = QwenDoubleStreamAttnProcessor2_0()
        for block in transformer.transformer_blocks:
            block.attn.set_processor(attn_processor)

        self.diffusers_pipeline.transformer = transformer

        self.transformer.train()
        for name, p in self.transformer.named_parameters():
            p.original_name = name

    def __getattr__(self, name):
        return getattr(self.diffusers_pipeline, name)

    def get_vae(self):
        return self.vae

    def get_text_encoders(self):
        return [self.text_encoder]

    def save_adapter(self, save_dir, peft_state_dict):
        self.peft_config.save_pretrained(save_dir)
        peft_state_dict = {'diffusion_model.'+k: v for k, v in peft_state_dict.items()}
        safetensors.torch.save_file(peft_state_dict, save_dir / 'adapter_model.safetensors', metadata={'format': 'pt'})

    def save_model(self, save_dir, state_dict):
        safetensors.torch.save_file(state_dict, save_dir / 'model.safetensors', metadata={'format': 'pt'})

    def get_preprocess_media_file_fn(self):
        return self.preprocess_media_file_fn

    def get_call_vae_fn(self, vae):
        def fn(*args):
            image = args[0]
            latents = vae.encode(image.to(vae.device, vae.dtype)).latent_dist.mode()
            latents = (latents - vae.latents_mean_tensor) / vae.latents_std_tensor
            result = {'latents': latents}
            if len(args) == 2:
                control_data = args[1]
                if isinstance(control_data, list):
                    control_latents_list = []
                    for control_image in control_data:
                        cl = vae.encode(control_image.to(vae.device, vae.dtype)).latent_dist.mode()
                        cl = (cl - vae.latents_mean_tensor) / vae.latents_std_tensor
                        control_latents_list.append(cl)
                    result['control_latents'] = control_latents_list
                else:
                    control_latents = vae.encode(control_data.to(vae.device, vae.dtype)).latent_dist.mode()
                    control_latents = (control_latents - vae.latents_mean_tensor) / vae.latents_std_tensor
                    result['control_latents'] = control_latents
            return result
        return fn

    def load_image_for_vlm(self, path):
        pil_img = Image.open(path)
        height, width = pil_img.height, pil_img.width

        if pil_img.mode not in ['RGB', 'RGBA'] and 'transparency' in pil_img.info:
            pil_img = pil_img.convert('RGBA')

        if pil_img.mode == 'RGBA':
            canvas = Image.new('RGBA', pil_img.size, (255, 255, 255))
            canvas.alpha_composite(pil_img)
            pil_img = canvas.convert('RGB')
        else:
            pil_img = pil_img.convert('RGB')

        scale_factor = self.vlm_image_size / math.sqrt(height*width)
        return ImageOps.scale(pil_img, scale_factor)

    def _get_qwen_prompt_embeds(
        self,
        prompt,
        control_files,
        device=None,
        dtype=None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if control_files is None:
            template = self.prompt_template_encode
            drop_idx = self.prompt_template_encode_start_idx
            txt = [template.format(e) for e in prompt]
            txt_tokens = self.tokenizer(
                txt, max_length=self.tokenizer_max_length + drop_idx, padding=True, truncation=True, return_tensors="pt"
            ).to(device)
            attention_mask = txt_tokens.attention_mask
            outputs = self.text_encoder(
                input_ids=txt_tokens.input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        else:
            template = self.prompt_template_encode_edit
            drop_idx = self.prompt_template_encode_start_idx_edit
            txt = [template.format(e) for e in prompt]
            images = [
                self.load_image_for_vlm(file)
                for file in control_files
            ]
            model_inputs = self.processor(
                text=txt,
                images=images,
                padding=True,
                return_tensors="pt",
            ).to(device)
            attention_mask = model_inputs.attention_mask
            outputs = self.text_encoder(
                input_ids=model_inputs.input_ids,
                attention_mask=attention_mask,
                pixel_values=model_inputs.pixel_values,
                image_grid_thw=model_inputs.image_grid_thw,
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]

        return split_hidden_states

    def get_call_text_encoder_fn(self, text_encoder):
        def fn(caption, is_video, control_file: list[str] | None):
            # args are lists
            assert not any(is_video)
            prompt_embeds = self._get_qwen_prompt_embeds(caption, control_file, device=text_encoder.device)
            return {'prompt_embeds': prompt_embeds}
        return fn

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        prompt_embeds = inputs['prompt_embeds']
        mask = inputs['mask']
        device = latents.device

        # prompt embeds are variable length
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.bool, device=device) for e in prompt_embeds]
        max_seq_len = max([e.size(0) for e in prompt_embeds])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in prompt_embeds]
        )
        prompt_embeds_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        max_text_len = prompt_embeds_mask.sum(dim=1).max().item()
        prompt_embeds = prompt_embeds[:, :max_text_len, :]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_text_len]

        bs, channels, num_frames, h, w = latents.shape

        num_channels_latents = self.transformer.config.in_channels // 4
        assert num_channels_latents == channels
        latents = self._pack_latents(latents, bs, num_channels_latents, h, w)

        if mask is not None:
            mask = mask.unsqueeze(1).expand((-1, num_channels_latents, -1, -1)) 
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')  
            mask = mask.unsqueeze(2) 
            mask = self._pack_latents(mask, bs, num_channels_latents, h, w)

        timestep_sample_method = self.model_config.get('timestep_sample_method', 'logit_normal')

        if timestep_sample_method == 'logit_normal':
            dist = torch.distributions.normal.Normal(0, 1)
        elif timestep_sample_method == 'uniform':
            dist = torch.distributions.uniform.Uniform(0, 1)
        else:
            raise NotImplementedError()

        if timestep_quantile is not None:
            t = dist.icdf(torch.full((bs,), timestep_quantile, device=device))
        else:
            t = dist.sample((bs,)).to(device)

        if timestep_sample_method == 'logit_normal':
            sigmoid_scale = self.model_config.get('sigmoid_scale', 1.0)
            t = t * sigmoid_scale
            t = torch.sigmoid(t)

        if shift := self.model_config.get('shift', None):
            t = (t * shift) / (1 + (shift - 1) * t)
        elif self.model_config.get('flux_shift', False):
            mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
            t = time_shift(mu, 1.0, t)

        x_1 = latents
        x_0 = torch.randn_like(x_1)
        t_expanded = t.view(-1, 1, 1)
        x_t = (1 - t_expanded) * x_1 + t_expanded * x_0
        target = x_0 - x_1

        img_shapes = [(1, h // 2, w // 2)]

        if 'control_latents' in inputs:
            control_data = inputs['control_latents']
            if isinstance(control_data, list):
                all_control_latents = []
                for control_latents in control_data:
                    cl = control_latents.float()
                    cl = self._pack_latents(cl, bs, num_channels_latents, h, w)
                    all_control_latents.append(cl)
                combined_control = torch.cat(all_control_latents, dim=1)
                img_seq_len = torch.tensor(x_t.shape[1], device=x_t.device).repeat((bs,))
                extra = (img_seq_len,)
                x_t = torch.cat([x_t, combined_control], dim=1)
                for _ in range(len(all_control_latents)):
                    img_shapes.append((1, h // 2, w // 2))
            else:
                control_latents = control_data.float()
                control_latents = self._pack_latents(control_latents, bs, num_channels_latents, h, w)
                assert control_latents.shape == latents.shape, (control_latents.shape, latents.shape)
                img_seq_len = torch.tensor(x_t.shape[1], device=x_t.device).repeat((bs,))
                extra = (img_seq_len,)
                x_t = torch.cat([x_t, control_latents], dim=1)
                img_shapes.append((1, h // 2, w // 2))
        else:
            extra = tuple()

        img_shapes = torch.tensor([img_shapes], dtype=torch.int32, device=device).repeat((bs, 1, 1))
        txt_seq_lens = torch.tensor([max_text_len], dtype=torch.int32, device=device).repeat((bs,))
        img_attention_mask = torch.ones((bs, x_t.shape[1]), dtype=torch.bool, device=device)
        attention_mask = torch.cat([prompt_embeds_mask, img_attention_mask], dim=1)
        attention_mask = attention_mask.view(bs, 1, 1, -1)
        assert attention_mask.dtype == torch.bool

        # Guidance
        guidance_scale = self.model_config.get('guidance_scale', 1.0)
        guidance = torch.full((bs,), guidance_scale, device=device, dtype=torch.float32)

        return (
            (x_t, prompt_embeds, attention_mask, t, img_shapes, txt_seq_lens, guidance) + extra,
            (target, mask),
        )

    def to_layers(self):
        transformer = self.transformer
        layers = [InitialLayer(transformer)]
        for i, block in enumerate(transformer.transformer_blocks):
            layers.append(TransformerLayer(block, i, self.offloader))
        layers.append(FinalLayer(transformer))
        return layers

    def enable_block_swap(self, blocks_to_swap):
        transformer = self.transformer
        blocks = transformer.transformer_blocks
        num_blocks = len(blocks)
        assert (
            blocks_to_swap <= num_blocks - 2
        ), f'Cannot swap more than {num_blocks - 2} blocks. Requested {blocks_to_swap} blocks to swap.'
        self.offloader = ModelOffloader(
            'TransformerBlock', blocks, num_blocks, blocks_to_swap, True, torch.device('cuda'), self.config['reentrant_activation_checkpointing']
        )
        transformer.transformer_blocks = None
        transformer.to('cuda')
        transformer.transformer_blocks = blocks
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
    def __init__(self, model):
        super().__init__()
        self.img_in = model.img_in
        self.txt_norm = model.txt_norm
        self.txt_in = model.txt_in
        self.time_text_embed = model.time_text_embed
        self.pos_embed = model.pos_embed
        self.guidance_embed = getattr(model, 'guidance_embed', None)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)

        hidden_states, encoder_hidden_states, attention_mask, timestep, img_shapes, txt_seq_lens, guidance, *extra = inputs

        if hidden_states.ndim == 5:
             B, C, F, H, W = hidden_states.shape
             hidden_states = hidden_states.permute(0, 2, 3, 4, 1).flatten(1, 3) 
        
        hidden_states = self.img_in(hidden_states)

        timestep = timestep.to(hidden_states.dtype)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        temb = self.time_text_embed(timestep, hidden_states)
        
        if self.guidance_embed is not None:
             if guidance is None:
                 guidance = torch.zeros_like(timestep)
             
             g_emb = get_timestep_embedding(guidance, 256)
             temb = temb + self.guidance_embed(g_emb.to(dtype=temb.dtype))

        if isinstance(img_shapes, torch.Tensor):
             img_shapes = img_shapes.tolist()
        if isinstance(txt_seq_lens, torch.Tensor):
             txt_seq_lens = txt_seq_lens.tolist()
             
        vid_freqs, txt_freqs = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)

        return make_contiguous(hidden_states, encoder_hidden_states, attention_mask, temb, vid_freqs, txt_freqs) + tuple(extra)

class TransformerLayer(nn.Module):
    def __init__(self, block, block_idx, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, encoder_hidden_states, attention_mask, temb, vid_freqs, txt_freqs, *extra = inputs

        self.offloader.wait_for_block(self.block_idx)
        encoder_hidden_states, hidden_states = self.block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=None,
            temb=temb,
            image_rotary_emb=(vid_freqs, txt_freqs),
            joint_attention_kwargs={'attention_mask': attention_mask},
        )
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(hidden_states, encoder_hidden_states, attention_mask, temb, vid_freqs, txt_freqs) + tuple(extra)


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.norm_out = model.norm_out
        self.proj_out = model.proj_out
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, encoder_hidden_states, attention_mask, temb, vid_freqs, txt_freqs, *extra = inputs
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)
        if len(extra) > 0:
            assert len(extra) == 1
            img_seq_len = extra[0][0].item()
            output = output[:, :img_seq_len, ...]
        return output
