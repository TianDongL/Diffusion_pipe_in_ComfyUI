import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from pathlib import Path
from einops import rearrange
from safetensors.torch import save_file

try:
    from diffusers import ZImageTransformer2DModel
except ImportError:
    ZImageTransformer2DModel = None

from diffusers import AutoencoderKL
from transformers import AutoModel, AutoTokenizer

from models.base import BasePipeline, make_contiguous
from utils.common import is_main_process, AUTOCAST_DTYPE, load_state_dict
from utils.offloading import ModelOffloader

from models.zimage_comfy import ZImagePipeline as ZImageComfyPipeline

class ZImageDiffusersPipeline(BasePipeline):
    name = 'z_image'
    
    adapter_target_modules = ['ZImageTransformerBlock']
    checkpointable_layers = ['TransformerWrapper']

    def __init__(self, config):
        if ZImageTransformer2DModel is None:
            raise ImportError("ZImageTransformer2DModel not found. Please install diffusers from source: pip install git+https://github.com/huggingface/diffusers")

        self.config = config
        self.model_config = self.config['model']
        self.offloader = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)
        
        dtype = self.model_config['dtype']
        checkpoint_path = self.model_config['checkpoint_path']
        checkpoint_path_obj = Path(checkpoint_path)
        
        if not checkpoint_path_obj.exists():
            raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
        
        if is_main_process():
            print(f"Loading Z-Image-Turbo model from {checkpoint_path}")

        # Load transformer
        transformer_path = checkpoint_path_obj / "transformer"
        if transformer_path.exists():
            self.transformer = ZImageTransformer2DModel.from_pretrained(
                str(transformer_path),
                torch_dtype=dtype,
                low_cpu_mem_usage=False,
            )
        else:
            self.transformer = ZImageTransformer2DModel.from_pretrained(
                str(checkpoint_path_obj),
                subfolder="transformer",
                torch_dtype=dtype,
                low_cpu_mem_usage=False,
            )
        
        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            str(checkpoint_path_obj),
            subfolder="vae",
            torch_dtype=dtype,
        )
        
        # Load Text Encoder & Tokenizer (Qwen)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint_path_obj),
            subfolder="tokenizer",
            trust_remote_code=True
        )
        
        self.text_encoder = AutoModel.from_pretrained(
            str(checkpoint_path_obj),
            subfolder="text_encoder",
            torch_dtype=dtype,
            trust_remote_code=True
        )
        
        # Store paths for potential reloading (e.g., in TrainingSampler)
        self.text_encoder_path = str(checkpoint_path_obj / "text_encoder") if hasattr(checkpoint_path_obj, '__truediv__') else f"{checkpoint_path_obj}/text_encoder"
        self.checkpoint_path = str(checkpoint_path_obj)
        
        self.transformer.train()
        self.text_encoder.train()
        
        for name, p in self.transformer.named_parameters():
            p.original_name = name

        if 'merge_adapters' in self.model_config:
            adapters = self.model_config['merge_adapters']
            if isinstance(adapters, str):
                adapters = [adapters]
                
            for adapter_path in adapters:
                if is_main_process():
                    print(f"Merging adapter: {adapter_path}")
                self.load_adapter_weights(adapter_path, fuse=True)
                self.transformer.unload_lora()

    def get_vae(self):
        return self.vae

    def get_text_encoders(self):
        return [self.text_encoder]

    def get_call_vae_fn(self, vae):
        def fn(tensor):
            latents = vae.encode(tensor.to(vae.device, vae.dtype)).latent_dist.mode()
            if hasattr(vae.config, 'shift_factor') and vae.config.shift_factor is not None:
                latents = latents - vae.config.shift_factor
            latents = latents * vae.config.scaling_factor
            return {'latents': latents}
        return fn

    def _extract_masked_hidden(self, hidden_states, attention_mask):
        res = []
        for i in range(hidden_states.shape[0]):
            valid = hidden_states[i][attention_mask[i] == 1]
            res.append(valid)
        return res

    def get_call_text_encoder_fn(self, text_encoder):
        prompt_template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        drop_idx = 34 
        
        def fn(caption, is_video):
            assert not any(is_video), "Z-Image-Turbo only supports image generation"
            
            device = text_encoder.device
            
            caption = [prompt_template.format(c) for c in caption]
            
            text_inputs = self.tokenizer(
                caption,
                padding="max_length",
                max_length=256 + drop_idx,  
                truncation=True,
                return_tensors="pt",
            )
            
            input_ids = text_inputs.input_ids.to(device)
            attention_mask = text_inputs.attention_mask.to(device)
            
            outputs = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            qwen_embeds = outputs.last_hidden_state  
            
            split_embeds = self._extract_masked_hidden(qwen_embeds, attention_mask)
            split_embeds = [e[drop_idx:] for e in split_embeds]
            
            pooled = torch.stack([e.mean(dim=0) for e in split_embeds])
            
            return {'encoder_hidden_states': split_embeds, 'pooled_projections': pooled}
        return fn

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        encoder_hidden_states = inputs['encoder_hidden_states'] # List[Tensor]
        pooled_projections = inputs['pooled_projections']
        mask = inputs.get('mask', torch.tensor([]))

        bs, c, h, w = latents.shape
        device = latents.device
        
        lengths = [e.shape[0] for e in encoder_hidden_states]
        max_len = max(lengths)
        embed_dim = encoder_hidden_states[0].shape[1]
        dtype = encoder_hidden_states[0].dtype
        
        padded_embeds = torch.zeros(bs, max_len, embed_dim, dtype=dtype, device=device)
        for i, e in enumerate(encoder_hidden_states):
            l = lengths[i]
            padded_embeds[i, :l] = e.to(device)
            
        txt_lens = torch.tensor(lengths, dtype=torch.long, device=device)

        img_ids = self._prepare_latent_image_ids(bs, h, w, device, dtype)
        if img_ids.ndim == 2:
            img_ids = img_ids.unsqueeze(0).repeat((bs, 1, 1))
        
        txt_ids = torch.zeros(bs, max_len, 3).to(device, dtype)

        # Timestep sampling
        timestep_sample_method = self.model_config.get('timestep_sample_method', 'logit_normal')
        
        if timestep_sample_method == 'logit_normal':
            dist = torch.distributions.normal.Normal(0, 1)
        else:
            dist = torch.distributions.uniform.Uniform(0, 1)

        if timestep_quantile is not None:
            t = dist.icdf(torch.full((bs,), timestep_quantile, device=device))
        else:
            t = dist.sample((bs,)).to(device)

        if timestep_sample_method == 'logit_normal':
            sigmoid_scale = self.model_config.get('sigmoid_scale', 1.0)
            t = t * sigmoid_scale
            t = torch.sigmoid(t)

        # Flow matching noise schedule
        x_1 = latents
        x_0 = torch.randn_like(x_1)
        t_expanded = t.view(-1, 1, 1, 1)
        x_t = (1 - t_expanded) * x_1 + t_expanded * x_0
        
        target = x_1 - x_0 
        
        # Flatten latents
        x_t = rearrange(x_t, "b c h w -> b (h w) c")
        target = rearrange(target, "b c h w -> b (h w) c")
        
        guidance = self.model_config.get('guidance', 0.0) 
        guidance_vec = torch.full((bs,), guidance, device=device, dtype=torch.float32)
        
        img_seq_len = torch.tensor(x_t.shape[1], device=device).repeat((bs,))
        model_t = 1.0 - t
        
        return (x_t, padded_embeds, pooled_projections, model_t, img_ids, txt_ids, guidance_vec, img_seq_len, txt_lens), (target, mask)

    def _prepare_latent_image_ids(self, batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]
        latent_image_ids = latent_image_ids.reshape(-1, 3)
        return latent_image_ids.to(device=device, dtype=dtype)

    def to_layers(self):
        transformer = self.transformer
        if hasattr(transformer, 'layers'):
            blocks = transformer.layers
        elif hasattr(transformer, 'blocks'):
            blocks = transformer.blocks
        elif hasattr(transformer, 'transformer_blocks'):
            blocks = transformer.transformer_blocks
        else:
            raise AttributeError("Could not find blocks in ZImageTransformer2DModel")

        layers = [EmbeddingWrapper(transformer)]
        for i, block in enumerate(blocks):
            layers.append(TransformerWrapper(block, i, self.offloader))
        layers.append(OutputWrapper(transformer))
        return layers

    def enable_block_swap(self, blocks_to_swap):
        transformer = self.transformer
        
        if hasattr(transformer, 'transformer_blocks'):
            block_attr = 'transformer_blocks'
        elif hasattr(transformer, 'blocks'):
            block_attr = 'blocks'
        elif hasattr(transformer, 'layers'):
            block_attr = 'layers'
        else:
            raise AttributeError("Could not find blocks in ZImageTransformer2DModel")
            
        blocks = getattr(transformer, block_attr)
        num_blocks = len(blocks)
        assert blocks_to_swap <= num_blocks - 2, f"Cannot swap more than {num_blocks - 2} blocks. Requested {blocks_to_swap} blocks to swap."

        self.offloader = ModelOffloader(
            'Block', blocks, num_blocks, blocks_to_swap, True, 
            torch.device('cuda'), self.config.get('reentrant_activation_checkpointing', False)
        )
        
        # Move non-block modules to GPU
        setattr(transformer, block_attr, None)
        transformer.to('cuda')
        setattr(transformer, block_attr, blocks)
        
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

    def get_param_groups(self, parameters):
        return [{'params': parameters}]

    def get_loss_fn(self):
        def loss_fn(output, label):
            if isinstance(output, list):
                output = torch.stack(output)
            
            if output.ndim == 5:
                output = rearrange(output, 'b c f h w -> b (f h w) c')            
            
            target, mask = label
            with torch.autocast('cuda', enabled=False):
                output = output.to(torch.float32)
                target = target.to(output.device, torch.float32)
                
                if random.random() < 0.01 and torch.distributed.get_rank() == 0:
                    print(f"DEBUG shapes - output: {output.shape}, target: {target.shape}")
                
                if 'pseudo_huber_c' in self.config:
                    c = self.config['pseudo_huber_c']
                    loss = torch.sqrt((output-target)**2 + c**2) - c
                else:
                    loss = F.mse_loss(output, target, reduction='none')
                
                if mask.numel() > 0:
                    mask = mask.to(output.device, torch.float32)
                    loss *= mask
                    
                loss = loss.mean()
            return loss
        return loss_fn

    def save_model(self, save_dir, diffusers_sd):
        print(f"Saving model to {save_dir / 'model.safetensors'}")
        save_file(diffusers_sd, save_dir / 'model.safetensors', metadata={"format": "pt"})

    def load_adapter_weights(self, adapter_path, fuse=False, fuse_weight=1.0):
        print(f'Loading adapter weights from path {adapter_path}')
        if adapter_path.endswith('.safetensors'):
            safetensors_file = adapter_path
        else:
            safetensors_files = list(Path(adapter_path).glob('*.safetensors'))
            if len(safetensors_files) == 0:
                raise RuntimeError(f'No safetensors file found in {adapter_path}')
            safetensors_file = safetensors_files[0]
            
        adapter_state_dict = load_state_dict(safetensors_file)
        
        # Try manual fusion if load_lora_weights is missing
        if not hasattr(self.transformer, 'load_lora_weights'):
            if fuse:
                self._manual_load_and_fuse_lora(adapter_state_dict, fuse_weight)
            else:
                raise NotImplementedError("Loading LoRA without fusing is not supported for this model manually yet.")
        else:
            self.transformer.load_lora_weights(adapter_state_dict, adapter_name='default')
            if fuse:
                self.transformer.fuse_lora(lora_scale=fuse_weight)

    def _convert_keys_to_comfyui_format(self, state_dict):
        """
        Convert LoRA keys from training format to ComfyUI format.
        
        Training format examples (from peft):
            base_model.model.context_refiner.0.attention.to_q.lora_A.weight
            transformer.context_refiner.0.attention.to_q.lora_A.weight
            
        ComfyUI format examples (target):
            diffusion_model.context_refiner.0.attention.qkv.lora_A.weight
            diffusion_model.context_refiner.0.attention.out.lora_A.weight
        """
        converted_dict = {}
        qkv_groups = {}  
        
        for key, value in state_dict.items():
            original_key = key
            
            if key.startswith("base_model.model."):
                key = key[len("base_model.model."):]
            elif key.startswith("transformer."):
                key = key[len("transformer."):]
            
            if (".attention.to_q." in key or ".attention.to_k." in key or ".attention.to_v." in key):
                if "lora_A" in key or "lora_B" in key:
                    if ".attention.to_q." in key:
                        qkv_type = "to_q"
                        base_key = key.replace(".attention.to_q.", ".attention.qkv.")
                    elif ".attention.to_k." in key:
                        qkv_type = "to_k"
                        base_key = key.replace(".attention.to_k.", ".attention.qkv.")
                    else:  # to_v
                        qkv_type = "to_v"
                        base_key = key.replace(".attention.to_v.", ".attention.qkv.")
                    
                    base_key = "diffusion_model." + base_key
                    
                    if base_key not in qkv_groups:
                        qkv_groups[base_key] = {}
                    qkv_groups[base_key][qkv_type] = value
                    continue
                elif "alpha" in key:
                    if ".attention.to_q." in key:
                        key = key.replace(".attention.to_q.", ".attention.qkv.")
                    elif ".attention.to_k." in key:
                        key = key.replace(".attention.to_k.", ".attention.qkv.")
                    elif ".attention.to_v." in key:
                        key = key.replace(".attention.to_v.", ".attention.qkv.")
            
            if ".attention.to_out.0." in key:
                key = key.replace(".attention.to_out.0.", ".attention.out.")
            
            if not key.startswith("diffusion_model."):
                key = "diffusion_model." + key
            
            converted_dict[key] = value
        
        for base_key, qkv_dict in qkv_groups.items():
            if len(qkv_dict) == 3:  
                if "lora_A" in base_key:
                    q_weight = qkv_dict["to_q"]
                    k_weight = qkv_dict["to_k"]
                    v_weight = qkv_dict["to_v"]
                    merged_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
                    converted_dict[base_key] = merged_weight
                elif "lora_B" in base_key:

                    
                    q_weight = qkv_dict["to_q"]
                    k_weight = qkv_dict["to_k"]
                    v_weight = qkv_dict["to_v"]
                    
                    out_dim, rank = q_weight.shape  
                    device = q_weight.device
                    dtype = q_weight.dtype
                    
                    merged_weight = torch.zeros(3 * out_dim, 3 * rank, device=device, dtype=dtype)
                    merged_weight[0:out_dim, 0:rank] = q_weight
                    merged_weight[out_dim:2*out_dim, rank:2*rank] = k_weight
                    merged_weight[2*out_dim:3*out_dim, 2*rank:3*rank] = v_weight
                    converted_dict[base_key] = merged_weight
                else:
                    print(f"Warning: Skipping unexpected key type: {base_key}")
            else:
                for qkv_type, weight in qkv_dict.items():
                    converted_dict[base_key.replace(".qkv.", f".{qkv_type}.")] = weight
        
        return converted_dict

    def save_adapter(self, save_dir, peft_state_dict):
        from safetensors.torch import save_file
        
        # Convert keys to ComfyUI format
        #print("Converting LoRA keys to ComfyUI format...")
        converted_state_dict = self._convert_keys_to_comfyui_format(peft_state_dict)
        
        output_path = save_dir / "lora.safetensors"
        print(f"Saving adapter to {output_path}")
        save_file(converted_state_dict, output_path, metadata={"format": "pt"})

    def _manual_load_and_fuse_lora(self, state_dict, fuse_weight=1.0):
        print("Manually fusing LoRA weights...")
        # Group keys by module
        lora_groups = {}
        for key, value in state_dict.items():
            if 'lora' not in key:
                continue      
            
            # Identify base module name
            # Pattern 1: ...lora.down.weight / ...lora.up.weight
            # Pattern 2: ...lora_A.weight / ...lora_B.weight
            
            if 'lora.down.weight' in key:
                base_key = key.replace('.lora.down.weight', '')
                type_ = 'down'
            elif 'lora.up.weight' in key:
                base_key = key.replace('.lora.up.weight', '')
                type_ = 'up'
            elif 'lora_A.weight' in key:
                base_key = key.replace('.lora_A.weight', '')
                type_ = 'down'
            elif 'lora_B.weight' in key:
                base_key = key.replace('.lora_B.weight', '')
                type_ = 'up'
            else:
                continue

            # Remove prefixes
            if base_key.startswith('diffusion_model.'):
                base_key = base_key[len('diffusion_model.'):]
            if base_key.startswith('transformer.'):
                base_key = base_key[len('transformer.'):]
                
            if base_key not in lora_groups:
                lora_groups[base_key] = {}
            lora_groups[base_key][type_] = value

        # Fuse
        fused_count = 0
        for base_key, weights in lora_groups.items():
            if 'down' not in weights or 'up' not in weights:
                continue
            
            # Find module
            module = self.transformer
            parts = base_key.split('.')
            
            try:
                for part in parts:
                    # Handle mapping from 'layers' to 'transformer_blocks' if needed
                    if part == 'layers' and not hasattr(module, 'layers') and hasattr(module, 'transformer_blocks'):
                        part = 'transformer_blocks'
                    module = getattr(module, part)
            except AttributeError:
                print(f"Warning: Could not find module for key {base_key} (part: {part}), skipping.")
                continue
            
            # Unwrap PEFT layer if present
            if hasattr(module, 'base_layer'):
                module = module.base_layer
                
            if not isinstance(module, (nn.Linear, nn.Conv2d)):
                # Fallback for quantized or custom layers that look like Linear
                if hasattr(module, 'weight') and module.weight.ndim == 2:
                     pass # Treat as Linear
                elif hasattr(module, 'weight') and module.weight.ndim == 4:
                     pass # Treat as Conv2d
                else:
                    print(f"Warning: Module {base_key} (Type: {type(module)}) is not Linear or Conv2d, skipping.")
                    continue
                
            # Calculate delta
            down = weights['down'].float().to(module.weight.device)
            up = weights['up'].float().to(module.weight.device)
            
            # Rank check
            rank = down.shape[0]
            
            
            if isinstance(module, nn.Conv2d) or (hasattr(module, 'weight') and module.weight.ndim == 4):
                delta = torch.mm(up.flatten(1), down.flatten(1)).reshape(module.weight.shape)
            else:
                delta = torch.mm(up, down)
                
            module.weight.data += delta * fuse_weight
            fused_count += 1
            
        print(f"Manually fused {fused_count} LoRA modules.")


class EmbeddingWrapper(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer
        self.t_embedder = transformer.t_embedder
        self.all_x_embedder = transformer.all_x_embedder
        self.noise_refiner = transformer.noise_refiner
        self.cap_embedder = transformer.cap_embedder
        self.context_refiner = transformer.context_refiner
        self.rope_embedder = transformer.rope_embedder

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)
        

        hidden_states, encoder_hidden_states, pooled_projections, timestep, img_ids, txt_ids, guidance, img_seq_len, txt_lens = inputs
        
        device = hidden_states.device
        dtype = next(self.transformer.parameters()).dtype
        
        hidden_states = hidden_states.to(device, dtype=dtype)
        encoder_hidden_states = encoder_hidden_states.to(device, dtype=dtype)
        timestep = timestep.to(device, dtype=dtype)
        img_ids = img_ids.to(device)
        guidance = guidance.to(device, dtype=dtype)
        img_seq_len = img_seq_len.to(device)
        txt_lens = txt_lens.to(device)
        
        bs, seq_len, dim = hidden_states.shape
        h = int(img_ids[0, :, 1].max().item()) + 1
        w = int(img_ids[0, :, 2].max().item()) + 1
        
        hidden_states_5d = hidden_states.transpose(1, 2).reshape(bs, dim, 1, h, w)
        
        x = [hidden_states_5d[i] for i in range(bs)]
        

        cap_feats = []
        for i in range(bs):
            l = txt_lens[i]
            cap_feats.append(encoder_hidden_states[i, :l])

        
        patch_size = 2
        f_patch_size = 1
        
        t = timestep * self.transformer.t_scale
        t = self.t_embedder(t)

        (
            x,
            cap_feats,
            x_size,
            x_pos_ids,
            cap_pos_ids,
            x_inner_pad_mask,
            cap_inner_pad_mask,
        ) = self.transformer.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)

        x_item_seqlens = [len(_) for _ in x]
        x_max_item_seqlen = max(x_item_seqlens)

        x = torch.cat(x, dim=0)
        x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x)

        adaln_input = t.type_as(x)
        x[torch.cat(x_inner_pad_mask)] = self.transformer.x_pad_token
        x = list(x.split(x_item_seqlens, dim=0))
        x_freqs_cis = list(self.rope_embedder(torch.cat(x_pos_ids, dim=0)).split(x_item_seqlens, dim=0))

        from torch.nn.utils.rnn import pad_sequence
        x = pad_sequence(x, batch_first=True, padding_value=0.0)
        x_freqs_cis = pad_sequence(x_freqs_cis, batch_first=True, padding_value=0.0)
        x_attn_mask = torch.zeros((bs, x_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(x_item_seqlens):
            x_attn_mask[i, :seq_len] = 1

        # Noise Refiner
        for layer in self.noise_refiner:
            x = layer(x, x_attn_mask, x_freqs_cis, adaln_input)

        cap_item_seqlens = [len(_) for _ in cap_feats]
        cap_max_item_seqlen = max(cap_item_seqlens)

        cap_feats = torch.cat(cap_feats, dim=0)
        cap_feats = self.cap_embedder(cap_feats)
        cap_feats[torch.cat(cap_inner_pad_mask)] = self.transformer.cap_pad_token
        cap_feats = list(cap_feats.split(cap_item_seqlens, dim=0))
        cap_freqs_cis = list(self.rope_embedder(torch.cat(cap_pos_ids, dim=0)).split(cap_item_seqlens, dim=0))

        cap_feats = pad_sequence(cap_feats, batch_first=True, padding_value=0.0)
        cap_freqs_cis = pad_sequence(cap_freqs_cis, batch_first=True, padding_value=0.0)
        cap_attn_mask = torch.zeros((bs, cap_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(cap_item_seqlens):
            cap_attn_mask[i, :seq_len] = 1

        # Context Refiner
        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, cap_attn_mask, cap_freqs_cis)

        unified = []
        unified_freqs_cis = []
        for i in range(bs):
            x_len = x_item_seqlens[i]
            cap_len = cap_item_seqlens[i]
            unified.append(torch.cat([x[i][:x_len], cap_feats[i][:cap_len]]))
            unified_freqs_cis.append(torch.cat([x_freqs_cis[i][:x_len], cap_freqs_cis[i][:cap_len]]))
        unified_item_seqlens = [a + b for a, b in zip(cap_item_seqlens, x_item_seqlens)]
        unified_max_item_seqlen = max(unified_item_seqlens)

        unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
        unified_freqs_cis = pad_sequence(unified_freqs_cis, batch_first=True, padding_value=0.0)
        unified_attn_mask = torch.zeros((bs, unified_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(unified_item_seqlens):
            unified_attn_mask[i, :seq_len] = 1
            
        return unified, unified_attn_mask, unified_freqs_cis, adaln_input, x_size, patch_size, f_patch_size


class TransformerWrapper(nn.Module):
    def __init__(self, block, block_idx, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        unified, unified_attn_mask, unified_freqs_cis, adaln_input, x_size, patch_size, f_patch_size = inputs
        
        self.offloader.wait_for_block(self.block_idx)
        
        unified = self.block(unified, unified_attn_mask, unified_freqs_cis, adaln_input)
        
        self.offloader.submit_move_blocks_forward(self.block_idx)
            
        return unified, unified_attn_mask, unified_freqs_cis, adaln_input, x_size, patch_size, f_patch_size


class OutputWrapper(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer
        self.all_final_layer = transformer.all_final_layer

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        unified, unified_attn_mask, unified_freqs_cis, adaln_input, x_size, patch_size, f_patch_size = inputs
        
        unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, adaln_input)
        unified = list(unified.unbind(dim=0))
        x = self.transformer.unpatchify(unified, x_size, patch_size, f_patch_size)
        return x


class ZImagePipeline:
    def __new__(cls, config):
        if 'diffusion_model' in config['model']:
            return ZImageComfyPipeline(config)
        else:
            return ZImageDiffusersPipeline(config)
