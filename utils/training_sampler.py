import os
from pathlib import Path
from typing import Optional, Any
import time

import torch
import torchvision
from PIL import Image
import deepspeed


from utils.common import is_main_process
from models.z_image import EmbeddingWrapper, OutputWrapper, TransformerWrapper



import math
from typing import Union
from torch.distributions import LogNormal
from diffusers import FlowMatchEulerDiscreteScheduler
import torch
import numpy as np
from einops import rearrange

default_weighing_scheme = [1.0] * 1000

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


class CustomFlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_noise_sigma = 1.0
        self.timestep_type = "linear"

        with torch.no_grad():
            num_timesteps = 1000
            
            x = torch.arange(num_timesteps, dtype=torch.float32)
            y = torch.exp(-2 * ((x - num_timesteps / 2) / num_timesteps) ** 2)

            y_shifted = y - y.min()

            bsmntw_weighing = y_shifted * (num_timesteps / y_shifted.sum())

            hbsmntw_weighing = y_shifted * (num_timesteps / y_shifted.sum())

            hbsmntw_weighing[num_timesteps //
                             2:] = hbsmntw_weighing[num_timesteps // 2:].max()

            timesteps = torch.linspace(1000, 1, num_timesteps, device='cpu')

            self.linear_timesteps = timesteps
            self.linear_timesteps_weights = bsmntw_weighing
            self.linear_timesteps_weights2 = hbsmntw_weighing
            pass

    def get_weights_for_timesteps(self, timesteps: torch.Tensor, v2=False, timestep_type="linear") -> torch.Tensor:
        step_indices = [(self.timesteps == t).nonzero().item()
                        for t in timesteps]

        if timestep_type == "weighted":
            weights = torch.tensor(
                [default_weighing_scheme[i] for i in step_indices],
                device=timesteps.device,
                dtype=timesteps.dtype
            )
        if v2:
            weights = self.linear_timesteps_weights2[step_indices].flatten()
        else:
            weights = self.linear_timesteps_weights[step_indices].flatten()

        return weights

    def get_sigmas(self, timesteps: torch.Tensor, n_dim, dtype, device) -> torch.Tensor:
        sigmas = self.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item()
                        for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)

        return sigma

    def add_noise(
            self,
            original_samples: torch.Tensor,
            noise: torch.Tensor,
            timesteps: torch.Tensor,
    ) -> torch.Tensor:
        t_01 = (timesteps / 1000).to(original_samples.device)
        noisy_model_input = (1.0 - t_01) * original_samples + t_01 * noise
        return noisy_model_input

    def scale_model_input(self, sample: torch.Tensor, timestep: Union[float, torch.Tensor]) -> torch.Tensor:
        return sample

    def set_train_timesteps(
        self,
        num_timesteps,
        device,
        timestep_type='linear',
        latents=None,
        patch_size=1,
        shift=1.0
    ):
        self.timestep_type = timestep_type
        if timestep_type == 'linear' or timestep_type == 'weighted':
            full_steps = torch.linspace(1000, 0, num_timesteps + 1, device=device)
            self.timesteps = full_steps[:-1]
            self.sigmas = full_steps / 1000.0
            return self.timesteps
        elif timestep_type == 'sigmoid':
            t = torch.sigmoid(torch.randn((num_timesteps,), device=device))

            timesteps = ((1 - t) * 1000)

            timesteps, _ = torch.sort(timesteps, descending=True)

            self.timesteps = timesteps.to(device=device)
            self.sigmas = timesteps / 1000.0
            self.sigmas = torch.cat([self.sigmas, torch.zeros(1, device=device)])

            return timesteps
        elif timestep_type in ['flux_shift', 'lumina2_shift', 'shift']:
            sigmas = torch.linspace(1.0, 0.0, num_timesteps + 1, device=device)
            self.sigmas = sigmas # N+1 points
            
            # Dynamic Shifting Logic from ai-toolkit
            if latents is not None:
                # Assuming Flux/Z-Image style
                h = latents.shape[2]
                w = latents.shape[3]
                # Patch size 1 or 2? ai-toolkit mentions doubling patch size for Flux?
                # But here we just assume seq len based on H*W.
                # Z-Image uses patch_size=2? Or 1? 
                # ai-toolkit defaults patch_size=1 in method arg.
                
                image_seq_len = h * w // (patch_size**2)
                
                # Default Flux params from ai-toolkit
                mu = calculate_shift(
                    image_seq_len,
                    base_seq_len=256,
                    max_seq_len=4096,
                    base_shift=0.5,
                    max_shift=1.16
                )
                
                # Apply shift: sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
                # But shift is 'mu' here.
                sigmas = mu * sigmas / (1 + (mu - 1) * sigmas)
            else:
                # Fallback to static shift
                sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
            
            self.timesteps = sigmas[:-1] * 1000.0
            self.sigmas = sigmas
            return self.timesteps



        elif timestep_type == 'lognorm_blend':
            alpha = 0.75
            lognormal = LogNormal(loc=0, scale=0.333)
            t1 = lognormal.sample((int(num_timesteps * alpha),)).to(device)
            t1 = ((1 - t1/t1.max()) * 1000)
            t2 = torch.linspace(1000, 1, int(num_timesteps * (1 - alpha)), device=device)
            timesteps = torch.cat((t1, t2))
            
            timesteps, _ = torch.sort(timesteps, descending=True)
            timesteps = timesteps.to(torch.int)
            self.timesteps = timesteps.to(device=device)
            return timesteps
        else:
            raise ValueError(f"Invalid timestep type: {timestep_type}")


class TrainingSampler:

    def __init__(
        self,
        model_pipeline,
        output_dir: str,
        sample_every_n_steps: Optional[int] = None,
        sample_every_n_epochs: Optional[int] = None,
        save_images: bool = True,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.0,
        guidance_value: float = 4.0,
        sample_prompt: str = "",
        tb_writer: Optional[Any] = None, 
        shift: float = 1.0,
        timestep_type: str = "linear",
        height: int = 1024,
        width: int = 1024,
    ):

        self.model_pipeline = model_pipeline
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_every_n_steps = sample_every_n_steps
        self.sample_every_n_epochs = sample_every_n_epochs
        self.save_images = save_images
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.guidance_value = guidance_value
        self.sample_prompt = sample_prompt
        self.tb_writer = tb_writer
        self.shift = shift
        self.timestep_type = timestep_type
        self.height = height
        self.width = width
        self.save_to_tensorboard = tb_writer is not None # Derived from tb_writer
        self.save_to_wandb = False # Original was False, keeping it as is.
        self._last_sampled_epoch = -1
        
        if is_main_process():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
    def should_sample(self, step: Optional[int] = None, epoch: Optional[int] = None) -> bool:

        if step is not None and self.sample_every_n_steps is not None:
            if step % self.sample_every_n_steps == 0:
                return True
                
        if epoch is not None and self.sample_every_n_epochs is not None:
            if epoch % self.sample_every_n_epochs == 0 and epoch != self._last_sampled_epoch:
                return True
                
        return False
    
    @torch.no_grad()
    def sample_from_latents(
        self,
        latents: torch.Tensor,
        captions: Optional[list] = None,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        prefix: str = "sample",
    ) -> Optional[Image.Image]:

        if not is_main_process():
            return None
        
        # Update last sampled epoch
        if epoch is not None:
            self._last_sampled_epoch = epoch
        
        latents = latents.detach().clone()
        
        if latents.shape[0] > 1:
            latents = latents[0:1]
        
        vae = self.model_pipeline.get_vae()
        
        try:
            if hasattr(vae, 'first_stage_model'):
                vae_model = vae.first_stage_model
                vae_params = list(vae_model.parameters()) if hasattr(vae_model, 'parameters') else []
                
                if vae_params and str(vae_params[0].device) == 'meta':
                    print("[Sampler] VAE first_stage_model is on meta device. Moving to CUDA...")
                    vae.first_stage_model = vae.first_stage_model.to('cuda')
                    print("[Sampler] Moved VAE first_stage_model to CUDA")
            
            try:
                image = self._decode_comfy_vae(latents, vae)
            except Exception as e:
                print(f"[Sampler] Error decoding latents: {e}")
                import traceback
                traceback.print_exc()
                return []
                 
        except Exception as e:
            print(f"[Sampler] Error handling VAE: {e}")
            import traceback
            traceback.print_exc()
            return []
        
        self._save_images(image, captions, step, epoch, prefix)
        
        return image
    
    def _decode_latents(self, latents: torch.Tensor, vae) -> Image.Image:

        if hasattr(vae, 'decode'):
            return self._decode_diffusers_vae(latents, vae)
        elif hasattr(vae, 'decode_tiled'):
            return self._decode_comfy_vae(latents, vae)
        else:
            raise NotImplementedError(f"unknown VAE type: {type(vae)}")
    
    def _decode_diffusers_vae(self, latents: torch.Tensor, vae) -> Image.Image:
        model_config = getattr(self.model_pipeline, 'model_config', {})
        if model_config.get('type') == 'z_image':
            scaling_factor = 0.3611
            shift_factor = 0.1159
        else:
            scaling_factor = getattr(vae.config, 'scaling_factor', 0.18215)
            shift_factor = getattr(vae.config, 'shift_factor', 0.0)

        latents = latents / scaling_factor
        latents = latents + shift_factor
        
        device = getattr(vae, 'device', latents.device)
        dtype = getattr(vae, 'dtype', latents.dtype)
        
        latents = latents.to(device, dtype)
        
        decoded = vae.decode(latents).sample
        
        img = decoded[0].cpu().float()
        img = ((img + 1) / 2).clamp(0, 1)
        pil_img = torchvision.transforms.functional.to_pil_image(img)
        
        return pil_img
    
    def _decode_comfy_vae(self, latents: torch.Tensor, vae) -> Image.Image:
        if hasattr(vae, 'first_stage_model'):
            vae_model = vae.first_stage_model
            try:
                is_meta = False
                for param in vae_model.parameters():
                    if str(param.device) == 'meta':
                        is_meta = True
                        break
                
                if is_meta:
                    print("[Sampler] VAE is on meta device. Reloading VAE from disk...")
                    
                    from pathlib import Path
                    import comfy.utils
                    import comfy.sd
                    
                    model_config = self.model_pipeline.model_config
                    model_dir = Path(model_config['checkpoint_path'])
                    vae_path = model_dir / 'vae' / 'diffusion_pytorch_model.safetensors'
                    
                    if vae_path.exists():
                        print("[Sampler] Reloading VAE using Diffusers...")
                        
                        import os
                        old_cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
                        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
                        
                        from diffusers import AutoencoderKL
                        
                        vae_dir = vae_path.parent
                        
                        try:
                            diffusers_vae = AutoencoderKL.from_pretrained(
                                str(vae_dir),
                                torch_dtype=torch.bfloat16,
                                device_map='cuda:0'
                            )
                            
                            print(f"[Sampler] Successfully loaded VAE from {vae_dir} to CUDA")
                            
                            if old_cuda_visible_devices is not None:
                                os.environ['CUDA_VISIBLE_DEVICES'] = old_cuda_visible_devices
                            else:
                                os.environ.pop('CUDA_VISIBLE_DEVICES', None)
                            
                            # Unscale latents
                            # Unscale latents
                            model_config = getattr(self.model_pipeline, 'model_config', {})
                            if model_config.get('type') == 'z_image':
                                scaling_factor = 0.3611
                                shift_factor = 0.1159
                            else:
                                scaling_factor = diffusers_vae.config.scaling_factor if hasattr(diffusers_vae.config, 'scaling_factor') else 0.18215
                                shift_factor = diffusers_vae.config.shift_factor if hasattr(diffusers_vae.config, 'shift_factor') else 0.0

                            single_latent = latents[0:1].to('cuda', torch.bfloat16) / scaling_factor
                            single_latent = single_latent + shift_factor
                            
                            decoded = diffusers_vae.decode(single_latent).sample
                            
                            img = decoded[0].cpu().float()
                            img = ((img + 1) / 2).clamp(0, 1)
                            pil_img = torchvision.transforms.functional.to_pil_image(img)
                            
                            return pil_img
                            
                        except Exception as e:
                            if old_cuda_visible_devices is not None:
                                os.environ['CUDA_VISIBLE_DEVICES'] = old_cuda_visible_devices
                            else:
                                os.environ.pop('CUDA_VISIBLE_DEVICES', None)
                                
                            print(f"[Sampler] Error loading VAE with Diffusers: {e}")
                            import traceback
                            traceback.print_exc()
                            return None
                    else:
                        print(f"[Sampler] Warning: VAE file not found at {vae_path}")
                        return None
            except Exception as e:
                print(f"[Sampler] Error checking VAE device: {e}")
        else:
            vae_model = vae
        
        single_latent = latents[0:1].to('cuda')
        
        try:
            first_param = next(vae_model.parameters())
            vae_dtype = first_param.dtype
            single_latent = single_latent.to(vae_dtype)
        except StopIteration:
            single_latent = single_latent.to(torch.float32)
        except Exception:
             single_latent = single_latent.to(torch.float32)

        scaling_factor = 0.18215 
        if hasattr(vae_model, 'scale_factor'):
             scaling_factor = vae_model.scale_factor
        elif hasattr(vae, 'config') and hasattr(vae.config, 'scaling_factor'):
             scaling_factor = vae.config.scaling_factor
        
        single_latent = single_latent / scaling_factor

        try:
            decoded = vae.decode(single_latent)
        except RuntimeError as e:
            if "device meta" in str(e):
                 from pathlib import Path
                 model_config = self.model_pipeline.model_config
                 model_dir = Path(model_config['checkpoint_path'])
                 vae_path = model_dir / 'vae' / 'diffusion_pytorch_model.safetensors'
                 
                 if vae_path.exists():
                    print("[Sampler] meta device error，Reloading VAE using Diffusers...")
                    import os
                    old_cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
                    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
                    from diffusers import AutoencoderKL
                    vae_dir = vae_path.parent
                    try:
                        diffusers_vae = AutoencoderKL.from_pretrained(
                            str(vae_dir),
                            torch_dtype=torch.bfloat16,
                            device_map='cuda:0'
                        )
                        if old_cuda_visible_devices is not None:
                            os.environ['CUDA_VISIBLE_DEVICES'] = old_cuda_visible_devices
                        else:
                            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
                        
                        # Unscale latents for fallback
                        scaling_factor = diffusers_vae.config.scaling_factor if hasattr(diffusers_vae.config, 'scaling_factor') else 0.18215
                        single_latent = latents[0:1].to('cuda', torch.bfloat16) / scaling_factor
                        
                        decoded = diffusers_vae.decode(single_latent).sample
                        img = decoded[0].cpu().float()
                        img = ((img + 1) / 2).clamp(0, 1)
                        pil_img = torchvision.transforms.functional.to_pil_image(img)
                        return pil_img
                    except Exception as load_e:
                        if old_cuda_visible_devices is not None:
                            os.environ['CUDA_VISIBLE_DEVICES'] = old_cuda_visible_devices
                        else:
                            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
                        print(f"[Sampler] Fallback reload failed: {load_e}")
                        raise e 
                 else:
                     raise e
            else:
                raise e
        
        if hasattr(decoded, 'sample'):
            decoded = decoded.sample
        
        img = decoded[0].cpu().float()
        
        img = ((img + 1) / 2).clamp(0, 1)
        pil_img = torchvision.transforms.functional.to_pil_image(img)
        
        return pil_img
    
    def _save_images(
        self,
        image: Image.Image,
        captions: Optional[list],
        step: Optional[int],
        epoch: Optional[int],
        prefix: str,
    ):
        triggered_by_step = False
        triggered_by_epoch = False
        
        if step is not None and self.sample_every_n_steps is not None:
            if step % self.sample_every_n_steps == 0:
                triggered_by_step = True
        
        if epoch is not None and self.sample_every_n_epochs is not None:
            if epoch % self.sample_every_n_epochs == 0:
                triggered_by_epoch = True
        
        if triggered_by_step and step is not None:
            filename = f"step{step}.png"
        elif triggered_by_epoch and epoch is not None:
            filename = f"epoch{epoch}.png"
        elif step is not None:
            filename = f"step{step}.png"
        elif epoch is not None:
            filename = f"epoch{epoch}.png"
        else:
            filename = f"{int(time.time())}.png"
        filepath = self.output_dir / filename
        image.save(filepath)
        
        
        print(f"saved image to {filepath}")

        if self.tb_writer is not None:
             try:
                 img_tensor = torchvision.transforms.ToTensor()(image)
                 tag = f"sampler/{prefix}"
                 if epoch is not None:
                     tag += f"_e{epoch}"
                 if step is not None:
                     tag += f"_s{step}"
                 
                 self.tb_writer.add_image(tag, img_tensor, global_step=step if step is not None else epoch)
             except Exception as e:
                 print(f"Error logging image to TensorBoard: {e}")
    
    @torch.no_grad()
    def sample_from_batch(
        self,
        batch: Any,
        model_output: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ):
        text_encoder = None
        
        try:
            if not self.should_sample(step, epoch):
                return
            
            if not is_main_process():
                return

            noisy_latents = None
            if isinstance(batch, (tuple, list)):
                if len(batch) == 2 and isinstance(batch[0], (tuple, list)):
                    inputs = batch[0]
                    if len(inputs) >= 1:
                        noisy_latents = inputs[0]
            elif isinstance(batch, dict):
                noisy_latents = batch.get('latents')
            
            if noisy_latents is None:
                # Try to find latents in a list of dicts or similar
                if isinstance(batch, (tuple, list)):
                     for item in batch:
                        if isinstance(item, dict) and 'latents' in item:
                            noisy_latents = item['latents']
                            break
            
            if noisy_latents is None:
                print("[Sampler] Could not find latents in batch to determine shape. Skipping.")
                return

        except Exception as e:
            print(f"[Sampler] Error preparing batch: {e}")
            return

        
        try:
            model = self.model_pipeline
            vae = model.get_vae()

            if not (hasattr(model, 'get_call_text_encoder_fn') and hasattr(model, 'get_text_encoders')):
                print("[Sampler] Model pipeline does not support text encoding. Skipping inference.")
                return []

            text_encoders = model.get_text_encoders()
            if len(text_encoders) == 0:
                return []
                
            call_text_encoder_fn = model.get_call_text_encoder_fn(text_encoders[0])
            
            text_encoder = text_encoders[0]
            
            try:
                first_param = next(text_encoder.parameters(), None)
                if first_param is None:
                    print("[Sampler] Text encoder has no parameters. Skipping.")
                    return []
                    
                if first_param.device.type == 'meta':
                    print("[Sampler] Text encoder is on meta device, materializing to CUDA...")
                    
                    try:
                        from transformers import AutoModel
                        import os
                        
                        text_encoder_path = None
                        
                        if hasattr(model, 'text_encoder_path'):
                            text_encoder_path = model.text_encoder_path
                        elif hasattr(model, 'checkpoint_path'):
                            text_encoder_path = os.path.join(model.checkpoint_path, "text_encoder")
                        elif hasattr(model, 'config') and hasattr(model.config, 'text_encoder'):
                            text_encoder_path = model.config.text_encoder
                        
                        if text_encoder_path and os.path.exists(text_encoder_path):
                            print(f"[Sampler] Reloading text encoder from {text_encoder_path}...")
                            text_encoder = AutoModel.from_pretrained(
                                text_encoder_path, 
                                torch_dtype=torch.bfloat16,
                                trust_remote_code=True
                            ).to('cuda')
                            if hasattr(model, 'text_encoder'):
                                model.text_encoder = text_encoder
                            call_text_encoder_fn = model.get_call_text_encoder_fn(text_encoder)
                            print("[Sampler] Text encoder reloaded successfully.")
                        else:
                            print(f"[Sampler] Cannot find text encoder path (tried: {text_encoder_path}), skipping inference.")
                            return []
                            
                    except Exception as e:
                        print(f"[Sampler] Failed to reload text encoder: {e}")
                        import traceback
                        traceback.print_exc()
                        print("[Sampler] Skipping inference.")
                        return []
                    
            except Exception as e:
                print(f"[Sampler] Cannot check/move text encoder device: {e}")
                print("[Sampler] Skipping full inference.")
                return []
            
            captions = None
            
            if self.sample_prompt:
                captions = [self.sample_prompt] * noisy_latents.shape[0]
            elif isinstance(batch, dict):
                captions = batch.get('caption')
            
            if captions is None:
                captions = [""] * noisy_latents.shape[0]
                
            num_inference_steps = self.num_inference_steps
            guidance_scale = self.guidance_scale
            guidance_value = self.guidance_value
            
            uncond_captions = [""] * len(captions)
            is_video = [False] * len(captions)
            
            with torch.no_grad():
                cond_outputs = call_text_encoder_fn(captions, is_video)
                cond_embeds = cond_outputs['encoder_hidden_states']
                cond_pooled = cond_outputs['pooled_projections']
                
                uncond_outputs = call_text_encoder_fn(uncond_captions, is_video)
                uncond_embeds = uncond_outputs['encoder_hidden_states']
                uncond_pooled = uncond_outputs['pooled_projections']

            scheduler = CustomFlowMatchEulerDiscreteScheduler(
                num_train_timesteps=1000,
                shift=self.shift,
            )
            scheduler.set_train_timesteps(
                num_inference_steps, 
                device='cuda', 
                timestep_type=self.timestep_type,
                shift=self.shift
            )
            
            if noisy_latents.ndim == 3:
                img_ids = None
                if isinstance(batch, (tuple, list)) and len(batch) == 2:
                    inputs = batch[0]
                    if len(inputs) >= 5:
                        img_ids = inputs[4]  

                if img_ids is not None:
                    h_ids = img_ids[..., 1].long()
                    w_ids = img_ids[..., 2].long()
                    H = int(h_ids.max().item()) + 1
                    W = int(w_ids.max().item()) + 1
                    B = noisy_latents.shape[0]
                    
                    
                    C = 16                   
                    downscale = 8
                    H_latent = self.height // downscale
                    W_latent = self.width // downscale
                    
                    latents = torch.randn(B, C, H_latent, W_latent, device='cuda', dtype=torch.float32)

                    if hasattr(model, '_prepare_latent_image_ids'):
                         pass 
                    
                    
                    if hasattr(model, '_prepare_latent_image_ids'):
                         img_ids = model._prepare_latent_image_ids(B, H_latent, W_latent, 'cuda', torch.bfloat16) # Assuming bfloat16
                    else:
                         latent_image_ids = torch.zeros(H_latent, W_latent, 3)
                         latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(H_latent)[:, None]
                         latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(W_latent)[None, :]
                         img_ids = latent_image_ids.reshape(-1, 3).to(device='cuda', dtype=torch.bfloat16).unsqueeze(0).repeat(B, 1, 1)

                else:
                    downscale = 8
                    H_latent = self.height // downscale
                    W_latent = self.width // downscale
                    C = noisy_latents.shape[1] if noisy_latents is not None else 4
                    latents = torch.randn(B, C, H_latent, W_latent, device='cuda', dtype=torch.float32)
            else:
                downscale = 8
                H_latent = self.height // downscale
                W_latent = self.width // downscale
                C = noisy_latents.shape[1]
                latents = torch.randn(B, C, H_latent, W_latent, device='cuda', dtype=torch.float32)
                img_ids = None 
            
            # Re-initialize scheduler with latents for dynamic shifting (Flux/Z-Image)
            scheduler.set_train_timesteps(
                num_inference_steps,
                device='cuda',
                timestep_type=self.timestep_type,
                latents=latents,
                shift=self.shift
            )

            # Re-initialize scheduler with latents for dynamic shifting (Flux/Z-Image)
            scheduler.set_train_timesteps(
                num_inference_steps,
                device='cuda',
                timestep_type=self.timestep_type,
                latents=latents,
                shift=self.shift
            )
            
            print(f"[Sampler] Starting full inference (CFG={guidance_scale}, Steps={num_inference_steps})...")
            
            merge_adapters = model.model_config.get('merge_adapters', [])
            if isinstance(merge_adapters, str):
                merge_adapters = [merge_adapters]
            
            for adapter_path in merge_adapters:
                print(f"[Sampler] Subtracting adapter {adapter_path} for sampling...")
                try:
                    model.load_adapter_weights(adapter_path, fuse=True, fuse_weight=-1.0)
                    if hasattr(model.transformer, 'unload_lora'):
                        model.transformer.unload_lora()
                except Exception as e:
                    print(f"[Sampler] Warning: Failed to subtract adapter {adapter_path}: {e}")

            was_training = model.transformer.training
            model.transformer.eval()
            try:
                for i, t in enumerate(scheduler.timesteps):
                    latent_model_input = scheduler.scale_model_input(latents, t)
                    
                    bs, c, h, w = latent_model_input.shape
                    device = latent_model_input.device
                    dtype = latent_model_input.dtype
                    
                    model_t = 1.0 - (t / 1000.0)
                    model_t = model_t.to(device, dtype)
                    if model_t.ndim == 0:
                         model_t = model_t.unsqueeze(0).repeat(bs)
                    

                    def run_model(x, encoder_hidden_states, pooled_projections):
                        lengths = [e.shape[0] for e in encoder_hidden_states]
                        max_len = max(lengths)
                        embed_dim = encoder_hidden_states[0].shape[1]
                        
                        padded_embeds = torch.zeros(bs, max_len, embed_dim, dtype=dtype, device=device)
                        for idx, e in enumerate(encoder_hidden_states):
                            l = lengths[idx]
                            padded_embeds[idx, :l] = e.to(device)
                        txt_lens = torch.tensor(lengths, dtype=torch.long, device=device)
                        
                        if hasattr(model, '_prepare_latent_image_ids'):
                            img_ids = model._prepare_latent_image_ids(bs, h, w, device, dtype)
                        else:
                            latent_image_ids = torch.zeros(h, w, 3)
                            latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(h)[:, None]
                            latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(w)[None, :]
                            img_ids = latent_image_ids.reshape(-1, 3).to(device=device, dtype=dtype)
                            
                        if img_ids.ndim == 2:
                            img_ids = img_ids.unsqueeze(0).repeat((bs, 1, 1))
                        
                        txt_ids = torch.zeros(bs, max_len, 3).to(device, dtype)
                        guidance_vec = torch.full((bs,), guidance_value, device=device, dtype=torch.float32)
                        img_seq_len = torch.tensor(h * w, device=device).repeat((bs,))
                        
                        x_flat = rearrange(x, "b c h w -> b (h w) c")
                        
                        inputs = (x_flat, padded_embeds, pooled_projections, model_t, img_ids, txt_ids, guidance_vec, img_seq_len, txt_lens)
                        
                        transformer = model.transformer
                        embedding_wrapper = EmbeddingWrapper(transformer)
                        
                        unified, unified_attn_mask, unified_freqs_cis, adaln_input, x_size, patch_size, f_patch_size = embedding_wrapper(inputs)
                        
                        if hasattr(transformer, 'transformer_blocks'):
                            blocks = transformer.transformer_blocks
                        elif hasattr(transformer, 'blocks'):
                            blocks = transformer.blocks
                        else:
                            blocks = transformer.layers
                            
                        for block in blocks:
                             unified = block(unified, unified_attn_mask, unified_freqs_cis, adaln_input)
                             
                        output_wrapper = OutputWrapper(transformer)
                        output_inputs = (unified, unified_attn_mask, unified_freqs_cis, adaln_input, x_size, patch_size, f_patch_size)
                        output = output_wrapper(output_inputs)
                        
                        if isinstance(output, list):
                            output = torch.stack(output) if len(output) > 0 else output[0]
                        
                        return output

                    with torch.no_grad():
                        noise_pred_cond = run_model(latent_model_input, cond_embeds, cond_pooled)
                        
                        if not isinstance(noise_pred_cond, torch.Tensor):
                            if isinstance(noise_pred_cond, list):
                                noise_pred_cond = torch.stack(noise_pred_cond) if len(noise_pred_cond) > 1 else noise_pred_cond[0]
                        
                        if noise_pred_cond.ndim == 5 and noise_pred_cond.shape[2] == 1:
                            noise_pred_cond = noise_pred_cond.squeeze(2)

                        if guidance_scale > 1.0:
                            noise_pred_uncond = run_model(latent_model_input, uncond_embeds, uncond_pooled)
                            
                            if not isinstance(noise_pred_uncond, torch.Tensor):
                                if isinstance(noise_pred_uncond, list):
                                    noise_pred_uncond = torch.stack(noise_pred_uncond) if len(noise_pred_uncond) > 1 else noise_pred_uncond[0]
                            
                            if noise_pred_uncond.ndim == 5 and noise_pred_uncond.shape[2] == 1:
                                noise_pred_uncond = noise_pred_uncond.squeeze(2)
                                
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                        else:
                            # For Turbo/Distilled models (scale <= 1.0, e.g. 0.0 or 1.0), skip uncond and use cond directly
                            noise_pred = noise_pred_cond
                        
                        noise_pred = -noise_pred
                        
                        latents = scheduler.step(noise_pred, t, latents).prev_sample
            finally:
                if was_training:
                    model.transformer.train()
                
                # Restore adapters
                for adapter_path in merge_adapters:
                    print(f"[Sampler] Restoring adapter {adapter_path}...")
                    try:
                        model.load_adapter_weights(adapter_path, fuse=True, fuse_weight=1.0)
                        if hasattr(model.transformer, 'unload_lora'):
                            model.transformer.unload_lora()
                    except Exception as e:
                        print(f"[Sampler] Warning: Failed to restore adapter {adapter_path}: {e}")
            
            print(f"[Sampler] Inference completed successfully! Saving samples...")

            self.sample_from_latents(
                latents=latents,
                captions=captions,
                step=step,
                epoch=epoch,
                prefix=f"inference_cfg{guidance_scale}_step{num_inference_steps}"
            )
            
            return []

        except Exception as e:
            print(f"[Sampler] Error during full inference: {str(e)}")
            import traceback
            traceback.print_exc()
            print("[Sampler] Falling back to reconstruction...")
            
        finally:
            if text_encoder is not None and str(text_encoder.device) != 'meta':
                 text_encoder.to('meta')
                 
                 if hasattr(model, 'text_encoder'):
                     model.text_encoder.to('meta')
                     
                 call_text_encoder_fn = None
                 torch.cuda.empty_cache()
