from pathlib import Path
import sys
import argparse
import os.path
sys.path.insert(0, os.path.abspath('submodules/HunyuanVideo'))

import torch
from PIL import Image
import torchvision

from utils.common import VIDEO_EXTENSIONS
from hyvideo.vae import load_vae


MODEL_BASE = Path('/home/anon/HunyuanVideo/ckpts')

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)

args = parser.parse_args()
assert args.input.is_file()
assert args.output.is_dir()


def vae_encode(tensor, vae):
    # tensor values already in range [-1, 1] here
    latents = vae.encode(tensor).latent_dist.sample()
    return latents * vae.config.scaling_factor


def vae_decode(latents, vae):
    # tensor values already in range [-1, 1] here
    latents = latents / vae.config.scaling_factor
    tensor = vae.decode(latents, return_dict=False)[0]
    return tensor


if __name__ == '__main__':
    vae, _, s_ratio, t_ratio = load_vae(
        '884-16c-hy',
        'bf16',
        vae_path=MODEL_BASE / 'hunyuan-video-t2v-720p/vae',
        device='cuda',
    )

    if args.input.suffix in VIDEO_EXTENSIONS:
        raise NotImplementedError()
    else:
        pil_img = Image.open(args.input)
        video = torchvision.transforms.functional.to_tensor(pil_img).unsqueeze(1).unsqueeze(0)

    video = (video * 2) - 1

    latents = vae_encode(video.to(vae.device, vae.dtype), vae)
    video = vae_decode(latents, vae).to('cpu', torch.float32)

    video = ((video + 1) / 2).clamp(0, 1)

    if video.shape[2] == 1:
        img = video.squeeze(2).squeeze(0)
        pil_img = torchvision.transforms.functional.to_pil_image(img)
        pil_img.save(args.output / args.input.name)
