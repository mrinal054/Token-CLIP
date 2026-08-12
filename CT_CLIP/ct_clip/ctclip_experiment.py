import os
import sys
import yaml
import torch
from box import Box

sys.path.append(os.getcwd() + "/utils/")

from ct_clip.ct_clip import CTCLIP  # package import
from nets.text_models import build_text_model
from nets.image_models import build_image_model


def test_dummy_3d_vit_encoder_shape(config_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # -------------------------
    # Load config (same as your training code)
    # -------------------------
    with open(config_path, "r") as f:
        cfg = Box(yaml.safe_load(f))

    # -------------------------
    # Build encoders from config
    # -------------------------
    tokenizer, text_encoder = build_text_model(cfg.text_encoder)
    image_encoder = build_image_model(cfg.image_encoder)  # <-- this should return your 3D ViT

    # -------------------------
    # Build CTCLIP with your 3D image encoder
    # -------------------------
    model = CTCLIP(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        dim_image=cfg.vlm.dim_image,
        dim_text=cfg.vlm.dim_text,
        dim_latent=cfg.vlm.dim_latent,
        extra_latent_projection=cfg.vlm.extra_latent_projection,
        use_mlm=cfg.vlm.use_mlm,
        downsample_image_embeds=cfg.vlm.downsample_image_embeds,
        use_all_token_embeds=cfg.vlm.use_all_token_embeds,
    ).to(device).eval()

    # -------------------------
    # Decide dummy volume shape
    # Prefer config if available; else use your example
    # -------------------------
    B = 2
    C = int(getattr(cfg.image_encoder, "in_channels", 1))

    # Try to infer spatial shape from config (common keys)
    # Fallback to your observed: D=64, H=W=196
    resize_shape = getattr(cfg.dataloader, "resize_shape", None)
    if resize_shape is not None and len(resize_shape) == 3:
        D, H, W = map(int, resize_shape)
    else:
        D, H, W = 64, 196, 196

    dummy_vol = torch.randn(B, C, D, H, W, device=device)
    print("\n[1] Dummy volume input shape:", dummy_vol.shape)

    # -------------------------
    # Call the 3D ViT directly (THIS is what you want)
    # -------------------------
    with torch.no_grad():
        enc_image = model.visual_transformer(dummy_vol, return_encoded_tokens=True)

    print("[2] 3D ViT raw encoder output shape:", tuple(enc_image.shape))

    # -------------------------
    # Mimic your CTCLIP.forward() post-processing
    # In your code:
    #   enc_image = torch.mean(enc_image, dim=1)
    #   enc_image = enc_image.view(enc_image.shape[0], -1)
    # -------------------------
    enc_mean = torch.mean(enc_image, dim=1)
    print("[3] After mean(dim=1) shape:", tuple(enc_mean.shape))

    enc_flat = enc_mean.view(enc_mean.shape[0], -1)
    print("[4] After flatten shape:", tuple(enc_flat.shape))


    # image_embeds = enc_image[:, :] if enc_image.ndim == 3 else enc_image

    # -------------------------
    # Latent projection (same as forward)
    # -------------------------
    with torch.no_grad():
        image_latents = model.to_visual_latent(enc_flat)

    print("[5] Projected image_latents shape:", tuple(image_latents.shape))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    print(args.config)

    test_dummy_3d_vit_encoder_shape(args.config)

    # Set the directory to /scripts/ and run the following command from the terminal

    # CUDA_VISIBLE_DEVICES=1 python -m ct_clip.ctclip_experiment --config /research/m324371/Project/Digital_Twin/CT-CLIP/scripts/config/clip_test.yaml

