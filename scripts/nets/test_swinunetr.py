#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 14:28:54 2026

@author: m324371

Instruction:
    Execute the following command from the root directory (/research/m324371/Project/Digital_Twin/CT-CLIP/scripts/):
        
        python -m nets.test_swinunetr

"""

import torch

# Fake config (matches what load_swinunetr expects)
cfg = {
    "img_size": [96, 96, 96],     # (D, H, W)
    "in_channels": 1,
    "out_channels": 1,
    "feature_size": 48,
    "dim": 512,
    "export_stage": 3,
    "depths": (2, 2, 2, 2),
    "num_heads": (3, 6, 12, 24),
    "normalize": True,
    "use_checkpoint": True,
    "pretrained_dir": "/research/m324371/Project/Digital_Twin/CT-CLIP/scripts/pretrained_weights/SwinUNETR/model_swinvit.pt",
}

# Build model
from nets.image_models import load_swinunetr   

model = load_swinunetr(cfg)
model.eval().cuda()

# Dummy input (CTCLIP-style)  [B, C, D, H, W]
x = torch.randn(2, 1, 96, 96, 96, device="cuda")

# Test CTViT-style output
with torch.no_grad():
    tokens = model(x, return_encoded_tokens=True)

print("Tokens shape:", tokens.shape) # Shape: [B, t, h, w, dim] <-- CLIP vision_transformer expects this shape

# Test what CTCLIP will actually consume
pooled = tokens.mean(dim=1)          # mean over t
flattened = pooled.view(pooled.size(0), -1)

print("After mean(dim=1):", pooled.shape)
print("Flattened shape:", flattened.shape)
