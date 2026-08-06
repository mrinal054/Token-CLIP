import torch
import torch.nn as nn


class SwinUNETREncoder(nn.Module):
    """
    Link: https://monai-dev.readthedocs.io/en/fixes-sphinx/networks.html#swinunetr
    
    SwinUNETR encoder wrapper that mimics CTViT output.

    When return_encoded_tokens=True, returns:
        tokens: [B, t, h, w, dim]
        
    ct_clip.py Line ~730: enc_image= self.visual_transformer(image, return_encoded_tokens=True)

    Input:
        x: [B, C, D, H, W]
    """

    def __init__(self, swinunetr, dim=512, export_stage=3, stage_channels=None):
        super().__init__()
        self.swinunetr = swinunetr
        self.export_stage = export_stage

        # Build projection lazily so we don't depend on swinunetr.feature_size existing
        self.proj = None
        self.dim = dim

    def forward(self, x, return_encoded_tokens=False):
        # Extract multi-scale encoder features
        # feats[k]: [B, Ck, Dk, Hk, Wk]
        feats = self.swinunetr.swinViT(x, self.swinunetr.normalize)

        feat = feats[self.export_stage]   # [B, C, D', H', W']

        # Lazily create projection on first forward pass
        if self.proj is None:
            in_dim = feat.shape[1]  # C
            self.proj = nn.Identity() if in_dim == self.dim else nn.Conv3d(
                in_dim, self.dim, kernel_size=1, bias=False
            ).to(feat.device)


        feat = self.proj(feat)            # [B, dim, D', H', W']

        if return_encoded_tokens: # required by CTCLIP
            # Match CTViT: [B, t, h, w, dim]
            # t <- D', h <- H', w <- W'
            return feat.permute(0, 2, 3, 4, 1).contiguous()

        # Safe default: pooled embedding
        return feat.mean(dim=(2, 3, 4))