import torch
from torch import nn
from pathlib import Path

from .svr import SpatioTemporalVisualTokenRefinerModelV2
from .filip_infoNCE import FILIPInfoNCELoss


class CLIPTokenRefined(nn.Module):
    def __init__(
        self,
        *,
        image_encoder: nn.Module,                 # must support forward(image, return_encoded_tokens=True)
        text_encoder: nn.Module,                  # HF BERT-like or my own

        # Explicit dims from config
        text_feat_dim: int,                       # e.g., 768
        image_feat_dim: int,                      # e.g., token refiner output dim
        shared_latent_dim: int = 512,             # shared projection dim (CTCLIP dim_latent)

        token_refiner_dict: dict,

        contrastive_loss_temperature: float = 0.07,
        filip_pool: str = "logsumexp",
        filip_lse_alpha: float = 10.0,
        filip_decoupled: bool = False,
        filip_max_logit_scale: float = 100.0,
    ):
        super().__init__()

        self.visual_transformer = image_encoder
        self.text_encoder = text_encoder

        # ----- Token refiner -----
        if token_refiner_dict is None:
            raise ValueError("token_refiner_dict must be provided")

        embed_size = token_refiner_dict["embed_size"]
        self.return_div_loss = bool(token_refiner_dict.get("return_div_loss", False))

        self.token_refiner = SpatioTemporalVisualTokenRefinerModelV2(
            embed_size=embed_size,
            num_heads=token_refiner_dict.get("num_heads", 8),
            num_layers=token_refiner_dict.get("num_layers", 4),
            top_k=token_refiner_dict.get("top_k", 224),
            use_multi_scale=token_refiner_dict.get("use_multi_scale", False),
            attn_type=token_refiner_dict.get("attn_type", "rma"),
            enable_diffts=token_refiner_dict.get("enable_diffts", False),
            enable_dmtp=token_refiner_dict.get("enable_dmtp", False),
            scales=token_refiner_dict.get("scales", [1]),
            tau=token_refiner_dict.get("tau", 1.0),
            context_bias=token_refiner_dict.get("context_bias", False),
            return_div_loss=self.return_div_loss,
            div_weight=token_refiner_dict.get("div_weight", 1e-3),
            # >>> MKD CHANGE: configurable RMA relative-position limits.
            spatial_max_seq_len=token_refiner_dict.get("spatial_max_seq_len", 512),
            temporal_max_seq_len=token_refiner_dict.get("temporal_max_seq_len", 512),
        )

        # ----- Projections to shared latent dim (CTCLIP style) -----
        self.to_text_latent = nn.Linear(int(text_feat_dim), int(shared_latent_dim), bias=False)
        self.to_visual_latent = nn.Linear(int(image_feat_dim), int(shared_latent_dim), bias=False)

        # ----- FILIP loss -----
        self.contrastive_criterion = FILIPInfoNCELoss(
            temperature=float(contrastive_loss_temperature),
            pool=filip_pool,
            lse_alpha=float(filip_lse_alpha),
            decoupled=bool(filip_decoupled),
            max_logit_scale=float(filip_max_logit_scale),
        )

    @staticmethod
    def _get_last_hidden_state(text_out: object) -> torch.Tensor:
        # returns (B, L, E_text)
        if isinstance(text_out, torch.Tensor):
            return text_out
        if hasattr(text_out, "last_hidden_state"):
            return text_out.last_hidden_state
        if isinstance(text_out, (tuple, list)) and len(text_out) > 0:
            return text_out[0]
        raise TypeError(f"Unsupported text encoder output type: {type(text_out)}")

    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    def load(self, path): 
        path = Path(path)
        assert path.exists()
        ckpt = torch.load(str(path), map_location="cpu")
        
        # Case 1: new-style checkpoint: {"model": ..., "optim": ..., "steps": ...}
        # Case 2: old-style checkpoint: plain state_dict
    
        if isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    
        self.load_state_dict(state_dict)


    def forward(self, text, image, return_loss: bool = False) -> dict:
        # text mask
        text_mask = text.attention_mask.bool()  # (B, L)

        # text encoding -> (B, L, E_text)
        text_out = self.text_encoder(input_ids=text.input_ids, attention_mask=text.attention_mask)
        enc_text = self._get_last_hidden_state(text_out)

        # image encoding -> (B, PD, PH, PW, E_img)
        enc_image = self.visual_transformer(image, return_encoded_tokens=True)

        # (B,PD,PH,PW,E) -> (B,PD,PH*PW,E)
        B, PD, PH, PW, E = enc_image.shape
        reshaped_enc_image = enc_image.reshape(B, PD, PH * PW, E)

        # token refinement -> (B, K, E_img_refined)
        if self.return_div_loss:
            refined_enc_image, div_loss = self.token_refiner(reshaped_enc_image) # Shape: [8,224,512] for top_k:128 & scale:[1,2,4] 
        else:
            refined_enc_image = self.token_refiner(reshaped_enc_image)
            div_loss = torch.zeros((), device=enc_image.device, dtype=enc_image.dtype)

        # project both to shared_latent_dim
        text_latents = self.to_text_latent(enc_text)              # (B, L, D)
        image_latents = self.to_visual_latent(refined_enc_image)  # (B, K, D)

        if return_loss: # Use during training
          # Training requires Bi == Bt for CLIP-style targets
          # FILIP loss
          contrastive_loss, contrastive_info = self.contrastive_criterion(
              image_latent_tokens=image_latents,
              text_latent_tokens=text_latents,
              text_mask=None, # text_mask, # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
          )
  
          loss_dict =  {
              "contrastive_loss": contrastive_loss,
              "div_loss": div_loss,
              "contrastive_info": contrastive_info,
          }
          
          return loss_dict

        # During inference: return logits without cross-entropy, image_latents, and text_latents.
        # sim: (Bi, Bt)
        sim = self.contrastive_criterion.filip_patch_similarity(
            image_tokens=image_latents,
            text_tokens=text_latents,
            text_mask=None, # text_mask, # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        )
    
        # apply same logit scale as loss uses
        logit_scale = self.contrastive_criterion.logit_scale.to(device=sim.device, dtype=sim.dtype)
        logit_scale = logit_scale.clamp(
            max=torch.log(torch.tensor(self.contrastive_criterion.max_logit_scale, device=sim.device, dtype=sim.dtype))
        )
        logits = sim * logit_scale.exp()  # (Bi, Bt)
    
        return logits, image_latents, text_latents