import torch
import torch.nn as nn
import torch.nn.functional as F


class FILIPInfoNCELoss(nn.Module):
    """
    Symmetric FILIP-style InfoNCE loss packaged as a torch.nn.Module.

    CTCLIP-aligned temperature:
      - we store a learnable "logit_scale" parameter
      - logits = sim * exp(logit_scale)
      - user still passes conventional temperature values (e.g., 0.07, 0.1)
        by initializing logit_scale = log(1 / temperature)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        pool: str = "logsumexp",                    # "max" | "logsumexp"
        lse_alpha: float = 10.0,                    # sharpness for logsumexp pooling
        decoupled: bool = False,                    # DCL-style (optional)
        max_logit_scale: float = 100.0,             # optional safety cap (like OpenCLIP)
    ):
        super().__init__()

        self.pool = pool
        self.lse_alpha = float(lse_alpha)
        self.decoupled = bool(decoupled)
        self.max_logit_scale = float(max_logit_scale)

        # --- CTCLIP / CLIP style ---
        # user passes temperature tau0 (e.g., 0.07)
        # we learn logit_scale s, where exp(s) = 1 / tau
        if isinstance(temperature, torch.Tensor):
            tau0 = float(temperature.detach().item())
        else:
            tau0 = float(temperature)

        tau0 = max(tau0, 1e-8)
        init_logit_scale = float(torch.log(torch.tensor(1.0 / tau0)).item())

        self.logit_scale = nn.Parameter(torch.tensor(init_logit_scale), requires_grad=True)

    @staticmethod
    def _matrix_diag(x: torch.Tensor) -> torch.Tensor:
        return torch.diagonal(x, offset=0, dim1=-2, dim2=-1)

    def filip_patch_similarity(
        self,
        image_tokens: torch.Tensor,   # (B, K, D)
        text_tokens: torch.Tensor,    # (B, L, D)
        text_mask=None,               # (B, L) True for valid tokens
    ) -> torch.Tensor:
        assert image_tokens.ndim == 3 and text_tokens.ndim == 3
        B, K, D = image_tokens.shape
        Bt, L, Dt = text_tokens.shape
 
        # assert B == Bt and D == Dt, "Batch size and embedding dim must match."

        image_tokens = F.normalize(image_tokens, dim=-1)
        text_tokens = F.normalize(text_tokens, dim=-1)

        S = torch.einsum("b k d, c l d -> b c k l", image_tokens, text_tokens)

        if text_mask is not None:
            assert text_mask.shape == (B, L)
            mask = text_mask[None, :, None, :].to(dtype=torch.bool, device=S.device)
            S = S.masked_fill(~mask, torch.finfo(S.dtype).min)

        if self.pool == "max":
            S_red_L = S.max(dim=-1).values
        elif self.pool == "logsumexp":
            alpha = self.lse_alpha
            S_red_L = torch.logsumexp(S * alpha, dim=-1) / alpha
        else:
            raise ValueError(f"Unknown pool='{self.pool}'. Use 'max' or 'logsumexp'.")

        sim = S_red_L.mean(dim=-1)
        return sim

    def forward(
        self,
        image_latent_tokens: torch.Tensor,  # (B, K, D)
        text_latent_tokens: torch.Tensor,   # (B, L, D)
        text_mask=None,                     # (B, L)
    ) -> tuple[torch.Tensor, dict]:

        B = image_latent_tokens.shape[0]
        device = image_latent_tokens.device

        sim = self.filip_patch_similarity(
            image_tokens=image_latent_tokens,
            text_tokens=text_latent_tokens,
            text_mask=text_mask,
        )

        # --- CTCLIP-aligned scaling: logits = sim * exp(logit_scale) ---
        # match dtype/device of sim
        logit_scale = self.logit_scale.to(device=device, dtype=sim.dtype)
        # optional safety cap to avoid runaway scales
        logit_scale = logit_scale.clamp(max=torch.log(torch.tensor(self.max_logit_scale, device=device, dtype=sim.dtype)))
        scale = logit_scale.exp()
        # ---------------------------------------------------------------

        logits_i2t = sim * scale
        logits_t2i = logits_i2t.t()

        targets = torch.arange(B, device=device)
        
        if not self.decoupled:
            loss_i2t = F.cross_entropy(logits_i2t, targets)
            loss_t2i = F.cross_entropy(logits_t2i, targets)
        else:
            neg_inf = torch.finfo(logits_i2t.dtype).min
            eye = torch.eye(B, device=device, dtype=torch.bool)

            li = logits_i2t.masked_fill(eye, neg_inf)
            log_denom_i = torch.logsumexp(li, dim=-1)
            pos_i = self._matrix_diag(logits_i2t)
            loss_i2t = (-pos_i + log_denom_i).mean()

            lt = logits_t2i.masked_fill(eye, neg_inf)
            log_denom_t = torch.logsumexp(lt, dim=-1)
            pos_t = self._matrix_diag(logits_t2i)
            loss_t2i = (-pos_t + log_denom_t).mean()

        loss = 0.5 * (loss_i2t + loss_t2i)

        # For logging: recover "temperature" tau = 1/scale
        tau = (1.0 / scale).detach()

        info = {
            "logits_i2t": logits_i2t,
            "logits_t2i": logits_t2i,
            "loss_i2t": loss_i2t.detach(),
            "loss_t2i": loss_t2i.detach(),
            "logit_scale": scale.detach(),
            "tau": tau,
        }
        return loss, info


if __name__ == "__main__":
    image_latent_tokens = torch.rand(8, 224, 512) # B,K,E -> B: Batch, K: No. of image tokens, E: Embedding dim
    text_latent_tokens = torch.rand(8, 512, 512)   # B,L,E -> B: Batch, L: No. of text tokens, E: Embedding dim

    loss_fn = FILIPInfoNCELoss(
        temperature=0.07,    # keep using conventional tau values
        pool="logsumexp",
        lse_alpha=10.0,
        decoupled=False,
        max_logit_scale=100,
    )

    loss, info = loss_fn(image_latent_tokens, text_latent_tokens)
    print("Loss:", loss.item())
    print("Learned logit_scale:", info["logit_scale"].item())
    print("Implied tau:", info["tau"].item())
