import torch
import torch.nn as nn

class GramDiversityLoss(nn.Module):
    """
    Encourage different heads to attend to different tokens using a Gram matrix penalty.
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        """
        :param weights: (b, L, R) softmax weights over tokens (sum over L is 1 for each head)
            b: Batch size
            L: No. of tokens
            R: Number of heads (or selection slots)

        :return: scalar diversity loss
        """
        b, L, R = weights.shape

        # W: (b, R, L)
        W = weights.permute(0, 2, 1)

        # L2 normalize each head vector over tokens
        W = W / (W.norm(p=2, dim=-1, keepdim=True) + self.eps)

        # Gram matrix: (b, R, R)
        G = torch.bmm(W, W.transpose(1, 2))

        # Off-diagonal penalty: make G close to I
        I = torch.eye(R, device=weights.device, dtype=weights.dtype).unsqueeze(0)  # (1, R, R)
        loss = ((G - I) ** 2).mean()

        return loss


if __name__ == "__main__":
    weights = torch.rand(2, 2000, 512)
    loss_fn = GramDiversityLoss(eps=1e-8)
    loss = loss_fn(weights)   # weights: (b, L, R)
