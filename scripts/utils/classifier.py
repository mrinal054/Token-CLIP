import torch
import torch.nn as nn
from classification_head import ClassificationHeadCTCLIP

#%% Imagelatents + Classification head
class ImageLatentsClassifier(nn.Module):
    def __init__(self, trained_model, 
                 latent_dim, 
                 num_classes, 
                 dropout_prob=0.3,
                 out_channels:list=None,
                 freeze_latents:bool=True):
        
        super(ImageLatentsClassifier, self).__init__()
        self.trained_model = trained_model

        # Freeze weights
        if freeze_latents:
            for param in self.trained_model.parameters():
                param.requires_grad = False

        # Add condition to handle None for out_channels
        if out_channels is not None:
            out_channels = [latent_dim] + out_channels
        else: out_channels = [latent_dim]

        # Classification head
        self.classifier = ClassificationHeadCTCLIP(num_classes=num_classes,
                                                 out_channels=out_channels,
                                                 dropout=dropout_prob)

    # MKD added the forward method <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<    
    def forward(self, text_tokens, image_inputs, **kwargs):
        kwargs['return_latents'] = True
        # device = image_inputs.device  # Ensure device is passed as positional arg
        kwargs['device'] = image_inputs.device 
        _, image_latents, _ = self.trained_model(text_tokens, image_inputs, **kwargs)
    
        return self.classifier(image_latents)
    
    def save(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load(self, file_path):
        loaded_state_dict = torch.load(file_path)
        self.load_state_dict(loaded_state_dict)


#%% Image latents only
class ImageLatents(nn.Module):
    def __init__(self, trained_model, 
                 latent_dim, 
                 dropout_prob=0.3,
                 out_channels:list=None,
                 freeze_latents:bool=True):
        
        super(ImageLatents, self).__init__()
        self.trained_model = trained_model

        # Freeze weights
        if freeze_latents:
            for param in self.trained_model.parameters():
                param.requires_grad = False

    # MKD added the forward method <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<    
    def forward(self, text_tokens, image_inputs, **kwargs):
        kwargs['return_latents'] = True
        # device = image_inputs.device  # Ensure device is passed as positional arg
        kwargs['device'] = image_inputs.device 
        _, image_latents, _ = self.trained_model(text_tokens, image_inputs, **kwargs)
    
        return image_latents
    
    def save(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load(self, file_path):
        loaded_state_dict = torch.load(file_path)
        self.load_state_dict(loaded_state_dict)
        
#%% Visual projection extractor
def resolve_layer(root: nn.Module, name: str) -> nn.Module:
    """Resolve dotted path with indices and DDP support."""
    m = root.module if hasattr(root, "module") else root
    for part in name.split("."):
        m = m[int(part)] if part.isdigit() else getattr(m, part)
    return m

class CTCLIPExtractor(nn.Module):
    def __init__(self, model: nn.Module, layer_name: str, pre_hook: bool = True):
        super().__init__()
        self.model = model
        self.layer_name = layer_name
        self.pre_hook = pre_hook
        self._buf = {}
        self._hook = None

        # attach hook
        target = resolve_layer(self.model, self.layer_name)

        def grab_pre(module, inputs):
            # inputs is a tuple
            self._buf["out"] = inputs[0]

        def grab_post(module, inputs, output):
            # output may be Tensor or tuple
            self._buf["out"] = output[0] if isinstance(output, (tuple, list)) else output

        self._hook = (target.register_forward_pre_hook(grab_pre)
                      if self.pre_hook else
                      target.register_forward_hook(grab_post))

    def forward(self, images, text_tokens, device, **kwargs):
        
        if device is None:
            device = images.device
        self._buf.clear()  # clear previous run
        _ = self.model(text_tokens, images, device=device, **kwargs)
        out = self._buf.get("out", None)
        if out is None:
            raise RuntimeError(f"Hook did not capture output from layer '{self.layer_name}'. "
                               f"Ensure that layer runs in this forward pass.")
        return out

    def remove_hook(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None




#%% Imagelatents + Classification head for u2tokenizer-based CLIP
import torch
import torch.nn as nn


class ImageLatentsClassifierTokenRefined(nn.Module):
    def __init__(
        self,
        trained_model,
        latent_dim,  # shared space dim (D)
        num_classes,
        dropout_prob=0.3,
        out_channels: list = None,
        freeze_latents: bool = True,
        pooling: str = "average",  # "average" or "attention"
    ):
        """
        Initialize the image latent classifier for linear probing or fine-tuning.

        :param trained_model: (nn.Module) Pretrained CLIP-based model that returns image latents of shape (B, K, D).
        :param latent_dim: (int) Dimensionality of the shared embedding space (D). 
        :param num_classes: (int) Number of output classes for the classification task. 
        :param dropout_prob: (float) Dropout probability applied within the classification head.
        :param out_channels: (list, optional) List defining intermediate hidden layer dimensions of the classification head. 
        :param freeze_latents: (bool) If True, all parameters of the pretrained image encoder are frozen.
        :param pooling: (str) Pooling strategy used to aggregate token-level image latents (B, K, D) into a global representation (B, D). 
                        Options: - "average": mean pooling across tokens.
                                 - "attention": learnable attention-weighted pooling.
        """

        super(ImageLatentsClassifierTokenRefined, self).__init__()
        self.trained_model = trained_model
        self.pooling = pooling.lower()

        # Freeze weights (linear probing if True)
        if freeze_latents:
            for param in self.trained_model.parameters():
                param.requires_grad = False

        # Add condition to handle None for out_channels
        if out_channels is not None:
            out_channels = [latent_dim] + out_channels
        else:
            out_channels = [latent_dim]

        # Attention pooling (tiny param head)
        # Takes tokens (B,K,D) -> scores (B,K,1) -> softmax over K -> weighted sum -> (B,D)
        if self.pooling == "attention":
            # Uncomment Option 1: Simpler. No non-linearity
            self.attn_pool = nn.Linear(latent_dim, 1)

            # # Uncomment Option 2: Slightly fancy. Adds non-linearity
            # self.attn_pool = nn.Sequential(
            #     nn.Linear(latent_dim, latent_dim // 2 if latent_dim >= 2 else 1),
            #     nn.Tanh(),
            #     nn.Linear(latent_dim // 2 if latent_dim >= 2 else 1, 1),
            # )
        else:
            self.attn_pool = None  # not used for average pooling

        # Classification head expects (B, D) ideally
        self.classifier = ClassificationHeadCTCLIP(
            num_classes=num_classes,
            out_channels=out_channels,
            dropout=dropout_prob,
        )

    def _pool_latents(self, image_latents: torch.Tensor) -> torch.Tensor:
        """
        Pool (B, K, D) -> (B, D) using either average or attention-weighted pooling.
        """
        if image_latents.dim() != 3:
            raise ValueError(
                f"Expected image_latents of shape (B,K,D), got {tuple(image_latents.shape)}"
            )

        if self.pooling == "average":
            return image_latents.mean(dim=1)                    # (B,D)

        if self.pooling == "attention":
            scores = self.attn_pool(image_latents)              # (B,K,1)
            weights = torch.softmax(scores, dim=1)              # (B,K,1)
            weighted_sum = (weights * image_latents).sum(dim=1) # (B,D)

            return weighted_sum

        raise ValueError(f"Unknown pooling='{self.pooling}'. Use 'average' or 'attention'.")

    # MKD added the forward method <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    def forward(self, text_tokens, image_inputs, **kwargs):
        return_loss = False
        _, image_latents, _ = self.trained_model(text_tokens, image_inputs, return_loss, **kwargs)

        # image_latents is assumed (B,K,D)
        pooled = self._pool_latents(image_latents)  # (B,D)

        return self.classifier(pooled)

    def save(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load(self, file_path, map_location="cpu"):
        loaded_state_dict = torch.load(file_path, map_location=map_location)
        self.load_state_dict(loaded_state_dict)
