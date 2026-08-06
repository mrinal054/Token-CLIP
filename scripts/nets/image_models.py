import torch
import torch.nn as nn

def load_swinunetr(cfg):
    """
    cfg corresponds to config.image_encoder

    MONAI SwinUNETR: https://monai-dev.readthedocs.io/en/fixes-sphinx/networks.html#swinunetr
    Pretrained weights: https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt
    Useful tutorial: https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_btcv_segmentation_3d.ipynb
    """
    
    from monai.networks.nets import SwinUNETR
    import inspect

    from nets.encoder_swinunetr import SwinUNETREncoder

    # Match CTViT naming
    dim = cfg.get("dim", 512)
    export_stage = cfg.get("export_stage", 3)

    # img_size: (D, H, W)
    img_size = cfg.get("img_size", None)
    img_size = tuple(img_size)

    model_kwargs = {
        "img_size": img_size,
        "in_channels": cfg.get("in_channels", 1),
        "out_channels": cfg.get("out_channels", 1),  # unused (decoder ignored)
        "feature_size": cfg.get("feature_size", 48),
        "use_checkpoint": cfg.get("use_checkpoint", False),
        "spatial_dims": cfg.get("spatial_dims", 3),
        "depths": cfg.get("depths", (2, 2, 2, 2)),
        "num_heads": cfg.get("num_heads", (3, 6, 12, 24)),
        "norm_name": cfg.get("norm_name", "instance"),
        "drop_rate": cfg.get("drop_rate", 0.0),
        "attn_drop_rate": cfg.get("attn_drop_rate", 0.0),
        "dropout_path_rate": cfg.get("dropout_path_rate", 0.0),
        "normalize": cfg.get("normalize", True),
        "downsample": cfg.get("downsample", "mergingv2"),
        "use_v2": cfg.get("use_v2", False),
    }
    
    sig = inspect.signature(SwinUNETR.__init__)
    allowed = set(sig.parameters.keys())
    filtered_kwargs = {k: v for k, v in model_kwargs.items() if k in allowed}
    swin = SwinUNETR(**filtered_kwargs)

    # Load pretrained weights
    if cfg.get("pretrained_dir"):
        weight = torch.load(cfg.get("pretrained_dir"), weights_only=True)
        swin.load_from(weights=weight)
        print("Using pretrained self-supervied Swin UNETR backbone weights!")


    return SwinUNETREncoder(
        swinunetr=swin,
        dim=dim,
        export_stage=export_stage,
    )


def load_ctvit(cfg):
    from transformer_maskgit import CTViT
    """
    cfg corresponds to config.image_encoder
    """
    # Pull parameters directly from cfg with sensible defaults
    dim = cfg.get("dim", 512)
    codebook_size = cfg.get("codebook_size", 8192)
    image_size = cfg.get("image_size", 128)
    patch_size = cfg.get("patch_size", 8)
    temporal_patch_size = cfg.get("temporal_patch_size", 8)
    spatial_depth = cfg.get("spatial_depth", 4)
    temporal_depth = cfg.get("temporal_depth", 4)
    dim_head = cfg.get("dim_head", 32)
    heads = cfg.get("heads", 8)

    # Sanity checks
    if image_size % patch_size != 0:
        raise ValueError(f"image_size {image_size} must be divisible by patch_size {patch_size}.")

    model = CTViT(
        dim=dim,
        codebook_size=codebook_size,
        image_size=image_size,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        spatial_depth=spatial_depth,
        temporal_depth=temporal_depth,
        dim_head=dim_head,
        heads=heads,
    )
    
    return model



# Registry of available image encoders
REGISTRY = {
    "ctvit": load_ctvit,  
    "swinunetr": load_swinunetr, 
    # resnet152: load_resnet152,
}

def build_image_model(cfg):
    """
    cfg is the dict under config.image_encoder
    """
    name = cfg.get("name", "")
    if name not in REGISTRY:
        raise ValueError(f"Unknown image encoder '{name}'. Available: {list(REGISTRY.keys())}")
    return REGISTRY[name](cfg)
