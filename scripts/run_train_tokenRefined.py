import sys
import os

sys.path.append(os.getcwd() + '/utils/') 

import torch
# from transformer_maskgit import CTViT
# from transformers import BertTokenizer, BertModel
from ct_clip import CLIPTokenRefined 
# from CLIPTrainerRefined import ClipTrainer # this does not have the unique report sampler
from CLIPTrainerRefined_v2 import ClipTrainer # this has the unique report sampler

from misc import save_config # misc is in utils
from nets.text_models import build_text_model
from nets.image_models import build_image_model
import yaml
from tqdm import tqdm
import argparse
from box import Box

# Function to read config file from command line
def get_config_from_args():
    parser = argparse.ArgumentParser(description="Pass config file")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML config file")
    args = parser.parse_args()
    return args

# Get the config file from command-line arguments
args = get_config_from_args()

with open(args.config, "r") as file:
    config = yaml.safe_load(file)
config = Box(config)

# Save config file
os.makedirs(config.train.save_ckpts, exist_ok=True)
save_config(config, save_dir=config.train.save_ckpts, filename="config.yaml", verbose=True)

# Text encoder
tokenizer, text_encoder = build_text_model(config.text_encoder)

tokenizer_eoncode_kwargs = config.text_encoder.get("tokenizer_encode_kwargs")

# Image encoder
image_encoder = build_image_model(config.image_encoder)

# CLIP model
clip = CLIPTokenRefined(
    image_encoder = image_encoder,
    text_encoder = text_encoder,
    image_feat_dim = config.vlm.image_feat_dim, # 131072, #2097152,           # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MKD changed
    text_feat_dim = config.vlm.text_feat_dim,
    shared_latent_dim = config.vlm.shared_latent_dim,
    token_refiner_dict = config.others.token_refiner_dict,
    contrastive_loss_temperature = config.loss.contrastive_loss_temperature,
    filip_pool = config.loss.filip_pool,
    filip_lse_alpha = config.loss.filip_lse_alpha,
    filip_decoupled = config.loss.filip_decoupled,
    filip_max_logit_scale = config.loss.filip_max_logit_scale,
)

# Trainer
trainer = ClipTrainer( 
    clip,
    reports_file_train = config.directories.reports_train,
    reports_file_valid = config.directories.reports_val,
    text_column = config.pathologies.text_column,
    use_random_window = config.text_encoder.use_random_window,
    end_bias = config.text_encoder.end_bias,
    data_train = config.directories.images_train,
    data_valid = config.directories.images_val,
    labels = config.directories.labels_val,
    batch_size = config.train.batch_size,
    num_train_steps = config.train.n_train_steps,
    num_workers = config.train.n_workers,
    save_results_every = config.train.save_results_every,
    save_model_every = config.train.save_model_every,
    results_folder = config.train.save_results,
    ckpts_folder = config.train.save_ckpts,
    resume_from = config.train.resume_from,
    pathologies = config.pathologies.labels,
    resample = config.dataloader.resample,
    n_zSlices = config.dataloader.n_zSlices,
    zSlices_pad_value = config.dataloader.zSlices_pad_value,
    clip = config.dataloader.intensity_clip,
    clip_percentile = config.dataloader.clip_percentile,
    normalize = config.dataloader.normalize,
    resize_shape = config.dataloader.resize_shape,
    transform = config.dataloader.transform,
    verbose = config.dataloader.verbose,
    tokenizer = tokenizer,
    tokenizer_kwargs = tokenizer_eoncode_kwargs,
    lr = config.train.lr, 
    wd = config.train.weight_decay,
    max_grad_norm = config.train.max_grad_norm,
    # accelerate_kwargs
)

trainer.train()

# Note !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Need to change input parameters of the dataloader

# Run command
# nohup accelerate launch --multi_gpu run_train.py >log_ctclip.log &


