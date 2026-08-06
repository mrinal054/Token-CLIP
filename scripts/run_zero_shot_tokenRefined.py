# MKD added following two lines to handle "too many files open" error.
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

import sys
import os

sys.path.append(os.getcwd() + '/utils/') 

import torch
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel

# from ct_clip import CTCLIP
# >>>>> Change:
from ct_clip import CLIPTokenRefined

# from zero_shot import CTClipInference
# >>>>> Change:
from zero_shot_tokenRefined import CTClipInference

import accelerate

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

# Text encoder
tokenizer, text_encoder = build_text_model(config.text_encoder)

# Image encoder
image_encoder = build_image_model(config.image_encoder)

# CLIP model
# clip = CTCLIP(...)
# >>>>> Change:
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


# Load checkpoint
clip.load(config.inference.load_ckpt)

# Uncomment for vocabfine inference 
inference = CTClipInference(
    clip,
    data_folder = config.directories.images_test,
    reports_file= config.directories.reports_test,
    text_column = config.pathologies.text_column,
    labels = config.directories.labels_test,
    batch_size = config.inference.batch_size,
    n_workers = config.inference.n_workers,
    results_folder= config.inference.save_results,
    resample = config.dataloader.resample,
    n_zSlices = config.dataloader.n_zSlices,
    zSlices_pad_value = config.dataloader.zSlices_pad_value,
    clip = config.dataloader.intensity_clip,
    clip_percentile=config.dataloader.clip_percentile,
    normalize = config.dataloader.normalize,
    resize_shape = config.dataloader.resize_shape,
    transform = None,
    verbose = config.dataloader.verbose,
    pathologies = config.pathologies.labels,
    tokenizer = tokenizer,
)

inference.infer()

# Run command
# nohup accelerate launch --multi_gpu run_zero_shot.py > log_ctclip_test2.log 2>&1 &