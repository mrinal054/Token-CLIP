import sys
import os

sys.path.append(os.getcwd() + '/utils/') 

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
# from data_inference import CTReportDatasetinfer
from data_inference import CTReportDatasetinferKLab

from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
# from ct_clip import CTCLIP
from ct_clip import CLIPTokenRefined
import torch.nn.functional as F
from src.args import parse_arguments
from src.models.utils import cosine_lr, torch_load, LabelSmoothing

import pandas as pd
import matplotlib.pyplot as plt

from misc import save_config # misc is in utils
from nets.text_models import build_text_model
from nets.image_models import build_image_model
import yaml
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

NUM_CHUNKS = (len(config.pathologies.labels) + config.train.chunk_size - 1) // config.train.chunk_size # ceil division

"""
# Note: How chunking works (MKD added)
If there are 15 pahtologies and chunk size is 5, then there will be 3 chunks.
Each will have 5 data points - 5, 5, 5.
For 18 pathologies, it will 4 chunks - 5, 5, 5, 3.
"""

# Helper function to create chunk <<<<<<<<<<< added by MKD
def chunkify(seq, size):
    """Yield successive size-sized chunks from seq."""
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

def get_lr(optimizer):
    # Function to get the current learning rate of the optimizer
    for param_group in optimizer.param_groups:
        return param_group['lr']

def finetune(config):

    # Text encoder
    tokenizer, text_encoder = build_text_model(config.text_encoder)

    tokenizer_eoncode_kwargs = config.text_encoder.get("tokenizer_encode_kwargs")

    text_encoder.resize_token_embeddings(len(tokenizer))

    # Image encoder
    image_encoder = build_image_model(config.image_encoder)

    # CLIP model
    # clip = CTCLIP(
    #     image_encoder = image_encoder,
    #     text_encoder = text_encoder,
    #     dim_image = config.vlm.dim_image,        
    #     dim_text = config.vlm.dim_text,
    #     dim_latent = config.vlm.dim_latent,
    #     extra_latent_projection = config.vlm.extra_latent_projection,        # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
    #     use_mlm = config.vlm.use_mlm,
    #     downsample_image_embeds = config.vlm.downsample_image_embeds,
    #     use_all_token_embeds = config.vlm.use_all_token_embeds,
    # )


    # token_refiner_dict = dict(config.get("token_refiner_dict", config.get("token_refiner", {})))

    # clip = CLIPTokenRefined(
    #     image_encoder=image_encoder,
    #     text_encoder=text_encoder,
    #     text_feat_dim=config.vlm.dim_text,
    #     image_feat_dim=token_refiner_dict.get("embed_size", config.vlm.dim_image),
    #     shared_latent_dim=config.vlm.dim_latent,
    #     token_refiner_dict=token_refiner_dict,
    #     contrastive_loss_temperature=config.vlm.get("contrastive_loss_temperature", 0.07),
    #     filip_pool=config.vlm.get("filip_pool", "logsumexp"),
    #     filip_lse_alpha=config.vlm.get("filip_lse_alpha", 10.0),
    #     filip_decoupled=config.vlm.get("filip_decoupled", False),
    #     filip_max_logit_scale=config.vlm.get("filip_max_logit_scale", 100.0),
    # )

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

    clip.load(config.train.load_contrastive_ckpt)

    # num_classes = 18  # Specify the number of classes <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    print('Fine-tuning end-to-end')
    model = clip
    for name, param in model.named_parameters():
        if "latent" in name:
            print(name, param.shape)
        else:
            param.requires_grad = False

    ds = CTReportDatasetinferKLab(
                        img_dir=config.directories.images_train,
                        report_file=config.directories.reports_train,
                        text_column = config.pathologies.text_column,
                        label_file=config.directories.labels_train,
                        label_cols=config.pathologies.labels,
                        resample=config.dataloader.resample,
                        n_zSlices=config.dataloader.n_zSlices,
                        zSlices_pad_value=config.dataloader.zSlices_pad_value,
                        clip=config.dataloader.intensity_clip,
                        normalize=config.dataloader.normalize,
                        resize_shape=config.dataloader.resize_shape,
                        transform=config.dataloader.transform,
                        verbose=config.dataloader.verbose,
                    )
   
    dl = DataLoader(ds, num_workers=config.train.n_workers, batch_size=config.train.batch_size, shuffle=True)
    num_batches = len(dl)

    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # <<<<<<<<<<<<<<<< commented out by MKD

    model.cuda()
    devices = list(range(torch.cuda.device_count()))
    print('Using devices', devices)
    model = torch.nn.DataParallel(model, device_ids=devices)
    model.train()

    loss_fn = torch.nn.MSELoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    scheduler = cosine_lr(optimizer, config.train.lr, config.train.warmup_length, config.train.finetune_epochs * num_batches)

    for epoch in range(config.train.finetune_epochs):
        for i, batch in tqdm.tqdm(enumerate(dl)):
            start_time = time.time()
            step = i + epoch * num_batches
            scheduler(step)

            inputs, _, labels, _ = batch

            logits = []
            labels_tensor_all = labels.float().to(torch.device('cuda'))

            for k in range(NUM_CHUNKS): # <<<<<<<<< mkd changed to NUM_CHUNKS from 3
                logits_list = []
                labels_list = []

                pathologies_all = config.pathologies.labels

                # <<<<<<<<<<<<<< MKD commented out manual chunking
                # pathologies = pathologies_all[k * 6:(k + 1) * 6]
                # labels_tensor = labels_tensor_all[0][k * 6:(k + 1) * 6]

                # MKD added following chunking
                start_idx = k * config.train.chunk_size
                end_idx = start_idx + config.train.chunk_size
                pathologies = pathologies_all[start_idx:end_idx] # end idx is automatically truncated by Python
                labels_tensor = labels_tensor_all[0][start_idx:end_idx]

                for l in range(len(labels_tensor)):
                    text_yes = ""
                    text_no = ""
                    if labels_tensor[l] == 1:
                        text_yes = text_yes + f"{pathologies[l]}. "
                        text_no = text_no + f"not {pathologies[l]}. "
                    if labels_tensor[l] == 0:
                        text_yes = text_yes + f"not {pathologies[l]}. "
                        text_no = text_no + f"{pathologies[l]}. "
                    
                    text = [text_yes, text_no]
                    text_tokens = tokenizer(text, **config.text_encoder.get("tokenizer_encode_kwargs")).to(torch.device('cuda')) # <<<<<<<<< MKD added
                    # output = model(text_tokens, inputs, device=torch.device('cuda'))
                    # logits = F.softmax(output, dim=0)
                    # labels = torch.tensor([1.0, 0.0]).cuda()

                    output, _, _ = model(text_tokens, inputs)
                    logits = F.softmax(output, dim=1)
                    labels_target = torch.tensor([[1.0, 0.0]], device=logits.device).expand(logits.size(0), -1)

                    logits_list.append(logits)
                    # labels_list.append(labels)
                    labels_list.append(labels_target)

                concat_logits = torch.cat(logits_list, dim=0)
                concat_labels = torch.cat(labels_list, dim=0)

                loss = loss_fn(concat_logits, concat_labels)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            # print(get_lr(optimizer)) # <<<<<<<<<<<<<<<<<<<<<< MKD commented out

            batch_time = time.time() - start_time

            if i % config.train.print_every == 0:
                percent_complete = 100 * i / len(dl)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dl)}]\t"
                    f"Loss: {loss.item():.6f}\tBatch (t) {batch_time:.3f}", flush=True
                )
            if i % config.train.save_model_every == 0:
                os.makedirs(config.train.save_ckpts, exist_ok=True)

                # Access the underlying model to avoid the 'module.' prefix in state_dict keys
                model_to_save = model.module if hasattr(model, 'module') else model

                model_path = os.path.join(config.train.save_ckpts, f'checkpoint_{i}_epoch_{epoch+1}.pt')
                print('Saving model to', model_path)

                # Save the state_dict of the unwrapped model
                torch.save(model_to_save.state_dict(), model_path)

                optim_path = os.path.join(config.train.save_ckpts, f'optim_{i}_epoch_{epoch+1}.pt')

                # Save the optimizer state
                torch.save(optimizer.state_dict(), optim_path)

        # Saving model
        if config.train.save_ckpts is not None:
            os.makedirs(config.train.save_ckpts , exist_ok=True)

            # Access the underlying model to avoid the 'module.' prefix in state_dict keys
            model_to_save = model.module if hasattr(model, 'module') else model

            model_path = os.path.join(config.train.save_ckpts , f'epoch_{epoch+1}.pt')
            print('Saving model to', model_path)

            # Save the state_dict of the unwrapped model
            torch.save(model_to_save.state_dict(), model_path)

            optim_path = os.path.join(config.train.save_ckpts , f'optim_{epoch+1}.pt')

            # Save the optimizer state
            torch.save(optimizer.state_dict(), optim_path)

    if config.train.save_ckpts  is not None:
        return model_path


if __name__ == '__main__':
    # args = parse_arguments()
    finetune(config)
