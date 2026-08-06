"""
Mrinal Kanti Dhar
August 1, 2025

Last modfied: January 27, 2026

v2: Classification_head added.
v3: Classification_head modified. Supports out_channels=None. Mimics CT-CLIP classification head.
v4: Image model checkpoint loading added.
"""
import sys
import os
sys.path.append(os.getcwd() + '/utils/') 

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_inference import CTReportDatasetinferKLab

from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP

# from classification_head import ClassificationHeadCTCLIP
from classifier import ImageLatentsClassifierTokenRefined
from misc import save_config # misc is in utils

from src.models.utils import cosine_lr

import pandas as pd
import matplotlib.pyplot as plt

from nets.text_models import build_text_model
from nets.image_models import build_image_model
import yaml
from tqdm import tqdm
import argparse
from box import Box

from ct_clip import CLIPTokenRefined

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

def finetune(config):
    
    # Text encoder
    tokenizer, text_encoder = build_text_model(config.text_encoder)

    tokenizer_eoncode_kwargs = config.text_encoder.get("tokenizer_encode_kwargs")

    text_encoder.resize_token_embeddings(len(tokenizer))

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

    contrastive_ckpt = config.train.load_contrastive_ckpt
    if contrastive_ckpt is not None:
        clip.load(contrastive_ckpt)
        print(f"Following conttrastive checkpoint loaded: {contrastive_ckpt}")
    else:
        print(f"No contrastive checkpoint loaded")  

    # Define the number of classes and initialize the image classifier
    num_classes = len(config.pathologies.labels)  

    image_classifier = ImageLatentsClassifierTokenRefined(clip, 
                                              latent_dim=config.vlm.shared_latent_dim,                                               
                                              num_classes=num_classes,
                                              dropout_prob=config.image_encoder.classifier_dropout,
                                              out_channels=config.image_encoder.classifier_out_chs,
                                              freeze_latents=config.image_encoder.classifier_freeze_latents,
                                              pooling="attention",)   

    # Load dataset for fine-tuning
    ds_train = CTReportDatasetinferKLab(
        img_dir=config.directories.images_train,
        report_file=config.directories.reports_train,
        text_column = config.pathologies.text_column,
        label_file=config.directories.labels_train,
        label_cols=config.pathologies.labels,
        resample=config.dataloader.resample,
        n_zSlices=config.dataloader.n_zSlices,
        zSlices_pad_value=config.dataloader.zSlices_pad_value,
        clip=config.dataloader.intensity_clip,
        clip_percentile=config.dataloader.clip_percentile,
        normalize=config.dataloader.normalize,
        resize_shape=config.dataloader.resize_shape,
        transform=config.dataloader.transform,
        verbose=config.dataloader.verbose,
    )

    dl_train = DataLoader(ds_train, 
                          num_workers=config.train.n_workers, 
                          batch_size=config.train.batch_size, 
                          shuffle=True) 
    
    ds_valid = CTReportDatasetinferKLab(
        img_dir=config.directories.images_val,
        report_file=config.directories.reports_val,
        text_column = config.pathologies.text_column,
        label_file=config.directories.labels_val,
        label_cols=config.pathologies.labels,
        resample=config.dataloader.resample,
        n_zSlices=config.dataloader.n_zSlices,
        zSlices_pad_value=config.dataloader.zSlices_pad_value,
        clip=config.dataloader.intensity_clip,
        clip_percentile=config.dataloader.clip_percentile,
        normalize=config.dataloader.normalize,
        resize_shape=config.dataloader.resize_shape,
        transform=None,
        verbose=config.dataloader.verbose,
    )    
    
    dl_valid = DataLoader(ds_valid, 
                          num_workers=config.train.n_workers, 
                          batch_size=config.train.batch_size, # 1, 
                          shuffle=False)
    
    num_batches = len(dl_train)

    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MKD commented out -- this holds cpu and gpu

    # Move model to GPU and set it to training mode
    model = image_classifier.cuda()
    devices = list(range(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model, device_ids=devices)
    model.train()

    # Define loss function and optimizer
    weights = torch.tensor(config.loss.weights).cuda()

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weights) 

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    scheduler = cosine_lr(optimizer, config.train.lr, config.train.warmup_length, config.train.finetune_epochs * num_batches)
    
    # Resume linearprobing checkpoint
    if config.train.get("load_lipro_ckpt") is not None:
        print(f"Loading LiPro model checkpoint from {config.train.load_lipro_ckpt}")

        ckpt = torch.load(config.train.load_lipro_ckpt, map_location="cuda", weights_only=True)

        # Important: load into underlying model (no 'module.' mismatch)
        model_to_load = model.module if hasattr(model, "module") else model
        model_to_load.load_state_dict(ckpt, strict=True)

        if config.train.get("load_lipro_optim_ckpt") is not None:
            print(f"Loading optimizer checkpoint from {config.train.load_lipro_optim_ckpt}")
            optim_ckpt = torch.load(config.train.load_lipro_optim_ckpt, map_location="cuda", weights_only=False,)
            optimizer.load_state_dict(optim_ckpt)

        print("LiPro checkpoint loaded successfully")

    # Start training loop
    total_train_loss = []
    total_valid_loss = []

    if config.train.load_lipro_ckpt is not None:
        start_epoch = config.train.get("resume_epoch", 0) 
    else:
        start_epoch = 0

    for epoch in range(start_epoch, config.train.finetune_epochs):
        train_losses = []
        valid_losses = []
        for i, batch in tqdm(enumerate(dl_train)):
            
            start_time = time.time()
            step = i + epoch * num_batches

            inputs, _, labels, _ = batch

            labels = labels.float().cuda()
            
            B = inputs.shape[0] # Batch size (MKD added)
            
            text_tokens = tokenizer([" "] * B, **tokenizer_eoncode_kwargs).to("cuda") # MKD added *B to support batchsize > 1. 

            data_time = time.time() - start_time
          
            logits = model(text_tokens, inputs) # MKD added <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
            loss = loss_fn(logits, labels)
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler(step)
            batch_time = time.time() - start_time

            if i % config.train.print_every == 0:
                percent_complete = 100 * i / len(dl_train)
                print(f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dl_train)}]\t"
                      f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True)

            if i % config.train.save_model_every == 0:
                
                # Access the underlying model to avoid the 'module.' prefix in state_dict keys
                model_to_save = model.module if hasattr(model, 'module') else model

                model_path = os.path.join(config.train.save_ckpts, f'checkpoint_{i}_epoch_{epoch+1}.pt')
                print('Saving model to', model_path)

                # Save the state_dict of the unwrapped model
                torch.save(model_to_save.state_dict(), model_path)

                optim_path = os.path.join(config.train.save_ckpts, f'optim_{i}_epoch_{epoch+1}.pt')

                # Save the optimizer state
                torch.save(optimizer.state_dict(), optim_path)
        
        # Validation Loop
        with torch.no_grad():
            for i, batch in tqdm(enumerate(dl_valid)):
                start_time = time.time()
                step = i + epoch * num_batches
                inputs, _, labels, _ = batch
                labels = labels.float().cuda()
                
                B = inputs.shape[0] # Batch size (MKD added)
                text_tokens = tokenizer([" "] * B, **tokenizer_eoncode_kwargs).to("cuda") # MKD added *B to support batchsize > 1. 
                # text_tokens = tokenizer([" "], return_tensors="pt", padding="max_length", truncation=True, max_length=512).to("cuda")
                # logits = model(text_tokens, inputs, device=torch.device('cuda')) # <<<<<<<<<<<<<<<<<<<<<<<<<< MKD commented out
                logits = model(text_tokens, inputs) # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MKD added
                loss = loss_fn(logits, labels)
                valid_losses.append(loss.item())

        total_train_loss.append(sum(train_losses) / len(train_losses))
        total_valid_loss.append(sum(valid_losses) / len(valid_losses))
        
        # Save final model
        if config.train.save_ckpts is not None:
            os.makedirs(config.train.save_ckpts, exist_ok=True)

            # Access the underlying model to avoid the 'module.' prefix in state_dict keys
            model_to_save = model.module if hasattr(model, 'module') else model

            model_path = os.path.join(config.train.save_ckpts, f'epoch_{epoch+1}.pt')
            print('Saving model to', model_path)

            # Save the state_dict of the unwrapped model
            torch.save(model_to_save.state_dict(), model_path)

            optim_path = os.path.join(config.train.save_ckpts, f'optim_epoch_{epoch+1}.pt')

            # Save the optimizer state
            torch.save(optimizer.state_dict(), optim_path)
        
        # Plot Loss Functions
        n = min(len(total_train_loss), len(total_valid_loss))
        losses_df = pd.DataFrame({
            "epoch": list(range(start_epoch + 1, start_epoch + 1 + n)),
            "train_loss": total_train_loss[:n],
            "valid_loss": total_valid_loss[:n],
        })
        losses_df.to_excel(os.path.join(config.train.save_ckpts, 'losses.xlsx'), index=False)

        plt.figure(figsize=(10, 5))
        plt.plot(total_train_loss, label='Training Loss', marker='o')
        plt.plot(total_valid_loss, label='Validation Loss', marker='o')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(config.train.save_ckpts, 'loss_plot.png'))  # Save the plot as a PNG file

if __name__ == '__main__':

    # args = parse_arguments()
    # finetune(args)

    finetune(config)


"""
# Terminal Command
CUDA_VISIBLE_DEVICES=1 nohup python ct_lipro_train.py \
    --lr 1e-5 \
    --wd 0.1 \
    --epochs 200 \
    --warmup_length 100 \
    --save /research/m324371/Project/Digital_Twin/CT-CLIP/my_exp/runs/ct_all_1 \
    --pretrained /research/m324371/Project/Digital_Twin/CT-CLIP/scripts/output_folder_contrastive/CTClip.100000.pt \
    --data-folder /research/m324371/Project/Digital_Twin/Classification/Dataset/CT-ALL \
    --reports-file /research/m324371/Project/Digital_Twin/CT-CLIP/Dataframes/Dataset_CT_train.xlsx \
    --labels /research/m324371/Project/Digital_Twin/CT-CLIP/Dataframes/Dataset_CT_train_label.xlsx \
    --val_data_folder /research/m324371/Project/Digital_Twin/Classification/Dataset/CT-ALL \
    --val_reports_file /research/m324371/Project/Digital_Twin/CT-CLIP/Dataframes/Dataset_CT_val.xlsx \
    --val_labels /research/m324371/Project/Digital_Twin/CT-CLIP/Dataframes/Dataset_CT_val_label.xlsx \
> log_ctclip0.log 2>&1 &
"""
