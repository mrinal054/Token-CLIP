"""
v2: ImageLatentsClassifier is called from utils/classifier.py
"""
import sys
import os 

sys.path.append(os.getcwd() + '/utils/') 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.args import parse_arguments
from transformers import BertTokenizer, BertModel
from transformer_maskgit import CTViT
from ct_clip import CLIPTokenRefined
# from data_inference import CTReportDatasetinfer
from data_inference import CTReportDatasetinferKLab
from eval import evaluate_internal, plot_roc, accuracy, sigmoid, bootstrap, compute_cis
import tqdm
import numpy as np
import pandas as pd
# from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, multilabel_confusion_matrix 

import copy

import pandas as pd
import matplotlib.pyplot as plt

# from classification_head import ClassificationHeadCTCLIP
from classifier import ImageLatentsClassifierTokenRefined
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


def sigmoid(tensor):
    return 1 / (1 + torch.exp(-tensor))


def evaluate_model(config, model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    model = model.to(device)
    correct = 0
    total = 0
    predictedall=[]
    realall=[]
    logits = []
    accs = []
    with torch.no_grad():

        for batch in tqdm.tqdm(dataloader):
            inputs, _, labels, acc_no = batch
            labels = labels.float().to(device)
            inputs = inputs.to(device)
            text_tokens = tokenizer("", **tokenizer_eoncode_kwargs).to(device)
            # output = model(False, text_tokens, inputs,  device=device)
            output = model(text_tokens, inputs).to(device) # MKD added <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            realall.append(labels.detach().cpu().numpy()[0]) # >>>>>>>>>>>>>>> !!! ATTENTION: Only taking the first element in the batch
            # save_out = sigmoid(torch.tensor(output)).cpu().numpy() # <<<<<<<<<<< MKD commented out
            save_out = torch.sigmoid(output).detach().cpu().numpy()  # <<<<<<<<<<< MKD added
            predictedall.append(save_out[0]) # >>>>>>>>>>>>>>> !!! ATTENTION: Only taking the first element in the batch
            accs.append(acc_no[0]) # >>>>>>>>>>>>>>> !!! ATTENTION: Only taking the first element in the batch
            print(acc_no[0], flush=True)

        plotdir = config.inference.save_results
        os.makedirs(plotdir, exist_ok=True)
        logits = np.array(logits)

        with open(f"{plotdir}accessions.txt", "w") as file:
            for item in accs:
                file.write(item[0] + "\n")

        pathologies = config.pathologies.labels # label names

        realall=np.array(realall)
        predictedall=np.array(predictedall)

        # Convert probabilities into predictions
        thresholds = np.array([0.5]*len(pathologies))
        assert len(pathologies) == len(thresholds), f"Length mismatch. {len(pathologies)} pathologies, whereas {len(thresholds)} thresholds."
        predLabelall = (predictedall >= thresholds).astype(int)        

        # Store gt and prediction in excel file
        df_probabilities = pd.DataFrame(predictedall, columns=pathologies)
        df_probabilities.insert(0, 'Name', accs)

        df_predLabelall = pd.DataFrame(predLabelall, columns=pathologies)
        df_predLabelall.insert(0, 'Name', accs)

        df_gts = pd.DataFrame(realall, columns=pathologies)
        df_gts.insert(0, 'Name', accs)

        with pd.ExcelWriter(f'{plotdir}results.xlsx', engine='xlsxwriter') as writer:
            df_probabilities.to_excel(writer, sheet_name="Probabilities", index=False)
            df_predLabelall.to_excel(writer, sheet_name="Predictions", index=False)
            df_gts.to_excel(writer, sheet_name="GTs", index=False)


        "Calculate metrics"
        # Initialize metrics storage
        metrics = {
            'Pathology': [],
            'Accuracy': [],
            'Specificity': [],
            'Precision': [],
            'Recall': [],
            'F1-score': [],
            'AUC': []
        }

        # Loop through each pathology
        for pathology in pathologies:
            y_true = df_gts[pathology].values
            y_pred = df_predLabelall[pathology].values
            y_prob = df_probabilities[pathology].values

            # Accuracy
            acc = accuracy_score(y_true, y_pred)

            # Confusion matrix to compute specificity
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0

            # Other metrics
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            # AUC
            try:
                auc = roc_auc_score(y_true, y_prob)
            except:
                auc = float('nan')  # AUC undefined if only one class present

            # Store
            metrics['Pathology'].append(pathology)
            metrics['Accuracy'].append(acc)
            metrics['Specificity'].append(specificity)
            metrics['Precision'].append(precision)
            metrics['Recall'].append(recall)
            metrics['F1-score'].append(f1)
            metrics['AUC'].append(auc)

        # Convert to DataFrame
        df_metrics = pd.DataFrame(metrics)

        # Save to Excel
        with pd.ExcelWriter(f'{plotdir}metrics.xlsx', engine='xlsxwriter') as writer:
            df_metrics.to_excel(writer, sheet_name='Metrics', index=False)


        # Back to original code <<<<<<<< MKD added
        np.savez(f"{plotdir}labels_weights.npz", data=realall)
        np.savez(f"{plotdir}predicted_weights.npz", data=predictedall)

        dfs=evaluate_internal(predictedall,realall,pathologies, plotdir)

        writer = pd.ExcelWriter(f'{plotdir}aurocs.xlsx', engine='xlsxwriter')

        dfs.to_excel(writer, sheet_name='Sheet1', index=False)

        writer.close()
        
        print(f"Liprobing inference saved to {plotdir}") 




if __name__ == '__main__':
 
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
    )

    num_classes = len(config.pathologies.labels)   

    image_classifier = ImageLatentsClassifierTokenRefined(clip, 
                                              latent_dim=config.vlm.shared_latent_dim,                                               
                                              num_classes=num_classes,
                                              dropout_prob=config.image_encoder.classifier_dropout,
                                              out_channels=config.image_encoder.classifier_out_chs,
                                              freeze_latents=config.image_encoder.classifier_freeze_latents,
                                              pooling="attention",) 
      
    zero_shot = copy.deepcopy(image_classifier)

    # Load checkpoint
    image_classifier.load(config.inference.load_ckpt)  

    # Prepare the evaluation dataset
    ds = CTReportDatasetinferKLab(
        img_dir=config.directories.images_test,
        report_file=config.directories.reports_test,
        text_column = config.pathologies.text_column,
        label_file=config.directories.labels_test,
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

    dl = DataLoader(ds, num_workers=config.inference.n_workers, batch_size=config.inference.batch_size, shuffle=False)

    # Evaluate the model
    evaluate_model(config, image_classifier, dl, torch.device('cuda'))

"""
# Terminal command
CUDA_VISIBLE_DEVICES=2 nohup python ct_lipro_inference.py \
    --save /research/m324371/Project/Digital_Twin/CT-CLIP/my_exp/outputs/ct_all_1 \
    --pretrained /research/m324371/Project/Digital_Twin/CT-CLIP/my_exp/runs/ct_all_1/checkpoint_6000_epoch_200.pt \
    --data-folder /research/m324371/Project/Digital_Twin/Classification/Dataset/CT-ALL \
    --reports-file /research/m324371/Project/Digital_Twin/CT-CLIP/Dataframes/Dataset_CT_test.xlsx \
    --labels /research/m324371/Project/Digital_Twin/CT-CLIP/Dataframes/Dataset_CT_test_label.xlsx \
> log_ctclipLiproInf.log 2>&1 &
"""