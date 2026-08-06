from pathlib import Path
from shutil import rmtree
from transformer_maskgit.optimizer import get_optimizer
from transformers import BertTokenizer, BertModel

from eval import evaluate_internal, plot_roc, accuracy, sigmoid, bootstrap, compute_cis

# from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, multilabel_confusion_matrix 

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from data_inference import CTReportDatasetinferKLab
import numpy as np
import tqdm
import pandas as pd

from einops import rearrange
import accelerate
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
import math
import torch.optim.lr_scheduler as lr_scheduler
# >>>>> Change: import the new token-refined CLIP model instead of the old CTCLIP model.
from ct_clip import CLIPTokenRefined

def exists(val):
    return val is not None

def noop(*args, **kwargs):
    pass

def cycle(dl):
    while True:
        for data in dl:
            yield data

def apply_softmax(array):
    """
    Applies softmax function to a torch array.

    Args:
        array (torch.Tensor): Input tensor array.

    Returns:
        torch.Tensor: Tensor array after applying softmax.
    """
    # >>>>> Change: apply softmax on the last dimension so it works for token-refined logits of shape (batch, 2).
    softmax = torch.nn.Softmax(dim=-1)
    softmax_array = softmax(array)
    return softmax_array


class CTClipInference(nn.Module):
    def __init__(
        self,
        # >>>>> Change: update the type annotation to the new token-refined CLIP model.
        CTClip: CLIPTokenRefined,
        *,
        # num_train_steps,
        batch_size,
        n_workers,
        data_folder: "external_valid",
        reports_file: "data_reports.xslx",
        text_column: "Text",
        results_folder = './results',
        labels = "labels.csv",
        accelerate_kwargs: dict = dict(),

        tokenizer = None, # <<<<<<<<<<<<<<<<<<<<<< MKD added (the followings also)
        pathologies = None, # list

        resample=(1.0, 1.0, 1.0),
        n_zSlices=None,
        zSlices_pad_value=0,
        clip=(-1000, 400),
        clip_percentile=None, 
        normalize=True,
        resize_shape=(64, 128, 128),
        transform=None,
        verbose=False

    ):
        super().__init__()
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], **accelerate_kwargs)
        self.CTClip = CTClip

        if tokenizer != None:
            self.tokenizer=tokenizer
        else:
            self.tokenizer=BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)

        
        self.results_folder = results_folder
        self.register_buffer('steps', torch.Tensor([0]))
        self.batch_size = batch_size
        self.pathologies = pathologies

        self.ds = CTReportDatasetinferKLab(
            img_dir=data_folder,
            report_file=reports_file,
            text_column = text_column,
            label_file=labels,
            label_cols=pathologies,
            resample=resample,
            n_zSlices=n_zSlices,
            zSlices_pad_value=zSlices_pad_value,
            clip=clip,
            clip_percentile=clip_percentile,
            normalize=normalize,
            resize_shape=resize_shape,
            transform=transform,
            verbose=verbose
        )

        # Split dataset into train and validation sets
        self.dl = DataLoader(
            self.ds,
            num_workers=n_workers,
            batch_size=self.batch_size, # <<<<<<<< MKD changed from 1 to self.batch_size
            shuffle = False, # <<<<<<<<<<<<<<<<<<<<<<< MKD changes from True to False
            # persistent_workers=False,   # MKD added
            # prefetch_factor=1,          # MKD added <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # pin_memory=False,           # MKD added
        )
        # prepare with accelerator
        self.device = self.accelerator.device
        self.CTClip.to(self.device)

        # >>>>> Change: prepare the dataloader itself, then build the iterator from the prepared dataloader.
        (
            self.dl,
            self.CTClip,
        ) = self.accelerator.prepare(
            self.dl,
            self.CTClip,
        )
        self.dl_iter=cycle(self.dl)

        self.result_folder_txt = self.results_folder
        self.results_folder = Path(results_folder)

        self.results_folder.mkdir(parents=True, exist_ok=True)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def infer(self, log_fn=noop):
        device = self.device

        steps = int(self.steps.item())

        logs = {}

        with torch.no_grad():

            models_to_evaluate = ((self.CTClip, str(steps)),)

            for model, filename in models_to_evaluate:
                model.eval()
                predictedall=[]
                realall=[]
                accession_names=[]
        
                # >>>>> Change: iterate over the number of batches, not dataset size, so batched inference stays aligned.
                for i in tqdm.tqdm(range(len(self.dl))):
                    valid_data, text, onehotlabels, acc_name = next(self.dl_iter)
                    # >>>>> Change: explicitly move the image batch to the same device as the token-refined CLIP model.
                    valid_data = valid_data.to(device)

                    plotdir = self.result_folder_txt
                    Path(plotdir).mkdir(parents=True, exist_ok=True)

                    # >>>>> Change: collect probabilities for the full batch for each pathology.
                    predictedlabels_batch=[]

                    for pathology in self.pathologies:
                        text = [f"{pathology}.", f"not {pathology}."]
                        text_tokens=self.tokenizer(
                                        text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)

                        # >>>>> Change: the new CLIP model returns (logits, image_latents, text_latents) and does not take a device arg.
                        output, _, _ = model(text_tokens, valid_data)

                        output = apply_softmax(output)

                        # >>>>> Change: for prompts [positive, negative], take column 0 as the positive probability for every image in the batch.
                        append_out=output[:, 0].detach().cpu().numpy()
                        predictedlabels_batch.append(append_out) # this is probability, not a nominal value

                    # >>>>> Change: convert from [num_pathologies][batch] to [batch][num_pathologies].
                    predictedlabels_batch = np.stack(predictedlabels_batch, axis=1)

                    # >>>>> Change: append all batch items instead of only the first sample.
                    predictedall.extend(predictedlabels_batch.tolist())
                    realall.extend(onehotlabels.detach().cpu().numpy())
                    accession_names.extend(list(acc_name))

                realall=np.array(realall)
                predictedall=np.array(predictedall)

                np.savez(f"{plotdir}labels_weights.npz", data=realall)
                np.savez(f"{plotdir}predicted_weights.npz", data=predictedall)
                with open(f"{plotdir}accessions.txt", "w") as file:
                    for item in accession_names:
                        file.write(item + "\n")


                dfs=evaluate_internal(predictedall,realall,self.pathologies, plotdir)

                writer = pd.ExcelWriter(f'{plotdir}aurocs.xlsx', engine='xlsxwriter')

                dfs.to_excel(writer, sheet_name='Sheet1', index=False)

                writer.close()

                # Convert probabilities into predictions
                thresholds = np.array([0.5]*len(self.pathologies))
                assert len(self.pathologies) == len(thresholds), f"Length mismatch. {len(self.pathologies)} pathologies, whereas {len(thresholds)} thresholds."
                predLabelall = (predictedall >= thresholds).astype(int)

                # Store gt and prediction in excel file
                df_probabilities = pd.DataFrame(predictedall, columns=self.pathologies)
                df_probabilities.insert(0, 'Name', accession_names)

                df_predLabelall = pd.DataFrame(predLabelall, columns=self.pathologies)
                df_predLabelall.insert(0, 'Name', accession_names)

                df_gts = pd.DataFrame(realall, columns=self.pathologies)
                df_gts.insert(0, 'Name', accession_names)

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
                for pathology in self.pathologies:
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

        self.steps += 1

        log_fn(logs)

        print(f'Inference complete. Stored to {plotdir}')
