"""
v2: Unique report sampler added
"""

from pathlib import Path
from shutil import rmtree
from datetime import timedelta

from transformer_maskgit.optimizer import get_optimizer
from transformers import BertTokenizer, BertModel

from eval import evaluate_internal, plot_roc, accuracy, sigmoid, bootstrap, compute_cis
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F

from data import CTReportDatasetKLab
from data_inference import CTReportDatasetinferKLab
from data_sampler import UniqueReportBatchSampler

import numpy as np
import pandas as pd
import tqdm

from einops import rearrange
import accelerate
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs

import math
import torch.optim.lr_scheduler as lr_scheduler
# from ct_clip import CTCLIP
from ct_clip import CLIPTokenRefined
import os

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MKD added following packages
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import csv
import random

# helpers

# <<<<<<<<<<<<<<< MKD added tokenize_random_window function [START]
def tokenize_random_window(
        tokenizer,
        texts,
        *, # anything after it should be keyward arguments.
        encode_kwargs: dict = None,
        end_bias: float = 0.0,
    ):
    """
    Model-agnostic random-window token cropping using HF tokenizers.

    :param tokenizer: HF tokenizer.
    :param texts: str or List[str]
    :param encode_kwargs: dict from config (uses max_length, padding)
    :param end_bias: bias crop start toward later parts [0, 1)
    :return: BatchEncoding with input_ids + attention_mask (pt tensors)
    """
    if isinstance(texts, str):
        texts = [texts]

    encode_kwargs = dict(encode_kwargs or {})
    max_length = int(encode_kwargs.get("max_length", 512))
    padding = encode_kwargs.get("padding", "max_length")
    return_tensors = encode_kwargs.get("return_tensors", "pt")

    end_bias = float(end_bias)
    end_bias = max(0.0, min(end_bias, 0.999)) # range 0.0~0.999

    # Tokenize full text without truncation/padding
    enc = tokenizer(
        texts,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_attention_mask=False,
        return_tensors=None,
    )

    input_ids_batch = []
    for ids in enc["input_ids"]:
        L = len(ids)

        # How many special tokens will be added for a single sequence?
        n_special = tokenizer.num_special_tokens_to_add(pair=False)
        crop_len = max_length - n_special

        if crop_len <= 0:
            raise ValueError(f"max_length={max_length} is too small for required special tokens (n={n_special}).")

        # Random window crop on raw ids (no specials yet)
        if L > crop_len:
            max_start = L - crop_len
            min_start = int(end_bias * max_start)
            start = random.randint(min_start, max_start)
            ids = ids[start:start + crop_len]

        # Add model-specific special tokens (BERT/Roberta/BOS-EOS/etc.)
        ids = tokenizer.build_inputs_with_special_tokens(ids)
        input_ids_batch.append(ids)

    # Pad to max_length; no truncation needed now because we cropped for specials
    batch = tokenizer.pad(
        {"input_ids": input_ids_batch},
        padding=padding,
        max_length=max_length,
        return_tensors=return_tensors,
    )

    return batch
# <<<<<<<<<<<<<<< MKD added tokenize_random_window function [END]



def apply_softmax(array):
    """
    Applies softmax function to a torch array.

    Args:
        array (torch.Tensor): Input tensor array.

    Returns:
        torch.Tensor: Tensor array after applying softmax.
    """
    softmax = torch.nn.Softmax(dim=0)
    softmax_array = softmax(array)
    return softmax_array



def tensor_to_nifti(tensor, path, affine=np.eye(4)):
    """
    Save tensor as a NIfTI file.

    Args:
        tensor (torch.Tensor): The input tensor with shape (D, H, W) or (C, D, H, W).
        path (str): The path to save the NIfTI file.
        affine (np.ndarray, optional): The affine matrix for the NIfTI file. Defaults to np.eye(4).
    """

    tensor = tensor.cpu()

    if tensor.dim() == 4:
        # Assume single channel data if there are multiple channels
        if tensor.size(0) != 1:
            print("Warning: Saving only the first channel of the input tensor")
        tensor = tensor.squeeze(0)
    tensor=tensor.swapaxes(0,2)
    numpy_data = tensor.detach().numpy().astype(np.float32)
    nifti_img = nib.Nifti1Image(numpy_data, affine)
    nib.save(nifti_img, path)

def exists(val):
    return val is not None

def noop(*args, **kwargs):
    pass

def cycle(dl):
    while True:
        for data in dl:
            yield data

def yes_or_no(question):
    # answer = input(f'{question} (y/n) ')      # <<<<<<<<<<<<<<<<<<<<<<<<<<<<< MKD commented
    # return answer.lower() in ('yes', 'y')     # <<<<<<<<<<<<<<<<<<<<<<<<<<<<< MKD commented
    
    # print(f"{question} (auto-answering 'y')")   # <<<<<<<<<<<<<<<<<<<<<<<<<<<<< MKD added
    # return True  
    
    print(f"{question} (auto-answering 'n')")   # <<<<<<<<<<<<<<<<<<<<<<<<<<<<< MKD added
    return False                                  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<< MKD added
    


def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

class CosineAnnealingWarmUpRestarts(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_warmup=10000, gamma=1.0, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_max = eta_max
        self.T_warmup = T_warmup
        self.gamma = gamma
        self.T_cur = 0
        self.lr_min = 0
        self.iteration = 0

        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.iteration < self.T_warmup:
            lr = self.eta_max * self.iteration / self.T_warmup
        else:
            self.T_cur = self.iteration - self.T_warmup
            T_i = self.T_0
            while self.T_cur >= T_i:
                self.T_cur -= T_i
                T_i *= self.T_mult
                self.lr_min = self.eta_max * (self.gamma ** self.T_cur)
            lr = self.lr_min + 0.5 * (self.eta_max - self.lr_min) * \
                 (1 + math.cos(math.pi * self.T_cur / T_i))

        self.iteration += 1
        return [lr for _ in self.optimizer.param_groups]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self._update_lr()
        self._update_T()

    def _update_lr(self):
        self.optimizer.param_groups[0]['lr'] = self.get_lr()[0]

    def _update_T(self):
        if self.T_cur == self.T_0:
            self.T_cur = 0
            self.lr_min = 0
            self.iteration = 0
            self.T_0 *= self.T_mult
            self.eta_max *= self.gamma

class ClipTrainer(nn.Module):
    def __init__(
        self,
        CTClip: CLIPTokenRefined, # Previously, CTCLIP
        *,
        num_train_steps,
        batch_size,
        data_train = "train",
        data_valid = "valid",
        reports_file_train = "data_reports.xslx",
        reports_file_valid = "data_reports.xslx",
        text_column = "Text",
        use_random_window = False,          # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MKD added 
        end_bias = 0.0,                     # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MKD added 
        labels = "labels.csv",
        tokenizer = None,
        tokenizer_kwargs = None,
        lr = 1.25e-6,
        wd = 0.,
        max_grad_norm = 0.5,
        save_results_every = 1000,
        save_model_every = 1000,
        results_folder = '/shares/menze.dqbm.uzh/ihamam/ctclip/',
        ckpts_folder = None,                # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MKD added (str)
        resume_from = None,                 # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MKD added
        pathologies = None,                 # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MKD added (list)
        num_workers = 8,
        resample = (1.0, 1.0, 1.0),         # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MKD added 
        n_zSlices = None,                   # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MKD added 
        zSlices_pad_value = 0,              # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MKD added 
        clip = (-1000, 400),                # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MKD added 
        clip_percentile = None,             # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MKD added 
        normalize = True,                   # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MKD added 
        resize_shape = (64, 128, 128),      # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MKD added 
        transform = None,                   # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MKD added 
        verbose = False,                    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MKD added 
        accelerate_kwargs: dict = dict()
    ):
        super().__init__()
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, kwargs], **accelerate_kwargs)
        self.CTClip = CTClip
        if tokenizer != None:
            self.tokenizer=tokenizer
        else:
            self.tokenizer=BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)

        self.tokenizer_kwargs = tokenizer_kwargs

        self.register_buffer('steps', torch.Tensor([0]))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size

        all_parameters = CTClip.parameters() # <<<<<<<<<<<<<<<<<<<<<<<<<< MKD changed. Previously: set(CTClip.parameters())

        self.optim = get_optimizer(all_parameters, lr=lr, wd=wd)

        self.max_grad_norm = max_grad_norm
        self.lr=lr

        self.pathologies = pathologies

        self.use_random_window = use_random_window
        self.end_bias = end_bias

        # Load the pre-trained weights
        # self.ds = CTReportDataset(data_folder=data_train, csv_file=reports_file_train)
        self.ds = CTReportDatasetKLab(img_dir=data_train, 
                                        report_file=reports_file_train,
                                        text_column=text_column,
                                        resample=resample,
                                        n_zSlices=n_zSlices,
                                        zSlices_pad_value=zSlices_pad_value,
                                        clip=clip,
                                        clip_percentile=clip_percentile,
                                        normalize=normalize,
                                        resize_shape=resize_shape,
                                        transform=transform,
                                        verbose=verbose,
                                        )

        self.valid_ds = CTReportDatasetinferKLab(
            img_dir=data_valid,
            report_file=reports_file_valid,
            text_column=text_column,
            label_file=labels,
            label_cols=pathologies,
            resample=resample,
            n_zSlices=n_zSlices,
            zSlices_pad_value=zSlices_pad_value,
            clip=clip,
            clip_percentile=clip_percentile,
            normalize=normalize,
            resize_shape=resize_shape,
            transform=None,
            verbose=verbose,
        )

        # self.valid_ds = CTReportDatasetinfer(data_folder=data_valid, csv_file=reports_file_valid, labels = labels)

        # self.dl = DataLoader(
        #     self.ds,
        #     num_workers=num_workers,
        #     batch_size=self.batch_size,
        #     shuffle = True,
        # )

        self.train_batch_sampler = UniqueReportBatchSampler(
            self.ds,
            batch_size=self.batch_size,
            report_id_column="Report ID", # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< hard-coded for now, but could be added to the config file. 
            fallback_to_text=True,
            drop_last=True,
            shuffle=True,
            seed=None,
        )

        self.dl = DataLoader(
            self.ds,
            num_workers=num_workers,
            batch_sampler=self.train_batch_sampler,
            pin_memory=True,
        )

        self.valid_dl = DataLoader(
            self.valid_ds,
            num_workers=num_workers,
            batch_size=1,
            shuffle = False,
        )

        # prepare with accelerator
        self.dl_iter=cycle(self.dl)
        self.valid_dl_iter=cycle(self.valid_dl)
        self.device = self.accelerator.device
        self.CTClip.to(self.device)

        (
 			self.dl_iter,
            self.valid_dl_iter,
            self.CTClip,
            self.optim,
        ) = self.accelerator.prepare(
            self.dl_iter,
            self.valid_dl_iter,
            self.CTClip,
            self.optim,
        )

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        self.results_folder = Path(results_folder)
        self.ckpts_folder = Path(ckpts_folder)

        if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
            rmtree(str(self.results_folder))
            
        if len([*self.ckpts_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
            rmtree(str(self.ckpts_folder))

        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.ckpts_folder.mkdir(parents=True, exist_ok=True)

        if resume_from is not None: # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MKD added
            self.print(f"Resuming from checkpoint: {resume_from}")
            self.load(resume_from)

        # <<<<<<<<<<<<<<<<<<<<<<<<<<< MKD added to plot loss [START]
        self.loss_steps = []

        # store each component separately
        self.contrastive_loss_history = []
        self.div_loss_history = []
        self.total_loss_history = []   # optional but handy

        self.save_loss_every = 200

        self.loss_csv_path = str(self.results_folder / "loss_curve.csv")
        self.loss_png_path = str(self.results_folder / "loss_curve.png")

        if self.is_main and not os.path.exists(self.loss_csv_path):
            with open(self.loss_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "contrastive_loss", "div_loss", "total_loss"])
        # <<<<<<<<<<<<<<<<<<<<<<<<<<< MKD added to plot loss [END]

    def save(self, path):
        if not self.accelerator.is_local_main_process:
            return

        pkg = dict(
            model=self.accelerator.get_state_dict(self.CTClip),
            optim=self.optim.state_dict(),
            steps=int(self.steps.item()), # <<<<<<<<<<<<<<<<<<<<<<<<< MKD added
        )
        torch.save(pkg, path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(path, map_location=self.device)

        CTClip = self.accelerator.unwrap_model(self.CTClip)
        CTClip.load_state_dict(pkg['model'])

        self.optim.load_state_dict(pkg['optim'])

        if 'steps' in pkg:                     # <<<<<<<<<<<<<<<<<<<<<< MKD added
            self.steps[...] = pkg['steps']     # keeps buffer type

    def print(self, msg):
        self.accelerator.print(msg)


    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def train_step(self):
        device = self.device

        steps = int(self.steps.item())

        self.CTClip.train()

        logs = {}

        # update CTClip model
        video, text = next(self.dl_iter)

        device=self.device
        video=video.to(device)
        mask = torch.ones((video.shape[0], video.shape[2])).bool().to(device)
        # text = text.to(device)
        text = list(text)

        if self.use_random_window: # <<<<<<<<<<<<<<<<<< MKD added the if block
            text_tokens = tokenize_random_window(
                self.tokenizer, text, encode_kwargs=self.tokenizer_kwargs, end_bias=self.end_bias,).to(device)
        else:
            text_tokens=self.tokenizer(text, **self.tokenizer_kwargs).to(device) # <<<<<<<<<<<<< MKD added kwargs

        #video = video
        with self.accelerator.autocast():
            loss_dict = self.CTClip(text_tokens, video, return_loss=True) # this clip return a loss dict
            contrastive_loss = loss_dict["contrastive_loss"]
            div_loss = loss_dict["div_loss"]
            loss = contrastive_loss + div_loss

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MKD commented out
        # self.accelerator.backward(loss)
        # accum_log(logs, {'loss': loss.item()})
        # if exists(self.max_grad_norm):
        #     self.accelerator.clip_grad_norm_(self.CTClip.parameters(), self.max_grad_norm)

        # self.optim.step()
        # self.optim.zero_grad()
        # self.print(f"{steps}: loss: {logs['loss']}")

        # ----- MKD added [start] -----
        self.accelerator.backward(loss)

        accum_log(logs, {
            'loss': float(loss.item()),
            'contrastive_loss': float(contrastive_loss.item()),
            'div_loss': float(div_loss.item()),
        })

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.CTClip.parameters(), self.max_grad_norm)

        self.optim.step()
        self.optim.zero_grad()

        self.print(
            f"{steps}: "
            f"total={logs['loss']:.6f} | "
            f"contrastive={logs['contrastive_loss']:.6f} | "
            f"div={logs['div_loss']:.6f}"
        )
        
        # Log loss
        step_int = int(self.steps.item())
        loss_val = float(loss.item())

        # Store in memory
        self.loss_steps.append(step_int)
        self.total_loss_history.append(float(loss.item()))
        self.contrastive_loss_history.append(float(contrastive_loss.item()))
        self.div_loss_history.append(float(div_loss.item()))

        # Write to CSV + plot occasionally (main process only)
        if self.is_main and (step_int % self.save_loss_every == 0): # currently hard-coded to self.save_loss_every=200
            # Append CSV row
            with open(self.loss_csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                # writer.writerow([step_int, loss_val])
                writer.writerow([
                    step_int,
                    float(contrastive_loss.item()),
                    float(div_loss.item()),
                    float(loss.item())
                ])

            # Plot loss curve 
            plt.figure()
            plt.plot(self.loss_steps, self.total_loss_history, label="Total")
            plt.plot(self.loss_steps, self.contrastive_loss_history, label="Contrastive")
            plt.plot(self.loss_steps, self.div_loss_history, label="Diversity")
            plt.legend()
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Training Loss Components")
            plt.tight_layout()
            plt.savefig(self.loss_png_path, dpi=200)
            plt.close()

            print("Loss value logged.")

        # ----- MKD added [end] -----

        if self.is_main and not (steps % self.save_results_every):
            with torch.no_grad():

                models_to_evaluate = ((self.CTClip, str(steps)),)

                for model, filename in models_to_evaluate:
                    model.eval()
                    predictedall=[]
                    realall=[]

                    # Fast inference on 100 images
                    for i in range(100):
                        valid_data, text, onehotlabels, name_acc = next(self.valid_dl_iter)
                        valid_data = valid_data.to(device)

                        if "module" in model.__dict__:
                            model = model.module


                        plotdir = str(self.results_folder / f'Clip_{steps}' )
                        plotdir = plotdir + "/"

                        Path(plotdir).mkdir(parents=True, exist_ok=True)

                        predictedlabels=[]
                        for pathology in self.pathologies:
                            text = [f"There is {pathology}.", f"There is no {pathology}."]
                            text_tokens=self.tokenizer(text, **self.tokenizer_kwargs).to(device)  # <<<<<<<<<<<<< MKD added kwargs
                            
                            # output = model(text_tokens, valid_data) # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MKD commented out
                            # output = apply_softmax(output) # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MKD commented out
                            # append_out=output.detach().cpu().numpy() # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MKD commented out
                            
                            
                            # New addition by MKD
                            logits, _, _ = model(text_tokens, valid_data, return_loss=False)  # (1, 2) for eval case
                            probs = F.softmax(logits, dim=-1)                           # softmax over the 2 text options
                            
                            # probs shape will be (1,2); take prob of "There is pathology." (index 0)
                            predictedlabels.append(float(probs[0, 0].detach().cpu().item()))
                            

                            # CT-CLIP unncessarily added the following
                            # if output[0]>output[1]:
                            #     predictedlabels.append(append_out[0])
                            # else:
                            #     predictedlabels.append(append_out[0])

                            # predictedlabels.append(append_out[0]) # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MKD commented out

                        predictedall.append(predictedlabels)
                        realall.append(onehotlabels.detach().cpu().numpy()[0])
                        # Print and save classification report
                    realall=np.array(realall)
                    predictedall=np.array(predictedall)

                    dfs=evaluate_internal(predictedall,realall,self.pathologies, plotdir)
                    realall = np.rint(realall).astype(int)
                    predictedall = np.rint(predictedall).astype(int)

                    print('Test F1 Accuracy: ', f1_score(realall, predictedall,average='micro'))
                    print('Test Flat Accuracy: ', accuracy_score(realall.flatten(), predictedall.flatten()),'\n')

                    writer = pd.ExcelWriter(f'{plotdir}aurocs.xlsx', engine='xlsxwriter')

                    dfs.to_excel(writer, sheet_name='Sheet1', index=False)

                    writer.close()

        # # save model every so often # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MKD commented out
        # if self.is_main and not (steps % self.save_model_every):
        #     model_save = self.accelerator.unwrap_model(self.CTClip)
        #     state_dict = model_save.state_dict()
        #     model_path = str(self.ckpts_folder / f'Clip.{steps}.pt')
        #     self.accelerator.save(state_dict, model_path)
        #     self.print(f'{steps}: saving model to {str(self.ckpts_folder)}')

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MKD added [START]
        if self.is_main and not (steps % self.save_model_every):
            model_path = str(self.ckpts_folder / f'Clip.{steps}.pt')
            self.save(model_path)   # <<<<<< saves model + optimizer + steps
            self.print(f'{steps}: saving model and optimizer to {model_path}')
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MKD added [END]

        self.steps += 1
        return logs



    def train(self, log_fn=noop):
        device = next(self.CTClip.parameters()).device
        device=torch.device('cuda')
        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print('training complete')
