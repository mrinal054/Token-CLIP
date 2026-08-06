"""
Cosine similarity-based utility for CTCLIP-style and token-refined CLIP models.

CHANGED FOR TOKEN-REFINED CLIP
------------------------------
This file now supports two model APIs:
1) CTCLIP-style global latent outputs via `return_latents=True`
2) token-level outputs (for example `CLIPTokenRefined`) via `return_loss=False`

For token-level outputs, the utility mean-pools image tokens and masked-mean-pools
text tokens to get one embedding per sample before cosine-similarity analysis.

What this file does
-------------------
This utility extracts the *final normalized CLIP latents* for images and texts,
then builds the similarity distributions:

1. matched pairs                 -> image_i vs text_i
2. random unmatched pairs        -> image_i vs random text_j, j != i
3. same-task / same-label pairs  -> image_i vs text_j where label_j == label_i
                                   for a chosen pathology/task column

Why keep this as a standalone utility?
--------------------------------------
- `data.py` already knows how to load and preprocess the MRI volumes.
- `ct_clip.py` already exposes normalized latents via `return_latents=True`.
- `CTCLIPTrainer.py` already owns the tokenizer and tokenizer kwargs.

So the least disruptive implementation is: leave our training code untouched,
and add one analysis file that can be called *after* a checkpoint is loaded.

Typical usage
-------------
from similarity_similarity import generate_similarity_from_trainer

artifacts = generate_similarity_from_trainer(
    trainer=trainer,
    report_file="/path/to/test.xlsx",
    img_dir="/path/to/nifti_dir",
    text_column="Radiology Report",
    label_columns=["Kidney cyst", "Liver cyst", "CKD"],
    output_dir="/path/to/results/similarity_type1",
    model_label="Type-1",
    n_random_unmatched=25,
    n_same_label_unmatched=25,
    exclude_same_id_cols=["MRN", "Accession"],
)

print(artifacts)

Notes
-----
- CHANGED FOR TOKEN-REFINED CLIP: supports both CTCLIP-style global latents
  and token-level outputs. Token-level outputs are pooled before cosine analysis.
- To compare Type-1 / Type-2 / Type-3 text models, run this once per model.
- The function also saves the extracted latents so we can reuse them later for
  UMAP / retrieval / neighborhood analyses without re-running model inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
import inspect
import math
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import CTReportDatasetKLab


# -----------------------------------------------------------------------------
# Small helper copied from the training file, so cosine similarity uses the same token
# cropping behavior when we explicitly request random-window tokenization.
# -----------------------------------------------------------------------------

def tokenize_random_window(
    tokenizer,
    texts,
    *,
    encode_kwargs: Optional[dict] = None,
    end_bias: float = 0.0,
):
    """
    Model-agnostic random-window token cropping using HF tokenizers.

    This is intentionally the same logic as in CTCLIPTrainer.py, copied here to
    keep this utility self-contained.
    """
    if isinstance(texts, str):
        texts = [texts]

    encode_kwargs = dict(encode_kwargs or {})
    max_length = int(encode_kwargs.get("max_length", 512))
    padding = encode_kwargs.get("padding", "max_length")
    return_tensors = encode_kwargs.get("return_tensors", "pt")

    end_bias = float(end_bias)
    end_bias = max(0.0, min(end_bias, 0.999))

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
        n_special = tokenizer.num_special_tokens_to_add(pair=False)
        crop_len = max_length - n_special

        if crop_len <= 0:
            raise ValueError(
                f"max_length={max_length} is too small for required special tokens (n={n_special})."
            )

        if L > crop_len:
            max_start = L - crop_len
            min_start = int(end_bias * max_start)
            start = random.randint(min_start, max_start)
            ids = ids[start : start + crop_len]

        ids = tokenizer.build_inputs_with_special_tokens(ids)
        input_ids_batch.append(ids)

    batch = tokenizer.pad(
        {"input_ids": input_ids_batch},
        padding=padding,
        max_length=max_length,
        return_tensors=return_tensors,
    )
    return batch


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def _coerce_to_list(x: Optional[Union[str, Sequence[str]]]) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


def _clean_id_value(value):
    """Convert pandas / numpy scalar values into something easy to store in CSV."""
    if pd.isna(value):
        return ""
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        # If a label was read as 0.0 / 1.0, store as int when possible.
        if float(value).is_integer():
            return int(value)
        return float(value)
    return str(value)



def _default_tokenizer_kwargs(tokenizer_kwargs: Optional[Mapping]) -> Dict:
    """
    Match the usual CTCLIP / HF inference settings if the caller did not supply
    tokenizer kwargs explicitly.
    """
    default = {
        "return_tensors": "pt",
        "padding": "max_length",
        "truncation": True,
        "max_length": 512,
    }
    if tokenizer_kwargs is not None:
        default.update(dict(tokenizer_kwargs))
    return default



def _tokenize_texts(
    *,
    tokenizer,
    texts: Sequence[str],
    tokenizer_kwargs: Mapping,
    device: torch.device,
    use_random_window: bool,
    end_bias: float,
):
    """Tokenize a batch of texts and move the resulting BatchEncoding to device."""
    if use_random_window:
        batch = tokenize_random_window(
            tokenizer,
            list(texts),
            encode_kwargs=dict(tokenizer_kwargs),
            end_bias=end_bias,
        )
    else:
        batch = tokenizer(list(texts), **dict(tokenizer_kwargs))
    return batch.to(device)



def _unwrap_model_from_trainer(trainer):
    """
    CHANGED FOR TOKEN-REFINED CLIP
    Support either of these trainer attributes:
    - trainer.CTClip  (original CTCLIP trainer)
    - trainer.clip    (common alternative naming)
    - trainer.model   (generic trainer wrapper)
    """
    model = None
    for attr_name in ("CTClip", "clip", "model"):
        if hasattr(trainer, attr_name):
            model = getattr(trainer, attr_name)
            break

    if model is None:
        raise AttributeError(
            "Could not find the CLIP model on trainer. Expected one of: "
            "trainer.CTClip, trainer.clip, or trainer.model."
        )

    if hasattr(trainer, "accelerator") and hasattr(trainer.accelerator, "unwrap_model"):
        return trainer.accelerator.unwrap_model(model)
    return model



def _get_device_from_trainer(trainer) -> torch.device:
    if hasattr(trainer, "device"):
        return torch.device(trainer.device)
    model = _unwrap_model_from_trainer(trainer)
    return next(model.parameters()).device


def _model_supports_kwarg(model, kwarg_name: str) -> bool:
    """
    CHANGED FOR TOKEN-REFINED CLIP
    Small introspection helper so we can branch on model forward API with
    minimal changes to the rest of the file.
    """
    try:
        sig = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return False
    return kwarg_name in sig.parameters



def _masked_mean_pool(token_latents: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    CHANGED FOR TOKEN-REFINED CLIP
    Convert token-level text latents [B, L, D] into one vector per sample [B, D].

    - If attention_mask is available, ignore padded tokens.
    - If the input is already [B, D], return it unchanged.
    """
    if token_latents.ndim == 2:
        return token_latents
    if token_latents.ndim != 3:
        raise ValueError(f"Expected text latents to have shape [B, D] or [B, L, D], got {tuple(token_latents.shape)}.")

    if attention_mask is None:
        return token_latents.mean(dim=1)

    if attention_mask.ndim != 2:
        raise ValueError(f"Expected attention_mask to have shape [B, L], got {tuple(attention_mask.shape)}.")

    mask = attention_mask.to(device=token_latents.device, dtype=token_latents.dtype).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (token_latents * mask).sum(dim=1) / denom



def _mean_pool_image_tokens(image_latents: torch.Tensor) -> torch.Tensor:
    """
    CHANGED FOR TOKEN-REFINED CLIP
    Convert token-level image latents [B, K, D] into one vector per sample [B, D].

    - If the input is already [B, D], return it unchanged.
    """
    if image_latents.ndim == 2:
        return image_latents
    if image_latents.ndim != 3:
        raise ValueError(f"Expected image latents to have shape [B, D] or [B, K, D], got {tuple(image_latents.shape)}.")
    return image_latents.mean(dim=1)



def _extract_similarity_latents(model, text_tokens, images, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CHANGED FOR TOKEN-REFINED CLIP
    Return one image embedding and one text embedding per sample for similarity analysis.

    Supported model APIs
    --------------------
    1) CTCLIP-style:
       model(text_tokens, images, device=device, return_latents=True)
       -> (text_latents [B, D], image_latents [B, D], ...)

    2) Token-refined CLIP:
       model(text_tokens, images, return_loss=False)
       -> (logits, image_token_latents [B, K, D], text_token_latents [B, L, D])

    For token-level outputs, we pool to one vector per sample so the rest of the
    cosine-similarity pipeline can stay unchanged.
    """
    if _model_supports_kwarg(model, "return_latents"):
        out = model(
            text_tokens,
            images,
            device=device,
            return_latents=True,
        )
        if not isinstance(out, (tuple, list)) or len(out) < 2:
            raise RuntimeError(
                "CTCLIP-style model with return_latents=True did not return the expected tuple."
            )
        text_latents = out[0]
        image_latents = out[1]
        return image_latents, text_latents

    if _model_supports_kwarg(model, "return_loss"):
        out = model(
            text_tokens,
            images,
            return_loss=False,
        )
        if not isinstance(out, (tuple, list)) or len(out) < 3:
            raise RuntimeError(
                "Token-refined model did not return the expected (logits, image_latents, text_latents) tuple."
            )

        _, image_token_latents, text_token_latents = out[:3]
        image_latents = _mean_pool_image_tokens(image_token_latents)
        text_latents = _masked_mean_pool(
            text_token_latents,
            attention_mask=getattr(text_tokens, "attention_mask", None),
        )
        return image_latents, text_latents

    raise RuntimeError(
        "Unsupported model API. Expected either a CTCLIP-style forward with return_latents, "
        "or a token-refined forward with return_loss."
    )



def _infer_dataset_kwargs_from_trainer(trainer) -> Dict:
    """
    Pull preprocessing knobs from trainer.ds if available so it
    uses the same MRI preprocessing as training.
    """
    ds = getattr(trainer, "ds", None)
    if ds is None:
        return {}

    dataset_kwargs = {}
    for attr in [
        "resample",
        "n_zSlices",
        "zSlices_pad_value",
        "clip",
        "clip_percentile",
        "normalize",
        "resize_shape",
        "transform",
        "verbose",
    ]:
        if hasattr(ds, attr):
            dataset_kwargs[attr] = getattr(ds, attr)
    return dataset_kwargs



def _normalise_label_columns(df: pd.DataFrame, label_columns: Sequence[str]) -> pd.DataFrame:
    """
    Force label columns to numeric if possible.

    That makes same-label sampling robust even if Excel loaded them as strings.
    """
    out = df.copy()
    for col in label_columns:
        if col not in out.columns:
            raise KeyError(f"Label column '{col}' was not found in the Excel file.")
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _select_row_indices(
    df: pd.DataFrame,
    *,
    max_samples: Optional[int] = None,
    stratify_by_label: Optional[str] = None,
    seed: int = 42,
) -> List[int]:
    """
    CHANGED: Select dataset rows, optionally with simple label-stratified sampling.

    - If max_samples is None, use all rows.
    - If stratify_by_label is provided, sample approximately evenly from each
      non-null label value in that column.
    """
    n_total = len(df)
    if max_samples is None or int(max_samples) >= n_total:
        return list(range(n_total))

    max_samples = int(max_samples)
    if max_samples <= 0:
        raise ValueError("max_samples must be a positive integer.")

    rng = np.random.default_rng(seed)

    if stratify_by_label is None:
        return sorted(rng.choice(n_total, size=max_samples, replace=False).tolist())

    if stratify_by_label not in df.columns:
        raise KeyError(f"stratify_by_label column '{stratify_by_label}' was not found in the Excel file.")

    label_series = pd.to_numeric(df[stratify_by_label], errors="coerce")
    valid_mask = label_series.notna()

    # Fall back to plain random sampling if the label column has no usable values.
    if valid_mask.sum() == 0:
        return sorted(rng.choice(n_total, size=max_samples, replace=False).tolist())

    valid_df = df.loc[valid_mask].copy()
    valid_df["_stratify_label"] = label_series.loc[valid_mask].to_numpy()
    groups = [g.index.to_numpy() for _, g in valid_df.groupby("_stratify_label", sort=True)]

    n_groups = len(groups)
    base = max_samples // n_groups
    remainder = max_samples % n_groups

    selected = []
    leftovers = []

    for i, group_idx in enumerate(groups):
        n_take = base + (1 if i < remainder else 0)
        n_take = min(n_take, len(group_idx))
        if n_take > 0:
            chosen = rng.choice(group_idx, size=n_take, replace=False).tolist()
            selected.extend(chosen)
            chosen_set = set(chosen)
            leftovers.extend([idx for idx in group_idx.tolist() if idx not in chosen_set])
        else:
            leftovers.extend(group_idx.tolist())

    # If some groups were too small, top up from the remaining valid rows first.
    if len(selected) < max_samples and leftovers:
        need = min(max_samples - len(selected), len(leftovers))
        selected.extend(rng.choice(np.array(leftovers), size=need, replace=False).tolist())

    # Final fallback: top up from any remaining dataset rows.
    if len(selected) < max_samples:
        remaining_pool = sorted(set(range(n_total)) - set(selected))
        if remaining_pool:
            need = min(max_samples - len(selected), len(remaining_pool))
            selected.extend(rng.choice(np.array(remaining_pool), size=need, replace=False).tolist())

    return sorted(selected[:max_samples])



def _sample_candidate_indices(
    candidate_indices: np.ndarray,
    *,
    n_to_sample: int,
    rng: np.random.Generator,
    allow_replacement_if_needed: bool = True,
) -> np.ndarray:
    """
    Sample candidate negative indices.

    We sample without replacement when possible.
    If a label bucket is tiny, we optionally fall back to sampling with
    replacement so every anchor still contributes some same-label negatives.
    """
    if n_to_sample <= 0 or candidate_indices.size == 0:
        return np.array([], dtype=np.int64)

    if candidate_indices.size >= n_to_sample:
        return rng.choice(candidate_indices, size=n_to_sample, replace=False)

    if allow_replacement_if_needed:
        return rng.choice(candidate_indices, size=n_to_sample, replace=True)

    return rng.choice(candidate_indices, size=candidate_indices.size, replace=False)



def _build_exclusion_mask(
    metadata_df: pd.DataFrame,
    *,
    anchor_index: int,
    exclude_same_id_cols: Sequence[str],
) -> np.ndarray:
    """
    Build a boolean mask of candidate rows that are allowed for negative pairing.

    Rules:
    - the true paired text (same row) is always excluded
    - optional columns such as MRN / Accession can also be excluded so a
      "negative" does not accidentally come from the same patient or same exam
    """
    n = len(metadata_df)
    mask = np.ones(n, dtype=bool)
    mask[anchor_index] = False

    for col in exclude_same_id_cols:
        if col not in metadata_df.columns:
            continue

        anchor_value = metadata_df.iloc[anchor_index][col]
        if pd.isna(anchor_value):
            continue

        same_value_mask = metadata_df[col].eq(anchor_value).to_numpy()
        mask = mask & (~same_value_mask)

    return mask


# -----------------------------------------------------------------------------
# Latent extraction
# -----------------------------------------------------------------------------

@torch.no_grad()
def extract_clip_latents_from_trainer(
    trainer,
    *,
    report_file: Union[str, Path],
    img_dir: Union[str, Path],
    text_column: str,
    batch_size: Optional[int] = None,
    num_workers: int = 4,
    tokenizer_kwargs: Optional[Mapping] = None,
    use_random_window: Optional[bool] = None,
    end_bias: Optional[float] = None,
    max_samples: Optional[int] = None,
    stratify_by_label: Optional[str] = None,
    seed: int = 42,
    **dataset_overrides,
) -> Tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
    """
    Run one forward pass over a dataset and return:
    - image_latents  : [N, D]
    - text_latents   : [N, D]
    - metadata_df    : dataframe aligned with those latent rows

    Important:
    - CTCLIP-style models already return shared-space sample embeddings.
    - Token-refined models return token-level latents, which we pool to one
      sample embedding per modality before cosine similarity.
    - We re-normalize both modalities below so downstream dot-products are
      always valid cosine similarities.
    """
    model = _unwrap_model_from_trainer(trainer)
    device = _get_device_from_trainer(trainer)
    tokenizer = trainer.tokenizer

    if batch_size is None:
        # Smaller evaluation batch sizes are safer for 3D MRI volumes.
        batch_size = min(4, int(getattr(trainer, "batch_size", 4)))

    tokenizer_kwargs = _default_tokenizer_kwargs(
        tokenizer_kwargs if tokenizer_kwargs is not None else getattr(trainer, "tokenizer_kwargs", None)
    )

    if use_random_window is None:
        use_random_window = bool(getattr(trainer, "use_random_window", False))
    if end_bias is None:
        end_bias = float(getattr(trainer, "end_bias", 0.0))

    dataset_kwargs = _infer_dataset_kwargs_from_trainer(trainer)
    dataset_kwargs.update(dataset_overrides)

    dataset = CTReportDatasetKLab(
        report_file=str(report_file),
        text_column=text_column,
        img_dir=str(img_dir),
        **dataset_kwargs,
    )

    # CHANGED: optional random / stratified row subsetting for faster, more balanced figures.
    selected_indices = _select_row_indices(
        dataset.df,
        max_samples=max_samples,
        stratify_by_label=stratify_by_label,
        seed=int(seed),
    )

    if len(selected_indices) < len(dataset):
        subset = Subset(dataset, selected_indices)
        metadata_df = dataset.df.iloc[selected_indices].reset_index(drop=True).copy()
        active_dataset = subset
    else:
        metadata_df = dataset.df.reset_index(drop=True).copy()
        active_dataset = dataset

    loader = DataLoader(
        active_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model_was_training = model.training
    model.eval()

    image_latents_all = []
    text_latents_all = []

    seen_rows = 0
    for images, texts in loader:
        texts = list(texts)
        images = images.to(device, non_blocking=True)

        text_tokens = _tokenize_texts(
            tokenizer=tokenizer,
            texts=texts,
            tokenizer_kwargs=tokenizer_kwargs,
            device=device,
            use_random_window=use_random_window,
            end_bias=float(end_bias),
        )

        # CHANGED FOR TOKEN-REFINED CLIP
        # Support either:
        # - CTCLIP-style global latents
        # - token-level latents that must be pooled to one vector per sample
        image_latents, text_latents = _extract_similarity_latents(
            model,
            text_tokens,
            images,
            device=device,
        )

        image_latents_all.append(image_latents.detach().cpu())
        text_latents_all.append(text_latents.detach().cpu())
        seen_rows += images.shape[0]

    if model_was_training:
        model.train()

    image_latents = torch.cat(image_latents_all, dim=0)
    text_latents = torch.cat(text_latents_all, dim=0)

    if image_latents.shape[0] != len(metadata_df) or text_latents.shape[0] != len(metadata_df):
        raise RuntimeError(
            "Latent count and metadata row count do not match. "
            f"image_latents={image_latents.shape[0]}, text_latents={text_latents.shape[0]}, metadata_rows={len(metadata_df)}"
        )

    # Re-normalize defensively so downstream cosine similarity is always safe,
    # even if the model code changes later.
    image_latents = F.normalize(image_latents.float(), dim=-1)
    text_latents = F.normalize(text_latents.float(), dim=-1)

    metadata_df = metadata_df.reset_index(drop=True)
    metadata_df.insert(0, "row_index", np.arange(len(metadata_df)))
    return image_latents, text_latents, metadata_df


# -----------------------------------------------------------------------------
# Similarity table construction
# -----------------------------------------------------------------------------

def build_similarity_table(
    *,
    image_latents: torch.Tensor,
    text_latents: torch.Tensor,
    metadata_df: pd.DataFrame,
    label_columns: Sequence[str],
    n_random_unmatched: int = 25,
    n_same_label_unmatched: int = 25,
    exclude_same_id_cols: Optional[Sequence[str]] = None,
    id_columns_to_export: Optional[Sequence[str]] = None,
    model_label: str = "",
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the long-format similarity table used for similarity.

    Returns
    -------
    sim_df : pd.DataFrame
        One row per evaluated pair.
    summary_df : pd.DataFrame
        Summary stats (count / mean / median / std / quartiles) for easy review.
    """
    if image_latents.shape != text_latents.shape:
        raise ValueError(
            f"image_latents and text_latents must have the same shape. "
            f"Got {tuple(image_latents.shape)} vs {tuple(text_latents.shape)}."
        )

    label_columns = _coerce_to_list(label_columns)
    exclude_same_id_cols = _coerce_to_list(exclude_same_id_cols)
    id_columns_to_export = _coerce_to_list(id_columns_to_export)

    metadata_df = _normalise_label_columns(metadata_df, label_columns)

    n = len(metadata_df)
    rng = np.random.default_rng(seed)

    # Since the latents are normalized, dot-product == cosine similarity.
    matched_scores = torch.sum(image_latents * text_latents, dim=1).cpu().numpy()

    rows = []
    skipped_same_label = {task: 0 for task in label_columns}

    for task in label_columns:
        task_values = metadata_df[task].to_numpy()

        for anchor_idx in range(n):
            anchor_row = metadata_df.iloc[anchor_idx]
            anchor_label = anchor_row[task]

            base_row = {
                "model_label": model_label,
                "task": task,
                "anchor_row_index": int(anchor_idx),
                "anchor_label": np.nan if pd.isna(anchor_label) else int(anchor_label),
            }

            # Preserve some human-readable identifiers so later we can inspect
            # interesting / hard cases without reconstructing the row mapping.
            for col in id_columns_to_export:
                if col in metadata_df.columns:
                    base_row[f"anchor_{col}"] = _clean_id_value(anchor_row[col])

            # --------------------------------------------------------------
            # 1) Matched pair: image_i vs text_i
            # --------------------------------------------------------------
            matched_row = dict(base_row)
            matched_row.update(
                {
                    "pair_type": "matched",
                    "candidate_row_index": int(anchor_idx),
                    "candidate_label": np.nan if pd.isna(anchor_label) else int(anchor_label),
                    "similarity": float(matched_scores[anchor_idx]),
                }
            )
            for col in id_columns_to_export:
                if col in metadata_df.columns:
                    matched_row[f"candidate_{col}"] = _clean_id_value(anchor_row[col])
            rows.append(matched_row)

            # Build the generic exclusion mask once; this is reused for random
            # negatives and same-label negatives.
            allowed_mask = _build_exclusion_mask(
                metadata_df,
                anchor_index=anchor_idx,
                exclude_same_id_cols=exclude_same_id_cols,
            )

            # --------------------------------------------------------------
            # 2) Random unmatched: image_i vs random text_j, j != i
            # --------------------------------------------------------------
            random_candidates = np.where(allowed_mask)[0]
            sampled_random = _sample_candidate_indices(
                random_candidates,
                n_to_sample=int(n_random_unmatched),
                rng=rng,
                allow_replacement_if_needed=True,
            )

            if sampled_random.size > 0:
                sims = torch.matmul(
                    image_latents[anchor_idx : anchor_idx + 1],
                    text_latents[sampled_random].T,
                ).squeeze(0).cpu().numpy()

                for candidate_idx, sim_val in zip(sampled_random.tolist(), sims.tolist()):
                    candidate_row = metadata_df.iloc[candidate_idx]
                    out_row = dict(base_row)
                    out_row.update(
                        {
                            "pair_type": "random_unmatched",
                            "candidate_row_index": int(candidate_idx),
                            "candidate_label": np.nan
                            if pd.isna(candidate_row[task])
                            else int(candidate_row[task]),
                            "similarity": float(sim_val),
                        }
                    )
                    for col in id_columns_to_export:
                        if col in metadata_df.columns:
                            out_row[f"candidate_{col}"] = _clean_id_value(candidate_row[col])
                    rows.append(out_row)

            # --------------------------------------------------------------
            # 3) Same-task / same-label unmatched:
            #    image_i vs text_j where label_j == label_i and j != i
            # --------------------------------------------------------------
            if pd.isna(anchor_label):
                skipped_same_label[task] += 1
                continue

            same_label_mask = metadata_df[task].eq(anchor_label).to_numpy()
            same_label_candidates = np.where(allowed_mask & same_label_mask)[0]
            sampled_same_label = _sample_candidate_indices(
                same_label_candidates,
                n_to_sample=int(n_same_label_unmatched),
                rng=rng,
                allow_replacement_if_needed=True,
            )

            if sampled_same_label.size == 0:
                skipped_same_label[task] += 1
                continue

            sims = torch.matmul(
                image_latents[anchor_idx : anchor_idx + 1],
                text_latents[sampled_same_label].T,
            ).squeeze(0).cpu().numpy()

            for candidate_idx, sim_val in zip(sampled_same_label.tolist(), sims.tolist()):
                candidate_row = metadata_df.iloc[candidate_idx]
                out_row = dict(base_row)
                out_row.update(
                    {
                        "pair_type": "same_label_unmatched",
                        "candidate_row_index": int(candidate_idx),
                        "candidate_label": np.nan
                        if pd.isna(candidate_row[task])
                        else int(candidate_row[task]),
                        "similarity": float(sim_val),
                    }
                )
                for col in id_columns_to_export:
                    if col in metadata_df.columns:
                        out_row[f"candidate_{col}"] = _clean_id_value(candidate_row[col])
                rows.append(out_row)

    sim_df = pd.DataFrame(rows)

    if sim_df.empty:
        raise RuntimeError("No similarity rows were generated. Please check the label columns and dataset.")

    summary_df = (
        sim_df.groupby(["model_label", "task", "pair_type"]) ["similarity"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )

    # Add quartiles because they are often easier to interpret than std.
    q_df = (
        sim_df.groupby(["model_label", "task", "pair_type"]) ["similarity"]
        .quantile([0.25, 0.75])
        .unstack(level=-1)
        .reset_index()
    )
    q_df.columns = [
        "model_label",
        "task",
        "pair_type",
        "q25",
        "q75",
    ]
    summary_df = summary_df.merge(q_df, on=["model_label", "task", "pair_type"], how="left")

    skipped_rows = pd.DataFrame(
        {
            "model_label": [model_label] * len(skipped_same_label),
            "task": list(skipped_same_label.keys()),
            "pair_type": ["same_label_unmatched"] * len(skipped_same_label),
            "count_skipped_anchors": list(skipped_same_label.values()),
        }
    )
    summary_df = summary_df.merge(
        skipped_rows,
        on=["model_label", "task", "pair_type"],
        how="left",
    )

    return sim_df, summary_df


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_similarity(
    sim_df: pd.DataFrame,
    *,
    output_png: Union[str, Path],
    title: Optional[str] = None,
    label_columns: Optional[Sequence[str]] = None,
    pair_type_order: Sequence[str] = ("matched", "random_unmatched", "same_label_unmatched"),
):
    """
    Create similarity as violin + box plots.

    One subplot per task / pathology label.
    """
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    if label_columns is None:
        label_columns = list(pd.unique(sim_df["task"]))
    else:
        label_columns = list(label_columns)

    pretty_names = {
        "matched": "Matched",
        "random_unmatched": "Random unmatched",
        "same_label_unmatched": "Same-task / same-label unmatched",
    }

    n_tasks = len(label_columns)
    if n_tasks == 0:
        raise ValueError("No label/task columns were provided for plotting.")

    fig_width = max(6, 5.0 * n_tasks)
    fig, axes = plt.subplots(1, n_tasks, figsize=(fig_width, 5.5), sharey=True)
    if n_tasks == 1:
        axes = [axes]

    for ax, task in zip(axes, label_columns):
        task_df = sim_df[sim_df["task"] == task].copy()

        series_list = []
        labels = []
        for pair_type in pair_type_order:
            vals = task_df.loc[task_df["pair_type"] == pair_type, "similarity"].dropna().to_numpy()
            if vals.size == 0:
                continue
            series_list.append(vals)
            labels.append(pretty_names.get(pair_type, pair_type))

        if not series_list:
            ax.set_title(f"{task}\n(no valid pairs)")
            ax.axis("off")
            continue

        positions = np.arange(1, len(series_list) + 1)

        # Violin plot shows the distribution shape.
        vp = ax.violinplot(series_list, positions=positions, widths=0.85, showmedians=True)
        # Keep default matplotlib styling to avoid over-encoding the figure.
        for body in vp["bodies"]:
            body.set_alpha(0.35)

        # Overlay a compact boxplot so median / IQR are easy to read.
        ax.boxplot(
            series_list,
            positions=positions,
            widths=0.18,
            showfliers=False,
            manage_ticks=False,
        )

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_title(task)
        ax.set_ylabel("Cosine similarity")
        ax.grid(True, axis="y", alpha=0.25)

        # Annotate medians and counts directly on the plot.
        y_top = max(float(np.max(v)) for v in series_list)
        y_range = max(float(np.max(v) - np.min(v)) for v in series_list)
        if y_range == 0:
            y_range = 0.05

        for pos, vals in zip(positions, series_list):
            median_val = float(np.median(vals))
            n_vals = int(len(vals))
            ax.text(
                pos,
                y_top + 0.05 * y_range,
                f"median={median_val:.3f}\nn={n_vals}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.set_ylim(top=y_top + 0.18 * y_range)

    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()

    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# High-level wrapper
# -----------------------------------------------------------------------------

def generate_similarity_from_trainer(
    trainer,
    *,
    report_file: Union[str, Path],
    img_dir: Union[str, Path],
    text_column: str,
    label_columns: Union[str, Sequence[str]],
    output_dir: Union[str, Path],
    model_label: str = "",
    output_prefix: str = "similarity",
    batch_size: Optional[int] = None,
    num_workers: int = 4,
    tokenizer_kwargs: Optional[Mapping] = None,
    use_random_window: Optional[bool] = None,
    end_bias: Optional[float] = None,
    n_random_unmatched: int = 25,
    n_same_label_unmatched: int = 25,
    exclude_same_id_cols: Optional[Sequence[str]] = ("MRN", "Accession"),
    id_columns_to_export: Optional[Sequence[str]] = ("MRN", "Accession", "Filename"),
    save_latents: bool = True,
    max_samples: Optional[int] = None,
    stratify_by_label: Optional[str] = None,
    seed: int = 42,
    figure_title: Optional[str] = None,
    **dataset_overrides,
) -> Dict[str, str]:
    """
    End-to-end similarity generation.

    This is the function we will normally call.

    Outputs
    -------
    Returns a dictionary containing:
    - latents_pt   : torch file with image/text latents + metadata
    - raw_csv      : long-form pairwise similarity table
    - summary_csv  : per-task summary stats
    - figure_png   : violin/box plot

    CHANGED:
    - max_samples limits the number of images / reports used
    - stratify_by_label enables simple balanced sampling (for example, 0/1)
    """
    label_columns = _coerce_to_list(label_columns)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_latents, text_latents, metadata_df = extract_clip_latents_from_trainer(
        trainer,
        report_file=report_file,
        img_dir=img_dir,
        text_column=text_column,
        batch_size=batch_size,
        num_workers=num_workers,
        tokenizer_kwargs=tokenizer_kwargs,
        use_random_window=use_random_window,
        end_bias=end_bias,
        max_samples=max_samples,
        stratify_by_label=stratify_by_label,
        seed=int(seed),
        **dataset_overrides,
    )

    sim_df, summary_df = build_similarity_table(
        image_latents=image_latents,
        text_latents=text_latents,
        metadata_df=metadata_df,
        label_columns=label_columns,
        n_random_unmatched=int(n_random_unmatched),
        n_same_label_unmatched=int(n_same_label_unmatched),
        exclude_same_id_cols=exclude_same_id_cols,
        id_columns_to_export=id_columns_to_export,
        model_label=model_label,
        seed=int(seed),
    )

    raw_csv_path = output_dir / f"{output_prefix}_raw_similarity.csv"
    summary_csv_path = output_dir / f"{output_prefix}_summary.csv"
    figure_png_path = output_dir / f"{output_prefix}.png"
    latents_pt_path = output_dir / f"{output_prefix}_latents.pt"

    sim_df.to_csv(raw_csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)

    if figure_title is None:
        figure_title = model_label.strip() or "Matched vs unmatched cosine similarity"

    plot_similarity(
        sim_df,
        output_png=figure_png_path,
        title=figure_title,
        label_columns=label_columns,
    )

    if save_latents:
        torch.save(
            {
                "image_latents": image_latents,
                "text_latents": text_latents,
                "metadata_df": metadata_df.to_dict(orient="list"),
                "label_columns": list(label_columns),
                "model_label": model_label,
            },
            latents_pt_path,
        )

    return {
        "raw_csv": str(raw_csv_path),
        "summary_csv": str(summary_csv_path),
        "figure_png": str(figure_png_path),
        "latents_pt": str(latents_pt_path) if save_latents else "",
    }


# -----------------------------------------------------------------------------
# Optional helper for later multi-model aggregation.
# This is intentionally simple: it combines multiple raw CSV files into one CSV.
# We can then make a larger manuscript figure externally.
# -----------------------------------------------------------------------------

def combine_similarity_csvs(
    csv_paths: Sequence[Union[str, Path]],
    *,
    output_csv: Union[str, Path],
) -> str:
    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)
        frames.append(df)
    merged = pd.concat(frames, axis=0, ignore_index=True)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    return str(output_csv)
