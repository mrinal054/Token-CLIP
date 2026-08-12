# Token-CLIP

Token-CLIP is a token-based contrastive learning framework for training and evaluating vision–language models on 3D medical images.

## Installation

### 1. Create a Python Environment

Create and activate a Python virtual environment before installing the required packages.

For instructions, see the [Python virtual environment setup guide](<virtual-environment-guide-link>).

### 2. Clone the Repository

Clone this repository to your desired directory:

```bash
git clone <repository-url>
```

Navigate to the project root directory:

```bash
cd Token-CLIP
```

### 3. Install the Required Dependencies

Install the packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Install CT-CLIP

Navigate to the `CT-CLIP` directory:

```bash
cd CT-CLIP
```

Install CT-CLIP in editable mode:

```bash
pip install -e .
```

Return to the project root directory and navigate to the `scripts` directory:

```bash
cd ../scripts
```

## Data Preparation

### Spreadsheet

Prepare the input spreadsheet by following the format provided in the [demo spreadsheet](<demo-spreadsheet-link>).

> **Note:** The current implementation expects all input images to be stored in the same directory.

### Configuration File

Create a configuration file based on the provided [sample configuration file](<sample-config-link>).

For example:

```text
config/template.yaml
```

Update the paths, training parameters, and other settings in the configuration file before running the experiments.

## Usage

The following commands demonstrate how to train and evaluate Token-CLIP.

### Token-Based Contrastive Learning

#### Training

```bash
nohup accelerate launch --multi_gpu run_train_tokenRefined.py \
    --config config/template.yaml \
    > log_MR_contrastive_train.log 2>&1 &
```

### Linear Probing

#### Training

```bash
CUDA_VISIBLE_DEVICES=0 nohup python lipro_train_tokenRefined_v1.py \
    --config config/template.yaml \
    > log_MR_liproTrain.log 2>&1 &
```

#### Inference

```bash
CUDA_VISIBLE_DEVICES=0 nohup python lipro_inference_tokenRefined_v1.py \
    --config config/template.yaml \
    > log_MR_liproInfer.log 2>&1 &
```

### Vocabulary Fine-Tuning

#### Training

For MSE-based vocabulary fine-tuning:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python vocabfine_TokenRefined_train.py \
    --config config/template.yaml \
    > log_MR_vocabfine.log 2>&1 &
```

#### Inference

```bash
ulimit -n 4096 && CUDA_VISIBLE_DEVICES=0 nohup accelerate launch --multi_gpu \
    run_zero_shot_tokenRefined.py \
    --config config/template.yaml \
    > log_MR_vocabfine_test.log 2>&1 &
```

## Monitoring Experiments

The commands above run in the background using `nohup`. To monitor a log file, use:

```bash
tail -f <log-file-name>
```

For example:

```bash
tail -f log_MR_contrastive_train.log
```

## Acknowledgments

This repository includes code adapted from **CT-CLIP** by Hamamci et al. and **μ²Tokenizer** by Li et al. <br>

CT-CLIP Reference: Hamamci, Ibrahim Ethem, et al. “Generalist foundation models from a multimodal dataset for 3D computed tomography.” *Nature Biomedical Engineering* (2026): 1–19. <br>

CT-CLIP Repository: https://github.com/ibrahimethemhamamci/CT-CLIP <br>

CT-CLIP is distributed under the Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International License (**CC BY-NC-SA 4.0**). The CT-CLIP code used in this repository was modified for token-based CLIP training. <br>

μ²Tokenizer Reference: Li, Siyou, et al. “μ²Tokenizer: Differentiable Multi-Scale Multi-Modal Tokenizer for Radiology Report Generation.” In *International Conference on Medical Image Computing and Computer-Assisted Intervention*, Springer Nature Switzerland, 2025. <br>

μ²Tokenizer Repository: https://github.com/Siyou-Li/u2Tokenizer <br>

The μ²Tokenizer code used in this repository was adapted for integration into the proposed training framework. <br>

Please cite the original CT-CLIP and μ²Tokenizer papers when using code or methods derived from these projects.