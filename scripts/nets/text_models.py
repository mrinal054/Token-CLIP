from transformers import BertTokenizer, BertModel
from transformers import LongformerTokenizer, LongformerModel


def load_bert(cfg):
    pretrained = cfg.get("pretrained_model", "bert-base-uncased")
    do_lower_case = cfg.get("do_lower_case", True)
    tok_kwargs = cfg.get("tokenizer_init_kwargs", {})
    mdl_kwargs = cfg.get("model_kwargs", {})

    tokenizer = BertTokenizer.from_pretrained(pretrained, do_lower_case=do_lower_case, **tok_kwargs)
    model = BertModel.from_pretrained(pretrained, **mdl_kwargs)

    if cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    return tokenizer, model


def load_longformer(cfg):
    pretrained = cfg.get("pretrained_model", "allenai/longformer-base-4096")
    do_lower_case = cfg.get("do_lower_case", True)
    tok_kwargs = cfg.get("tokenizer_init_kwargs", {})
    mdl_kwargs = cfg.get("model_kwargs", {})

    tokenizer = LongformerTokenizer.from_pretrained(pretrained, do_lower_case=do_lower_case, **tok_kwargs)
    model = LongformerModel.from_pretrained(pretrained, **mdl_kwargs)

    if cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    return tokenizer, model


def load_biolongformer(cfg):
    pretrained = cfg.get("pretrained_model", "allenai/biolongformer-base")
    do_lower_case = cfg.get("do_lower_case", True)
    tok_kwargs = cfg.get("tokenizer_init_kwargs", {})
    mdl_kwargs = cfg.get("model_kwargs", {})

    tokenizer = LongformerTokenizer.from_pretrained(pretrained, do_lower_case=do_lower_case, **tok_kwargs)
    model = LongformerModel.from_pretrained(pretrained, **mdl_kwargs)

    if cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    return tokenizer, model


def load_bioclinicalbert(cfg):
    pretrained = cfg.get("pretrained_model", "emilyalsentzer/Bio_ClinicalBERT")
    do_lower_case = cfg.get("do_lower_case", True)
    tok_kwargs = cfg.get("tokenizer_init_kwargs", {})
    mdl_kwargs = cfg.get("model_kwargs", {})

    tokenizer = BertTokenizer.from_pretrained(pretrained, do_lower_case=do_lower_case, **tok_kwargs)
    model = BertModel.from_pretrained(pretrained, **mdl_kwargs)

    if cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    return tokenizer, model


def load_pubmedbert(cfg):
    pretrained = cfg.get("pretrained_model", "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    do_lower_case = cfg.get("do_lower_case", True)
    tok_kwargs = cfg.get("tokenizer_init_kwargs", {})
    mdl_kwargs = cfg.get("model_kwargs", {})

    tokenizer = BertTokenizer.from_pretrained(pretrained, do_lower_case=do_lower_case, **tok_kwargs)
    model = BertModel.from_pretrained(pretrained, **mdl_kwargs)

    if cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    return tokenizer, model


def load_bluebert(cfg):
    # Common BlueBERT checkpoints:
    # "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
    # "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12"
    pretrained = cfg.get(
        "pretrained_model",
        "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
    )
    do_lower_case = cfg.get("do_lower_case", True)  # BlueBERT is typically uncased
    tok_kwargs = cfg.get("tokenizer_init_kwargs", {})
    mdl_kwargs = cfg.get("model_kwargs", {})

    tokenizer = BertTokenizer.from_pretrained(pretrained, do_lower_case=do_lower_case, **tok_kwargs)
    model = BertModel.from_pretrained(pretrained, **mdl_kwargs)

    if cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    return tokenizer, model


# unified registry
REGISTRY = {
    "bert": load_bert,
    "longformer": load_longformer,
    "biolongformer": load_biolongformer,
    "bioclinicalbert": load_bioclinicalbert,
    "pubmedbert": load_pubmedbert,
    "bluebert": load_bluebert,        
}

def build_text_model(cfg):
    name = cfg.get("name", "") 
    if name not in REGISTRY:
        raise ValueError(f"Unknown text encoder '{name}'. Available: {list(REGISTRY.keys())}")
    return REGISTRY[name](cfg)
