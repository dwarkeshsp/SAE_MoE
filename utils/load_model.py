import torch
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    MixtralModel,
    MixtralConfig,
)
from torch import nn
import collections
import time
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

double_quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
)

layer = 15

config = MixtralConfig(
    num_experts_per_tok=8,
    num_hidden_layers=layer,
)

model = MixtralModel(config).from_pretrained(
    "mistralai/Mixtral-8x7B-v0.1",
    quantization_config=double_quant_config,
    attn_implementation="flash_attention_2",
)
