# %%
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


start = time.time()
print("started")


def load_generated_dataset(
    filename, batch_size=32, dataset_relative_path="../dataset/"
):
    dataset = pd.read_csv(f"{dataset_relative_path}{filename}")
    # print(dataset.head(10))
    dataset_list = dataset["text"].tolist()
    dataloader = torch.utils.data.DataLoader(dataset_list, batch_size=batch_size)
    return dataloader


print(f"function defined: {time.time() - start}")

# load_generated_dataset("wikitext_44836.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"device set, to load tok: {time.time() - start}")

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
print(tokenizer.eos_token)

print(f"tokenizer loaded: {time.time() - start}")

double_quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
)

print(f"quantize config done: {time.time() - start}")

# model = AutoModelForCausalLM.from_pretrained(
#     "mistralai/Mixtral-8x7B-v0.1",
#     quantization_config=double_quant_config,
#     attn_implementation="flash_attention_2",
# )

layer = 15


config = MixtralConfig(
    num_experts_per_tok=8,
    num_hidden_layers=layer,
)

print(f"config done: {time.time() - start}")

# model = MixtralModel(config).from_pretrained("quantized")

model = MixtralModel(config).from_pretrained(
    "mistralai/Mixtral-8x7B-v0.1",
    quantization_config=double_quant_config,
    attn_implementation="flash_attention_2",
)

print(f"end: {time.time() - start}")

# %%

output_dir = "quantized/"
model.save_pretrained(output_dir)


# %%


activation = collections.defaultdict(list)


def getActivation(name):
    def hook(model, input, output):
        assert type(output) is not tuple
        activation[name].append(output.detach())

    return hook


experts = model.layers[15].block_sparse_moe.experts

for expert_idx in range(8):
    experts[expert_idx].w3.register_forward_hook(
        getActivation(f"layer_{layer}_expert_{expert_idx}_w_3")
    )
    experts[expert_idx].act_fn.register_forward_hook(
        getActivation(f"layer_{layer}_expert_{expert_idx}_act_w1")
    )
    # experts[expert_idx].w2.register_forward_hook(
    #     getActivation(f"layer_15_expert_{expert_idx}_w_2")
    # )

dataset = load_generated_dataset("wikitext_44836.csv", batch_size=1)

i = 0
for batch in tqdm(dataset):
    if i > 10:
        break
    i += 1

    tokenizer.pad_token = tokenizer.eos_token
    batch_tokens = tokenizer(
        batch,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).to(device)

    print(batch_tokens["input_ids"].shape)

    # generated_ids = model.generate(**batch_tokens, max_new_tokens=1, do_sample=False)
    output = model(batch_tokens["input_ids"])
    print(output)

    # print(tokenizer.batch_decode(generated_ids))


# %%

for k, v in activation.items():
    print(k, len(v), v[0].shape)

# %%
