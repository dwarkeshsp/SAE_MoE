# %%
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch import nn
import collections


def load_generated_dataset(
    filename, batch_size=32, dataset_relative_path="../dataset/"
):
    dataset = pd.read_csv(f"{dataset_relative_path}{filename}")
    # print(dataset.head(10))
    dataset_list = dataset["text"].tolist()
    dataloader = torch.utils.data.DataLoader(dataset_list, batch_size=batch_size)
    return dataloader


# load_generated_dataset("wikitext_44836.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

double_quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x7B-v0.1",
    quantization_config=double_quant_config,
    attn_implementation="flash_attention_2",
)

# %%

activation = collections.defaultdict(list)


def getActivation(name):
    def hook(model, input, output):
        assert type(output) is not tuple
        activation[name].append(output.detach())

    return hook


experts = model.model.layers[15].block_sparse_moe.experts

for expert_idx in range(8):
    experts[expert_idx].w3.register_forward_hook(
        getActivation(f"layer_15_expert_{expert_idx}_w_3")
    )
    experts[expert_idx].act_fn.register_forward_hook(
        getActivation(f"layer_15_expert_{expert_idx}_act_w1")
    )
    experts[expert_idx].w2.register_forward_hook(
        getActivation(f"layer_15_expert_{expert_idx}_w_2")
    )

for batch in tqdm(load_generated_dataset("wikitext_44836.csv")):
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
    tokenizer.pad_token = tokenizer.eos_token
    batch_tokens = tokenizer(
        batch,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).to(device)

    print(batch_tokens["input_ids"].shape)

    generated_ids = model.generate(**batch_tokens, max_new_tokens=1, do_sample=False)
    print(tokenizer.batch_decode(generated_ids))

    # print(batch)
    break

# %%
