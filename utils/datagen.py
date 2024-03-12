# %%
from datasets import load_dataset, concatenate_datasets

import random
import pandas as pd


def generate_dataset(num_samples=110000, datasets=[], data_splits=[]):
    for i, dataset in enumerate(datasets):
        data = load_dataset(
            dataset,
            "wikitext-2-v1",
            # split="train",
        )
        print(data)
        all = concatenate_datasets([dataset for dataset in data.values()])
        length = len(all)
        data_samples = int(min(num_samples, length) * data_splits[i])
        print(length)
        print(data_samples)
        indices = random.sample(range(length), data_samples)
        sampled_data = all.select(indices)
        sampled_data = pd.DataFrame(sampled_data)
        sampled_data.replace("", pd.NA, inplace=True)
        sampled_data.replace("\n", pd.NA, inplace=True)
        sampled_data = sampled_data.dropna()
        filename = dataset.split("/")[-1]
        sampled_data.to_csv(
            f"{dataset_relative_path}{filename}_{data_samples}.csv", index=False
        )
        print(f"Generated {dataset}_{data_samples}.csv")


# generate_dataset(
#     num_samples=1800000,
#     # datasets=["cais/mmlu", "bigcode/the-stack-v2", "wikitext"],
#     datasets=["wikitext"],
#     data_splits=[1],
# )

load_dataset("wikitext_44836.csv")
