# %%
from datasets import load_dataset, concatenate_datasets

import random
import pandas as pd


def generate_dataset(num_samples=110000, datasets=[], data_splits=[]):
    for i, dataset in enumerate(datasets):
        data = load_dataset(dataset, "all")
        print(len(data))
        all = concatenate_datasets([dataset for dataset in data.values()])
        length = len(all)
        data_samples = int(num_samples * data_splits[i])
        print(length)
        print(data_samples)
        indices = random.sample(range(length), data_samples)
        sampled_data = all.select(indices)
        sampled_data = pd.DataFrame(sampled_data)
        sampled_data.to_csv(f"../dataset/{dataset}_{data_samples}.csv", index=False)
        print(f"Generated {dataset}_{data_samples}.csv")


generate_dataset(
    num_samples=110000,
    datasets=["cais/mmlu", "bigcode/the-stack-v2", "wikitext"],
    data_splits=[0.10, 0.40, 0.5],
)
