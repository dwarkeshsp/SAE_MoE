from datasets import load_dataset

import pandas as pd
import torch

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")


def clean_df(samples_df):
    samples_df.replace("", pd.NA, inplace=True)
    samples_df.replace("\n", pd.NA, inplace=True)
    samples_df = samples_df.dropna()
    return samples_df


def generate_dataset(num_samples, dataset_relative_path="../dataset/"):
    data = load_dataset("monology/pile-uncopyrighted", split="train", streaming="true")
    i = 0
    samples = []
    for item in iter(data):
        sample = item["text"]
        samples.append(sample)
        i += 1
        if i == num_samples:
            break
    samples_df = pd.DataFrame(samples)
    print(samples_df.head(5))

    samples_df = clean_df(samples_df)

    def split_sequence(sequence, max_length):
        tokens = tokenizer.encode(sequence)
        sequences = []
        start_idx = 0
        while start_idx < len(tokens):
            end_idx = min(start_idx + max_length, len(tokens))
            sequences.append(tokenizer.decode(tokens[start_idx:end_idx]))
            start_idx = end_idx
        return sequences

    max_length = 128  # Specify the desired maximum sequence length
    samples_df = samples_df[0].apply(lambda x: split_sequence(x, max_length))
    samples_df = samples_df.explode("text")

    samples_df = clean_df(samples_df)

    print(samples_df.head(5))

    samples_df.to_csv(f"{dataset_relative_path}/pile.csv", index=False)


def load_generated_dataset(
    filename, batch_size=32, dataset_relative_path="../dataset/"
):
    dataset = pd.read_csv(f"{dataset_relative_path}/{filename}")
    print(dataset.head(10))
    dataset_list = dataset["0"].tolist()
    for i in range(len(dataset_list)):
        print(i, dataset_list[i])
    dataloader = torch.utils.data.DataLoader(dataset_list, batch_size=batch_size)
    return dataloader


dataset = load_generated_dataset("pile.csv", batch_size=2)

# generate_dataset(
#     num_samples=1000,
#     # datasets=["cais/mmlu", "bigcode/the-stack-v2", "wikitext"],
#     # datasets=["wikitext"],
#     # data_splits=[1],
# )

# load_dataset("wikitext_44836.csv")
