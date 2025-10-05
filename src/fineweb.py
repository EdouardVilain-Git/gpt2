"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb_10B"
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

from transformers import AutoTokenizer

localdir = "./edu_fineweb10B/"
remote_name = "sample-10BT"
dataset_limit = 1000000
block_size = 256

# download the dataset
fw = load_dataset(f"HuggingFaceFW/fineweb-edu", name=remote_name, split=f"train[:{dataset_limit}]")

# init the tokenizer from HF
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Tokenize examples
def tokenize_example(examples):
    return tokenizer(examples["text"], add_special_tokens=False)
fw_tokenized = fw.map(
    tokenize_example,
    batched=True,
    num_proc=4,
    remove_columns=fw.column_names,
)

# Fuse documents together to create chunks of constant size block_size
def pack_examples(examples):
    """
    Pack all sequences to a max length of block_size.
    """
    # First merge everything together
    merged = []
    for ids in examples["input_ids"]:
        merged.extend(ids)
        if merged[-1] != tokenizer.eos_token_id:
            merged.append(tokenizer.eos_token_id)
    
    # Split merged into fixed blocks of size block_size
    split_input_ids = [merged[i:i+block_size] for i in range(0, len(merged), block_size)]
    split_attention_mask = [[1 for _ in range(len(input_ids))] for input_ids in split_input_ids]

    return {"input_ids": split_input_ids, "attention_mask": split_attention_mask}

packed_ds = fw_tokenized.map(
    pack_examples,
    batched=True,
    num_proc=4
)

# Split into train and test
split_ds = packed_ds.train_test_split(test_size=.1)

# Save to local_dir
tokenizer.save_pretrained(os.path.join(localdir, "tokenizer"))
split_ds["train"].to_parquet(os.path.join(localdir, "train.parquet"))
split_ds["test"].to_parquet(os.path.join(localdir, "test.parquet"))
