import time
import math
import os

from model import GPTConfig, GPT

from datasets import load_dataset
import tiktoken
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
def get_dataloader(
    train_batch_size: int = 16,
    val_batch_size: int = 32,
    localdir: str = "./edu_fineweb10B/",
):
    # First load datasets
    data_files = {
        "train": os.path.join(localdir, "train.parquet"),
        "val": os.path.join(localdir, "test.parquet"),
    }
    ds = load_dataset("parquet", data_files=data_files)

    # Instantiate collator
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(localdir, "tokenizer"))
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="pt")

    # Create dataloaders
    train_loader = DataLoader(
        ds["train"],
        batch_size=train_batch_size,
        collate_fn=data_collator,
    )
    val_loader = DataLoader(
        ds["val"],
        batch_size=train_batch_size,
        collate_fn=data_collator,
    )

    return train_loader, val_loader

class DataLoaderLite:
    def __init__(self, B, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        with open("input.txt", "r", encoding="utf-8") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B,T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = buf[:-1].view(B,T) # inputs
        y = buf[1:].view(B,T) # targets
        self.current_position += B * T * self.num_processes 
        # if loading the next batch is out of bounds, reset
        if self.current_position + (B * T *self.num_processes) + 1 > len(self.tokens):
            self.current_position = 0
        return x,y
# ---------------------------------------------------------------------------

# Setup DDP
# torchrun command sets the env variables RANK, LOCAL_RANK, WORLD_SIZE
print(os.environ.get("RANK", -1))
ddp = int(os.environ.get("RANK", -1)) != -1 # is this a ddp run?
if ddp:
    # Use of DDP demands CUDA, we set the device appropriately according to rank
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_world_size = 1
    ddp_rank = 0
    ddp_local_rank = 0
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")


torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# Get data
total_batch_size = 16384
train_batch_size = 16
val_batch_size = 32
T = 256

# Assert that the total batch size is divisible by B*T
assert total_batch_size % (train_batch_size * T * ddp_world_size) == 0, "make sure that the total batch size is divisible by B * T * DDP World Size"
grad_accum_steps = total_batch_size // (train_batch_size * T * ddp_world_size)

if master_process:
    print(f"Total desired batch size: {total_batch_size}")
    print(f"=> Gradient accumulation steps: {grad_accum_steps}")

# Get DataLoader
train_loader, val_loader = get_dataloader(train_batch_size=train_batch_size, val_batch_size=val_batch_size)

# Get model
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
#model = torch.compile(model)

# Wrap into DDP
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

# Define LR Scheduler
def get_lr_cosine_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float = 0.06, horizon_ratio: float = 0.1):
    def get_lambda(current_step: int):
        # Warmup phase
        warmup_steps = int(warmup_ratio * total_steps)
        if current_step < warmup_steps:
            return current_step / (warmup_steps - 1)
        else:
            normalised_step = current_step - warmup_steps
            total_decay_steps = (total_steps - warmup_steps - 1)
            return (1-horizon_ratio) * (1 + math.cos(math.pi * normalised_step / total_decay_steps)) / 2 + horizon_ratio
    return LambdaLR(optimizer, get_lambda)

# Run training
epochs = 1
evaluate_on_steps = 100
val_batch_evals = 20
print(f"Will run for {epochs} epoch with {len(train_loader)} steps per epoch.")

# Instantiate data loader and run optimization
total_steps = len(train_loader) * epochs
optimizer = raw_model.configure_optimizer(weight_decay=.01, learning_rate=6e-4)
scheduler = get_lr_cosine_scheduler(optimizer, total_steps, warmup_ratio=.06, horizon_ratio=.1)

for epoch in range(epochs):
    running_loss = 0.0
    t0 = time.time()

    for step, batch in enumerate(train_loader):
        # Move all tensors to device
        batch = {k: v.to(device) for k,v in batch.items()}
        x = batch["input_ids"]
        y = batch["labels"]

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        running_loss += loss.detach()

        # In DDP, only synchronise gradients when a step is taken
        if ddp:
            model.require_backward_grad_sync = (step % grad_accum_steps == 0)
        loss.backward()

        if (step+1) % grad_accum_steps == 0:
            # Gradient update
            real_step = (step+1) // grad_accum_steps
            if ddp:
                dist.all_reduce(running_loss, op=dist.ReduceOp.AVG)

            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            t1 = time.time()
            if master_process:
                tokens_processed = ddp_world_size * grad_accum_steps * train_batch_size * T
                print(f"Step: {real_step} | Loss: {running_loss:.5f} | Norm: {norm:.4f} | {1000 * (t1-t0):.2f} ms/step | {grad_accum_steps * train_batch_size * T / (t1-t0):.2f} tokens/s")
            running_loss = 0.0
            t0 = time.time()

            # Evaluate step
            if real_step % evaluate_on_steps == 0:
                model.eval()
                with torch.no_grad():
                    running_eval_loss = 0.0
                    for val_step, batch in enumerate(val_loader):
                        batch = {k: v.to(device) for k,v in batch.items()}
                        x, y = batch["input_ids"], batch["labels"]
                        logits, loss = model(x, y)
                        running_eval_loss += loss.item()
                        if val_step + 1 == val_batch_evals: break
                eval_loss = running_eval_loss / val_batch_evals
                print(f"| Evaluation Step - Loss: {eval_loss:.4f} - Perplexity: {math.exp(eval_loss)}")
                model.train()


if ddp:
    dist.destroy_process_group()
