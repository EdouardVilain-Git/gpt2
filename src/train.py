from model import GPTConfig, GPT
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open("input.txt", "r", encoding="utf-8") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        self.current_position = 0

    def next_batch(self):
        B,T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = buf[:-1].view(B,T) # inputs
        y = buf[1:].view(B,T) # targets
        self.current_position += B*T 
        # if loading the next batch is out of bounds, reset
        if self.current_position + B*T + 1 > len(self.tokens):
            self.current_position = 0
        return x,y
# ---------------------------------------------------------------------------

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

# Get model
enc = tiktoken.get_encoding("gpt2")

# Get data
train_loader = DataLoaderLite(B=4,T=32)

# Get model
model = GPT(GPTConfig())
model.to(device)

# Run training on using data loader
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for epoch in range(50):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)

    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"Step {epoch} - Loss: {loss.item():.5f}")
