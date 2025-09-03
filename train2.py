import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from typing import Union

torch.manual_seed(1337)

# Constants.
batch_size = 32
block_size = 8
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_interval = 300
eval_iters = 200
learning_rate = 1e-2
max_iters = 3000

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
itos = {i: c for i, c in enumerate(chars)}
stoi = {c: i for i, c in itos.items()}


def encode(text: str):
    return [stoi[c] for c in text]


def decode(tokens) -> str:
    return "".join([itos[int(token)] for token in tokens])


data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]


def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    data = train_data if split == "train" else val_data
    indices = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in indices])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in indices])
    x = x.to(device)
    y = y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss() -> dict[str, int]:
    out = {}
    model.eval()  # just set mode. No effect.
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # just set mode back. No effect.
    return out


xb, yb = get_batch("train")


class BigramLanguageModule(nn.Module):

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self, indices: torch.Tensor, targets: Union[torch.Tensor, None] = None
    ) -> tuple[torch.Tensor, Union[torch.Tensor, None]]:
        logits = self.token_embedding_table(indices)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape  # [32, 8, 65]
            # Input and target pairing as (index, index + 1). So could be understand as
            # logits[index, embed_vector] <-> target[index + 1]
            logits = logits.view(B * T, C)  # [32 * 8 = 256, 65]
            targets = targets.view(B * T)  # [32 * 8 = 256]
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, indices: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, loss = self(indices)
            logits = logits[:, -1, :]  # [1, 65]
            probs = F.softmax(logits, dim=1)  # [1,65]
            index_next = torch.multinomial(probs, num_samples=1)  # [1,1]
            indices = torch.cat((indices, index_next), dim=1)  # [1,2]
        return indices


model = BigramLanguageModule(vocab_size)
m = model.to(device)

# optimizer = torch.optim.SGD(m.parameters(), lr=1e-3)
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)


now = datetime.now()
for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


print("training time elpased: ", datetime.now() - now)

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))

# x(input)b(bag)o(of)w(words)
xbow = torch.zeros((batch_size, block_size, vocab_size))
for b in range(batch_size):
    for t in range(block_size):
        xprev = x[b, : t + 1]  # [1, t, 65]
        xbow[b, t] = torch.mean(xprev, 0)
