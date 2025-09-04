import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from typing import Union

torch.manual_seed(1337)

# Constants.
batch_size = 32
block_size = 8
device = "cuda" if torch.cuda.is_available() else "mps"
eval_interval = 500
eval_iters = 200
learning_rate = 1e-3
max_iters = 5000
n_embed = 32

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
    indices = torch.randint(low=0, high=len(data) - block_size, size=(batch_size,))
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


class Head(nn.Module):
    # k & q is token metadata. v is token data;
    # k is what labels the current content has;
    # q is what label the current content is looking for;
    # v is what the token embedding represents, but transformed from
    #   n_embed to head_size by a linear layer;
    # n_embed is usually dividable by head_size. So for multi-head attention,
    # head could be concated [head_1_size, head_2_size, head_3_size, ...] = [n_embed]
    #
    # Note, each head's size is usually the same.
    def __init__(self, head_size: int) -> None:
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        # register_buffer will not store the var in parameters. So it won't be updated.
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, indices: torch.Tensor):
        B, T, C = indices.shape

        # [[1 0]  <- I. 'I' has label feature.
        #  [1 1]  <- love. 'love' has 2nd label.
        #  [0 1]] <- dog. 'dog'. dog has both labels.
        k = self.key(indices)  # batch_size, block_size, head_size

        # [[1 0]  <- I. 'I' interests in the first label in attention
        #  [0 1]  <- love. 'love' interests in the 2nd label in attention
        #  [1 1]] <- dog. 'dog' interests in both labels in attention
        q = self.query(indices)  # batch_size, block_size, head_size

        v = self.value(indices)  # batch_size, block_size, head_size

        # q@k^T has a overall weight on what token (represented by row), interests on other
        # [[1 1 0]  <- I
        #  [0 1 1]  <- love
        #  [1 2 1]] <- dog
        # 'I' is related to 'love' but not 'dog'
        # 'love is related to 'dog'
        # 'dog is related to 'I' and 'love', espeically 'love'
        #
        # d_k is head_size. The expectation of q@k will be larger if the head_size is getting
        # larger. So we need to normalize the value. If not, the softmax will be more
        # leaning to 0 and 1 which cause gradient to lost.
        #
        # batch_size, block_size, block_size
        weights = q @ k.transpose(-2, -1) * (C**-0.5)

        # must reset weight. masked_fill will not update in place.
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        output = weights @ v  # batch_size, block_size, n_embed
        return output


class BigramLanguageModule(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        # keep head_size the same as embedding to align.
        self.sa_head = Head(head_size=n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(
        self, indices: torch.Tensor, targets: Union[torch.Tensor, None] = None
    ) -> tuple[torch.Tensor, Union[torch.Tensor, None]]:
        B, T = indices.shape

        token_embeddings = self.token_embedding_table(
            indices
        )  # (batch, block, n_embed)
        position_embeddings = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (block, n_embed)
        x = token_embeddings + position_embeddings  # (batch, block, n_embed)
        x = self.sa_head(x)
        logits = self.lm_head(x)  # (batch, block, vocab_size)

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
            # Can only look back up to block_size, because we use position encoding.
            indices_cond = indices[:, -block_size:]
            logits, loss = self(indices_cond[:, -block_size:])
            logits = logits[:, -1, :]  # [1, 65]
            probs = F.softmax(logits, dim=1)  # [1,65]
            index_next = torch.multinomial(probs, num_samples=1)  # [1,1]
            indices = torch.cat((indices, index_next), dim=1)  # [1,2]
        return indices


model = BigramLanguageModule()
m = model.to(device)

# optimizer = torch.optim.SGD(m.parameters(), lr=1e-3)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)


now = datetime.now()
for iter in range(max_iters):

    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("training time elpased: ", datetime.now() - now)

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))
