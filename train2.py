import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from typing import Union

torch.manual_seed(1337)

# Constants.
batch_size = 64
block_size = 256
device = "cuda" if torch.cuda.is_available() else "mps"
dropout = 0.2
eval_interval = 500
eval_iters = 500
learning_rate = 3e-4
max_iters = 5000
n_embed = 384
n_head = 6
n_layer = 6

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

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # [[1 0]  <- I. 'I' has label feature.
        #  [1 1]  <- love. 'love' has 2nd label.
        #  [0 1]] <- dog. 'dog'. dog has both labels.
        k = self.key(x)  # batch_size, block_size, head_size

        # [[1 0]  <- I. 'I' interests in the first label in attention
        #  [0 1]  <- love. 'love' interests in the 2nd label in attention
        #  [1 1]] <- dog. 'dog' interests in both labels in attention
        q = self.query(x)  # batch_size, block_size, head_size

        v = self.value(x)  # batch_size, block_size, head_size

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
        weights = self.dropout(weights)
        output = weights @ v  # batch_size, block_size, n_embed
        return output


class MultiHeadAttention(nn.Module):

    def __init__(self, n_head: int, head_size: int) -> None:
        super().__init__()
        # Need to register ModuleList so each sub-Module's params could be
        # discovered by optimizer.
        self.heads = nn.ModuleList([Head(head_size) for i in range(n_head)])

        # torch.concat just physically concat the info from multi heads, but they
        # are isolated from each other. Having a project layer merges the info
        # in mutli-heads.
        self.projection = nn.Linear(n_embed, n_embed)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.cat([head(x) for head in self.heads], dim=-1)
        output = self.projection(output)
        return self.dropout(output)


# Header / Multiheader lacks non-linear acivation. Its softmax
# softmax(q@k^T * d^-0.5) @ v
# occurs before dot product with value (v). So technically, it is
# still just a linear transformation.
class FeedForward(nn.Module):

    def __init__(self, n_embed: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed * 4),
            nn.ReLU(),
            # This is also a projection layer. The feedforward usually
            # expands to high dimension to learn more features. So we need
            # another linear layer to map it back to n_embed.
            nn.Linear(n_embed * 4, n_embed),
            # to avoid overfitting by drop some random nerous.
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):

    def __init__(self, n_embed: int, n_head: int) -> None:
        super().__init__()
        head_size = n_embed // n_head
        self.multi_head = MultiHeadAttention(n_head, head_size)
        self.feed_forward = FeedForward(n_embed)
        # just like batch norm, this is normalize the data to avoid grad explosion.
        # this is applied together with residual connection.
        self.layer_norm1 = nn.LayerNorm(n_embed)
        self.layer_norm2 = nn.LayerNorm(n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # This is residual connections. x = x + layer(x)
        # It helps to preserve the graident in very deep neural network. Because
        # it is "+". The grad is distributed between the layer and the x itself.
        # So if the layer didn't learn anything. The grad will at least be 1 instead
        # of 0. Residual connection is to help grad -> 0.
        # To help grad too large, batchNorm is the way to go.
        x = x + self.multi_head(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x


class BigramLanguageModule(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        # keep head_size the same as embedding to align.
        # self.sa_head = Head(head_size=n_embed)
        # self.multi_head = MultiHeadAttention(n_head=4, head_size=n_embed // 4)
        # self.feed_forward = FeedForward(n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head) for _ in range(n_layer)])
        self.layer_norm = nn.LayerNorm(n_embed)
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
        x = self.blocks(x)
        x = self.layer_norm(x)
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
print(decode(m.generate(context, max_new_tokens=1000)[0].tolist()))
