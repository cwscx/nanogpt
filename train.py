import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32 # number of indepdent sequence to process in parallel
block_size = 128 # max context length for prediction
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'mps'
eval_iters = 200
n_embed = 384
n_head = 4
n_layer = 4
dropout = 0.2

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars: list[int] = sorted(list(set(text)))
vocab_size: int = len(chars)

charToIndex: dict[str, int] = {}
indexToChar: dict[int, str] = {}

for i, c in enumerate(chars):
    charToIndex[c] = i
    indexToChar[i] = c

def encode(text: str) -> list[int]:
    tokens: list[int] = []
    for c in text:
        tokens.append(charToIndex.get(c))
    return tokens

def decode(tokens: list[int]) -> str:
    chars: list[str] = []
    for token in tokens:
        chars.append(indexToChar.get(token))
    return "".join(chars)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n + 1:]

def get_batch(split: str) -> torch.tensor:
    if (split == 'train'):
        data = train_data
    else:
        data = val_data

    start_offsets = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i : i + block_size] for i in start_offsets])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in start_offsets])
    x = x.to(device)
    y = y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss() -> dict[str, float]:
    out = {}
    model.eval()
    for split in ['train', 'eval']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):

    def __init__ (self, head_size: int):
        super().__init__();
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.tensor:
        B,T,C = x.shape
        k = self.key(x) # B,T,C
        q = self.query(x) # B,T,C
        v = self.value(x) # B,T,C

        weights = q @ k.transpose(-2, -1) * (n_embed ** -0.5) # B,T,T
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # B,T,T
        weights = F.softmax(weights, dim=-1) # B,T,T
        weights = self.dropout(weights)

        out = weights @ v
        return out

class MultiHeadAttention(nn.Module):

    def __init__ (self, num_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)  # n_embed = num_heads * head_size
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):

    def __init__(self, n_embed: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),  # projection layer
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self, n_embed: int, n_head: int):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):

    def __init__ (self):
        super().__init__();
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        # self.sa_head = MultiHeadAttention(4, n_embed//4)
        # self.ffwd = FeedForward(n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx: list[int], targets: torch.tensor = None) -> tuple[torch.tensor, torch.tensor]:
        B,T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # B,T,C
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # torch.arange => tensor([0, 1, ..., T - 1]). (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets == None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets .view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: list[int], max_token_size: int) -> torch.tensor:
        for _ in range(max_token_size): 
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim = 1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['eval']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


context = torch.zeros((1, 1), dtype=torch.long, device = device)
print(decode(m.generate(context, max_token_size=500)[0].tolist()))



