import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
itos = {i:c for i, c in enumerate(chars)}
stoi = {c:i for i, c in itos.items()}
device = 'mps'

def encode(text: str):
    return [stoi[c] for c in text]

def decode(tokens) -> str:
    return ''.join([itos[int(token)] for token in tokens])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]

block_size = 8
batch_size = 4

def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    data = train_data if split == 'train' else val_data
    indices = torch.randint(0, len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in indices])
    y = torch.stack([data[i+1:i+block_size+1] for i in indices])
    return x, y

xb, yb = get_batch('train')
print(xb.shape)

class BigramLanguageModule(nn.Module):
    
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, indices: torch.Tensor, targets=None) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.embedding_table(indices)
       
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, indices, max_new_tokens: int) -> torch.Tensor :
        for _ in range(max_new_tokens):
            logits, loss = self(indices)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=1)
            index_next = torch.multinomial(probs, num_samples=1)
            indices = torch.cat((indices, index_next), dim=1)
        return indices
            


m = BigramLanguageModule(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss.shape)
print(loss)
print(decode(m.generate(torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0].tolist()))