import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


########################
###       Model      ###
########################

# See https://raw.githubusercontent.com/karpathy/ng-video-lecture/refs/heads/master/gpt.py
# From the tutorial Let's build GPT: from scratch, in code, spelled out by Andrej Karpathy
# https://www.youtube.com/watch?v=kCc8FmEb1nY
# Modified to use F.scaled_dot_product_attention to increase speed by computing every head at once.


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # (batch_size, n_heads, seq_len, head_dim)
        q = self.q_proj(x).view(batch_size, seq_len,
                                self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len,
                                self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len,
                                self.n_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v)

        attn_out = attn_out.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)
        return self.out_proj(attn_out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()

        self.sa = MultiHeadAttention(d_model, n_heads)
        self.ffwd = FeedForward(d_model, d_ff)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model, n_heads, d_ff, n_layers):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(seq_len, d_model)

        self.blocks = nn.Sequential(
            *[
                Block(d_model, n_heads, d_ff)
                for _ in range(n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, tokens):
        B, T = tokens.shape

        token_embeddings = self.token_embedding(tokens)

        positions = torch.arange(T, device=tokens.device)
        position_embeddings = self.position_embedding(positions)

        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        x = self.ln_f(x)

        return self.output(x)

########################
###       Train      ###
########################


def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        inputs = batch["encrypted_tokens"].to(DEVICE)
        targets = batch["tokens"].to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct_chars = 0
    total_chars = 0
    correct_seqs = 0
    total_seqs = 0

    for batch in dataloader:
        inputs = batch["encrypted_tokens"].to(DEVICE)
        targets = batch["tokens"].to(DEVICE)

        logits = model(inputs)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        total_loss += loss.item() * inputs.size(0)

        preds = torch.argmax(logits, dim=-1)

        correct_chars += (preds == targets).sum().item()
        total_chars += targets.numel()
        correct_seqs += (preds == targets).all(dim=-1).sum().item()
        total_seqs += targets.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    char_acc = correct_chars / total_chars
    seq_acc = correct_seqs / total_seqs

    return avg_loss, char_acc, seq_acc


########################
### Execution Script ###
########################



def train(train_dataloader, test_dataloader, EPOCHS=10, LR=1e-4, save_path="model/full_rank.pth"):
    model = TransformerModel(
        vocab_size=32,
        seq_len=32,
        d_model=768,
        n_heads=12,
        d_ff=3072,
        n_layers=12,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in tqdm(range(1, EPOCHS + 1)):
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion)
        val_loss, val_char_acc, val_seq_acc = evaluate(model, test_dataloader, criterion)

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Char Acc: {val_char_acc * 100:.2f}% | Val Seq Acc: {val_seq_acc * 100:.2f}%")

    torch.save(model.state_dict(), save_path)
    return model