"""Training loop for the word-level transformer language model (v1 architecture).

Usage (from the repository root):
    python training_scripts/train_v1.py --data input_texts/the_office.txt
    python training_scripts/train_v1.py --data input_texts/the_office.txt --max-iters 5000 --batch-size 64
"""
import argparse
import math
import os
import sys

import torch

# Make the repo root importable so `models` and `data` resolve when this
# script is run directly from training_scripts/.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.transformer_v1 import TransformerLM
from data.tokenizer import prune_speakers, build_vocab, encode

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train a word-level transformer LM")
parser.add_argument('--data', type=str, default=os.path.join(ROOT, 'input_texts', 'the_office.txt'),
                    help="Path to the training text file")
parser.add_argument('--out-dir', type=str, default=os.path.join(ROOT, 'trained_weights'),
                    help="Directory to save model checkpoints")
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--block-size', type=int, default=64)
parser.add_argument('--max-iters', type=int, default=10000)
parser.add_argument('--eval-interval', type=int, default=1000)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--n-embd', type=int, default=256)
parser.add_argument('--n-head', type=int, default=4)
parser.add_argument('--n-layer', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--min-freq', type=int, default=10,
                    help="Minimum word frequency to include in vocabulary")
parser.add_argument('--top-k-speakers', type=int, default=30,
                    help="Number of top speakers to keep (rest become <GUEST>)")
parser.add_argument('--seed', type=int, default=1337)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f"Using device: {device}")

torch.manual_seed(args.seed)

# ---------------------------------------------------------------------------
# Data loading & vocabulary
# ---------------------------------------------------------------------------
print("Loading raw script...")
with open(args.data, 'r', encoding='utf-8') as f:
    text = f.read()

print("Pruning minor characters to <GUEST>...")
text = prune_speakers(text, top_k_speakers=args.top_k_speakers)

vocab_words, stoi, itos, speaker_tokens = build_vocab(text, min_freq=args.min_freq)
vocab_size = len(vocab_words)
print(f"Total speaker tokens after pruning: {len(speaker_tokens)}")
print(f"Vocab size: {vocab_size}")

print("Encoding text into tensors...")
data = torch.tensor(encode(text, stoi, skip_normalize=True), dtype=torch.long)
print(f"Total tokens: {len(data):,}")
print(f"Tokens per vocab word: {len(data) / vocab_size:.1f} (target: 100+)")

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# ---------------------------------------------------------------------------
# Batching & evaluation helpers
# ---------------------------------------------------------------------------
eval_iters = 50


def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - args.block_size, (args.batch_size,))
    x = torch.stack([d[i:i + args.block_size] for i in ix])
    y = torch.stack([d[i + 1:i + args.block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it):
    warmup = 100
    if it < warmup:
        return args.lr * it / warmup
    decay_ratio = (it - warmup) / (args.max_iters - warmup)
    return args.lr * 0.5 * (1.0 + math.cos(math.pi * decay_ratio))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = TransformerLM(
    vocab_size=vocab_size,
    n_embd=args.n_embd,
    block_size=args.block_size,
    n_head=args.n_head,
    n_layer=args.n_layer,
    dropout=args.dropout,
).to(device)

param_count = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {param_count:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
best_val_loss = float('inf')

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
print("\nStarting training...")
for it in range(args.max_iters):
    if it % args.eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            os.makedirs(args.out_dir, exist_ok=True)
            save_path = os.path.join(args.out_dir, 'best_model.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab_words': vocab_words,
                'args': vars(args),
            }, save_path)
            print(f"  New best model saved (val loss: {best_val_loss:.4f})")

    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(it)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()

print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
