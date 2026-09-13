"""train_v2.py — training loop updated to use transformer_v2.py

Changes from train_v1.py:
    - imports TransformerLM and GPTConfig from transformer_v2
    - builds GPTConfig from argparse instead of passing args individually
    - uses model.configure_optimizers() instead of bare AdamW
    - checkpoint now saves config dataclass instead of vars(args)

Usage (from the repository root):
    python training_scripts/train_v2.py --data input_texts/the_office.txt
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

# [CHANGED] import from transformer_v2 instead of transformer_v1
from models.transformer_v2 import TransformerLM, GPTConfig
from data.tokenizer import prune_speakers, build_vocab, encode

# ---------------------------------------------------------------------------
# CLI — same flags as before
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--data',            type=str,   default=os.path.join(ROOT, 'input_texts', 'the_office.txt'))
parser.add_argument('--out-dir',         type=str,   default=os.path.join(ROOT, 'trained_weights'))
parser.add_argument('--batch-size',      type=int,   default=128)
parser.add_argument('--block-size',      type=int,   default=64)
parser.add_argument('--max-iters',       type=int,   default=10000)
parser.add_argument('--eval-interval',   type=int,   default=1000)
parser.add_argument('--lr',              type=float, default=3e-4)
parser.add_argument('--n-embd',          type=int,   default=256)
parser.add_argument('--n-head',          type=int,   default=4)
parser.add_argument('--n-layer',         type=int,   default=4)
parser.add_argument('--dropout',         type=float, default=0.2)
parser.add_argument('--weight-decay',    type=float, default=0.1)
parser.add_argument('--min-freq',        type=int,   default=10)
parser.add_argument('--top-k-speakers',  type=int,   default=30)
parser.add_argument('--seed',            type=int,   default=1337)

# Parse once, at the very end of your arguments!
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
if torch.backends.mps.is_available():
    device = 'mps'
    device_type = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
    device_type = 'cuda'
else:
    device = 'cpu'
    device_type = 'cpu'
print(f"Using device: {device}")

torch.manual_seed(args.seed)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
print("Loading raw script...")
with open(args.data, 'r', encoding='utf-8') as f:
    text = f.read()

print("Pruning minor characters to <GUEST>...")
text = prune_speakers(text, top_k_speakers=args.top_k_speakers)

vocab_words, stoi, itos, speaker_tokens = build_vocab(text, min_freq=args.min_freq)
vocab_size = len(vocab_words)
print(f"Vocab size: {vocab_size}")

data = torch.tensor(encode(text, stoi, skip_normalize=True), dtype=torch.long)
print(f"Total tokens: {len(data):,}")
print(f"Tokens per vocab word: {len(data) / vocab_size:.1f}")

n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

# ---------------------------------------------------------------------------
# Batching
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
# [CHANGED] build GPTConfig first, pass config object to model
# BEFORE: TransformerLM(vocab_size=x, n_embd=y, block_size=z, ...)
# AFTER:  config = GPTConfig(...); TransformerLM(config)
config = GPTConfig(
    block_size = args.block_size,
    vocab_size  = vocab_size,
    n_layer     = args.n_layer,
    n_head      = args.n_head,
    n_embd      = args.n_embd,
    dropout     = args.dropout,
)

model = TransformerLM(config).to(device)

# [CHANGED] use model.configure_optimizers() instead of bare AdamW
# BEFORE: optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
# AFTER:  weight decay applied to weight matrices only, not biases or LN params
optimizer = model.configure_optimizers(
    weight_decay=args.weight_decay,
    lr=args.lr,
    device_type=device_type,
)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
best_val_loss = float('inf')

print("\nStarting training...")
for it in range(args.max_iters):

    if it % args.eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            os.makedirs(args.out_dir, exist_ok=True)
            save_path = os.path.join(args.out_dir, 'best_model_v2.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab_words':      vocab_words,
                # [CHANGED] save config dataclass instead of vars(args)
                # generate.py can now reconstruct the exact model without
                # manually matching any arguments
                'config':           config,
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

"""
Running output:
Using device: mps
Loading raw script...
Pruning minor characters to <GUEST>...
Vocab size: 4213
Total tokens: 625,732
Tokens per vocab word: 148.5
Model parameters: 4,226,560

Starting training...
step 0: train loss 8.3771, val loss 8.3764
  New best model saved (val loss: 8.3764)
step 1000: train loss 4.3062, val loss 4.5879
  New best model saved (val loss: 4.5879)
step 2000: train loss 3.9891, val loss 4.4639
  New best model saved (val loss: 4.4639)
step 3000: train loss 3.7529, val loss 4.4286
  New best model saved (val loss: 4.4286)
step 4000: train loss 3.5656, val loss 4.4526
step 5000: train loss 3.4042, val loss 4.4696
step 6000: train loss 3.2867, val loss 4.5068
step 7000: train loss 3.1988, val loss 4.5300
step 8000: train loss 3.1421, val loss 4.5340
step 9000: train loss 3.1176, val loss 4.5330

Training complete. Best val loss: 4.4286



"""