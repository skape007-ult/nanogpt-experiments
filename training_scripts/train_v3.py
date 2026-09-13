"""train_v3.py — output-quality iteration on top of train_v2.py.

Same model code as v2 (models/transformer_v2.py); the changes target what the
chat actually produces:

    - v3 tokenizer (data/tokenizer.py): punctuation split into its own tokens,
      one tokenize() shared by training and chat, casing restored on decode.
      <UNK> falls from 12.7% to ~3% of training tokens.
    - <UNK> targets are masked out of the loss (ignore_index=-1), so the model
      spends no capacity learning to predict <UNK>.
    - Bigger model and longer context: 6 layers / 6 heads / 384-d, 128 tokens
      (several lines of dialogue instead of ~2).
    - nanoGPT "shakespeare-char" style schedule: lr 1e-3, 200-step warmup,
      cosine decay to min_lr 1e-4 (v2 decayed all the way to 0).
    - Finer evaluation (every 250 steps, 100 batches) with early stopping once
      validation loss stops improving for --patience evals.
    - The tokenizer (vocab + casing table) is saved inside the checkpoint, so
      generate.py no longer needs the training text to rebuild the vocabulary.

Usage (from the repository root):
    python training_scripts/train_v3.py --data input_texts/the_office.txt
"""
import argparse
import math
import os
import sys
import time

import torch

# Make the repo root importable so `models` and `data` resolve when this
# script is run directly from training_scripts/.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.transformer_v2 import TransformerLM, GPTConfig
from data.tokenizer import (prune_speakers, tokenize, build_casing, build_vocab_v3,
                            is_speaker_tag, V3_TAG_PATTERN)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train the v3 Office dialogue model")
parser.add_argument('--data',            type=str,   default=os.path.join(ROOT, 'input_texts', 'the_office.txt'))
parser.add_argument('--out-dir',         type=str,   default=os.path.join(ROOT, 'trained_weights'))
parser.add_argument('--batch-size',      type=int,   default=64)
parser.add_argument('--block-size',      type=int,   default=128)
parser.add_argument('--max-iters',       type=int,   default=8000)
parser.add_argument('--eval-interval',   type=int,   default=250)
parser.add_argument('--eval-iters',      type=int,   default=100)
parser.add_argument('--patience',        type=int,   default=6,
                    help="Stop after this many evals without a new best val loss")
parser.add_argument('--lr',              type=float, default=1e-3)
parser.add_argument('--min-lr',          type=float, default=1e-4)
parser.add_argument('--warmup-iters',    type=int,   default=200)
parser.add_argument('--n-embd',          type=int,   default=384)
parser.add_argument('--n-head',          type=int,   default=6)
parser.add_argument('--n-layer',         type=int,   default=6)
parser.add_argument('--dropout',         type=float, default=0.25)
parser.add_argument('--weight-decay',    type=float, default=0.1)
parser.add_argument('--min-freq',        type=int,   default=5)
parser.add_argument('--top-k-speakers',  type=int,   default=30)
parser.add_argument('--seed',            type=int,   default=1337)
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
# Data
# ---------------------------------------------------------------------------
print("Loading raw script...")
with open(args.data, 'r', encoding='utf-8') as f:
    text = f.read()

text = prune_speakers(text, top_k_speakers=args.top_k_speakers, tag_pattern=V3_TAG_PATTERN)

cased_tokens = tokenize(text, keep_case=True)
casing = build_casing(cased_tokens)
tokens = [t if is_speaker_tag(t) else t.lower() for t in cased_tokens]
del cased_tokens

vocab_words, stoi, itos, speaker_tokens = build_vocab_v3(tokens, min_freq=args.min_freq)
vocab_size = len(vocab_words)

data = torch.tensor([stoi.get(t, 0) for t in tokens], dtype=torch.long)
unk_rate = (data == 0).float().mean().item()
print(f"Vocab size: {vocab_size} ({len(speaker_tokens)} speaker tags)")
print(f"Total tokens: {len(data):,}")
print(f"Tokens per vocab word: {len(data) / vocab_size:.1f}")
print(f"<UNK> rate: {unk_rate:.2%}")

n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------
def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - args.block_size, (args.batch_size,))
    x = torch.stack([d[i:i + args.block_size] for i in ix])
    y = torch.stack([d[i + 1:i + args.block_size + 1] for i in ix])
    y[y == 0] = -1   # mask <UNK> targets out of the loss
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def get_lr(it):
    if it < args.warmup_iters:
        return args.lr * (it + 1) / args.warmup_iters
    decay_ratio = min(1.0, (it - args.warmup_iters) / max(1, args.max_iters - args.warmup_iters))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return args.min_lr + coeff * (args.lr - args.min_lr)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
config = GPTConfig(
    block_size = args.block_size,
    vocab_size = vocab_size,
    n_layer    = args.n_layer,
    n_head     = args.n_head,
    n_embd     = args.n_embd,
    dropout    = args.dropout,
)
model = TransformerLM(config).to(device)
optimizer = model.configure_optimizers(weight_decay=args.weight_decay, lr=args.lr, device_type=device)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
best_val_loss = float('inf')
evals_since_best = 0
os.makedirs(args.out_dir, exist_ok=True)
save_path = os.path.join(args.out_dir, 'best_model_v3.pt')
t0 = time.time()

print("\nStarting training...")
for it in range(args.max_iters + 1):

    if it % args.eval_interval == 0:
        losses = estimate_loss(model)
        elapsed = (time.time() - t0) / 60
        print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} "
              f"(val ppl {math.exp(losses['val']):.1f}, lr {get_lr(it):.2e}, {elapsed:.1f} min)", flush=True)

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            evals_since_best = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'config':           config,
                'tokenizer': {
                    'version':  'v3',
                    'vocab':    vocab_words,
                    'casing':   casing,
                    'speakers': speaker_tokens,
                },
                'train_info': {
                    'iter':      it,
                    'val_loss':  best_val_loss,
                    'train_loss': losses['train'],
                    'unk_rate':  unk_rate,
                    'args':      vars(args),
                },
            }, save_path)
            print(f"  New best model saved (val loss: {best_val_loss:.4f})", flush=True)
        else:
            evals_since_best += 1
            if evals_since_best >= args.patience:
                print(f"Early stopping: no improvement in {args.patience} evals")
                break

    if it == args.max_iters:
        break

    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(it)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()

print(f"\nTraining complete in {(time.time() - t0) / 60:.1f} min. "
      f"Best val loss: {best_val_loss:.4f} (ppl {math.exp(best_val_loss):.1f}) -> {save_path}")
