"""Interactive chat with a trained Office character model.

Works with checkpoints from either training script — the architecture is
detected from the checkpoint contents:
    - train_v1.py saves 'args'   (dict)       -> models.transformer_v1
    - train_v2.py saves 'config' (GPTConfig)  -> models.transformer_v2

Usage:
    python generate.py --checkpoint trained_weights/best_model.pt    --data input_texts/the_office.txt
    python generate.py --checkpoint trained_weights/best_model_v2.pt --data input_texts/the_office.txt
"""
import argparse
import dataclasses

import torch

from data.tokenizer import prune_speakers, build_vocab, encode, decode

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Chat with a trained Office character")
parser.add_argument('--checkpoint', type=str, default='trained_weights/best_model.pt',
                    help="Path to saved model checkpoint (v1 or v2)")
parser.add_argument('--data', type=str, default='input_texts/the_office.txt',
                    help="Path to the training text (needed to rebuild vocabulary)")
parser.add_argument('--temperature', type=float, default=0.6)
parser.add_argument('--top-k', type=int, default=20)
parser.add_argument('--max-tokens', type=int, default=60)
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

# ---------------------------------------------------------------------------
# Load checkpoint to get the training hyperparameters
# ---------------------------------------------------------------------------
# weights_only=False: v2 checkpoints pickle a GPTConfig dataclass, which the
# safe loader (default since torch 2.6) rejects. Only load checkpoints you trust.
checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
is_v2 = 'config' in checkpoint

if is_v2:
    config = checkpoint['config']
    train_args = dataclasses.asdict(config)
else:
    train_args = checkpoint.get('args', {})

# v2 checkpoints don't record the data-prep flags, so fall back to the defaults
# shared by both training scripts.
top_k_speakers = train_args.get('top_k_speakers', 30)
min_freq = train_args.get('min_freq', 10)

# ---------------------------------------------------------------------------
# Rebuild vocabulary from the same training data
# ---------------------------------------------------------------------------
print("Rebuilding vocabulary...")
with open(args.data, 'r', encoding='utf-8') as f:
    text = f.read()

text = prune_speakers(text, top_k_speakers=top_k_speakers)
vocab_words, stoi, itos, speaker_tokens = build_vocab(text, min_freq=min_freq)
vocab_size = len(vocab_words)
special_tokens = ['<UNK>', '<YOU>'] + speaker_tokens

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
if is_v2:
    from models.transformer_v2 import TransformerLM
    assert config.vocab_size == vocab_size, (
        f"Checkpoint vocab ({config.vocab_size}) != rebuilt vocab ({vocab_size}); "
        "use the same --data file the model was trained on")
    model = TransformerLM(config).to(device)
else:
    from models.transformer_v1 import TransformerLM
    model = TransformerLM(
        vocab_size=vocab_size,
        n_embd=train_args.get('n_embd', 256),
        block_size=train_args.get('block_size', 64),
        n_head=train_args.get('n_head', 4),
        n_layer=train_args.get('n_layer', 4),
        dropout=train_args.get('dropout', 0.2),
    ).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Loaded {'v2' if is_v2 else 'v1'} model from {args.checkpoint}")

# ---------------------------------------------------------------------------
# Character selection
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
print("DUNDER MIFFLIN AI INITIALIZED. Type 'quit' to exit.")
print("=" * 50 + "\n")

clean_speakers = [s.strip('<>') for s in speaker_tokens if s not in ['<GUEST>', '<UNK>']]
print("Main characters:", ", ".join(clean_speakers[:10]) + "...\n")

target_char = input("Who do you want to talk to? (e.g. MICHAEL): ").strip().upper()
target_tag = f"<{target_char}>"

if target_tag not in speaker_tokens:
    print(f"[{target_char} not found. Defaulting to MICHAEL.]")
    target_tag = "<MICHAEL>"
    target_char = "MICHAEL"

print(f"\n--- Chatting with {target_char} ---\n")

# UNK is at index 0 — ban it from generation
unk_id = stoi.get('<UNK>', 0)

# ---------------------------------------------------------------------------
# Chat loop
# ---------------------------------------------------------------------------
while True:
    try:
        user_input = input("You: ")
    except EOFError:
        break
    if user_input.lower() in ('quit', 'exit'):
        break

    prompt = f"<MICHAEL> I am the best boss in the world <YOU> {user_input} {target_tag}"
    input_ids = encode(prompt, stoi)
    if not input_ids:
        input_ids = [0]

    context = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        output = model.generate(
            idx=context,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            ban_token_ids=[unk_id],
        )

    reply_ids = output[0][len(input_ids):].tolist()
    raw_reply = decode(reply_ids, itos)

    # Stop at the next speaker tag
    for tag in special_tokens:
        if tag in raw_reply:
            raw_reply = raw_reply.split(tag)[0]

    reply = raw_reply.strip()
    if reply:
        reply = reply[0].upper() + reply[1:]

    print(f"{target_char}: {reply}\n")
