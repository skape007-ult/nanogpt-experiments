"""Chat with a trained Office character model.

The checkpoint format decides how it is loaded:
    - train_v3.py saves 'tokenizer' (vocab + casing) -> v3 tokenizer, multi-turn history
    - train_v2.py saves 'config' (GPTConfig)          -> legacy tokenizer, transformer_v2
    - train_v1.py saves 'args'   (dict)               -> legacy tokenizer, transformer_v1
Legacy (v1/v2) checkpoints don't store their vocabulary, so they also need --data.

Usage:
    # interactive
    python generate.py --checkpoint trained_weights/best_model_v3.pt

    # scripted conversation (prints a transcript and exits)
    python generate.py --character DWIGHT --you-are JIM --seed 7 \\
        --message "Dwight, where is my stapler?" --message "Why is it in jello?"

    # legacy checkpoints
    python generate.py --checkpoint trained_weights/best_model_v2.pt --data input_texts/the_office.txt
"""
import argparse

import torch

from data.tokenizer import (prune_speakers, build_vocab, encode, decode,
                            encode_v3, detokenize)

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Chat with a trained Office character")
parser.add_argument('--checkpoint', type=str, default='trained_weights/best_model_v3.pt',
                    help="Path to saved model checkpoint (v1, v2 or v3)")
parser.add_argument('--data', type=str, default='input_texts/the_office.txt',
                    help="Training text — only needed to rebuild the vocabulary for v1/v2 checkpoints")
parser.add_argument('--character', type=str, default=None,
                    help="Character to talk to (e.g. MICHAEL); asked interactively if omitted")
parser.add_argument('--you-are', type=str, default='GUEST',
                    help="Speaker tag used for your lines (v3 only), e.g. JIM or GUEST")
parser.add_argument('--message', action='append', default=None,
                    help="Scripted message; repeat for a multi-turn conversation. Skips interactive mode")
parser.add_argument('--temperature', type=float, default=0.8)
parser.add_argument('--top-k', type=int, default=50)
parser.add_argument('--top-p', type=float, default=0.92, help="Nucleus sampling threshold (v3 only)")
parser.add_argument('--repetition-penalty', type=float, default=1.15, help="v3 only; 1.0 disables")
parser.add_argument('--max-tokens', type=int, default=80)
parser.add_argument('--seed', type=int, default=None)
args = parser.parse_args()

if args.seed is not None:
    torch.manual_seed(args.seed)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def display_name(tag):
    return tag.strip('<>').replace('_', ' ').title()


# ---------------------------------------------------------------------------
# v3: shared tokenizer stored in the checkpoint, conversation history kept
# ---------------------------------------------------------------------------
class V3Chat:
    def __init__(self, checkpoint):
        from models.transformer_v2 import TransformerLM
        tok = checkpoint['tokenizer']
        self.model = TransformerLM(checkpoint['config']).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.block_size = checkpoint['config'].block_size
        self.itos = tok['vocab']
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.casing = tok['casing']
        self.speakers = tok['speakers']
        self.speaker_ids = [self.stoi[s] for s in self.speakers]
        self.history = []
        self.you = self.character = None

    def set_roles(self, character_tag, you_tag):
        if you_tag not in self.speakers or you_tag == character_tag:
            you_tag = '<GUEST>' if character_tag != '<GUEST>' else '<JIM>'
        self.character, self.you = character_tag, you_tag
        self.history = []

    def reply(self, message):
        ids = (self.history + [self.stoi[self.you]] + encode_v3(message, self.stoi)
               + [self.stoi[self.character]])
        ids = ids[-self.block_size:]
        context = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            output = self.model.generate(
                idx=context,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                ban_token_ids=[0],                      # never emit <UNK>
                stop_token_ids=self.speaker_ids,        # reply ends at the next speaker
            )

        new_ids = output[0, len(ids):].tolist()
        if new_ids and new_ids[-1] in self.speaker_ids:
            new_ids = new_ids[:-1]
        self.history = ids + new_ids
        return detokenize([self.itos[i] for i in new_ids], self.casing)


# ---------------------------------------------------------------------------
# v1 / v2: legacy whitespace tokenizer, vocabulary rebuilt from --data
# ---------------------------------------------------------------------------
class LegacyChat:
    def __init__(self, checkpoint):
        is_v2 = 'config' in checkpoint
        if is_v2:
            import dataclasses
            train_args = dataclasses.asdict(checkpoint['config'])
        else:
            train_args = checkpoint.get('args', {})

        print("Rebuilding vocabulary...")
        with open(args.data, 'r', encoding='utf-8') as f:
            text = f.read()
        text = prune_speakers(text, top_k_speakers=train_args.get('top_k_speakers', 30))
        _, self.stoi, self.itos, self.speakers = build_vocab(text, min_freq=train_args.get('min_freq', 10))
        self.special_tokens = ['<UNK>', '<YOU>'] + self.speakers

        saved_vocab = checkpoint.get('vocab_words')
        if saved_vocab is not None and len(saved_vocab) != len(self.itos):
            raise SystemExit(f"Checkpoint vocab ({len(saved_vocab)}) != rebuilt vocab ({len(self.itos)}); "
                             "use the same --data file the model was trained on")

        if is_v2:
            from models.transformer_v2 import TransformerLM
            self.model = TransformerLM(checkpoint['config']).to(device)
        else:
            from models.transformer_v1 import TransformerLM
            self.model = TransformerLM(
                vocab_size=len(self.itos),
                n_embd=train_args.get('n_embd', 256),
                block_size=train_args.get('block_size', 64),
                n_head=train_args.get('n_head', 4),
                n_layer=train_args.get('n_layer', 4),
                dropout=train_args.get('dropout', 0.2),
            ).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.character = None

    def set_roles(self, character_tag, you_tag):
        self.character = character_tag

    def reply(self, message):
        prompt = f"<MICHAEL> I am the best boss in the world <YOU> {message} {self.character}"
        input_ids = encode(prompt, self.stoi) or [0]
        context = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            output = self.model.generate(idx=context, max_new_tokens=args.max_tokens,
                                         temperature=args.temperature, top_k=args.top_k,
                                         ban_token_ids=[self.stoi.get('<UNK>', 0)])
        raw_reply = decode(output[0][len(input_ids):].tolist(), self.itos)
        for tag in self.special_tokens:
            if tag in raw_reply:
                raw_reply = raw_reply.split(tag)[0]
        reply = raw_reply.strip()
        return reply[0].upper() + reply[1:] if reply else reply


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
# weights_only=False: checkpoints pickle a GPTConfig dataclass, which the safe
# loader (default since torch 2.6) rejects. Only load checkpoints you trust.
checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
chat = V3Chat(checkpoint) if 'tokenizer' in checkpoint else LegacyChat(checkpoint)

# ---------------------------------------------------------------------------
# Character selection
# ---------------------------------------------------------------------------
interactive = not args.message
if interactive:
    print("\n" + "=" * 50)
    print("DUNDER MIFFLIN AI INITIALIZED. Type 'quit' to exit.")
    print("=" * 50 + "\n")
    main_cast = [s.strip('<>') for s in chat.speakers if s != '<GUEST>']
    print("Characters:", ", ".join(main_cast) + "\n")

target_char = args.character or input("Who do you want to talk to? (e.g. MICHAEL): ")
target_tag = f"<{target_char.strip().upper().replace(' ', '_')}>"
if target_tag not in chat.speakers:
    print(f"[{target_char} not found. Defaulting to MICHAEL.]")
    target_tag = "<MICHAEL>"

chat.set_roles(target_tag, f"<{args.you_are.strip().upper()}>")
you_name = display_name(chat.you) if getattr(chat, 'you', None) else "You"
char_name = display_name(target_tag)

# ---------------------------------------------------------------------------
# Chat loop
# ---------------------------------------------------------------------------
if interactive:
    print(f"\n--- Chatting with {char_name} ---\n")
    while True:
        try:
            user_input = input(f"{you_name}: ")
        except EOFError:
            break
        if user_input.lower() in ('quit', 'exit'):
            break
        print(f"{char_name}: {chat.reply(user_input)}\n")
else:
    for message in args.message:
        print(f"{you_name}: {message}")
        print(f"{char_name}: {chat.reply(message)}\n")
