# NanoGPT Experiments — The Office Dialogue Model

A word-level adaptation of Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT), rebuilt from scratch and trained on the complete dialogue of *The Office (US)* to generate in-character, multi-speaker conversation.

nanoGPT is a minimal GPT-2-style character/BPE language model. This project keeps its core recipe (decoder-only transformer, causal self-attention, pre-LayerNorm blocks, weight tying, AdamW with cosine LR decay) and adapts it to a different problem: **modelling a TV script as a conversation between named speakers**, with a word-level tokenizer, speaker tokens, and an interactive chat front end.

The code is organised as two iterations:

- **v1** is a first-principles implementation in the style of Karpathy's "Let's build GPT" lecture (per-head attention modules, arguments passed individually).
- **v2** refactors v1 to line up with nanoGPT's `model.py`. Every change is annotated in the source with `[NEW]` / `[CHANGED]` comments, so the two files can be read side by side as a diff.

---

## What was accomplished

### Adaptations of nanoGPT

| Area | nanoGPT | This project |
|---|---|---|
| Data | Shakespeare / OpenWebText | *The Office* scripts, one `<SPEAKER> line` per row |
| Tokenization | Characters or GPT-2 BPE | Custom **word-level** vocab with frequency threshold (`min_freq`) and `<UNK>` |
| Speakers | — | Each character is a single `<NAME>` token; top-*k* speakers kept, the rest collapsed into `<GUEST>` |
| Inference | Unconditional sampling | **Interactive chat**: pick a character; the model replies as them; `<UNK>` is banned and output stops at the next speaker tag |
| Hardware | CUDA-first | Runs on Apple Silicon (`mps`), CUDA, or CPU |

### Carried over from nanoGPT (implemented in v2)

- Fused Q/K/V projection (`c_attn`) and batched multi-head attention via `F.scaled_dot_product_attention` (flash attention)
- `GPTConfig` dataclass saved inside the checkpoint so the model can be rebuilt exactly
- Full `_init_weights` pass plus scaled init (`0.02 / sqrt(2·n_layer)`) on residual projections
- `configure_optimizers()`: weight decay only on ≥2-D tensors (none on biases or LayerNorm); AdamW `betas=(0.9, 0.95)`
- Bias-free Linear/LayerNorm option, embedding dropout, and nanoGPT naming (`wte`, `wpe`, `h`, `ln_f`)
- Shared with v1: weight tying, gradient clipping at 1.0, linear warmup followed by cosine LR decay, best-val-loss checkpointing

### Results (v2, default settings, Apple M-series GPU)

| Metric | Value |
|---|---|
| Vocabulary | 4,213 words (incl. speaker tags) |
| Training tokens | 625,732 (≈148 tokens per vocab word) |
| Parameters | 4.23 M (4 layers, 4 heads, 256-d, context 64) |
| Best validation loss | **4.43** at step 3,000 (from 8.38 at init) |

After step 3,000 the training loss kept falling (3.12 by step 9,000) while validation loss rose to about 4.53. That is classic overfitting on a dataset of this size, and the checkpointing logic keeps the step-3,000 model.

### Key findings

- Word-level tokenization removed the character-level "gibberish word" problem, but noisy dialogue data needed aggressive frequency filtering (`min_freq=10`).
- Pruning to the top 30 speakers and collapsing the rest into `<GUEST>` significantly reduced vocabulary bloat.
- The tokens-per-vocab-word ratio was the strongest predictor of overfitting; keeping it above ~100 kept the model generalizing.
- Banning `<UNK>` at sampling time and cutting the reply at the next speaker tag gave the cleanest multi-turn dialogue.

---

## Project structure

```
nanogpt-experiments/
├── data/
│   ├── __init__.py
│   ├── prepare_office.py      # Kaggle CSV → tagged text file
│   └── tokenizer.py           # Speaker pruning, vocabulary, encode/decode
├── models/
│   ├── __init__.py
│   ├── transformer_v1.py      # From-scratch GPT (lecture-style)
│   └── transformer_v2.py      # nanoGPT-aligned refactor, annotated vs v1
├── training_scripts/
│   ├── train_v1.py            # Trains transformer_v1 → best_model.pt
│   └── train_v2.py            # Trains transformer_v2 → best_model_v2.pt
├── generate.py                # Interactive character chat (v1 or v2 checkpoints)
├── requirements.txt
├── LICENSE
└── README.md
```

## What each script does

### `data/prepare_office.py`
A one-off data preparation CLI. It reads `The-Office-Lines-V4.csv` from Kaggle with pandas and drops rows with no speaker or line. Each speaker name is normalised to an uppercase tag (`Michael Scott` → `<MICHAEL_SCOTT>`, non-alphanumerics removed), and every line is written as `<SPEAKER> dialogue` to a plain text file (default `input_texts/the_office.txt`). This file is the only input the rest of the pipeline needs.

### `data/tokenizer.py`
The shared word-level tokenizer used by training and generation. It downloads the NLTK WordNet data on import.
- `prune_speakers(text, top_k_speakers=30)` counts every `<TAG>`, keeps the *k* most frequent speakers, and rewrites all other tags to `<GUEST>`.
- `build_vocab(text, min_freq=10)` builds the vocabulary: `<UNK>` (id 0), `<YOU>` (the human user in chat), all speaker tags, then every whitespace-separated word seen at least `min_freq` times, sorted. It returns `(vocab_words, stoi, itos, speaker_tokens)`.
- `normalize(text)` lowercases text, strips `[stage directions]` and punctuation, and lemmatizes words with WordNet while protecting speaker tags from being altered.
- `encode(text, stoi, skip_normalize=False)` and `decode(ids, itos)` convert between text and token ids. Unknown words map to `<UNK>`.

### `models/transformer_v1.py`
The first from-scratch GPT implementation.
- `Head` is a single causal attention head (separate key/query/value linears plus `scaled_dot_product_attention`).
- `MultiHeadAttention` runs *n* `Head`s, concatenates their outputs, then applies a projection and dropout.
- `FeedForward` is a 4× MLP with GELU. `Block` is a pre-LayerNorm residual block (attention followed by MLP).
- `TransformerLM` combines token and positional embeddings, `nn.Sequential` blocks, a final LayerNorm and a tied `lm_head`, with scaled init on residual projections. Its `forward()` returns logits and the cross-entropy loss. Its `generate()` samples autoregressively with temperature, top-k, and optional banned token ids.

### `models/transformer_v2.py`
The nanoGPT-aligned rewrite of v1. It exposes the same `forward`/`generate` interface, but models are built from a `GPTConfig`.
- `GPTConfig` is a dataclass of all architecture hyperparameters and is stored in checkpoints.
- `CausalSelfAttention` uses one fused `c_attn` projection, reshapes to `(B, n_head, T, head_size)`, runs all heads in a single flash-attention call, and applies attention and residual dropout.
- `FeedForward`, `Block`: same roles as in v1, with a configurable bias.
- `TransformerLM` is laid out as an `nn.ModuleDict` (`wte`, `wpe`, `drop`, `h`, `ln_f`) plus a tied `lm_head`. It also has `_init_weights`, `get_num_params()` (which excludes position embeddings), a `block_size` assertion in `forward`, and `configure_optimizers()` with separate decay/no-decay parameter groups.
- Comments throughout mark each `[NEW]` / `[CHANGED]` item and show the v1 code it replaced.

### `training_scripts/train_v1.py`
The training loop for `transformer_v1`.
1. Picks a device (`mps` → `cuda` → `cpu`) and sets the random seed.
2. Loads the text file, prunes speakers, builds the vocabulary, encodes the whole corpus, and prints tokens per vocab word as an overfitting sanity check.
3. Splits the data 90/10 into train and validation sets and samples random `block_size` windows as batches.
4. Trains with AdamW (weight decay 0.01 on all parameters), 100-step linear warmup followed by cosine decay, and gradient clipping at 1.0.
5. Every `--eval-interval` steps, averages loss over 50 batches per split and saves `best_model.pt` (state dict, vocab, CLI args) whenever validation loss improves.

### `training_scripts/train_v2.py`
Same pipeline as v1, but uses `transformer_v2`. It builds a `GPTConfig` from the CLI flags, creates the optimizer with `model.configure_optimizers()` (default `--weight-decay 0.1`, decay on matrices only), and saves `best_model_v2.pt` containing the `config` object. The bottom of the file keeps the log from the full reference training run summarised in [Results](#results-v2-default-settings-apple-m-series-gpu).

### `generate.py`
An interactive chat CLI.
1. Loads a checkpoint and detects the architecture: a checkpoint with `config` uses v2, one with `args` uses v1.
2. Rebuilds the exact vocabulary by re-running `prune_speakers` and `build_vocab` on the same text file. Vocabularies are not stored separately, so `--data` must match the file used for training.
3. Lists the main characters and asks which one to talk to.
4. For each user message, builds a prompt `<MICHAEL> I am the best boss in the world <YOU> {message} <CHARACTER>`, samples up to `--max-tokens` with `--temperature` and `--top-k`, bans `<UNK>`, and cuts the reply at the next speaker tag.

Type `quit` or `exit` to leave.

---

## Usage

All commands are run from the repository root.

### 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare data

The transcript dataset is **not included** in this repository. Download [The Office US Complete Dialogue](https://www.kaggle.com/datasets/nasirkhalid24/the-office-us-complete-dialoguetranscript) from Kaggle, then run:

```bash
mkdir -p input_texts
python data/prepare_office.py --csv path/to/The-Office-Lines-V4.csv --output input_texts/the_office.txt
```

### 3. Train

```bash
# nanoGPT-aligned model (recommended)
python training_scripts/train_v2.py --data input_texts/the_office.txt

# original from-scratch model
python training_scripts/train_v1.py --data input_texts/the_office.txt
```

Checkpoints are written to `trained_weights/`. Every hyperparameter (`--n-layer`, `--n-head`, `--n-embd`, `--block-size`, `--batch-size`, `--lr`, `--dropout`, `--min-freq`, `--top-k-speakers`, …) is a CLI flag; run either script with `--help` for the full list.

### 4. Chat

```bash
python generate.py --checkpoint trained_weights/best_model_v2.pt --data input_texts/the_office.txt
```

Lower `--temperature` makes replies safer and more repetitive; higher values make them more varied. `--top-k` limits sampling to the *k* most likely words.

---

## Known limitations

- **Tokenization mismatch at inference.** Training encodes the raw text (`skip_normalize=True`, which keeps case and punctuation attached to words), but `generate.py` passes the user prompt through `normalize()` (lowercase + lemmatization). Some prompt words may therefore map to `<UNK>` or to a different token than expected.
- **Small model, small data.** About 626k tokens is small for a 4M-parameter model. Validation loss bottoms out early (step ~3k), and replies are stylistically "Office-like" rather than coherent.
- **Short context.** `block_size=64` words covers only a few lines of dialogue.
- **Checkpoints are pickles.** `generate.py` loads them with `weights_only=False` (needed for the `GPTConfig` object), so only load checkpoints you trained yourself.

## Acknowledgements

- [nanoGPT](https://github.com/karpathy/nanoGPT) and the "Let's build GPT: from scratch" lecture by Andrej Karpathy, the basis for the model architecture and training recipe.
- [The Office US Complete Dialogue](https://www.kaggle.com/datasets/nasirkhalid24/the-office-us-complete-dialoguetranscript) dataset on Kaggle. *The Office* is the property of its respective rights holders; no transcript data is distributed here.

## License

MIT; see [LICENSE](LICENSE). Portions adapted from nanoGPT (MIT, © Andrej Karpathy).
