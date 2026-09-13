# NanoGPT Experiments — The Office Dialogue Model

A word-level adaptation of Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT), rebuilt from scratch and trained on the complete dialogue of *The Office (US)* to hold in-character, multi-turn conversations.

nanoGPT is a minimal GPT-2-style character/BPE language model. This project keeps its core recipe (decoder-only transformer, causal self-attention, pre-LayerNorm blocks, weight tying, AdamW with cosine LR decay) and adapts it to a different problem: **modelling a TV script as a conversation between named speakers**, with a speaker-aware word-level tokenizer and an interactive chat front end.

The project developed in three iterations:

| Iteration | Focus | Files |
|---|---|---|
| **v1** | First-principles GPT in the style of Karpathy's "Let's build GPT" lecture | `models/transformer_v1.py`, `training_scripts/train_v1.py` |
| **v2** | Refactor aligned with nanoGPT's `model.py`; every change annotated `[NEW]` / `[CHANGED]` | `models/transformer_v2.py`, `training_scripts/train_v2.py` |
| **v3** | **Output quality**: new tokenizer, bigger context and model, better sampling, multi-turn chat | `data/tokenizer.py` (v3 section), `training_scripts/train_v3.py`, `generate.py` |

---

## What was accomplished

### Adaptations of nanoGPT

| Area | nanoGPT | This project |
|---|---|---|
| Data | Shakespeare / OpenWebText | *The Office* scripts, one `<SPEAKER> line` per row |
| Tokenization | Characters or GPT-2 BPE | Custom **word-level** vocab: punctuation split into its own tokens, lowercase words with casing restored on decode, frequency threshold with `<UNK>` |
| Speakers | — | Each character is a single `<NAME>` token; top-*k* speakers kept, the rest collapsed into `<GUEST>` |
| Inference | Unconditional sampling | **Interactive multi-turn chat**: you speak as one character, the model replies as another, and replies stop at the next speaker tag |
| Hardware | CUDA-first | Runs on Apple Silicon (`mps`), CUDA, or CPU |

### Carried over from nanoGPT (v2 onward)

- Fused Q/K/V projection (`c_attn`) and batched multi-head attention via `F.scaled_dot_product_attention` (flash attention)
- `GPTConfig` dataclass saved inside the checkpoint so the model can be rebuilt exactly
- Full `_init_weights` pass plus scaled init (`0.02 / sqrt(2·n_layer)`) on residual projections
- `configure_optimizers()`: weight decay only on ≥2-D tensors; AdamW `betas=(0.9, 0.95)`
- Bias-free Linear/LayerNorm, embedding dropout, `ignore_index=-1` in the loss, and nanoGPT naming (`wte`, `wpe`, `h`, `ln_f`)
- Weight tying, gradient clipping at 1.0, warmup followed by cosine LR decay, best-val-loss checkpointing

### v3: diagnosing and fixing the output quality

The v2 model reached a reasonable validation loss, but its chat replies were mostly word salad. Looking at the pipeline turned up six concrete causes:

| Problem in v1/v2 | Effect | v3 fix |
|---|---|---|
| Whitespace-only tokenization, so `office`, `office.` and `office,` were different words | **12.7% of all training tokens were `<UNK>`** | Punctuation split into separate tokens; `<UNK>` falls to **3.2%** |
| Training used raw text, but chat prompts were lowercased and lemmatized | Prompt words mapped to different ids, or to `<UNK>` | One `tokenize()` shared by training and chat |
| The chat prompt used a `<YOU>` token that never appears in the training data | The model was conditioned on an untrained, random embedding | You speak as a real speaker tag (`--you-are JIM`, default `<GUEST>`) |
| `<UNK>` counted in the loss | Capacity spent learning to predict `<UNK>` | `<UNK>` targets masked out (`ignore_index=-1`) |
| 64-token context, and every reply started from a fixed prompt with no memory | The model saw about 2 lines and forgot the conversation | 128-token context; chat history is kept between turns |
| Sampling always ran `max_new_tokens`; no repetition control; all-lowercase output | Rambling replies, loops, hard to read | Stops at the next speaker tag; nucleus (top-p) sampling; repetition penalty; casing and punctuation restored |

Training changes in `train_v3.py`: a 6-layer / 6-head / 384-d model (12.7M parameters), lr 1e-3 decaying to 1e-4 (nanoGPT's small-data recipe), evaluation every 250 steps, and early stopping.

### Results

| | v2 | v3 |
|---|---|---|
| Vocabulary | 4,213 | 5,398 |
| Training tokens | 625,732 | 778,865 |
| `<UNK>` share of tokens | 12.7% | **3.2%** |
| Parameters | 4.2M | 12.7M |
| Context (tokens) | 64 | 128 |
| Best validation loss | 4.43 (step 3,000) | **4.22** (step 1,500, perplexity 68) |
| Training time (Apple M-series GPU) | — | 29 min (early stopped at step 3,000) |

The two losses are not directly comparable: v3 uses a different vocabulary, counts punctuation as tokens, and excludes `<UNK>`. The conversations below are the better comparison. Both models get the same three messages (you speak as Jim, the model replies as Dwight), with `--seed 1`. The samples are not cherry-picked.

**v2** (legacy chat defaults: temperature 0.6, top-k 20; an empty reply means the model immediately jumped to another speaker tag)
```
You: Dwight, have you seen my stapler?
Dwight:

You: Why is it in jello again?
Dwight: In the break room.

You: Okay, I'm telling Michael.
Dwight:
```

**v3**
```
Jim: Dwight, have you seen my stapler?
Dwight: Oh, man. No, no. I'm sorry. You can't let me do this, I'll take a picture. I have the chance that you have... I don't even know.

Jim: Why is it in jello again?
Dwight: Jim, Pam. Pam, stop!

Jim: Okay, I'm telling Michael.
Dwight: Jim, I don't know. I did, Jim!
```

Another v3 example (you speak as Pam, the model replies as Michael, seed 2):
```
Pam: Michael, corporate is on the phone.
Michael: Hi, Michael Scott.

Pam: They want to know about the budget.
Michael: Oh, hey, you know what? I think you should be doing this for a while. It's all about your time.
```

Both models still overfit this small dataset. In the v3 run, validation loss bottomed out at step 1,500 (4.22) while training loss kept falling, from 3.34 to 2.52 by step 3,000. Both scripts keep the best checkpoint, and `train_v3.py` stops early. More dropout or more data is the obvious next lever.

### Key findings

- **Check the `<UNK>` rate before tuning the model.** Splitting punctuation from words did more for output quality than any architecture change, taking `<UNK>` from 1 in 8 tokens to 1 in 30.
- **Train and inference tokenization must be identical.** A mismatch fails silently: the model still runs, just on the wrong ids.
- **Never condition on a token the model hasn't seen.** Replacing the untrained `<YOU>` tag with a real speaker tag made replies respond to the prompt.
- Pruning to the top 30 speakers and collapsing the rest into `<GUEST>` kept speaker tags well trained.
- Tokens per vocabulary word was a good early warning for overfitting; keeping it above ~100 kept the model generalizing.
- Stopping at the next speaker tag, plus a mild repetition penalty, gave the cleanest turn-taking.

---

## Project structure

```
nanogpt-experiments/
├── data/
│   ├── __init__.py
│   ├── prepare_office.py      # Kaggle CSV → tagged text file
│   └── tokenizer.py           # Speaker pruning; legacy (v1/v2) and v3 tokenizers
├── models/
│   ├── __init__.py
│   ├── transformer_v1.py      # From-scratch GPT (lecture-style)
│   └── transformer_v2.py      # nanoGPT-aligned model (used by v2 and v3)
├── training_scripts/
│   ├── train_v1.py            # Trains transformer_v1 → best_model.pt
│   ├── train_v2.py            # Trains transformer_v2 → best_model_v2.pt
│   └── train_v3.py            # v3 tokenizer + training recipe → best_model_v3.pt
├── generate.py                # Chat CLI (v3 multi-turn; v1/v2 legacy)
├── requirements.txt
├── LICENSE
└── README.md
```

## What each script does

### `data/prepare_office.py`
A one-off data preparation CLI. It reads `The-Office-Lines-V4.csv` from Kaggle with pandas and drops rows with no speaker or line. Each speaker name is normalised to an uppercase tag (`Michael Scott` → `<MICHAEL_SCOTT>`, non-alphanumerics removed), and every line is written as `<SPEAKER> dialogue` to a plain text file (default `input_texts/the_office.txt`). This file is the only input the rest of the pipeline needs.

### `data/tokenizer.py`
The tokenizers shared by training and chat.
- **`prune_speakers(text, top_k_speakers=30, tag_pattern=...)`** counts every `<TAG>`, keeps the *k* most frequent speakers, and rewrites the rest to `<GUEST>`. All versions use it.
- **Legacy tokenizer (v1/v2):**
  - `build_vocab()` builds a vocabulary over whitespace-split words: `<UNK>`, `<YOU>`, speaker tags, then words seen at least `min_freq` times.
  - `normalize()` lowercases, strips punctuation and lemmatizes with NLTK WordNet, which loads lazily on first use.
  - `encode()` and `decode()` convert between text and token ids.
- **v3 tokenizer:**
  - `tokenize(text, keep_case=False)` splits text into speaker tags, words (keeping contractions and hyphenated words together) and individual punctuation tokens, and fixes mis-encoded apostrophes from the CSV.
  - `build_casing(tokens)` learns each word's usual mid-sentence spelling (`jim` → `Jim`, `tv` → `TV`), skipping sentence-initial words.
  - `build_vocab_v3(tokens, min_freq=5)` returns `<UNK>` (id 0), speaker tags, then frequent words.
  - `encode_v3(text, stoi)` converts text to token ids.
  - `detokenize(tokens, casing)` rebuilds readable text: restores casing, capitalizes sentence starts and "I", attaches punctuation and quotes, and renders speaker tags as `Name:` lines.

### `models/transformer_v1.py`
The first from-scratch GPT implementation.
- `Head` is a single causal attention head (separate key/query/value linears plus `scaled_dot_product_attention`).
- `MultiHeadAttention` concatenates *n* heads, then applies a projection and dropout.
- `FeedForward` is a 4× GELU MLP. `Block` is a pre-LayerNorm residual block.
- `TransformerLM` combines token and positional embeddings, `nn.Sequential` blocks, a final LayerNorm and a tied `lm_head`, with scaled residual init. `generate()` samples with temperature, top-k and banned token ids.

### `models/transformer_v2.py`
The nanoGPT-aligned rewrite of v1, used by both v2 and v3. Comments mark each `[NEW]` / `[CHANGED]` item and show the code it replaced.
- `GPTConfig` is a dataclass of all architecture hyperparameters and is stored in checkpoints.
- `CausalSelfAttention` uses one fused `c_attn` projection, reshapes to `(B, n_head, T, head_size)`, and runs all heads in a single flash-attention call, with attention and residual dropout.
- `FeedForward`, `Block`: same roles as in v1, with a configurable bias.
- `TransformerLM` is laid out as an `nn.ModuleDict` (`wte`, `wpe`, `drop`, `h`, `ln_f`) plus a tied `lm_head`. It also has:
  - `_init_weights`
  - `get_num_params()`
  - a `block_size` assertion
  - a loss with `ignore_index=-1`
  - `configure_optimizers()` with separate decay/no-decay groups
  - `generate()` with temperature, top-k, **top-p**, **repetition penalty**, banned tokens, and **stop tokens**

### `training_scripts/train_v1.py`
The training loop for `transformer_v1`.
1. Picks a device (`mps` → `cuda` → `cpu`) and sets the seed.
2. Loads the text, prunes speakers, builds the legacy vocabulary, and encodes the corpus.
3. Splits the data 90/10 into train and validation sets and samples random `block_size` windows.
4. Trains with AdamW (weight decay 0.01 on all parameters), warmup followed by cosine decay, and gradient clipping.
5. Every `--eval-interval` steps, estimates loss and saves `best_model.pt` (state dict, vocab, CLI args) whenever validation loss improves.

### `training_scripts/train_v2.py`
The same pipeline using `transformer_v2`. It builds a `GPTConfig` from the CLI flags, creates the optimizer with `model.configure_optimizers()` (weight decay 0.1 on matrices only), and saves `best_model_v2.pt` containing the `config`. The bottom of the file keeps the log from the reference training run.

### `training_scripts/train_v3.py`
The output-quality iteration, built on `transformer_v2`.
1. Tokenizes the corpus with the v3 tokenizer, learns the casing table, and builds the vocabulary (`min_freq=5`). It prints the vocabulary size, tokens per word and `<UNK>` rate.
2. Samples 128-token windows and sets `<UNK>` targets to `-1` so they are ignored by the loss.
3. Trains the 6-layer / 6-head / 384-d model with lr 1e-3 (200-step warmup, cosine decay to 1e-4), dropout 0.25, weight decay 0.1 and gradient clipping.
4. Every 250 steps, averages loss over 100 batches per split and logs perplexity, learning rate and elapsed time. It saves `best_model_v3.pt` (weights, `config`, vocabulary, casing table, speakers, training info) on each new best, and stops early after `--patience` evaluations without improvement.

### `generate.py`
The chat CLI.
1. Loads a checkpoint and chooses the pipeline from its contents: a `tokenizer` means v3, a `config` means v2, `args` means v1. v1/v2 checkpoints rebuild their vocabulary from `--data` and check that its size matches.
2. Picks the character to talk to (`--character`, or asks interactively) and your own speaker tag (`--you-are`, v3 only).
3. **v3:** each turn appends `<YOU_TAG> message <CHARACTER_TAG>` to the running history (the last 128 tokens) and samples a reply. Sampling uses temperature, top-k, top-p and a repetition penalty, bans `<UNK>`, stops at the next speaker tag, and is detokenized into readable text.
4. **v1/v2:** the original single-turn behaviour (fixed prompt with `<YOU>`, reply cut at the next tag).
5. Runs interactively (type `quit` to exit), or non-interactively with one or more `--message` flags, printing a transcript. `--seed` makes runs reproducible.

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
# recommended: v3
python training_scripts/train_v3.py --data input_texts/the_office.txt

# earlier iterations
python training_scripts/train_v2.py --data input_texts/the_office.txt
python training_scripts/train_v1.py --data input_texts/the_office.txt
```

Checkpoints go to `trained_weights/`. Every hyperparameter is a CLI flag; run any script with `--help` for the full list. On a smaller machine, try `--n-layer 4 --n-head 4 --n-embd 256`.

### 4. Chat

```bash
python generate.py --checkpoint trained_weights/best_model_v3.pt --you-are JIM
```

Or script a conversation:

```bash
python generate.py --character DWIGHT --you-are JIM --seed 7 --message "Where is my stapler?" --message "Why is it in jello?"
```

Sampling flags:
- `--temperature` (default 0.8): lower is safer, higher is more varied.
- `--top-k` (50) and `--top-p` (0.92): limit sampling to the most likely words.
- `--repetition-penalty` (1.15): 1.0 turns it off.
- `--max-tokens` (80): the longest a reply can be.

---

## Known limitations

- **Small data.** Under 1M tokens is small for a 12.7M-parameter model. Validation loss bottoms out early, and replies are recognisably in each character's voice and on topic for a line or two, but not reliably coherent.
- **Rare words are lost.** Words seen fewer than 5 times become `<UNK>` and are never generated. A subword (BPE) tokenizer would fix this, at the cost of the simple word-level design.
- **Speaker identity is a single token.** Guest characters all share `<GUEST>`, so the model can't tell them apart.
- **The validation split is the end of the series.** The last 10% of tokens is from late seasons, so the validation set partly measures how well the model handles later storylines.
- **Checkpoints are pickles.** `generate.py` loads them with `weights_only=False` (needed for `GPTConfig`), so only load checkpoints you trained yourself.

## Acknowledgements

- [nanoGPT](https://github.com/karpathy/nanoGPT) and the "Let's build GPT: from scratch" lecture by Andrej Karpathy, the basis for the model architecture and training recipe.
- [The Office US Complete Dialogue](https://www.kaggle.com/datasets/nasirkhalid24/the-office-us-complete-dialoguetranscript) dataset on Kaggle. *The Office* is the property of its respective rights holders; no transcript data is distributed here.

## License

MIT; see [LICENSE](LICENSE). Portions adapted from nanoGPT (MIT, © Andrej Karpathy).
