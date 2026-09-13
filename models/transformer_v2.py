"""
transformer_v2.py — nanoGPT-aligned transformer implementation.

Differences from transformer_v1.py (models/transformer_v1.py) are marked with:
    # [CHANGED] — something that existed before but was modified
    # [NEW]     — something that didn't exist before
    # [REMOVED] — something that was removed and why
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# [NEW] GPTConfig dataclass
# ---------------------------------------------------------------------------
# BEFORE: hyperparameters lived in argparse and were passed individually to
#         TransformerLM(vocab_size=x, n_embd=y, block_size=z, ...)
#         This meant generate.py had to manually reconstruct the same args,
#         which caused size-mismatch errors when they drifted out of sync.
#
# AFTER:  a single config object travels with the model everywhere.
#         train_v2.py builds it from argparse, saves it in the checkpoint,
#         and generate.py reconstructs the model from it with no guesswork.
# ---------------------------------------------------------------------------
@dataclass
class GPTConfig:
    block_size: int = 64
    vocab_size: int = 4213
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.2
    bias: bool = False      # GPT-2 uses no bias in linears — slightly cleaner


# ---------------------------------------------------------------------------
# [CHANGED] Attention — from looped Head modules to single fused operation
# ---------------------------------------------------------------------------
# BEFORE: separate Head class, n_head individual forward passes:
#
#   class Head(nn.Module):
#       def __init__(self, n_embd, head_size, dropout): ...
#       def forward(self, x):
#           k, q, v = self.key(x), self.query(x), self.value(x)
#           return F.scaled_dot_product_attention(q, k, v, ...)
#
#   class MultiHeadAttention(nn.Module):
#       self.heads = nn.ModuleList([Head(...) for _ in range(num_heads)])
#       def forward(self, x):
#           return torch.cat([h(x) for h in self.heads], dim=-1)  # n_head loops
#
# AFTER:  one Linear produces Q, K, V for ALL heads at once, then we split
#         and reshape. This is one GPU kernel instead of n_head kernels.
#         On MPS this is ~2-4x faster for the attention step.
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # [CHANGED] three separate key/query/value linears → one fused projection
        # BEFORE: self.key   = nn.Linear(n_embd, head_size, bias=False)
        #         self.query = nn.Linear(n_embd, head_size, bias=False)
        #         self.value = nn.Linear(n_embd, head_size, bias=False)
        # AFTER:  one matrix that produces all three at once
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # output projection (same as before, just renamed to match nanoGPT)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.shape

        # [CHANGED] single matrix multiply → split into Q, K, V
        # BEFORE: k = self.key(x); q = self.query(x); v = self.value(x)
        # AFTER:  one matmul, then split along the last dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # [CHANGED] reshape into (B, n_head, T, head_size) for batched attention
        # BEFORE: this reshape happened inside each Head's forward, separately
        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)

        # flash attention — same call as before, now covers all heads at once
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
        )

        # [CHANGED] reassemble heads — previously torch.cat([h(x) for h in self.heads])
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.c_proj(out))


# ---------------------------------------------------------------------------
# FeedForward — same as before, minor rename to match nanoGPT conventions
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            # [CHANGED] bias=config.bias instead of hardcoded True
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Block — same structure, now takes config instead of individual args
# ---------------------------------------------------------------------------
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        # [CHANGED] MultiHeadAttention(n_embd, n_head, head_size, dropout)
        #        →  CausalSelfAttention(config)
        self.sa = CausalSelfAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.ln2 = nn.LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# TransformerLM — main model, now aligned with nanoGPT
# ---------------------------------------------------------------------------
class TransformerLM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config    # [NEW] store config on the model for later access

        self.transformer = nn.ModuleDict({
            # [CHANGED] renamed to match nanoGPT: token_embedding → wte, position → wpe
            # purely cosmetic but makes it easier to cross-reference nanoGPT code
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'drop': nn.Dropout(config.dropout),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd, bias=config.bias),
        })

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying — same as before
        self.transformer.wte.weight = self.lm_head.weight

        # [CHANGED] weight initialization — from a single residual-only loop
        #           to a full _init_weights pass over every module
        # BEFORE:
        #   for pn, p in self.named_parameters():
        #       if pn.endswith('proj.weight'):
        #           torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))
        #
        # AFTER:  self.apply() walks every submodule recursively and sets
        #         sensible defaults for Linear and Embedding layers first,
        #         then the residual override runs on top.
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"Model parameters: {self.get_num_params():,}")

    # [NEW] full weight initializer — handles every layer type explicitly
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # [NEW] parameter count utility — useful for comparing models
    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # subtract position embedding since it doesn't contribute to generation capacity
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.shape
        assert T <= self.config.block_size, \
            f"Sequence length {T} exceeds block_size {self.config.block_size}"

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(torch.arange(T, device=device))
        x = self.transformer.drop(tok_emb + pos_emb)

        # [CHANGED] iterating over nn.ModuleList instead of nn.Sequential
        # BEFORE: self.blocks = nn.Sequential(...)  → x = self.blocks(x)
        # AFTER:  self.transformer.h is a ModuleList → explicit loop
        # ModuleList is preferred when you might want per-block outputs later
        # (e.g. for interpretability / activation analysis)
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        # [CHANGED] ignore_index=-1 (as in nanoGPT) — train_v3.py sets <UNK>
        # targets to -1 so the model is never rewarded for predicting <UNK>
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T), ignore_index=-1)
        return logits, loss

    # [NEW] configure_optimizers — separates decay and no-decay parameter groups
    # BEFORE: optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.01)
    #         This incorrectly applied weight decay to biases and LayerNorm weights,
    #         which actively hurts training. Weight decay should only apply to
    #         weight matrices (dim >= 2), never to biases or LN scale/shift params.
    def configure_optimizers(self, weight_decay, lr, device_type):
        decay_params = []
        nodecay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                # weight matrices — apply decay
                decay_params.append(param)
            else:
                # biases, layernorm weights (1D) — no decay
                nodecay_params.append(param)

        optim_groups = [
            {'params': decay_params,   'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]

        # [CHANGED] betas=(0.9, 0.95) instead of default (0.9, 0.999)
        # nanoGPT uses 0.95 for beta2 — better for language model training
        # as it gives more weight to recent gradient history
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95))
        return optimizer

    # [CHANGED] generate() gained three optional sampling controls (defaults keep
    #           the old behaviour):
    #   repetition_penalty — CTRL-style penalty on tokens already generated in
    #                        this call, breaks "no no no no" loops
    #   top_p              — nucleus sampling: keep the smallest set of tokens
    #                        whose probability mass exceeds top_p
    #   stop_token_ids     — stop as soon as one is sampled (e.g. the next
    #                        speaker tag), instead of always running max_new_tokens
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40, ban_token_ids=None,
                 top_p=None, repetition_penalty=1.0, stop_token_ids=None):
        prompt_len = idx.size(1)
        stop = set(stop_token_ids or [])

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if ban_token_ids:
                for tid in ban_token_ids:
                    logits[:, tid] = -float('Inf')

            if repetition_penalty != 1.0 and idx.size(1) > prompt_len:
                seen = torch.unique(idx[:, prompt_len:])
                scores = logits[:, seen]
                logits[:, seen] = torch.where(scores > 0, scores / repetition_penalty,
                                              scores * repetition_penalty)

            logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                # drop a token if the mass *before* it already exceeds top_p
                remove = (torch.cumsum(sorted_probs, dim=-1) - sorted_probs) > top_p
                sorted_logits[remove] = -float('Inf')
                logits = torch.full_like(logits, -float('Inf')).scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            if stop and idx.size(0) == 1 and idx_next.item() in stop:
                break

        return idx


