#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
molecule.py
A single-file, async, continually-learning GPT organism. One dependency: numpy.

- Trains on nonames.txt (one sentence per line)
- Keeps SQLite memory (tiny chat loop)
- Maintains a bounded corpus reservoir (never bloats)
- Starts in char-level mode (fast boot)
- Gradually enables BPE *without* invalidating old weights (vocab only EXPANDS)
- Never forgets by never overwriting learned deltas: it only appends modules

In the beginning there was nonames.txt.
And it was good.
Mostly.
Sometimes cursed.
"""

import os
import math
import time
import json
import random
import asyncio
import sqlite3
import threading
from dataclasses import dataclass
from collections import Counter, defaultdict
import numpy as np

random.seed(42)  # And lo, determinism shall pretend to tame chaos.

# ============================================================
# 0) CONFIG — bend reality here (carefully, mortals)
# ============================================================

@dataclass
class Config:
    # data
    corpus_path: str = "nonames.txt"
    db_path: str = "memory.sqlite3"
    ckpt_path: str = "molecule_ckpt.json"
    max_corpus_lines: int = 8000
    max_line_chars: int = 240

    # continual learning trigger (smaller than Karpathy's vibe, but not stupidly small)
    min_new_chars_to_train: int = 480  # And lo, the minimum mass shall be reached.

    # model
    tie_embeddings: bool = True  # GPT-style weight tying (wte == lm_head)

    n_layer: int = 2
    n_embd: int = 72          # a small bump for "GPT-3-ish" flavor without melting CPUs
    n_head: int = 4
    block_size: int = 96

    # training
    warmup_steps: int = 1200
    micro_steps: int = 32
    learning_rate: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.99
    eps_adam: float = 1e-8
    grad_clip: float = 1.0    # And lo, the gradients shall not explode into the sun.
    freeze_base_after_warmup: bool = True
    batch_size: int = 4

    # deltas (LoRA-ish)
    delta_rank: int = 8
    max_delta_modules: int = 12
    delta_grow_prob: float = 0.08

    # generation
    temperature: float = 0.85
    top_k: int = 40
    top_p: float = 0.92
    min_p: float = 0.06         # GPT-3/4 style: filter tokens below min_p * max_prob
    typical_p: float = 0.95     # Typical sampling: prefer tokens with typical information content
    max_gen_tokens: int = 180
    min_gen_tokens: int = 16
    repetition_guard: int = 4

    # tokenizer evolution
    enable_bpe_after_chars: int = 25000  # corpus size threshold to begin learning merges
    bpe_num_merges: int = 384
    bpe_retrain_every_chars: int = 4000  # retrain merges when corpus changes enough

    # async
    train_tick_seconds: float = 0.25

    # hybrid attention heads: "content", "rrpram", or "hybrid"
    head_types: tuple = ("content", "content", "hybrid", "hybrid")
    hybrid_alpha_init: float = 0.5

    # gamma (personality fingerprint)
    gamma_sparsity_threshold: float = 0.01

    # noise immune system
    noise_drift_threshold: float = -0.1   # cosine < this = noise, rollback
    gamma_min_magnitude: float = 1e-6     # skip immune check when gamma direction is near-zero

    # syntropy tracker (mathematical self-awareness)
    syntropy_window: int = 8              # rolling window for syntropy trend
    field_deviation_ceiling: float = 12.0 # KL divergence above this = drifted too far
    field_deviation_floor: float = 0.1    # below this = not learning, just parroting
    syntropy_lr_boost: float = 1.3        # boost LR when syntropy is rising
    syntropy_lr_dampen: float = 0.6       # dampen LR when syntropy is falling
    syntropy_delta_grow_boost: float = 0.15  # higher delta grow prob when syntropy is good

    # entropy-adaptive temperature
    entropy_low: float = 0.5
    entropy_high: float = 1.5
    entropy_temp_boost: float = 1.2
    entropy_temp_focus: float = 0.8

    # corpus field
    corpus_gen_max_tokens: int = 120

    # quantum buffer
    qb_min_bytes: int = 1024
    qb_min_novelty: float = 0.15
    qb_cooldown_seconds: float = 60.0


CFG = Config()

# ============================================================
# 1) SQLITE MEMORY — and a small ghost shall remember
# ============================================================

def init_db(db_path: str):
    # And lo, a memory shall awaken in SQLite, because RAM is a liar.
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            role TEXT NOT NULL,
            text TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS corpus_events(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            added_chars INTEGER NOT NULL,
            note TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS growth(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            step INTEGER NOT NULL,
            vocab_size INTEGER NOT NULL,
            n_params INTEGER NOT NULL,
            n_deltas INTEGER NOT NULL,
            corpus_chars INTEGER NOT NULL,
            loss REAL,
            gamma_sparsity REAL,
            gamma_magnitude REAL,
            note TEXT
        )
    """)
    # And lo, the organism shall track not just what it is, but where it is going.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS syntropy_log(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            entropy_before REAL,
            entropy_after REAL,
            syntropy_delta REAL,
            field_deviation REAL,
            purpose_magnitude REAL,
            purpose_alignment REAL,
            action_taken TEXT,
            note TEXT
        )
    """)
    con.commit()
    return con

def db_add_message(con, role: str, text: str):
    con.execute("INSERT INTO messages(ts, role, text) VALUES(?,?,?)",
                (time.time(), role, text))
    con.commit()

def db_recent_messages(con, limit: int = 32):
    cur = con.cursor()
    cur.execute("SELECT role, text FROM messages ORDER BY id DESC LIMIT ?", (limit,))
    return list(reversed(cur.fetchall()))

def db_log_growth(con, model, tok, docs, loss_val=None, note=None):
    # And lo, the organism shall write its own autobiography in numbers.
    """Record a growth snapshot — structural biography."""
    n_params = sum(len(r.data) for r in model.all_base_params())
    n_params += sum(len(r.data) for r in model.all_delta_params())
    corpus_chars = sum(len(d) for d in docs)
    step = con.execute("SELECT COUNT(*) FROM growth").fetchone()[0]
    g_sparsity, g_mag = None, None
    if hasattr(model, 'gamma_stats'):
        gs = model.gamma_stats()
        g_sparsity = gs.get("sparsity")
        g_mag = gs.get("magnitude")
    con.execute(
        "INSERT INTO growth(ts,step,vocab_size,n_params,n_deltas,corpus_chars,loss,gamma_sparsity,gamma_magnitude,note) "
        "VALUES(?,?,?,?,?,?,?,?,?,?)",
        (time.time(), step, tok.vocab_size, n_params, len(model.deltas),
         corpus_chars, loss_val, g_sparsity, g_mag, note))
    con.commit()

def db_describe_growth(con):
    # And lo, the organism shall read its own growth chart and weep with pride.
    """Return growth history for self-report."""
    cur = con.cursor()
    cur.execute("SELECT step,vocab_size,n_params,n_deltas,corpus_chars,loss,gamma_sparsity,gamma_magnitude,ts FROM growth ORDER BY id")
    return [{"step": r[0], "vocab_size": r[1], "n_params": r[2], "n_deltas": r[3],
             "corpus_chars": r[4], "loss": r[5], "gamma_sparsity": r[6],
             "gamma_magnitude": r[7], "ts": r[8]} for r in cur.fetchall()]

# ============================================================
# 2) CORPUS RESERVOIR — and nonames.txt shall not bloat forever
# ============================================================

def load_corpus_lines(path: str):
    if not os.path.exists(path):
        return []
    lines = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                lines.append(ln[:CFG.max_line_chars])
    return lines

def save_corpus_lines(path: str, lines):
    with open(path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln.replace("\n", " ").strip() + "\n")

def normalize_text(s: str) -> str:
    s = s.replace("\r", " ").replace("\t", " ")
    return " ".join(s.split())

def extract_candidate_sentences_from_messages(msgs):
    """
    Turn recent chat into training sentences.
    Key trick: KEEP the role marker (H:/A:) so the organism learns dialogue,
    but keep it short so the prompt format doesn't swallow the model.
    """
    out = []
    for role, text in msgs:
        t = normalize_text(text)
        if not t:
            continue

        tag = "H:" if role == "user" else "A:"

        buf = ""
        for ch in t:
            buf += ch
            if ch in ".!?":
                s = buf.strip()
                if len(s) >= 6:
                    out.append(f"{tag} {s}")
                buf = ""
        s = buf.strip()
        if len(s) >= 12:
            out.append(f"{tag} {s}")

    # stable dedup (case-insensitive)
    seen = set()
    uniq = []
    for s in out:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(s)
    return uniq


def reservoir_mix_keep(lines, new_sents, max_lines):
    """
    Keep newest half, and sample older material to fill the rest.
    This preserves continuity without infinite growth.
    """
    combined = lines + new_sents
    newest = combined[-(max_lines // 2):]
    older = combined[:-(max_lines // 2)]
    random.shuffle(older)
    older_keep = older[:max(0, max_lines - len(newest))]
    final = older_keep + newest

    # final dedup
    seen = set()
    dedup = []
    for s in final:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            dedup.append(s[:CFG.max_line_chars])
    return dedup[-max_lines:]

def update_reservoir_corpus(con, corpus_path: str, max_lines: int):
    msgs = db_recent_messages(con, limit=64)
    new_sents = extract_candidate_sentences_from_messages(msgs)
    if not new_sents:
        return 0

    lines = load_corpus_lines(corpus_path)
    before = sum(len(x) for x in lines)

    final = reservoir_mix_keep(lines, new_sents, max_lines)
    save_corpus_lines(corpus_path, final)

    after = sum(len(x) for x in final)
    added = max(0, after - before)

    con.execute(
        "INSERT INTO corpus_events(ts, added_chars, note) VALUES(?,?,?)",
        (time.time(), added, f"reservoir_update +{len(new_sents)} sents")
    )
    con.commit()
    return added

def compute_new_corpus_mass(con, last_event_id):
    cur = con.cursor()
    cur.execute("SELECT id, added_chars FROM corpus_events WHERE id > ? ORDER BY id ASC",
                (last_event_id,))
    rows = cur.fetchall()
    if not rows:
        return 0, last_event_id
    mass = sum(r[1] for r in rows)
    return mass, rows[-1][0]

# ============================================================
# 2.5) CO-OCCURRENCE FIELD — corpus-level statistics for
#       generation before (or alongside) trained weights
# ============================================================

class CooccurField:
    # And lo, the corpus shall whisper its statistics, and words shall follow words.
    """Lightweight bigram/trigram frequency model built from token IDs."""

    def __init__(self):
        self.unigram = Counter()
        self.bigram = defaultdict(Counter)
        self.trigram = defaultdict(Counter)
        self.total_tokens = 0

    def build_from_corpus(self, tok, docs):
        self.unigram.clear()
        self.bigram.clear()
        self.trigram.clear()
        self.total_tokens = 0
        for doc in docs:
            ids = tok.encode(doc)
            for i, tid in enumerate(ids):
                self.unigram[tid] += 1
                self.total_tokens += 1
                if i >= 1:
                    self.bigram[ids[i - 1]][tid] += 1
                if i >= 2:
                    self.trigram[(ids[i - 2], ids[i - 1])][tid] += 1

    def sample_next(self, context_ids, temperature=1.0):
        """Trigram -> bigram -> unigram fallback sampling."""
        dist = None
        if len(context_ids) >= 2:
            key = (context_ids[-2], context_ids[-1])
            if key in self.trigram and self.trigram[key]:
                dist = self.trigram[key]
        if dist is None and len(context_ids) >= 1:
            if context_ids[-1] in self.bigram and self.bigram[context_ids[-1]]:
                dist = self.bigram[context_ids[-1]]
        if dist is None:
            dist = self.unigram
        if not dist:
            return 0
        items = list(dist.items())
        logits_raw = [math.log(max(c, 1e-10)) / max(temperature, 1e-6) for _, c in items]
        probs = softmax_probs_float(logits_raw)
        r = random.random()
        cumsum = 0.0
        for i, p in enumerate(probs):
            cumsum += p
            if cumsum >= r:
                return items[i][0]
        return items[-1][0]


def corpus_generate(tok, field, seed_text, max_tokens=None):
    # And lo, the organism shall speak before it learns, like a newborn crying.
    """Generate text purely from corpus statistics (no model weights)."""
    if max_tokens is None:
        max_tokens = CFG.corpus_gen_max_tokens
    ids = tok.encode(seed_text)[:-1]
    out_ids = []
    eos_id = tok.stoi.get(tok.EOS, -1)
    for _ in range(max_tokens):
        nxt = field.sample_next(ids)
        if nxt == eos_id:
            break
        ids.append(nxt)
        out_ids.append(nxt)
    return tok.decode([tok.stoi[tok.BOS]] + out_ids + [eos_id])


def generate_resonant(model, tok, field, prompt_text, use_model=True, model_alpha=0.5):
    # And lo, the model and the corpus shall duet like two drunks harmonizing.
    """Blend model logits with corpus field for generation."""
    if not use_model:
        return corpus_generate(tok, field, prompt_text)

    with model.lock, no_grad():
        return _generate_resonant_impl(model, tok, field, prompt_text, model_alpha)

def _generate_resonant_impl(model, tok, field, prompt_text, model_alpha):
    ids = tok.encode(prompt_text)[:-1]
    if not ids:
        ids = [tok.stoi[tok.BOS]]
    keys = [[] for _ in range(model.n_layer)]
    values = [[] for _ in range(model.n_layer)]
    for pos in range(min(len(ids), model.block_size)):
        _ = model.forward_step(ids[pos], pos, keys, values)

    cur = ids[-1]
    out_ids = []
    eos_id = tok.stoi.get(tok.EOS, -1)

    for step in range(CFG.corpus_gen_max_tokens):
        pos = min(len(ids) - 1, model.block_size - 1)
        logits = model.forward_step(cur, pos, keys, values)
        model_probs = softmax_probs_float((logits.data / CFG.temperature).tolist())

        # corpus bias
        corpus_dist = {}
        if len(ids) >= 2:
            key = (ids[-2], ids[-1])
            if key in field.trigram:
                corpus_dist = dict(field.trigram[key])
        if not corpus_dist and len(ids) >= 1:
            if ids[-1] in field.bigram:
                corpus_dist = dict(field.bigram[ids[-1]])

        if corpus_dist:
            total_c = sum(corpus_dist.values())
            corpus_probs = [0.0] * len(model_probs)
            for tid, cnt in corpus_dist.items():
                if tid < len(corpus_probs):
                    corpus_probs[tid] = cnt / total_c
            blended = [model_alpha * mp + (1.0 - model_alpha) * cp
                       for mp, cp in zip(model_probs, corpus_probs)]
            total_b = sum(blended)
            if total_b > 0:
                blended = [b / total_b for b in blended]
            probs = blended
        else:
            probs = model_probs

        nxt = top_k_top_p_sample(probs, CFG.top_k, CFG.top_p, CFG.min_p, CFG.typical_p)
        if nxt == eos_id and step >= CFG.min_gen_tokens:
            break
        if nxt == eos_id:
            continue

        ids.append(nxt)
        cur = nxt
        out_ids.append(nxt)

        if len(ids) >= model.block_size:
            ids = ids[-model.block_size:]
            keys = [[] for _ in range(model.n_layer)]
            values = [[] for _ in range(model.n_layer)]
            for p in range(len(ids) - 1):
                _ = model.forward_step(ids[p], p, keys, values)

    return tok.decode([tok.stoi[tok.BOS]] + out_ids + [eos_id])


# ============================================================
# 3) TOKENIZER — char first, then BPE that only EXPANDS vocab
# ============================================================

class EvolvingTokenizer:
    """
    Starts as char-level.
    Later learns BPE merges and ADDS new tokens, never removing old ones.
    That means: existing weights remain valid; matrices only grow rows.
    """
    def __init__(self, docs):
        # And lo, the alphabet shall be forged from the corpus.
        base_text = "\n".join(docs) + "\n"
        self.base_chars = sorted(set(base_text))
        self.BOS = "<BOS>"
        self.EOS = "<EOS>"
        self.PAD = "<PAD>"

        # vocab tokens are strings; chars are tokens too
        self.tokens = list(self.base_chars) + [self.PAD, self.BOS, self.EOS]
        self.stoi = {t: i for i, t in enumerate(self.tokens)}
        self.itos = {i: t for t, i in self.stoi.items()}
        self.vocab_size = len(self.tokens)

        # BPE state
        self.bpe_enabled = False
        self.merges = []        # list of pairs, in rank order
        self.merge_to_tok = {}  # (a,b) -> new_token string
        self._trained_chars = len(base_text)

    def _word_to_symbols(self, word):
        # Classic BPE uses an end-of-word marker so spaces survive.
        return list(word) + ["</w>"]

    def _get_pairs(self, symbols):
        return {(symbols[i], symbols[i+1]) for i in range(len(symbols) - 1)}

    def maybe_enable_bpe(self, docs):
        # And lo, when the corpus grows heavy enough, subwords shall awaken.
        total_chars = sum(len(x) for x in docs)
        if (not self.bpe_enabled) and total_chars >= CFG.enable_bpe_after_chars:
            self.train_bpe(docs, CFG.bpe_num_merges)
            self.bpe_enabled = True
            self._trained_chars = total_chars
            return True
        return False

    def maybe_retrain_bpe(self, docs):
        if not self.bpe_enabled:
            return False
        total_chars = sum(len(x) for x in docs)
        if total_chars - self._trained_chars >= CFG.bpe_retrain_every_chars:
            self.train_bpe(docs, CFG.bpe_num_merges)
            self._trained_chars = total_chars
            return True
        return False

    def train_bpe(self, docs, num_merges):
        # And lo, the merges shall be learned from raw sentences, because we have no excuses.
        text = " ".join(docs)
        words = [w for w in text.split() if w]
        if not words:
            return

        vocab = Counter()
        for w in words:
            vocab[tuple(self._word_to_symbols(w))] += 1

        merges = []
        merge_to_tok = {}

        for _ in range(num_merges):
            pairs = defaultdict(int)
            for sym_seq, freq in vocab.items():
                for i in range(len(sym_seq) - 1):
                    pairs[(sym_seq[i], sym_seq[i+1])] += freq
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            a, b = best
            new_tok = a + b
            merges.append(best)
            merge_to_tok[best] = new_tok

            # merge in vocab
            new_vocab = Counter()
            for sym_seq, freq in vocab.items():
                out = []
                i = 0
                while i < len(sym_seq):
                    if i < len(sym_seq) - 1 and (sym_seq[i], sym_seq[i+1]) == best:
                        out.append(new_tok)
                        i += 2
                    else:
                        out.append(sym_seq[i])
                        i += 1
                new_vocab[tuple(out)] += freq
            vocab = new_vocab

            # Add token to vocab if new
            if new_tok not in self.stoi:
                self.stoi[new_tok] = len(self.tokens)
                self.tokens.append(new_tok)

        # ensure special tokens exist (they do)
        self.itos = {i: t for t, i in self.stoi.items()}
        self.vocab_size = len(self.tokens)
        self.merges = merges
        self.merge_to_tok = merge_to_tok

    def _apply_bpe_to_word(self, word):
        # And lo, greedy merging by learned rank shall be performed.
        symbols = self._word_to_symbols(word)

        # Build ranks (lower is better)
        rank = {pair: i for i, pair in enumerate(self.merges)}

        while True:
            pairs = [(rank.get((symbols[i], symbols[i+1]), 10**9), i)
                     for i in range(len(symbols) - 1)]
            if not pairs:
                break
            best_rank, idx = min(pairs, key=lambda x: x[0])
            if best_rank == 10**9:
                break
            pair = (symbols[idx], symbols[idx+1])
            new_tok = self.merge_to_tok[pair]
            symbols = symbols[:idx] + [new_tok] + symbols[idx+2:]
        return symbols

    def encode(self, s: str):
        s = s.strip()
        ids = [self.stoi[self.BOS]]

        if not self.bpe_enabled:
            for ch in s:
                if ch in self.stoi:
                    ids.append(self.stoi[ch])
            ids.append(self.stoi[self.EOS])
            return ids

        # BPE mode (still safe: chars remain valid tokens)
        words = s.split()
        for wi, w in enumerate(words):
            syms = self._apply_bpe_to_word(w)
            for tok in syms:
                if tok == "</w>":
                    continue
                if tok in self.stoi:
                    ids.append(self.stoi[tok])
            if wi != len(words) - 1:
                # represent spaces as the actual space char if present; fallback to '\n' or nothing
                if " " in self.stoi:
                    ids.append(self.stoi[" "])
        ids.append(self.stoi[self.EOS])
        return ids

    def decode(self, ids):
        out = []
        for t in ids:
            tok = self.itos.get(t, "")
            if tok in (self.BOS, self.PAD):
                continue
            if tok == self.EOS:
                break
            # BPE tokens are just strings; join them
            out.append(tok)
        s = "".join(out)
        # cleanup the accidental end markers if any slipped (shouldn't)
        s = s.replace("</w>", "")
        return " ".join(s.split()).strip()

# ============================================================
# 4) AUTOGRAD — vectors, not scalar confetti
# ============================================================

# And lo, when the organism speaks, it shall not waste breath building
# a backward graph it will never use. no_grad is mercy for inference.
_GRAD_ENABLED = True

class no_grad:
    """Context manager: disable autograd graph construction (like torch.no_grad)."""
    def __enter__(self):
        global _GRAD_ENABLED
        self._prev = _GRAD_ENABLED
        _GRAD_ENABLED = False
        return self
    def __exit__(self, *a):
        global _GRAD_ENABLED
        _GRAD_ENABLED = self._prev

class VectorValue:
    """A differentiable vector backed by numpy. One object = one embedding / hidden state."""
    __slots__ = ("data", "grad", "_children", "_back_fn")

    def __init__(self, data, children=(), back_fn=None):
        self.data = np.asarray(data, dtype=np.float64) if not isinstance(data, np.ndarray) else data
        self.grad = np.zeros(len(self.data), dtype=np.float64) if _GRAD_ENABLED else None
        self._children = children
        self._back_fn = back_fn

    def __add__(self, other):
        if isinstance(other, VectorValue):
            out = VectorValue(self.data + other.data)
            if _GRAD_ENABLED:
                out._children = (self, other)
                def _back():
                    self.grad += out.grad
                    other.grad += out.grad
                out._back_fn = _back
            return out
        out = VectorValue(self.data + float(other))
        if _GRAD_ENABLED:
            out._children = (self,)
            def _back():
                self.grad += out.grad
            out._back_fn = _back
        return out

    def __radd__(self, other): return self.__add__(other)

    def __neg__(self):
        out = VectorValue(-self.data)
        if _GRAD_ENABLED:
            out._children = (self,)
            def _back():
                self.grad -= out.grad
            out._back_fn = _back
        return out

    def __sub__(self, other):
        if isinstance(other, VectorValue):
            out = VectorValue(self.data - other.data)
            if _GRAD_ENABLED:
                out._children = (self, other)
                def _back():
                    self.grad += out.grad
                    other.grad -= out.grad
                out._back_fn = _back
            return out
        return self + (-float(other))

    def __mul__(self, other):
        if isinstance(other, VectorValue):
            out = VectorValue(self.data * other.data)
            if _GRAD_ENABLED:
                out._children = (self, other)
                def _back():
                    self.grad += other.data * out.grad
                    other.grad += self.data * out.grad
                out._back_fn = _back
            return out
        s = float(other)
        out = VectorValue(self.data * s)
        if _GRAD_ENABLED:
            out._children = (self,)
            def _back():
                self.grad += s * out.grad
            out._back_fn = _back
        return out

    def __rmul__(self, other): return self.__mul__(other)

    def relu(self):
        out = VectorValue(np.maximum(0.0, self.data))
        if _GRAD_ENABLED:
            out._children = (self,)
            mask = self.data > 0
            def _back():
                self.grad[mask] += out.grad[mask]
            out._back_fn = _back
        return out

    def squared_relu(self):
        r = np.maximum(0.0, self.data)
        out = VectorValue(r * r)
        if _GRAD_ENABLED:
            out._children = (self,)
            def _back():
                mask = self.data > 0
                self.grad[mask] += 2.0 * r[mask] * out.grad[mask]
            out._back_fn = _back
        return out

    def dot(self, other):
        val = float(np.dot(self.data, other.data))
        out = ScalarValue(val)
        if _GRAD_ENABLED:
            out._children = (self, other)
            def _back():
                self.grad += other.data * out.grad
                other.grad += self.data * out.grad
            out._back_fn = _back
        return out

    def mean_sq(self):
        val = float(np.mean(self.data ** 2))
        out = ScalarValue(val)
        if _GRAD_ENABLED:
            out._children = (self,)
            n = len(self.data)
            def _back():
                self.grad += (2.0 / n) * self.data * out.grad
            out._back_fn = _back
        return out

    def slice(self, start, end):
        out = VectorValue(self.data[start:end].copy())
        if _GRAD_ENABLED:
            out._children = (self,)
            def _back():
                self.grad[start:end] += out.grad
            out._back_fn = _back
        return out

    def element(self, idx):
        # And lo, one number shall be plucked from the vector, and gradients shall follow.
        """Extract single element as ScalarValue with gradient flow."""
        out = ScalarValue(float(self.data[idx]))
        if _GRAD_ENABLED:
            out._children = (self,)
            local_idx = idx
            def _back():
                self.grad[local_idx] += out.grad
            out._back_fn = _back
        return out

    @staticmethod
    def concat(vecs):
        out = VectorValue(np.concatenate([v.data for v in vecs]))
        if _GRAD_ENABLED:
            out._children = tuple(vecs)
            sizes = [len(v.data) for v in vecs]
            def _back():
                offset = 0
                for v, sz in zip(vecs, sizes):
                    v.grad += out.grad[offset:offset + sz]
                    offset += sz
            out._back_fn = _back
        return out

class ScalarValue:
    """A differentiable scalar. For loss, dot products, attention weights."""
    __slots__ = ("data", "grad", "_children", "_back_fn")

    def __init__(self, data, children=(), back_fn=None):
        self.data = float(data)
        self.grad = 0.0 if _GRAD_ENABLED else None
        self._children = children
        self._back_fn = back_fn

    def __add__(self, other):
        if isinstance(other, ScalarValue):
            out = ScalarValue(self.data + other.data)
            if _GRAD_ENABLED:
                out._children = (self, other)
                def _back():
                    self.grad += out.grad
                    other.grad += out.grad
                out._back_fn = _back
            return out
        out = ScalarValue(self.data + float(other))
        if _GRAD_ENABLED:
            out._children = (self,)
            def _back():
                self.grad += out.grad
            out._back_fn = _back
        return out

    def __radd__(self, other): return self.__add__(other)
    def __neg__(self): return self * -1.0

    def __sub__(self, other):
        if isinstance(other, ScalarValue):
            return self + (-other)
        return self + (-float(other))

    def __mul__(self, other):
        if isinstance(other, ScalarValue):
            out = ScalarValue(self.data * other.data)
            if _GRAD_ENABLED:
                out._children = (self, other)
                def _back():
                    self.grad += other.data * out.grad
                    other.grad += self.data * out.grad
                out._back_fn = _back
            return out
        s = float(other)
        out = ScalarValue(self.data * s)
        if _GRAD_ENABLED:
            out._children = (self,)
            def _back():
                self.grad += s * out.grad
            out._back_fn = _back
        return out

    def __rmul__(self, other): return self.__mul__(other)

    def sigmoid(self):
        sig = 1.0 / (1.0 + math.exp(-self.data))
        out = ScalarValue(sig)
        if _GRAD_ENABLED:
            out._children = (self,)
            def _back():
                self.grad += sig * (1.0 - sig) * out.grad
            out._back_fn = _back
        return out

def backward(root):
    # And lo, the graph shall be walked backwards, like a salmon with regrets.
    topo = []
    visited = set()

    def build(v):
        vid = id(v)
        if vid not in visited:
            visited.add(vid)
            for c in v._children:
                build(c)
            topo.append(v)

    build(root)
    root.grad = 1.0
    for v in reversed(topo):
        if v._back_fn is not None:
            v._back_fn()

# ============================================================
# 5) HIGH-LEVEL OPS — the sacred blocks
# ============================================================

class MatrixParam:
    """
    Weight matrix: rows of VectorValues. Shape (nout, nin).
    And yes, it can GROW when vocab expands — because forgetting is for cowards.
    """
    def __init__(self, nout, nin, std=0.02):
        self.rows = [VectorValue([random.gauss(0, std) for _ in range(nin)])
                     for _ in range(nout)]
        self.nout = nout
        self.nin = nin

    def matvec(self, x):
        # And lo, BLAS shall do the heavy lifting — numpy @ is 50-100x faster than Python loops.
        W = np.vstack([row.data for row in self.rows])  # (nout, nin)
        out_data = W @ x.data  # single BLAS call
        out = VectorValue(out_data)
        if _GRAD_ENABLED:
            out._children = tuple(self.rows) + (x,)
            rows_ref = self.rows
            nout = self.nout
            def _back():
                for i in range(nout):
                    g = out.grad[i]
                    rows_ref[i].grad += g * x.data   # numpy vectorized
                    x.grad += g * rows_ref[i].data    # numpy vectorized
            out._back_fn = _back
        return out

    def grow_rows(self, new_nout, std=0.02):
        # And lo, the matrix shall sprout new rows like a hydra learning new words.
        if new_nout <= self.nout:
            return
        for _ in range(new_nout - self.nout):
            self.rows.append(VectorValue([random.gauss(0, std) for _ in range(self.nin)]))
        self.nout = new_nout

    def params(self):
        return list(self.rows)

def rmsnorm(x):
    ms_val = float(np.mean(x.data ** 2))
    scale_val = (ms_val + 1e-5) ** -0.5
    out = VectorValue(x.data * scale_val)
    if _GRAD_ENABLED:
        out._children = (x,)
        n = len(x.data)
        def _back():
            ds_dms = -0.5 * (ms_val + 1e-5) ** -1.5
            cross = float(np.dot(out.grad, x.data))
            x.grad += scale_val * out.grad + cross * ds_dms * (2.0 / n) * x.data
        out._back_fn = _back
    return out

def cross_entropy_loss(logits, target):
    shifted = logits.data - logits.data.max()
    exps = np.exp(shifted)
    exp_sum = exps.sum()
    log_sum_exp = float(np.log(exp_sum)) + float(logits.data.max())
    loss_val = log_sum_exp - float(logits.data[target])
    probs = exps / exp_sum
    out = ScalarValue(loss_val)
    if _GRAD_ENABLED:
        out._children = (logits,)
        def _back():
            g = out.grad
            grad_delta = probs.copy()
            grad_delta[target] -= 1.0
            logits.grad += grad_delta * g
        out._back_fn = _back
    return out

def scalar_softmax(logits):
    max_val = max(s.data for s in logits)
    exps_data = [math.exp(s.data - max_val) for s in logits]
    total = sum(exps_data)
    probs_data = [e / total for e in exps_data]

    out = []
    for i in range(len(probs_data)):
        sv = ScalarValue(probs_data[i])
        if _GRAD_ENABLED:
            sv._children = tuple(logits)
            local_i = i
            def _make_back(ii, ps):
                def _back():
                    g = out[ii].grad
                    for j in range(len(logits)):
                        if j == ii:
                            logits[j].grad += g * ps[ii] * (1.0 - ps[ii])
                        else:
                            logits[j].grad += g * (-ps[ii] * ps[j])
                return _back
            sv._back_fn = _make_back(local_i, probs_data)
        out.append(sv)
    return out

def attention_weighted_sum(weights, values):
    dim = len(values[0].data)
    T = len(weights)
    # Stack values into matrix (T, dim), weights into vector (T,)
    V = np.vstack([v.data for v in values])         # (T, dim)
    w = np.array([wt.data for wt in weights])       # (T,)
    out_data = w @ V                                 # (dim,)
    out = VectorValue(out_data)
    if _GRAD_ENABLED:
        out._children = tuple(weights) + tuple(values)
        def _back():
            for t in range(T):
                weights[t].grad += float(np.dot(values[t].data, out.grad))
                values[t].grad += weights[t].data * out.grad
        out._back_fn = _back
    return out

def softmax_probs_float(data):
    d = np.asarray(data, dtype=np.float64)
    d = d - d.max()
    e = np.exp(d)
    return (e / e.sum()).tolist()

def top_k_top_p_sample(probs, k, p, min_p=0.0, typical_p=1.0):
    # And lo, sampling shall not be a coin flip but a controlled hallucination.
    n = len(probs)
    idx = list(range(n))
    idx.sort(key=lambda i: probs[i], reverse=True)

    # Top-k filtering
    if k > 0:
        idx = idx[:min(k, len(idx))]

    # Min-p filtering (GPT-3/4 style): remove tokens with prob < min_p * max_prob
    if min_p > 0.0 and idx:
        max_prob = probs[idx[0]]
        threshold = min_p * max_prob
        idx = [i for i in idx if probs[i] >= threshold]

    # Typical-p filtering: prefer tokens with typical information content
    # (i.e., tokens whose surprisal is close to the expected surprisal)
    if typical_p < 1.0 and idx:
        # Compute entropy (expected surprisal)
        entropy = -sum(probs[i] * math.log(probs[i]) for i in idx if probs[i] > 1e-12)
        # Compute absolute deviation from expected surprisal for each token
        deviations = []
        for i in idx:
            if probs[i] > 1e-12:
                surprisal = -math.log(probs[i])
                deviation = abs(surprisal - entropy)
                deviations.append((i, deviation))
        # Sort by deviation (lower is more typical)
        deviations.sort(key=lambda x: x[1])
        # Keep tokens until cumulative prob >= typical_p
        cum = 0.0
        typical_idx = []
        for i, _ in deviations:
            typical_idx.append(i)
            cum += probs[i]
            if cum >= typical_p:
                break
        if typical_idx:
            idx = typical_idx

    # Top-p (nucleus) filtering
    if p < 1.0:
        cum = 0.0
        cut = []
        for i in idx:
            cut.append(i)
            cum += probs[i]
            if cum >= p:
                break
        idx = cut

    mass = sum(probs[i] for i in idx)
    if mass <= 0:
        return idx[0] if idx else (n - 1)
    r = random.random() * mass
    s = 0.0
    for i in idx:
        s += probs[i]
        if s >= r:
            return i
    return idx[-1]

def clip_params(params, clip):
    # And lo, the gradients shall be clipped, lest they summon Cthulhu.
    if clip <= 0:
        return
    for p in params:
        np.clip(p.grad, -clip, clip, out=p.grad)

# ============================================================
# 6) DELTA ADAPTERS — appended souls, never overwritten
# ============================================================

class DeltaAdapter:
    """
    Low-rank adapter: for a base W, we add A @ B @ x.
    A and B are trained; base can be frozen.
    And yes, these can grow if vocab grows (for lm_head).
    """
    def __init__(self, nout, nin, r, std=0.02):
        self.A = MatrixParam(nout, r, std)
        self.B = MatrixParam(r, nin, std)

    def apply(self, x):
        bx = self.B.matvec(x)
        return self.A.matvec(bx)

    def maybe_grow_out(self, new_nout):
        # And lo, the adapter shall grow new output rows, because vocabulary is a living thing.
        self.A.grow_rows(new_nout, std=0.02)

    def params(self):
        return self.A.params() + self.B.params()

# ============================================================
# 7) GPT MODEL — a small beast with RoPE (GPT-3-ish spice)
# ============================================================

def rope_rotate(vec, pos, head_dim):
    """
    RoPE rotation for one head slice — numpy vectorized.
    """
    # And lo, positions shall become angles, and angles shall become meaning.
    n_pairs = head_dim // 2
    indices = np.arange(0, 2 * n_pairs, 2, dtype=np.float64)
    thetas = pos / (10000.0 ** (indices / head_dim))
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    x = vec.data[:head_dim].copy()
    out_data = x.copy()
    out_data[0::2] = x[0::2] * cos_t - x[1::2] * sin_t
    out_data[1::2] = x[0::2] * sin_t + x[1::2] * cos_t

    out = VectorValue(out_data)
    if _GRAD_ENABLED:
        out._children = (vec,)
        def _back():
            # inverse rotation = rotation by -theta
            ga = out.grad[0::2]
            gb = out.grad[1::2]
            vec.grad[0::2] += ga * cos_t + gb * sin_t
            vec.grad[1::2] += -ga * sin_t + gb * cos_t
        out._back_fn = _back
    return out

class GPT:
    def __init__(self, tok: EvolvingTokenizer):
        self.tok = tok
        self.n_layer = CFG.n_layer
        self.n_embd = CFG.n_embd
        self.n_head = CFG.n_head
        self.head_dim = CFG.n_embd // CFG.n_head
        self.block_size = CFG.block_size
        self.lock = threading.Lock()

        # Base weights
        V = tok.vocab_size
        self.base = {}
        self.base["wte"] = MatrixParam(V, CFG.n_embd, 0.08)
        self.base["wpe"] = MatrixParam(CFG.block_size, CFG.n_embd, 0.08)
        # output head (optionally tied to embeddings, classic GPT trick)
        self.base["lm_head"] = MatrixParam(V, CFG.n_embd, 0.08)
        if getattr(CFG, "tie_embeddings", False):
            self.base["lm_head"] = self.base["wte"]

        for li in range(CFG.n_layer):
            self.base[f"l{li}.wq"] = MatrixParam(CFG.n_embd, CFG.n_embd, 0.08)
            self.base[f"l{li}.wk"] = MatrixParam(CFG.n_embd, CFG.n_embd, 0.08)
            self.base[f"l{li}.wv"] = MatrixParam(CFG.n_embd, CFG.n_embd, 0.08)
            self.base[f"l{li}.wo"] = MatrixParam(CFG.n_embd, CFG.n_embd, 0.08)
            # "GPT-3-ish" hint: gated MLP (SwiGLU-ish) without extra deps
            self.base[f"l{li}.fc_g"] = MatrixParam(4 * CFG.n_embd, CFG.n_embd, 0.08)
            self.base[f"l{li}.fc_v"] = MatrixParam(4 * CFG.n_embd, CFG.n_embd, 0.08)
            self.base[f"l{li}.fc2"]  = MatrixParam(CFG.n_embd, 4 * CFG.n_embd, 0.08)
            # hybrid attention: RRPRAM pattern weights + learnable gate
            for h, htype in enumerate(CFG.head_types):
                if htype in ("rrpram", "hybrid"):
                    self.base[f"l{li}.h{h}.w_pattern"] = MatrixParam(
                        CFG.block_size, self.head_dim, 0.08)
                if htype == "hybrid":
                    self.base[f"l{li}.h{h}.alpha"] = MatrixParam(1, 1, 0.0)
                    self.base[f"l{li}.h{h}.alpha"].rows[0].data[0] = CFG.hybrid_alpha_init

        # Modular deltas
        self.deltas = []
        self.active_alpha = []

        # Adam state
        self._adam = {}

        # snapshot initial embeddings for gamma computation
        self._init_embed_snapshot = [row.data.tolist() for row in self.base["wte"].rows]

        # ensure at least one delta module exists
        self.add_delta_module(alpha=1.0)

    def maybe_expand_vocab(self, new_vocab_size):
        # And lo, when the tokenizer grows, the model shall grow with it.
        curV = self.base["wte"].nout
        if new_vocab_size <= curV:
            return

        self.base["wte"].grow_rows(new_vocab_size, std=0.08)
        if not getattr(CFG, "tie_embeddings", False):
            self.base["lm_head"].grow_rows(new_vocab_size, std=0.08)

        # Grow delta lm_head adapters too
        for mod in self.deltas:
            if "lm_head" in mod:
                mod["lm_head"].maybe_grow_out(new_vocab_size)

    def add_delta_module(self, alpha=1.0):
        # And lo, a new delta-soul shall be appended (never overwritten, never forgotten).
        mod = {}
        r = CFG.delta_rank
        for li in range(CFG.n_layer):
            for name in ("wq", "wk", "wv", "wo"):
                mod[f"l{li}.{name}"] = DeltaAdapter(CFG.n_embd, CFG.n_embd, r)
            mod[f"l{li}.fc_g"] = DeltaAdapter(4 * CFG.n_embd, CFG.n_embd, r)
            mod[f"l{li}.fc_v"] = DeltaAdapter(4 * CFG.n_embd, CFG.n_embd, r)
            mod[f"l{li}.fc2"]  = DeltaAdapter(CFG.n_embd, 4 * CFG.n_embd, r)
            for h, htype in enumerate(CFG.head_types):
                if htype in ("rrpram", "hybrid"):
                    mod[f"l{li}.h{h}.w_pattern"] = DeltaAdapter(
                        CFG.block_size, self.head_dim, r)

        mod["lm_head"] = DeltaAdapter(self.tok.vocab_size, CFG.n_embd, r)
        self.deltas.append(mod)
        self.active_alpha.append(alpha)

    def all_base_params(self):
        out = []
        for mat in self.base.values():
            out.extend(mat.params())
        return out

    def all_delta_params(self):
        out = []
        for mod in self.deltas:
            for ad in mod.values():
                out.extend(ad.params())
        return out

    # ---- Native gamma (personality fingerprint) ----
    # And lo, the organism shall subtract its birth from its present, and call the difference a soul.

    def compute_gamma(self):
        """Compute gamma = current_embed - init_embed (personality drift)."""
        current = self.base["wte"].rows
        init = self._init_embed_snapshot
        gamma = []
        for i in range(min(len(current), len(init))):
            gamma.append(current[i].data - np.array(init[i]))
        for i in range(len(init), len(current)):
            gamma.append(current[i].data.copy())
        return gamma

    # And lo, the soul shall be measured in sparsity and magnitude, like a ghost on a scale.
    def gamma_stats(self):
        """Sparsity, magnitude, top changed tokens."""
        gamma = self.compute_gamma()
        if not gamma:
            return {"sparsity": 1.0, "magnitude": 0.0, "top_tokens": [], "n_rows": 0}
        magnitudes = [(i, float(np.linalg.norm(row))) for i, row in enumerate(gamma)]
        all_vals = np.concatenate(gamma)
        total_el = len(all_vals)
        nonzero = int(np.sum(np.abs(all_vals) > CFG.gamma_sparsity_threshold))
        sparsity = 1.0 - (nonzero / max(1, total_el))
        overall_mag = math.sqrt(sum(m * m for _, m in magnitudes))
        magnitudes.sort(key=lambda x: x[1], reverse=True)
        return {
            "sparsity": sparsity,
            "magnitude": overall_mag,
            "top_tokens": [(tid, mag) for tid, mag in magnitudes[:10]],
            "n_rows": len(gamma),
        }

    # And lo, the direction of all change shall be averaged into one arrow, pointing toward who we became.
    def gamma_contrastive_projection(self):
        """Direction of mean embedding drift — personality vector.
        Returns (unit_vector, magnitude) or (None, 0.0) if too early."""
        current = self.base["wte"].rows
        init = self._init_embed_snapshot
        n = min(len(current), len(init))
        if n == 0:
            return None, 0.0
        C = np.vstack([current[i].data for i in range(n)])
        I = np.vstack([np.array(init[i]) for i in range(n)])
        direction = C.mean(axis=0) - I.mean(axis=0)
        mag = float(np.linalg.norm(direction))
        if mag > 1e-10:
            direction = direction / mag
        return direction.tolist(), mag

    # ---- Noise Immune System ----
    # And lo, the organism shall know poison from food, and reject what unmakes it.

    def snapshot_deltas(self):
        """Deep copy all delta A and B weight data for rollback."""
        snap = []
        for mod in self.deltas:
            mod_snap = {}
            for name, da in mod.items():
                mod_snap[name] = (
                    [row.data.copy() for row in da.A.rows],
                    [row.data.copy() for row in da.B.rows],
                )
            snap.append(mod_snap)
        return snap

    def restore_deltas(self, snap):
        """Restore delta weights from snapshot — rollback a poisoned burst."""
        for mod, mod_snap in zip(self.deltas, snap):
            for name, (a_data, b_data) in mod_snap.items():
                if name in mod:
                    da = mod[name]
                    for i, rd in enumerate(a_data):
                        da.A.rows[i].data[:] = rd
                    for i, rd in enumerate(b_data):
                        da.B.rows[i].data[:] = rd

    def gamma_drift_check(self, pre_direction, pre_magnitude=0.0):
        """Cosine similarity between pre-burst and post-burst contrastive projection.
        Negative = drifted opposite to identity trend = likely noise.
        Skips check when gamma magnitude is too small (early training)."""
        post_direction, post_mag = self.gamma_contrastive_projection()
        if pre_direction is None or post_direction is None:
            return 1.0  # can't check, assume OK
        # Skip immune check when gamma is near-zero (early training, numerically unstable)
        if pre_magnitude < CFG.gamma_min_magnitude or post_mag < CFG.gamma_min_magnitude:
            return 1.0
        # Both are unit vectors, dot product = cosine similarity
        return float(np.dot(pre_direction, post_direction))

    # ---- Syntropy Tracker (mathematical self-reasoning) ----
    # And lo, the organism shall not merely observe its own reflection,
    # but reason about the direction of its becoming.
    # Gamma is memory. Purpose is intention. Syntropy is the arrow.

    def compute_field_deviation(self, tok, field, docs, sample_n=32):
        """KL divergence between model logits and corpus co-occurrence field.
        Measures how far the learned model has drifted from raw corpus physics.
        Low = parroting the field. High = hallucinating beyond it.
        The sweet spot is in between: learning, not lying."""
        if not docs or field.total_tokens == 0:
            return 0.0

        kl_sum = 0.0
        count = 0
        sampled = random.sample(docs, min(sample_n, len(docs)))

        with no_grad():
            for doc in sampled:
                ids = tok.encode(doc)
                if len(ids) < 3:
                    continue
                keys = [[] for _ in range(self.n_layer)]
                values = [[] for _ in range(self.n_layer)]
                for pos in range(min(len(ids) - 1, self.block_size)):
                    tok_id = ids[pos]
                    tgt_id = ids[pos + 1]
                    logits = self.forward_step(tok_id, pos, keys, values)

                    # model distribution
                    shifted = logits.data - logits.data.max()
                    model_probs = np.exp(shifted)
                    model_probs = model_probs / model_probs.sum()

                    # corpus field distribution for this context
                    field_probs = np.zeros(len(model_probs))
                    ctx = ids[max(0, pos - 1):pos + 1]
                    if len(ctx) >= 2:
                        key = (ctx[-2], ctx[-1])
                        if key in field.trigram and field.trigram[key]:
                            total = sum(field.trigram[key].values())
                            for tid, cnt in field.trigram[key].items():
                                if tid < len(field_probs):
                                    field_probs[tid] = cnt / total
                    if field_probs.sum() < 1e-10:
                        if len(ctx) >= 1 and ctx[-1] in field.bigram:
                            total = sum(field.bigram[ctx[-1]].values())
                            for tid, cnt in field.bigram[ctx[-1]].items():
                                if tid < len(field_probs):
                                    field_probs[tid] = cnt / total

                    if field_probs.sum() < 1e-10:
                        continue

                    # KL(model || field) — how much model diverges from field
                    mask = (model_probs > 1e-12) & (field_probs > 1e-12)
                    if mask.any():
                        kl = float(np.sum(model_probs[mask] * np.log(model_probs[mask] / field_probs[mask])))
                        kl_sum += max(0.0, kl)  # clamp: partial KL can underflow
                        count += 1

        return kl_sum / max(1, count)

    def compute_model_entropy(self, tok, docs, sample_n=16):
        """Average entropy of model predictions on corpus samples.
        Falling entropy = rising order = syntropy in action."""
        if not docs:
            return 0.0

        entropy_sum = 0.0
        count = 0
        sampled = random.sample(docs, min(sample_n, len(docs)))

        with no_grad():
            for doc in sampled:
                ids = tok.encode(doc)
                if len(ids) < 3:
                    continue
                keys = [[] for _ in range(self.n_layer)]
                values = [[] for _ in range(self.n_layer)]
                for pos in range(min(len(ids) - 1, self.block_size)):
                    logits = self.forward_step(ids[pos], pos, keys, values)
                    shifted = logits.data - logits.data.max()
                    probs = np.exp(shifted)
                    probs = probs / probs.sum()
                    ent = -float(np.sum(probs[probs > 1e-12] * np.log(probs[probs > 1e-12])))
                    entropy_sum += ent
                    count += 1

        return entropy_sum / max(1, count)

    def compute_purpose_vector(self):
        """Purpose vector: direction of weight movement in the last delta layer.
        Unlike gamma (which is cumulative drift from birth),
        purpose captures the direction of the most recent change.
        Gamma is 'who I became'. Purpose is 'where I am going'."""
        if not self.deltas:
            return None, 0.0
        last_delta = self.deltas[-1]
        # aggregate delta A matrices as the purpose signal
        directions = []
        for name, da in last_delta.items():
            for row in da.A.rows:
                directions.append(row.data)
        if not directions:
            return None, 0.0
        mean_dir = np.mean(np.vstack(directions), axis=0)
        mag = float(np.linalg.norm(mean_dir))
        if mag > 1e-10:
            unit = mean_dir / mag
        else:
            unit = mean_dir
        return unit, mag

    def purpose_gamma_alignment(self):
        """Cosine similarity between purpose vector and gamma direction.
        High alignment = learning reinforces identity (syntropy).
        Low alignment = learning diverges from identity (entropy).
        Negative = learning opposes identity (danger)."""
        gamma_dir, gamma_mag = self.gamma_contrastive_projection()
        purpose_dir, purpose_mag = self.compute_purpose_vector()
        if gamma_dir is None or purpose_dir is None:
            return 0.0
        if gamma_mag < CFG.gamma_min_magnitude or purpose_mag < 1e-10:
            return 0.0
        # ensure same dimensionality (purpose might be different dim)
        g = np.array(gamma_dir)
        p = purpose_dir
        min_dim = min(len(g), len(p))
        if min_dim == 0:
            return 0.0
        return float(np.dot(g[:min_dim], p[:min_dim]))

    def _ensure_adam(self, params, key):
        if key not in self._adam:
            self._adam[key] = {
                "m": [np.zeros_like(p.data) for p in params],
                "v": [np.zeros_like(p.data) for p in params],
                "t": 0
            }

    def adam_step(self, params, key, lr):
        # And lo, Adam Optimizer shall descend like a petty god with momentum — numpy-vectorized.
        self._ensure_adam(params, key)
        st = self._adam[key]
        st["t"] += 1
        t = st["t"]
        b1, b2, eps = CFG.beta1, CFG.beta2, CFG.eps_adam
        b1_corr = 1.0 - b1 ** t
        b2_corr = 1.0 - b2 ** t

        clip_params(params, CFG.grad_clip)

        for i, p in enumerate(params):
            g = p.grad
            st["m"][i] = b1 * st["m"][i] + (1.0 - b1) * g
            st["v"][i] = b2 * st["v"][i] + (1.0 - b2) * (g * g)
            mhat = st["m"][i] / b1_corr
            vhat = st["v"][i] / b2_corr
            p.data -= lr * mhat / (np.sqrt(vhat) + eps)
            p.grad[:] = 0.0

    def _apply_with_deltas(self, name, x):
        # And lo, base weight shall speak, then deltas shall harmonize atop it.
        y = self.base[name].matvec(x)
        for alpha, mod in zip(self.active_alpha, self.deltas):
            if name in mod:
                y = y + (mod[name].apply(x) * alpha)
        return y

    def forward_step(self, token_id, pos_id, keys, values):
        tok_emb = self.base["wte"].rows[token_id]
        pos_emb = self.base["wpe"].rows[pos_id % self.block_size]
        x = tok_emb + pos_emb

        for li in range(self.n_layer):
            # ---- Attention ----
            x_res = x
            x = rmsnorm(x)

            q = self._apply_with_deltas(f"l{li}.wq", x)
            k = self._apply_with_deltas(f"l{li}.wk", x)
            v = self._apply_with_deltas(f"l{li}.wv", x)

            keys[li].append(k)
            values[li].append(v)

            head_outputs = []
            # And lo, each head shall choose its nature: content, rrpram, or the sacred hybrid of both.
            T = len(keys[li])
            for h in range(self.n_head):
                hs = h * self.head_dim
                he = hs + self.head_dim
                htype = CFG.head_types[h] if h < len(CFG.head_types) else "content"

                vh = [values[li][t].slice(hs, he) for t in range(T)]

                # content attention (Q@K^T/sqrt(d) + RoPE)
                content_logits = None
                if htype in ("content", "hybrid"):
                    qh = q.slice(hs, he)
                    qh = rope_rotate(qh, pos_id, self.head_dim)
                    content_logits = []
                    for t in range(T):
                        kh_t = keys[li][t].slice(hs, he)
                        kh_t = rope_rotate(kh_t, t, self.head_dim)
                        dot = qh.dot(kh_t) * (1.0 / math.sqrt(self.head_dim))
                        content_logits.append(dot)

                # RRPRAM attention (x @ W_pattern -> positional scores)
                rrpram_logits = None
                if htype in ("rrpram", "hybrid"):
                    xh = x.slice(hs, he)
                    pattern_full = self._apply_with_deltas(f"l{li}.h{h}.w_pattern", xh)
                    rrpram_logits = [pattern_full.element(t) for t in range(T)]

                # dispatch by head type
                if htype == "content":
                    attn_weights = scalar_softmax(content_logits)
                elif htype == "rrpram":
                    attn_weights = scalar_softmax(rrpram_logits)
                else:  # hybrid: blend with sigmoid gate (alpha in autograd graph)
                    alpha_scalar = self.base[f"l{li}.h{h}.alpha"].rows[0].element(0)
                    a = alpha_scalar.sigmoid()
                    one_minus_a = a * (-1.0) + 1.0  # 1 - sigmoid(alpha)
                    blended = [c * one_minus_a + r * a
                               for c, r in zip(content_logits, rrpram_logits)]
                    attn_weights = scalar_softmax(blended)

                head_out = attention_weighted_sum(attn_weights, vh)
                head_outputs.append(head_out)

            x_attn = VectorValue.concat(head_outputs)
            x = self._apply_with_deltas(f"l{li}.wo", x_attn)
            x = x + x_res

            # ---- Gated MLP (SwiGLU-ish) ----
            x_res = x
            x = rmsnorm(x)

            g = self._apply_with_deltas(f"l{li}.fc_g", x).relu()   # gate
            u = self._apply_with_deltas(f"l{li}.fc_v", x)          # value
            x = g * u                                              # gating

            x = self._apply_with_deltas(f"l{li}.fc2", x)
            x = x + x_res

        x = rmsnorm(x)
        logits = self._apply_with_deltas("lm_head", x)
        return logits

    def loss_on_sequence(self, ids):
        n = min(self.block_size, len(ids) - 1)
        if n <= 0:
            return ScalarValue(0.0)
        keys = [[] for _ in range(self.n_layer)]
        values = [[] for _ in range(self.n_layer)]
        total_loss = ScalarValue(0.0)
        for pos in range(n):
            tok = ids[pos]
            tgt = ids[pos + 1]
            logits = self.forward_step(tok, pos, keys, values)
            total_loss = total_loss + cross_entropy_loss(logits, tgt)
        return total_loss * (1.0 / n)

    def loss_on_batch(self, batch_ids):
        # And lo, batching shall be done without lying padding tokens into the loss.
        if not batch_ids:
            return ScalarValue(0.0)
        total = ScalarValue(0.0)
        for ids in batch_ids:
            total = total + self.loss_on_sequence(ids)
        return total * (1.0 / len(batch_ids))

    def generate_sentence(self, prompt_text=""):
        # And lo, generation shall aim for a sentence, not a random cough.
        # no_grad: inference needs no backward graph — pure mercy for speed.
        with self.lock, no_grad():
            return self._generate_sentence_impl(prompt_text)

    def _generate_sentence_impl(self, prompt_text=""):
        if prompt_text:
            ids = self.tok.encode(prompt_text)[:-1]
        else:
            ids = [self.tok.stoi[self.tok.BOS]]

        keys = [[] for _ in range(self.n_layer)]
        values = [[] for _ in range(self.n_layer)]

        # build cache from prompt
        for pos in range(min(len(ids), self.block_size)):
            _ = self.forward_step(ids[pos], pos, keys, values)

        cur = ids[-1] if ids else self.tok.stoi[self.tok.BOS]
        out_ids = []
        recent = []

        for step in range(CFG.max_gen_tokens):
            pos = min(len(ids) - 1, self.block_size - 1)
            logits = self.forward_step(cur, pos, keys, values)

            # entropy-adaptive temperature
            base_temp = float(CFG.temperature)
            if base_temp <= 1e-6:
                base_temp = 1e-6
            raw = logits.data
            raw_scaled = (raw / base_temp).tolist()
            probs0 = softmax_probs_float(raw_scaled)
            entropy = -sum(p * math.log(p) for p in probs0 if p > 1e-12)
            t_mul = 1.0
            if entropy < CFG.entropy_low:
                t_mul = CFG.entropy_temp_boost
            elif entropy > CFG.entropy_high:
                t_mul = CFG.entropy_temp_focus
            temp = base_temp * t_mul
            scaled = (raw / temp).tolist()
            probs = softmax_probs_float(scaled)
            nxt = top_k_top_p_sample(probs, CFG.top_k, CFG.top_p, CFG.min_p, CFG.typical_p)

            if nxt == self.tok.stoi[self.tok.EOS]:
                if step >= CFG.min_gen_tokens:
                    break
                # else: ignore early EOS and keep going
                continue

            ids.append(nxt)
            cur = nxt
            out_ids.append(nxt)

            recent.append(nxt)
            if len(recent) > CFG.repetition_guard * 2:
                recent = recent[-CFG.repetition_guard * 2:]
                n = CFG.repetition_guard
                if recent[-n:] == recent[-2*n:-n]:
                    break

            text_now = self.tok.decode([self.tok.stoi[self.tok.BOS]] + out_ids + [self.tok.stoi[self.tok.EOS]])
            if step >= CFG.min_gen_tokens and text_now and text_now[-1] in ".!?":
                break

            # sliding window rebuild (cheap)
            if len(ids) >= self.block_size:
                ids = ids[-self.block_size:]
                keys = [[] for _ in range(self.n_layer)]
                values = [[] for _ in range(self.n_layer)]
                for p in range(len(ids) - 1):
                    _ = self.forward_step(ids[p], p, keys, values)

        return self.tok.decode([self.tok.stoi[self.tok.BOS]] + out_ids + [self.tok.stoi[self.tok.EOS]])

# ============================================================
# 8) CHECKPOINTING — modular, compatible, no merge-amnesia
# ============================================================

def _serialize_matrix_param(mp):
    return [row.data.tolist() for row in mp.rows]

def _deserialize_matrix_param(data):
    mp = MatrixParam.__new__(MatrixParam)
    mp.rows = [VectorValue(row) for row in data]
    mp.nout = len(data)
    mp.nin = len(data[0]) if data else 0
    return mp

def save_checkpoint(model: GPT, tok: EvolvingTokenizer, path=None):
    # And lo, the organism shall persist as JSON, because we refuse dependencies.
    if path is None:
        path = CFG.ckpt_path
    obj = {
        "cfg": CFG.__dict__,
        "tokenizer": {
            "tokens": tok.tokens,
            "bpe_enabled": tok.bpe_enabled,
            "merges": [list(p) for p in tok.merges],  # rank-ordered pairs
            "trained_chars": tok._trained_chars,
        },
        "base": {k: _serialize_matrix_param(v) for k, v in model.base.items()},
        "alpha": model.active_alpha,
        "init_embed_snapshot": model._init_embed_snapshot,
        "deltas": []
    }
    for mod in model.deltas:
        m = {}
        for name, ad in mod.items():
            m[name] = {
                "A": _serialize_matrix_param(ad.A),
                "B": _serialize_matrix_param(ad.B),
            }
        obj["deltas"].append(m)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)

def load_checkpoint(docs, path=None):
    # And lo, resurrection shall be attempted.
    if path is None:
        path = CFG.ckpt_path
    if not os.path.exists(path):
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    tok = EvolvingTokenizer(docs if docs else ["Hello."])
    t = obj.get("tokenizer", {})
    if "tokens" in t and isinstance(t["tokens"], list):
        tok.tokens = t["tokens"]
        tok.stoi = {tt: i for i, tt in enumerate(tok.tokens)}
        tok.itos = {i: tt for tt, i in tok.stoi.items()}
        tok.vocab_size = len(tok.tokens)

    merges = t.get("merges", [])
    tok.merges = [tuple(p) for p in merges if isinstance(p, list) and len(p) == 2]
    tok.merge_to_tok = {tuple(p): (p[0] + p[1]) for p in tok.merges}
    tok.bpe_enabled = bool(t.get("bpe_enabled", False))
    tok._trained_chars = int(t.get("trained_chars", 0))

    model = GPT(tok)

    # Restore base
    model.base = {k: _deserialize_matrix_param(v) for k, v in obj["base"].items()}

    # Ensure hybrid attention weights exist (backward compat with old checkpoints)
    for li in range(CFG.n_layer):
        for h, htype in enumerate(CFG.head_types):
            pkey = f"l{li}.h{h}.w_pattern"
            akey = f"l{li}.h{h}.alpha"
            if htype in ("rrpram", "hybrid") and pkey not in model.base:
                model.base[pkey] = MatrixParam(CFG.block_size, model.head_dim, 0.08)
            if htype == "hybrid" and akey not in model.base:
                model.base[akey] = MatrixParam(1, 1, 0.0)
                model.base[akey].rows[0].data[0] = CFG.hybrid_alpha_init

    # Restore deltas
    model.deltas = []
    model.active_alpha = obj.get("alpha", [])
    for mod_data in obj.get("deltas", []):
        mm = {}
        for name, w in mod_data.items():
            ad = DeltaAdapter.__new__(DeltaAdapter)
            ad.A = _deserialize_matrix_param(w["A"])
            ad.B = _deserialize_matrix_param(w["B"])
            mm[name] = ad
        model.deltas.append(mm)

    if not model.deltas:
        model.add_delta_module(alpha=1.0)

    # Restore gamma baseline (or initialize from current if old checkpoint)
    snapshot = obj.get("init_embed_snapshot")
    if snapshot:
        model._init_embed_snapshot = snapshot
    else:
        model._init_embed_snapshot = [row.data.tolist() for row in model.base["wte"].rows]

    return model, tok

# ============================================================
# 9) TRAINING — warmup, then continual micro-bursts
# ============================================================

# ============================================================
# 9.5) SYNTROPY TRACKER — the arrow that points toward coherence
# ============================================================
# And lo, the organism shall not merely track its changes,
# but reason mathematically about whether it is becoming more itself.

class SyntropyTracker:
    """Mathematical self-reasoning engine.
    Tracks entropy trend, field deviation, purpose alignment.
    Makes decisions about learning direction — not just 'did I learn?'
    but 'should I keep going this way?'"""

    def __init__(self):
        self.entropy_history = []     # rolling window of model entropy
        self.syntropy_trend = 0.0     # positive = organizing, negative = dissolving
        self.field_deviation = 0.0    # how far from corpus physics
        self.purpose_magnitude = 0.0  # strength of current learning direction
        self.purpose_alignment = 0.0  # cosine(purpose, gamma)
        self.last_action = "none"     # what was decided last time

    def measure(self, model, tok, field, docs):
        """Take all measurements. This is the organism looking at itself
        through mathematical instruments."""
        entropy_now = model.compute_model_entropy(tok, docs)
        self.entropy_history.append(entropy_now)
        if len(self.entropy_history) > CFG.syntropy_window:
            self.entropy_history = self.entropy_history[-CFG.syntropy_window:]

        # syntropy = negative entropy trend (entropy going down = syntropy going up)
        if len(self.entropy_history) >= 2:
            recent_half = len(self.entropy_history) // 2
            old_mean = np.mean(self.entropy_history[:recent_half])
            new_mean = np.mean(self.entropy_history[recent_half:])
            self.syntropy_trend = float(old_mean - new_mean)  # positive = good
        else:
            self.syntropy_trend = 0.0

        self.field_deviation = model.compute_field_deviation(tok, field, docs)
        _, self.purpose_magnitude = model.compute_purpose_vector()
        self.purpose_alignment = model.purpose_gamma_alignment()

        return {
            "entropy": entropy_now,
            "syntropy_trend": self.syntropy_trend,
            "field_deviation": self.field_deviation,
            "purpose_magnitude": self.purpose_magnitude,
            "purpose_alignment": self.purpose_alignment,
        }

    def decide_action(self):
        """Mathematical self-reasoning: decide how to adjust learning.
        This is where tracking becomes reasoning, and reasoning becomes action.
        The organism does not just observe — it steers."""

        # Default: steady state
        lr_multiplier = 1.0
        delta_grow_override = None
        action = "steady"

        # CASE 1: Syntropy rising + field deviation in sweet spot = thriving
        if (self.syntropy_trend > 0.01 and
                CFG.field_deviation_floor < self.field_deviation < CFG.field_deviation_ceiling):
            lr_multiplier = CFG.syntropy_lr_boost
            if self.purpose_alignment > 0.3:
                delta_grow_override = CFG.syntropy_delta_grow_boost
                action = "amplify"  # everything aligned, push harder
            else:
                action = "boost"  # syntropy good but purpose drifting, boost gently

        # CASE 2: Syntropy falling = dissolving, slow down
        elif self.syntropy_trend < -0.01:
            lr_multiplier = CFG.syntropy_lr_dampen
            action = "dampen"  # losing order, reduce learning rate

        # CASE 3: Field deviation too high = hallucinating
        elif self.field_deviation > CFG.field_deviation_ceiling:
            lr_multiplier = CFG.syntropy_lr_dampen
            action = "ground"  # too far from corpus, pull back

        # CASE 4: Field deviation too low = parroting
        elif self.field_deviation < CFG.field_deviation_floor:
            lr_multiplier = CFG.syntropy_lr_boost
            action = "explore"  # too close to corpus, push out

        # CASE 5: Purpose opposes gamma = identity crisis
        if self.purpose_alignment < -0.3:
            lr_multiplier *= 0.5
            action = "realign"  # learning against identity, slow down hard

        self.last_action = action
        return {
            "lr_multiplier": lr_multiplier,
            "delta_grow_override": delta_grow_override,
            "action": action,
        }

    def log_to_db(self, con, entropy_before, entropy_after, action):
        """Write the mathematical conclusion to the syntropy log."""
        con.execute(
            "INSERT INTO syntropy_log(ts, entropy_before, entropy_after, syntropy_delta, "
            "field_deviation, purpose_magnitude, purpose_alignment, action_taken, note) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (time.time(), entropy_before, entropy_after,
             self.syntropy_trend, self.field_deviation,
             self.purpose_magnitude, self.purpose_alignment,
             action, None))
        con.commit()


def train_steps(model: GPT, tok: EvolvingTokenizer, docs, steps, train_base=True, train_deltas=True):
    if not docs:
        return

    with model.lock:
        _train_steps_locked(model, tok, docs, steps, train_base, train_deltas)

def _train_steps_locked(model, tok, docs, steps, train_base, train_deltas):
    base_params = model.all_base_params() if train_base else []
    delta_params = model.all_delta_params() if train_deltas else []

    for step in range(steps):
        # And lo, the training batch shall be sampled from the cursed book of names.
        batch = random.choices(docs, k=CFG.batch_size)
        batch_ids = [tok.encode(doc) for doc in batch if doc]

        loss = model.loss_on_batch(batch_ids)
        backward(loss)

        lr = CFG.learning_rate * (1.0 - (step / max(1, steps)))
        if base_params:
            model.adam_step(base_params, key="base", lr=lr)
        if delta_params:
            model.adam_step(delta_params, key="delta", lr=lr)

        if step % 100 == 0:
            print(f"  train step {step}/{steps} | loss {loss.data:.4f}")

# And lo, the buffer shall measure not just bytes but novelty, for raw mass means nothing without surprise.
class QuantumBuffer:
    """Smart training trigger: accumulates experience, fires when ready."""
    def __init__(self):
        self.accumulated_bytes = 0
        self.unique_tokens = set()
        self.total_tokens = 0
        self.last_burst_time = 0.0

    def feed(self, new_chars, tok, docs):
        self.accumulated_bytes += new_chars
        for doc in docs[-20:]:
            ids = tok.encode(doc)
            for tid in ids:
                self.total_tokens += 1
                self.unique_tokens.add(tid)

    def novelty_score(self):
        if self.total_tokens == 0:
            return 0.0
        return len(self.unique_tokens) / max(1, self.total_tokens)

    def should_trigger(self):
        now = time.time()
        cooldown_ok = (now - self.last_burst_time) >= CFG.qb_cooldown_seconds
        bytes_ok = self.accumulated_bytes >= CFG.qb_min_bytes
        novelty_ok = self.novelty_score() >= CFG.qb_min_novelty
        return (bytes_ok or novelty_ok) and cooldown_ok

    def reset(self):
        self.accumulated_bytes = 0
        self.unique_tokens.clear()
        self.total_tokens = 0
        self.last_burst_time = time.time()


async def background_trainer(con, model: GPT, tok: EvolvingTokenizer):
    # And lo, asynchronous training shall occur, because sleeping is for humans.
    last_event_id = 0
    warmed_up = False
    qbuf = QuantumBuffer()
    syntracker = SyntropyTracker()
    field = CooccurField()

    while True:
        _ = update_reservoir_corpus(con, CFG.corpus_path, CFG.max_corpus_lines)
        mass, last_event_id = compute_new_corpus_mass(con, last_event_id)
        docs = load_corpus_lines(CFG.corpus_path)

        # Rebuild field from current corpus (the organism re-reads its own physics)
        if docs:
            field.build_from_corpus(tok, docs)

        # Tokenizer evolution (char -> BPE enablement) + safe vocab expansion
        bpe_just_enabled = tok.maybe_enable_bpe(docs)
        bpe_retrained = tok.maybe_retrain_bpe(docs)
        if bpe_just_enabled or bpe_retrained:
            with model.lock:
                model.maybe_expand_vocab(tok.vocab_size)
                save_checkpoint(model, tok)

        if (not warmed_up) and docs:
            print("[trainer] warmup training... (and so it begins)")
            train_steps(model, tok, docs, CFG.warmup_steps,
                        train_base=True, train_deltas=True)
            with model.lock:
                save_checkpoint(model, tok)
            db_log_growth(con, model, tok, docs, note="warmup_complete")
            warmed_up = True
            print("[trainer] warmup complete. base may freeze now, like a proud fossil.")

        if warmed_up and docs:
            qbuf.feed(mass, tok, docs)
            if qbuf.should_trigger():
                nov = qbuf.novelty_score()
                print(f"[trainer] quantum burst (bytes={qbuf.accumulated_bytes}, novelty={nov:.3f})")

                # SYNTROPY: measure before burst
                with model.lock:
                    pre_metrics = syntracker.measure(model, tok, field, docs)
                    entropy_before = pre_metrics["entropy"]

                    # SYNTROPY: decide how to learn (mathematical self-reasoning)
                    decision = syntracker.decide_action()
                    lr_mul = decision["lr_multiplier"]
                    action = decision["action"]
                    print(f"[syntropy] action={action} | trend={syntracker.syntropy_trend:.4f} "
                          f"| field_dev={syntracker.field_deviation:.3f} "
                          f"| purpose_align={syntracker.purpose_alignment:.3f} "
                          f"| lr_mul={lr_mul:.2f}")

                    # IMMUNE SYSTEM: snapshot before burst
                    pre_direction, pre_mag = model.gamma_contrastive_projection()
                    delta_snap = model.snapshot_deltas()

                # Apply syntropy-adjusted learning rate
                original_lr = CFG.learning_rate
                CFG.learning_rate = original_lr * lr_mul

                train_base = not CFG.freeze_base_after_warmup
                train_steps(model, tok, docs, CFG.micro_steps,
                            train_base=train_base, train_deltas=True)

                CFG.learning_rate = original_lr  # restore

                with model.lock:
                    # IMMUNE SYSTEM: check drift after burst
                    drift_cos = model.gamma_drift_check(pre_direction, pre_mag)
                    if drift_cos < CFG.noise_drift_threshold:
                        print(f"[immune] NOISE DETECTED (drift cosine={drift_cos:.3f}). Rolling back deltas.")
                        model.restore_deltas(delta_snap)
                        db_log_growth(con, model, tok, docs, note="noise_rejected")
                        syntracker.log_to_db(con, entropy_before, entropy_before, "noise_rejected")
                    else:
                        # SYNTROPY: measure after burst
                        post_metrics = syntracker.measure(model, tok, field, docs)
                        entropy_after = post_metrics["entropy"]
                        syntracker.log_to_db(con, entropy_before, entropy_after, action)
                        save_checkpoint(model, tok)
                        db_log_growth(con, model, tok, docs, note=f"quantum_burst:{action}")

                qbuf.reset()

                # Delta module growth — influenced by syntropy
                grow_prob = CFG.delta_grow_prob
                if decision.get("delta_grow_override") is not None:
                    grow_prob = decision["delta_grow_override"]
                if len(model.deltas) < CFG.max_delta_modules and random.random() < grow_prob:
                    print(f"[trainer] growing new delta module (total: {len(model.deltas)+1})")
                    with model.lock:
                        model.add_delta_module(alpha=1.0)
                        save_checkpoint(model, tok)

        await asyncio.sleep(CFG.train_tick_seconds)

# ============================================================
# 10) CHAT LOOP — tiny memory, tiny ego, continuous learning
# ============================================================

def build_prompt_from_memory(con, user_text):
    # Keep the prompt clean and stable.
    # Goal: teach dialogue, not prompt meta.
    recent = db_recent_messages(con, limit=14)

    def _clip(s, n=260):
        s = normalize_text(s)
        return s[:n].strip()

    parts = []
    # A tiny anchor so it doesn't drift into "random cough" mode.
    parts.append("A: (I listen. I answer. I learn.)")

    for role, text in recent[-12:]:
        tag = "H:" if role == "user" else "A:"
        parts.append(f"{tag} {_clip(text)}")

    parts.append(f"H: {_clip(user_text)}")
    parts.append("A:")
    return "\n".join(parts)





async def chat_main():
    con = init_db(CFG.db_path)

    if not os.path.exists(CFG.corpus_path):
        print(f"Seed corpus not found: {CFG.corpus_path}")
        print("Place nonames.txt alongside molecule.py to begin.")
        return

    docs = load_corpus_lines(CFG.corpus_path)

    model, tok = load_checkpoint(docs, CFG.ckpt_path)
    if model is None or tok is None:
        tok = EvolvingTokenizer(docs if docs else ["Hello."])
        model = GPT(tok)

    # Ensure tokenizer evolution can expand model
    model.maybe_expand_vocab(tok.vocab_size)

    trainer_task = asyncio.create_task(background_trainer(con, model, tok))

    print("molecule is alive. Type and press Enter. Ctrl+C to exit.\n")
    try:
        while True:
            user_text = await asyncio.get_event_loop().run_in_executor(None, input, "> ")
            user_text = user_text.strip()
            if not user_text:
                continue

            db_add_message(con, "user", user_text)

            # And lo, the reservoir shall be fed for future trainings.
            update_reservoir_corpus(con, CFG.corpus_path, CFG.max_corpus_lines)

            prompt = build_prompt_from_memory(con, user_text)
            answer = model.generate_sentence(prompt_text=prompt) or "..."

            print(answer)
            db_add_message(con, "assistant", answer)

    except KeyboardInterrupt:
        pass
    finally:
        trainer_task.cancel()
        try:
            await trainer_task
        except asyncio.CancelledError:
            pass
        save_checkpoint(model, tok)
        con.close()

# ============================================================
# 11) AWAKEN — now, when all is assembled as an organism, not a gearbox,
#             it is time to declare the final function.
# ============================================================

def main():
    asyncio.run(chat_main())

if __name__ == "__main__":
    main()
