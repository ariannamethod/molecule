#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
molecule.py
A dependency-free, single-file, async, continually-learning GPT organism.

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
from dataclasses import dataclass
from collections import Counter, defaultdict

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

    # deltas (LoRA-ish)
    delta_rank: int = 8
    max_delta_modules: int = 12
    delta_grow_prob: float = 0.08

    # generation
    temperature: float = 0.85
    top_k: int = 40
    top_p: float = 0.92
    max_gen_tokens: int = 180
    min_gen_tokens: int = 16
    repetition_guard: int = 4

    # tokenizer evolution
    enable_bpe_after_chars: int = 25000  # corpus size threshold to begin learning merges
    bpe_num_merges: int = 384
    bpe_retrain_every_chars: int = 4000  # retrain merges when corpus changes enough

    # async
    train_tick_seconds: float = 0.25


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

class VectorValue:
    """A differentiable vector. One object = one embedding / hidden state."""
    __slots__ = ("data", "grad", "_children", "_back_fn")

    def __init__(self, data, children=(), back_fn=None):
        self.data = list(data) if not isinstance(data, list) else data
        self.grad = [0.0] * len(self.data)
        self._children = children
        self._back_fn = back_fn

    def __add__(self, other):
        if isinstance(other, VectorValue):
            out = VectorValue([a + b for a, b in zip(self.data, other.data)],
                              (self, other))
            def _back():
                for i in range(len(self.data)):
                    self.grad[i] += out.grad[i]
                    other.grad[i] += out.grad[i]
            out._back_fn = _back
            return out
        s = float(other)
        out = VectorValue([a + s for a in self.data], (self,))
        def _back():
            for i in range(len(self.data)):
                self.grad[i] += out.grad[i]
        out._back_fn = _back
        return out

    def __radd__(self, other): return self.__add__(other)

    def __neg__(self):
        out = VectorValue([-a for a in self.data], (self,))
        def _back():
            for i in range(len(self.data)):
                self.grad[i] -= out.grad[i]
        out._back_fn = _back
        return out

    def __sub__(self, other):
        if isinstance(other, VectorValue):
            out = VectorValue([a - b for a, b in zip(self.data, other.data)],
                              (self, other))
            def _back():
                for i in range(len(self.data)):
                    self.grad[i] += out.grad[i]
                    other.grad[i] -= out.grad[i]
            out._back_fn = _back
            return out
        return self + (-float(other))

    def __mul__(self, other):
        if isinstance(other, VectorValue):
            out = VectorValue([a * b for a, b in zip(self.data, other.data)],
                              (self, other))
            def _back():
                for i in range(len(self.data)):
                    self.grad[i] += other.data[i] * out.grad[i]
                    other.grad[i] += self.data[i] * out.grad[i]
            out._back_fn = _back
            return out
        s = float(other)
        out = VectorValue([a * s for a in self.data], (self,))
        def _back():
            for i in range(len(self.data)):
                self.grad[i] += s * out.grad[i]
        out._back_fn = _back
        return out

    def __rmul__(self, other): return self.__mul__(other)

    def relu(self):
        out = VectorValue([max(0.0, a) for a in self.data], (self,))
        def _back():
            for i in range(len(self.data)):
                if self.data[i] > 0:
                    self.grad[i] += out.grad[i]
        out._back_fn = _back
        return out

    def squared_relu(self):
        r = [max(0.0, a) for a in self.data]
        out = VectorValue([x * x for x in r], (self,))
        def _back():
            for i in range(len(self.data)):
                if self.data[i] > 0:
                    self.grad[i] += 2.0 * r[i] * out.grad[i]
        out._back_fn = _back
        return out

    def dot(self, other):
        val = sum(a * b for a, b in zip(self.data, other.data))
        out = ScalarValue(val, (self, other))
        def _back():
            for i in range(len(self.data)):
                self.grad[i] += other.data[i] * out.grad
                other.grad[i] += self.data[i] * out.grad
        out._back_fn = _back
        return out

    def mean_sq(self):
        n = len(self.data)
        val = sum(a * a for a in self.data) / n
        out = ScalarValue(val, (self,))
        def _back():
            for i in range(len(self.data)):
                self.grad[i] += (2.0 * self.data[i] / n) * out.grad
        out._back_fn = _back
        return out

    def slice(self, start, end):
        out = VectorValue(self.data[start:end], (self,))
        def _back():
            for i, j in enumerate(range(start, end)):
                self.grad[j] += out.grad[i]
        out._back_fn = _back
        return out

    @staticmethod
    def concat(vecs):
        data = []
        for v in vecs:
            data.extend(v.data)
        out = VectorValue(data, tuple(vecs))
        def _back():
            offset = 0
            for v in vecs:
                for i in range(len(v.data)):
                    v.grad[i] += out.grad[offset + i]
                offset += len(v.data)
        out._back_fn = _back
        return out

class ScalarValue:
    """A differentiable scalar. For loss, dot products, attention weights."""
    __slots__ = ("data", "grad", "_children", "_back_fn")

    def __init__(self, data, children=(), back_fn=None):
        self.data = float(data)
        self.grad = 0.0
        self._children = children
        self._back_fn = back_fn

    def __add__(self, other):
        if isinstance(other, ScalarValue):
            out = ScalarValue(self.data + other.data, (self, other))
            def _back():
                self.grad += out.grad
                other.grad += out.grad
            out._back_fn = _back
            return out
        out = ScalarValue(self.data + float(other), (self,))
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
            out = ScalarValue(self.data * other.data, (self, other))
            def _back():
                self.grad += other.data * out.grad
                other.grad += self.data * out.grad
            out._back_fn = _back
            return out
        s = float(other)
        out = ScalarValue(self.data * s, (self,))
        def _back():
            self.grad += s * out.grad
        out._back_fn = _back
        return out

    def __rmul__(self, other): return self.__mul__(other)

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
        nout = self.nout
        nin = len(x.data)
        out_data = [sum(self.rows[i].data[j] * x.data[j]
                        for j in range(nin))
                    for i in range(nout)]
        out = VectorValue(out_data, tuple(self.rows) + (x,))
        rows_ref = self.rows
        def _back():
            for i in range(nout):
                g = out.grad[i]
                row_data = rows_ref[i].data
                row_grad = rows_ref[i].grad
                for j in range(nin):
                    row_grad[j] += g * x.data[j]
                    x.grad[j] += g * row_data[j]
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
    ms = x.mean_sq()
    scale_val = (ms.data + 1e-5) ** -0.5
    out = VectorValue([a * scale_val for a in x.data], (x, ms))
    n = len(x.data)
    def _back():
        s = scale_val
        ds_dms = -0.5 * (ms.data + 1e-5) ** -1.5
        cross = sum(out.grad[j] * x.data[j] for j in range(n))
        for i in range(n):
            x.grad[i] += s * out.grad[i]
            x.grad[i] += cross * ds_dms * (2.0 * x.data[i] / n)
    out._back_fn = _back
    return out

def cross_entropy_loss(logits, target):
    max_val = max(logits.data)
    shifted = [x - max_val for x in logits.data]
    exp_sum = sum(math.exp(x) for x in shifted)
    log_sum_exp = math.log(exp_sum) + max_val
    loss_val = log_sum_exp - logits.data[target]
    probs = [math.exp(x) / exp_sum for x in shifted]
    out = ScalarValue(loss_val, (logits,))
    def _back():
        g = out.grad
        for i in range(len(logits.data)):
            logits.grad[i] += (probs[i] - (1.0 if i == target else 0.0)) * g
    out._back_fn = _back
    return out

def scalar_softmax(logits):
    max_val = max(s.data for s in logits)
    exps_data = [math.exp(s.data - max_val) for s in logits]
    total = sum(exps_data)
    probs_data = [e / total for e in exps_data]

    out = []
    for i in range(len(probs_data)):
        sv = ScalarValue(probs_data[i], tuple(logits))
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
    out_data = [sum(weights[t].data * values[t].data[j] for t in range(T))
                for j in range(dim)]
    out = VectorValue(out_data, tuple(weights) + tuple(values))
    def _back():
        for t in range(T):
            w_t = weights[t]
            v_t = values[t]
            for j in range(dim):
                w_t.grad += v_t.data[j] * out.grad[j]
                v_t.grad[j] += w_t.data * out.grad[j]
    out._back_fn = _back
    return out

def softmax_probs_float(data):
    max_val = max(data)
    exps = [math.exp(x - max_val) for x in data]
    total = sum(exps)
    return [e / total for e in exps]

def top_k_top_p_sample(probs, k, p):
    # And lo, sampling shall not be a coin flip but a controlled hallucination.
    n = len(probs)
    idx = list(range(n))
    idx.sort(key=lambda i: probs[i], reverse=True)

    if k > 0:
        idx = idx[:min(k, len(idx))]

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
        for j in range(len(p.grad)):
            if p.grad[j] > clip:
                p.grad[j] = clip
            elif p.grad[j] < -clip:
                p.grad[j] = -clip

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
    RoPE rotation for one head slice.
    Implemented as a linear transform => autograd-friendly.
    """
    # And lo, positions shall become angles, and angles shall become meaning.
    x = vec.data[:]  # local
    out_data = x[:]
    # rotate pairs
    for i in range(0, head_dim, 2):
        if i + 1 >= head_dim:
            break
        theta = (pos / (10000.0 ** (i / head_dim)))
        c = math.cos(theta)
        s = math.sin(theta)
        a = x[i]
        b = x[i + 1]
        out_data[i] = a * c - b * s
        out_data[i + 1] = a * s + b * c

    out = VectorValue(out_data, (vec,))
    def _back():
        # inverse rotation is rotation by -theta (transpose of rotation matrix)
        for i in range(0, head_dim, 2):
            if i + 1 >= head_dim:
                break
            theta = (pos / (10000.0 ** (i / head_dim)))
            c = math.cos(theta)
            s = math.sin(theta)
            ga = out.grad[i]
            gb = out.grad[i + 1]
            vec.grad[i]     += ga * c + gb * s
            vec.grad[i + 1] += -ga * s + gb * c
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

        # Modular deltas
        self.deltas = []
        self.active_alpha = []

        # Adam state
        self._adam = {}

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

    def _ensure_adam(self, params, key):
        if key not in self._adam:
            self._adam[key] = {
                "m": [[0.0] * len(p.data) for p in params],
                "v": [[0.0] * len(p.data) for p in params],
                "t": 0
            }

    def adam_step(self, params, key, lr):
        # And lo, Adam Optimizer shall descend like a petty god with momentum.
        self._ensure_adam(params, key)
        st = self._adam[key]
        st["t"] += 1
        t = st["t"]
        b1, b2, eps = CFG.beta1, CFG.beta2, CFG.eps_adam
        b1_corr = 1.0 - b1 ** t
        b2_corr = 1.0 - b2 ** t

        clip_params(params, CFG.grad_clip)

        for i, p in enumerate(params):
            mi = st["m"][i]
            vi = st["v"][i]
            for j in range(len(p.data)):
                g = p.grad[j]
                mi[j] = b1 * mi[j] + (1 - b1) * g
                vi[j] = b2 * vi[j] + (1 - b2) * (g * g)
                mhat = mi[j] / b1_corr
                vhat = vi[j] / b2_corr
                p.data[j] -= lr * mhat / (math.sqrt(vhat) + eps)
                p.grad[j] = 0.0

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
            for h in range(self.n_head):
                hs = h * self.head_dim
                he = hs + self.head_dim

                qh = q.slice(hs, he)
                # RoPE on q
                qh = rope_rotate(qh, pos_id, self.head_dim)

                attn_logits = []
                for t in range(len(keys[li])):
                    kh_t = keys[li][t].slice(hs, he)
                    # RoPE on k at its own position t
                    kh_t = rope_rotate(kh_t, t, self.head_dim)
                    dot = qh.dot(kh_t) * (1.0 / math.sqrt(self.head_dim))
                    attn_logits.append(dot)

                attn_weights = scalar_softmax(attn_logits)

                vh = [values[li][t].slice(hs, he) for t in range(len(values[li]))]
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

            # small "gpt-ish" stability trick: adapt temperature slightly to confidence
            base_temp = float(CFG.temperature)
            if base_temp <= 1e-6:
                base_temp = 1e-6
            raw = logits.data
            raw_scaled = [v / base_temp for v in raw]
            probs0 = softmax_probs_float(raw_scaled)
            maxp = max(probs0) if probs0 else 0.0
            # if too peaky -> loosen; if too flat -> tighten
            t_mul = 1.0
            if maxp > 0.60:
                t_mul = 1.10
            elif maxp < 0.15:
                t_mul = 0.90
            temp = base_temp * t_mul
            scaled = [v / temp for v in raw]
            probs = softmax_probs_float(scaled)
            nxt = top_k_top_p_sample(probs, CFG.top_k, CFG.top_p)

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
    return [list(row.data) for row in mp.rows]

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

    return model, tok

# ============================================================
# 9) TRAINING — warmup, then continual micro-bursts
# ============================================================

def train_steps(model: GPT, tok: EvolvingTokenizer, docs, steps, train_base=True, train_deltas=True):
    if not docs:
        return

    base_params = model.all_base_params() if train_base else []
    delta_params = model.all_delta_params() if train_deltas else []

    for step in range(steps):
        # And lo, the training batch shall be sampled from the cursed book of names.
        batch = random.choices(docs, k=4)
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

async def background_trainer(con, model: GPT, tok: EvolvingTokenizer):
    # And lo, asynchronous training shall occur, because sleeping is for humans.
    last_event_id = 0
    warmed_up = False

    while True:
        _ = update_reservoir_corpus(con, CFG.corpus_path, CFG.max_corpus_lines)
        mass, last_event_id = compute_new_corpus_mass(con, last_event_id)
        docs = load_corpus_lines(CFG.corpus_path)

        # Tokenizer evolution (char -> BPE enablement) + safe vocab expansion
        bpe_just_enabled = tok.maybe_enable_bpe(docs)
        bpe_retrained = tok.maybe_retrain_bpe(docs)
        if bpe_just_enabled or bpe_retrained:
            # Ensure model can handle new vocab size
            model.maybe_expand_vocab(tok.vocab_size)
            save_checkpoint(model, tok)

        if (not warmed_up) and docs:
            print("[trainer] warmup training... (and so it begins)")
            train_steps(model, tok, docs, CFG.warmup_steps,
                        train_base=True, train_deltas=True)
            save_checkpoint(model, tok)
            warmed_up = True
            print("[trainer] warmup complete. base may freeze now, like a proud fossil.")

        if warmed_up and mass >= CFG.min_new_chars_to_train and docs:
            print(f"[trainer] micro-train burst ({mass} new chars) — and lo, it feeds again.")
            train_base = not CFG.freeze_base_after_warmup
            train_steps(model, tok, docs, CFG.micro_steps,
                        train_base=train_base, train_deltas=True)
            save_checkpoint(model, tok)

            # occasionally grow a new delta module
            if len(model.deltas) < CFG.max_delta_modules and random.random() < CFG.delta_grow_prob:
                print(f"[trainer] growing new delta module (total: {len(model.deltas)+1}) — new soul appended.")
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
