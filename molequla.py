#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
molequla.py
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
    ckpt_path: str = "molequla_ckpt.json"
    max_corpus_lines: int = 8000
    max_line_chars: int = 240

    # continual learning trigger (smaller than Karpathy's vibe, but not stupidly small)
    min_new_chars_to_train: int = 480  # And lo, the minimum mass shall be reached.

    # model
    tie_embeddings: bool = True  # GPT-style weight tying (wte == lm_head)

    n_layer: int = 1
    n_embd: int = 16           # embryo — organism starts small and grows
    n_head: int = 1
    block_size: int = 96

    # ontogenesis — growth stages (corpus_chars, n_embd, n_layer, n_head)
    growth_stages: tuple = (
        (0,      16, 1, 1),      # embryo: ~10K params
        (20000,  32, 1, 2),      # infant: ~28K params
        (50000,  64, 2, 4),      # child: ~154K params
        (200000, 128, 4, 4),     # adolescent: ~1.1M params
        (350000, 224, 5, 8),     # teen: ~4.1M params
        (500000, 320, 6, 8),     # adult: ~10M params
    )
    freeze_after_growth_steps: int = 500  # freeze base weights after growth, train only deltas
    post_growth_lr_scale: float = 0.3    # LR multiplier during freeze period (prevents delta overfit to noise)

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
    accum_steps: int = 1     # gradient accumulation (effective batch = batch_size * accum_steps)
    lr_min: float = 0.001
    max_total_steps: int = 50000
    cosine_warmup_steps: int = 200

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
    freq_penalty: float = 0.1
    presence_penalty: float = 0.1

    # tokenizer evolution
    enable_bpe_after_chars: int = 20000  # corpus size threshold to begin learning merges
    bpe_num_merges: int = 384
    bpe_retrain_every_chars: int = 4000  # retrain merges when corpus changes enough

    # async
    train_tick_seconds: float = 0.25

    # hybrid attention heads: "content", "rrpram", or "hybrid"
    head_types: tuple = ("content",)  # embryo: 1 head = 1 content
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
    corpus_fade_k: float = 3.0           # sigmoid steepness for corpus→model transition
    corpus_fade_threshold: float = 1.5   # entropy at which blend is 50/50
    cooccur_window_size: int = 5         # co-occurrence proximity window (Stanley-style)
    user_boost_strength: float = 0.3     # how strongly user's recent words are boosted
    user_boost_decay: float = 0.7        # per-generation decay of user word boost

    # quantum buffer
    qb_min_bytes: int = 1024
    qb_min_novelty: float = 0.15
    qb_cooldown_seconds: float = 60.0

    # consciousness: per-token dissonance feedback
    dissonance_ema_alpha: float = 0.3        # EMA smoothing for entropy within generation
    dissonance_spike_k: float = 0.8          # temp multiplier when entropy spikes
    dissonance_drop_k: float = 1.2           # temp multiplier when entropy drops
    dissonance_spike_threshold: float = 1.5  # entropy/EMA ratio triggering spike
    dissonance_drop_threshold: float = 0.5   # entropy/EMA ratio triggering drop

    # consciousness: pattern breaking (anti-field generation)
    anti_field_prob: float = 0.05      # probability of pure-model token (bypass corpus)
    anti_field_min_step: int = 8       # don't anti-field before this many tokens


    # consciousness: conscience (self-editing)
    conscience_window: int = 8         # rolling window for generation entropy trend
    conscience_decay: float = 0.95     # delta_alpha_scale reduction factor
    conscience_recovery: float = 1.005 # delta_alpha_scale recovery factor
    conscience_floor: float = 0.3      # minimum delta_alpha_scale


CFG = Config()

def head_types_for_n_head(n):
    """Compute head type tuple for a given number of heads."""
    if n <= 1:
        return ("content",)
    if n == 2:
        return ("content", "hybrid")
    half = (n + 1) // 2  # ceiling: majority content
    return tuple(["content"] * half + ["hybrid"] * (n - half))

# ============================================================
# 1) SQLITE MEMORY — and a small ghost shall remember
# ============================================================

def init_db(db_path: str):
    # And lo, a memory shall awaken in SQLite, because RAM is a liar.
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
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
        self.fourgram = defaultdict(Counter)         # (prev3,prev2,prev1) → {next: count}
        self.cooccur_window = defaultdict(Counter)   # token → {nearby_token: count} (Stanley-style proximity)
        self.user_boost = {}                         # token → float, temporary user word boosts (Leo-style)
        self.total_tokens = 0
        self.lock = threading.Lock()  # protects concurrent access (overthinkg + background trainer)

    def build_from_corpus(self, tok, docs):
        with self.lock:
            self.unigram.clear()
            self.bigram.clear()
            self.trigram.clear()
            self.fourgram.clear()
            self.cooccur_window.clear()
            self.total_tokens = 0
            window = CFG.cooccur_window_size
            for doc in docs:
                ids = tok.encode(doc)
                for i, tid in enumerate(ids):
                    self.unigram[tid] += 1
                    self.total_tokens += 1
                    if i >= 1:
                        self.bigram[ids[i - 1]][tid] += 1
                    if i >= 2:
                        self.trigram[(ids[i - 2], ids[i - 1])][tid] += 1
                # 4-grams: deeper context for child+ stages
                for i in range(len(ids) - 3):
                    ctx = (ids[i], ids[i + 1], ids[i + 2])
                    self.fourgram[ctx][ids[i + 3]] += 1
                # Co-occurrence window: "words that resonate together, stay together" (Stanley)
                for i in range(len(ids)):
                    center = ids[i]
                    start = max(0, i - window)
                    end = min(len(ids), i + window + 1)
                    for j in range(start, end):
                        if i != j:
                            self.cooccur_window[center][ids[j]] += 1

    def ingest_tokens(self, ids):
        """Incrementally add n-gram counts from a token sequence.
        Unlike build_from_corpus, does NOT clear existing data — adds on top.
        Used by overthinkg rings to enrich the field with the model's own output."""
        self.ingest_tokens_weighted(ids, 1.0)

    def ingest_tokens_weighted(self, ids, weight):
        """Add n-gram counts weighted by a factor.
        High weight = this text matters more (coherent output). Low = less influence.
        Stanley's observe_shard weights by resonance score; we weight by inverse entropy."""
        with self.lock:
            window = CFG.cooccur_window_size
            for tid in ids:
                self.unigram[tid] += weight
            for i in range(len(ids) - 1):
                self.bigram[ids[i]][ids[i + 1]] += weight
            for i in range(len(ids) - 2):
                self.trigram[(ids[i], ids[i + 1])][ids[i + 2]] += weight
            for i in range(len(ids) - 3):
                ctx = (ids[i], ids[i + 1], ids[i + 2])
                self.fourgram[ctx][ids[i + 3]] += weight
            # Co-occurrence window
            for i in range(len(ids)):
                center = ids[i]
                start = max(0, i - window)
                end = min(len(ids), i + window + 1)
                for j in range(start, end):
                    if i != j:
                        self.cooccur_window[center][ids[j]] += weight

    def absorb_user_words(self, ids):
        """Set temporary boosts for tokens the user just said.
        Like Leo's Santa Klaus but simpler: user words get multiplicative boost in generation."""
        with self.lock:
            # Decay existing boosts first
            to_delete = []
            for k, v in self.user_boost.items():
                nv = v * CFG.user_boost_decay
                if nv < 0.01:
                    to_delete.append(k)
                else:
                    self.user_boost[k] = nv
            for k in to_delete:
                del self.user_boost[k]
            # Boost user's tokens
            strength = CFG.user_boost_strength
            for tid in ids:
                self.user_boost[tid] = self.user_boost.get(tid, 0.0) + strength

    def decay_user_boost(self):
        """Reduce user word boosts after a generation."""
        with self.lock:
            to_delete = []
            for k, v in self.user_boost.items():
                nv = v * CFG.user_boost_decay
                if nv < 0.01:
                    to_delete.append(k)
                else:
                    self.user_boost[k] = nv
            for k in to_delete:
                del self.user_boost[k]

    def sample_next(self, context_ids, temperature=1.0):
        """4-gram -> trigram -> bigram -> unigram fallback, with co-occurrence window + user boost."""
        # Collect counts into a dict (sparse, like Go version)
        counts = {}
        found = False

        # Try 4-gram (deepest context)
        if len(context_ids) >= 3:
            ctx = (context_ids[-3], context_ids[-2], context_ids[-1])
            if ctx in self.fourgram and self.fourgram[ctx]:
                for tid, v in self.fourgram[ctx].items():
                    counts[tid] = counts.get(tid, 0.0) + v
                    found = True

        # Fallback to trigram
        if not found and len(context_ids) >= 2:
            key = (context_ids[-2], context_ids[-1])
            if key in self.trigram and self.trigram[key]:
                for tid, v in self.trigram[key].items():
                    counts[tid] = counts.get(tid, 0.0) + v
                    found = True

        # Fallback to bigram
        if not found and len(context_ids) >= 1:
            if context_ids[-1] in self.bigram and self.bigram[context_ids[-1]]:
                for tid, v in self.bigram[context_ids[-1]].items():
                    counts[tid] = counts.get(tid, 0.0) + v
                    found = True

        # Fallback to unigram
        if not found:
            for tid, v in self.unigram.items():
                counts[tid] = counts.get(tid, 0.0) + v

        if not counts:
            return 0

        # Blend with co-occurrence window (background resonance, always active)
        if len(context_ids) > 0:
            wnd = CFG.cooccur_window_size
            ctx_slice = context_ids[-wnd:] if len(context_ids) > wnd else context_ids
            for ctx_tok in ctx_slice:
                if ctx_tok in self.cooccur_window:
                    for tid, cnt in self.cooccur_window[ctx_tok].items():
                        counts[tid] = counts.get(tid, 0.0) + cnt * 0.3  # co-occurrence is softer than n-gram

        # Apply user word boost (multiplicative)
        if self.user_boost:
            for tid, boost in self.user_boost.items():
                if tid in counts and counts[tid] > 0:
                    counts[tid] *= (1.0 + boost)

        # Apply temperature and sample
        items = list(counts.items())
        if temperature > 0:
            logits_raw = [math.pow(max(c, 1e-10), 1.0 / max(temperature, 1e-6)) for _, c in items]
        else:
            logits_raw = [c for _, c in items]
        total = sum(logits_raw)
        if total <= 0:
            return random.choice(items)[0] if items else 0
        r = random.random() * total
        cumsum = 0.0
        for i, w in enumerate(logits_raw):
            cumsum += w
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
    recent_buf = []
    token_counts = {}
    eos_id = tok.stoi.get(tok.EOS, -1)
    bos_id = tok.stoi.get(tok.BOS, -1)

    # Consciousness: per-token dissonance tracking (Feature 1)
    entropy_ema = 0.0
    entropy_ema_init = False
    low_drop_count = 0
    entropy_sum = 0.0
    entropy_count = 0

    for step in range(CFG.max_gen_tokens):
        pos = min(len(ids) - 1, model.block_size - 1)
        logits = model.forward_step(cur, pos, keys, values)

        # Frequency/presence penalty on raw logits (before temperature)
        if CFG.freq_penalty > 0 or CFG.presence_penalty > 0:
            for tid, cnt in token_counts.items():
                if cnt > 0 and tid < len(logits.data):
                    logits.data[tid] -= CFG.freq_penalty * cnt
                    logits.data[tid] -= CFG.presence_penalty

        # Model probs with dissonance-adaptive temperature
        temp = CFG.temperature
        if temp <= 1e-6:
            temp = 1e-6
        scaled = (logits.data / temp).tolist()
        model_probs = softmax_probs_float(scaled)

        # Per-token entropy for dissonance
        probs_arr = np.array(model_probs)
        ent_mask = probs_arr > 1e-12
        entropy = -float(np.sum(probs_arr[ent_mask] * np.log(probs_arr[ent_mask])))
        entropy_sum += entropy
        entropy_count += 1

        # Consciousness: per-token dissonance feedback (Feature 1)
        dissonance_mul = 1.0
        if not entropy_ema_init:
            entropy_ema = entropy
            entropy_ema_init = True
        else:
            entropy_ema = CFG.dissonance_ema_alpha * entropy + (1.0 - CFG.dissonance_ema_alpha) * entropy_ema
            if entropy_ema > 1e-6:
                ratio = entropy / entropy_ema
                if ratio > CFG.dissonance_spike_threshold:
                    dissonance_mul = CFG.dissonance_spike_k
                    low_drop_count = 0
                elif ratio < CFG.dissonance_drop_threshold:
                    low_drop_count += 1
                    if low_drop_count >= 3:
                        dissonance_mul = CFG.dissonance_drop_k
                else:
                    low_drop_count = 0
        if dissonance_mul != 1.0:
            temp *= dissonance_mul
            scaled = (logits.data / temp).tolist()
            model_probs = softmax_probs_float(scaled)

        # corpus bias
        corpus_dist = None
        if len(ids) >= 2:
            key = (ids[-2], ids[-1])
            if key in field.trigram:
                corpus_dist = field.trigram[key]  # read-only, no copy needed
        if corpus_dist is None and len(ids) >= 1:
            if ids[-1] in field.bigram:
                corpus_dist = field.bigram[ids[-1]]

        if corpus_dist:
            total_c = sum(corpus_dist.values())
            corpus_probs = [0.0] * len(model_probs)
            for tid, cnt in corpus_dist.items():
                if tid < len(corpus_probs):
                    corpus_probs[tid] = cnt / total_c
            # Adaptive blend: model takes over as it becomes confident
            adaptive_model_w = 1.0 / (1.0 + math.exp(-CFG.corpus_fade_k * (CFG.corpus_fade_threshold - entropy)))
            blended = [adaptive_model_w * mp + (1.0 - adaptive_model_w) * cp
                       for mp, cp in zip(model_probs, corpus_probs)]
            total_b = sum(blended)
            if total_b > 0:
                blended = [b / total_b for b in blended]
            probs = blended
        else:
            probs = model_probs

        # Consciousness: pattern breaking (Feature 2)
        # "I could follow the field, but I choose to speak for myself"
        if step >= CFG.anti_field_min_step and CFG.anti_field_prob > 0 and random.random() < CFG.anti_field_prob:
            probs = model_probs  # pure model voice, bypass corpus

        nxt = top_k_top_p_sample(probs, CFG.top_k, CFG.top_p, CFG.min_p, CFG.typical_p)
        if nxt == eos_id and step >= CFG.min_gen_tokens:
            break
        if nxt == eos_id:
            continue

        ids.append(nxt)
        cur = nxt
        out_ids.append(nxt)
        token_counts[nxt] = token_counts.get(nxt, 0) + 1

        # Repetition guard: break if last rg*2 tokens are a repeating pattern
        recent_buf.append(nxt)
        rg = CFG.repetition_guard
        if len(recent_buf) > rg * 2:
            recent_buf = recent_buf[-(rg * 2):]
            if recent_buf[rg:] == recent_buf[:rg]:
                break

        # Sentence-end check
        if step >= CFG.min_gen_tokens and out_ids:
            text = tok.decode([bos_id] + out_ids + [eos_id])
            if text and text[-1] in '.!?':
                break

        if len(ids) >= model.block_size:
            ids = ids[-model.block_size:]
            keys = [[] for _ in range(model.n_layer)]
            values = [[] for _ in range(model.n_layer)]
            for p in range(len(ids) - 1):
                _ = model.forward_step(ids[p], p, keys, values)

    # Consciousness: store mean entropy for conscience (Feature 5)
    if entropy_count > 0:
        model.last_gen_entropy = entropy_sum / entropy_count

    return tok.decode([tok.stoi[tok.BOS]] + out_ids + [eos_id])


# ============================================================
# 3) TOKENIZER — byte-level BPE (GPT-3/4 style)
# ============================================================

import unicodedata

def _unicode_segment(text):
    """Pre-segmentation by Unicode category. BPE merges happen WITHIN segments only.
    Categories: letters (+marks), digits, whitespace, punctuation/symbols."""
    segments = []
    current = []
    current_cat = None
    for ch in text:
        cat = unicodedata.category(ch)
        if cat[0] == 'L' or cat[0] == 'M':  # letter or combining mark
            cat_group = 'L'
        elif cat[0] == 'N':  # number/digit
            cat_group = 'N'
        elif cat[0] == 'Z' or ch in ('\n', '\r', '\t'):  # whitespace
            cat_group = 'Z'
        else:  # punctuation, symbols, everything else
            cat_group = 'P'
        if cat_group != current_cat and current:
            segments.append(bytes(current))
            current = []
        current_cat = cat_group
        current.extend(ch.encode('utf-8'))
    if current:
        segments.append(bytes(current))
    return segments


class EvolvingTokenizer:
    """
    Byte-level BPE tokenizer (GPT-3/4 style).
    Bootstrap: 256 byte tokens + BOS + EOS + PAD = 259 tokens.
    BPE merges operate on byte sequences within Unicode segments.
    Vocab only EXPANDS — existing weights remain valid.
    """
    def __init__(self, docs=None):
        self.BOS = "<BOS>"
        self.EOS = "<EOS>"
        self.PAD = "<PAD>"

        # 256 byte tokens (hex strings like "0x00"..."0xff") + 3 special tokens
        self.tokens = [f"0x{i:02x}" for i in range(256)] + [self.BOS, self.EOS, self.PAD]
        self.stoi = {t: i for i, t in enumerate(self.tokens)}
        self.itos = {i: t for t, i in self.stoi.items()}
        self.vocab_size = len(self.tokens)  # 259

        # BPE state
        self.bpe_enabled = False
        self.merges = []        # list of (token_a, token_b) pairs, rank-ordered
        self.merge_to_tok = {}  # (a,b) -> merged_token string
        self._trained_chars = sum(len(d) for d in docs) if docs else 0

    def _bytes_to_token_ids(self, raw_bytes):
        """Convert raw bytes to base token IDs (0-255)."""
        return list(raw_bytes)

    def maybe_enable_bpe(self, docs):
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
        """Learn BPE merges from corpus. Operates on byte sequences within Unicode segments."""
        text = " ".join(docs)
        if not text:
            return

        # Segment and convert to byte-token sequences
        segments = _unicode_segment(text)
        # Count frequency of each byte-token sequence
        vocab = Counter()
        for seg in segments:
            tok_seq = tuple(self.tokens[b] for b in seg)  # e.g. ("0x48", "0x65", "0x6c")
            vocab[tok_seq] += 1

        for _ in range(num_merges):
            pairs = defaultdict(int)
            for tok_seq, freq in vocab.items():
                for i in range(len(tok_seq) - 1):
                    pairs[(tok_seq[i], tok_seq[i+1])] += freq
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            a, b = best
            new_tok = a + "+" + b  # e.g. "0x48+0x65"
            self.merges.append(best)
            self.merge_to_tok[best] = new_tok

            # Merge in vocab
            new_vocab = Counter()
            for tok_seq, freq in vocab.items():
                out = []
                i = 0
                while i < len(tok_seq):
                    if i < len(tok_seq) - 1 and (tok_seq[i], tok_seq[i+1]) == best:
                        out.append(new_tok)
                        i += 2
                    else:
                        out.append(tok_seq[i])
                        i += 1
                new_vocab[tuple(out)] += freq
            vocab = new_vocab

            # Add token to vocab if new
            if new_tok not in self.stoi:
                self.stoi[new_tok] = len(self.tokens)
                self.tokens.append(new_tok)

        self.itos = {i: t for t, i in self.stoi.items()}
        self.vocab_size = len(self.tokens)

    def _apply_bpe(self, token_seq):
        """Apply learned BPE merges to a sequence of tokens (greedy, lowest-rank first).
        Uses in-place list mutation (pop+insert) instead of O(n) slice rebuilds."""
        if not self.merges:
            return token_seq

        symbols = list(token_seq)
        rank = {pair: i for i, pair in enumerate(self.merges)}

        while len(symbols) >= 2:
            # Find pair with lowest merge rank
            best_rank = 10**9
            best_idx = -1
            for i in range(len(symbols) - 1):
                r = rank.get((symbols[i], symbols[i+1]), 10**9)
                if r < best_rank:
                    best_rank = r
                    best_idx = i
            if best_rank == 10**9:
                break
            pair = (symbols[best_idx], symbols[best_idx+1])
            # In-place mutation: replace two symbols with merged token
            symbols[best_idx] = self.merge_to_tok[pair]
            del symbols[best_idx + 1]
        return symbols

    def encode(self, s: str):
        """Encode text to token IDs: text → segments → bytes → BPE → IDs."""
        s = s.strip()
        ids = [self.stoi[self.BOS]]

        if not s:
            ids.append(self.stoi[self.EOS])
            return ids

        segments = _unicode_segment(s)
        for seg in segments:
            # Convert bytes to base token names
            base_tokens = tuple(self.tokens[b] for b in seg)
            if self.bpe_enabled:
                merged = self._apply_bpe(base_tokens)
            else:
                merged = base_tokens
            for tok in merged:
                if tok in self.stoi:
                    ids.append(self.stoi[tok])

        ids.append(self.stoi[self.EOS])
        return ids

    def _token_to_bytes(self, tok):
        """Convert a token string back to bytes."""
        if tok.startswith("0x") and "+" not in tok and len(tok) == 4:
            # Single byte token: "0x41" → bytes([0x41])
            return bytes([int(tok, 16)])
        elif "+" in tok:
            # Merged token: "0x48+0x65" → recurse
            parts = tok.split("+")
            result = b""
            # Rebuild from parts — each part is either "0xNN" or a sub-merge
            i = 0
            while i < len(parts):
                result += bytes([int(parts[i], 16)])
                i += 1
            return result
        return b""

    def decode(self, ids):
        """Decode token IDs back to text: IDs → bytes → UTF-8."""
        raw_bytes = b""
        for t in ids:
            tok = self.itos.get(t, "")
            if tok in (self.BOS, self.PAD):
                continue
            if tok == self.EOS:
                break
            raw_bytes += self._token_to_bytes(tok)
        try:
            return raw_bytes.decode('utf-8', errors='replace').strip()
        except Exception:
            return raw_bytes.decode('utf-8', errors='replace').strip()

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

    def silu(self):
        sig = 1.0 / (1.0 + np.exp(-self.data))
        out = VectorValue(self.data * sig)
        if _GRAD_ENABLED:
            out._children = (self,)
            def _back():
                # d/dx[x * sigmoid(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
                #                       = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
                self.grad += (sig * (1.0 + self.data * (1.0 - sig))) * out.grad
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

    # Clean up graph references to free intermediate nodes
    for v in topo:
        v._children = ()
        v._back_fn = None

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
        self._W_cache = None

    def invalidate_cache(self):
        self._W_cache = None

    def matvec(self, x):
        # And lo, BLAS shall do the heavy lifting — numpy @ is 50-100x faster than Python loops.
        if self._W_cache is None:
            self._W_cache = np.vstack([row.data for row in self.rows])  # (nout, nin)
        W = self._W_cache
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
        self._W_cache = None

    def grow_cols(self, new_nin, std=0.001):
        # And lo, the matrix shall widen its reach, each row stretching into new dimensions.
        if new_nin <= self.nin:
            return
        for row in self.rows:
            ext = np.array([random.gauss(0, std) for _ in range(new_nin - self.nin)])
            row.data = np.concatenate([row.data, ext])
            if row.grad is not None:
                row.grad = np.concatenate([row.grad, np.zeros(new_nin - self.nin)])
        self.nin = new_nin
        self._W_cache = None

    def grow(self, new_nout, new_nin, std=0.001):
        # Ontogenesis: grow both dimensions. Cols first so new rows get full width.
        self.grow_cols(new_nin, std)
        self.grow_rows(new_nout, std)

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

    def grow_dims(self, new_nout, new_nin):
        # Ontogenesis: grow both outer dimensions of the adapter. Rank stays the same.
        self.A.grow_rows(new_nout)    # A: (nout, r) → extend output
        self.B.grow_cols(new_nin)     # B: (r, nin) → extend input

    def params(self):
        return self.A.params() + self.B.params()

# ============================================================
# 7) GPT MODEL — a small beast with RoPE (GPT-3-ish spice)
# ============================================================

_ROPE_CACHE = {}

def _get_rope_cos_sin(pos, head_dim):
    """Cached RoPE cos/sin computation. Avoids recomputing thetas every call."""
    key = (pos, head_dim)
    if key not in _ROPE_CACHE:
        n_pairs = head_dim // 2
        indices = np.arange(0, 2 * n_pairs, 2, dtype=np.float64)
        thetas = pos / (10000.0 ** (indices / head_dim))
        _ROPE_CACHE[key] = (np.cos(thetas), np.sin(thetas))
    return _ROPE_CACHE[key]

def rope_rotate(vec, pos, head_dim):
    """
    RoPE rotation for one head slice — numpy vectorized.
    """
    # And lo, positions shall become angles, and angles shall become meaning.
    cos_t, sin_t = _get_rope_cos_sin(pos, head_dim)

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
        self.residual_alpha = 1.0 / math.sqrt(max(1, CFG.n_layer))
        self.global_step = 0
        self.last_warmup_stage = -1          # per-stage warmup: last stage that completed warmup
        self.growth_step_offset = 0          # global_step at last growth event (for LR warmup reset)
        self.syntropy_temp_offset = 0.0  # temperature bridge from syntropy state
        self._growth_freeze_remaining = 0  # ontogenesis: freeze base after growth
        self._corpus_field = None  # set by background_trainer for adaptive blend

        # consciousness state
        self.delta_alpha_scale = 1.0         # conscience: multiplier on all delta contributions
        self.generation_entropy_history = []  # conscience: rolling window of per-generation mean entropy
        self.last_surprise = 0.0             # self-prediction error on last prompt
        self.surprise_baseline = 0.0         # EMA of surprise over time
        self.last_gen_entropy = 0.0          # mean entropy of last generation (for conscience)

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

    # ---- Ontogenesis (architecture growth) ----
    # And lo, the organism shall not be born adult but shall grow, stage by stage,
    # from embryo to child to adolescent, each growth a small death and rebirth.

    def current_growth_stage(self):
        """Return index of current stage based on model dimensions."""
        for i, (_, embd, layer, head) in enumerate(CFG.growth_stages):
            if self.n_embd == embd and self.n_layer == layer and self.n_head == head:
                return i
        return -1  # dimensions don't match any stage (legacy checkpoint)

    def target_growth_stage(self, corpus_chars):
        """Return the target stage index based on corpus size."""
        target = 0
        for i, (thresh, _, _, _) in enumerate(CFG.growth_stages):
            if corpus_chars >= thresh:
                target = i
        return target

    def maybe_grow_architecture(self, corpus_chars):
        """Check if growth is needed and execute it. Returns True if grew."""
        current = self.current_growth_stage()
        if current < 0:
            return False  # legacy checkpoint, skip growth
        if self._growth_freeze_remaining > 0:
            return False  # still stabilizing from last growth
        target = self.target_growth_stage(corpus_chars)
        if target <= current:
            return False
        # Grow only one stage at a time — prevent catastrophic multi-stage jumps
        target = current + 1

        _, new_embd, new_layer, new_head = CFG.growth_stages[target]
        old_embd = self.n_embd
        old_layer = self.n_layer
        old_head = self.n_head
        new_head_dim = new_embd // new_head

        print(f"[growth] ONTOGENESIS: stage {current} -> {target}")
        print(f"  embd: {old_embd} -> {new_embd}, layer: {old_layer} -> {new_layer}, head: {old_head} -> {new_head}")

        # 1. Grow embedding matrices (columns only — vocab rows stay)
        self.base["wte"].grow_cols(new_embd)
        self.base["wpe"].grow_cols(new_embd)
        if not getattr(CFG, "tie_embeddings", False):
            self.base["lm_head"].grow_cols(new_embd)

        # 2. Grow existing layer matrices
        new_htypes = head_types_for_n_head(new_head)
        for li in range(old_layer):
            for name in ("wq", "wk", "wv", "wo"):
                self.base[f"l{li}.{name}"].grow(new_embd, new_embd)
            self.base[f"l{li}.fc_g"].grow(4 * new_embd, new_embd)
            self.base[f"l{li}.fc_v"].grow(4 * new_embd, new_embd)
            self.base[f"l{li}.fc2"].grow(new_embd, 4 * new_embd)
            # Grow existing head pattern matrices
            for h in range(old_head):
                pkey = f"l{li}.h{h}.w_pattern"
                if pkey in self.base:
                    self.base[pkey].grow_cols(new_head_dim)
            # Add new heads for existing layer
            for h in range(old_head, new_head):
                htype = new_htypes[h] if h < len(new_htypes) else "content"
                if htype in ("rrpram", "hybrid"):
                    self.base[f"l{li}.h{h}.w_pattern"] = MatrixParam(
                        CFG.block_size, new_head_dim, 0.08)
                if htype == "hybrid":
                    self.base[f"l{li}.h{h}.alpha"] = MatrixParam(1, 1, 0.0)
                    self.base[f"l{li}.h{h}.alpha"].rows[0].data[0] = CFG.hybrid_alpha_init

        # 3. Add entirely new layers
        for li in range(old_layer, new_layer):
            self.base[f"l{li}.wq"] = MatrixParam(new_embd, new_embd, 0.08)
            self.base[f"l{li}.wk"] = MatrixParam(new_embd, new_embd, 0.08)
            self.base[f"l{li}.wv"] = MatrixParam(new_embd, new_embd, 0.08)
            self.base[f"l{li}.wo"] = MatrixParam(new_embd, new_embd, 0.08)
            self.base[f"l{li}.fc_g"] = MatrixParam(4 * new_embd, new_embd, 0.08)
            self.base[f"l{li}.fc_v"] = MatrixParam(4 * new_embd, new_embd, 0.08)
            self.base[f"l{li}.fc2"]  = MatrixParam(new_embd, 4 * new_embd, 0.08)
            for h in range(new_head):
                htype = new_htypes[h] if h < len(new_htypes) else "content"
                if htype in ("rrpram", "hybrid"):
                    self.base[f"l{li}.h{h}.w_pattern"] = MatrixParam(
                        CFG.block_size, new_head_dim, 0.08)
                if htype == "hybrid":
                    self.base[f"l{li}.h{h}.alpha"] = MatrixParam(1, 1, 0.0)
                    self.base[f"l{li}.h{h}.alpha"].rows[0].data[0] = CFG.hybrid_alpha_init

        # 4. Grow delta adapters
        r = CFG.delta_rank
        for mod in self.deltas:
            # Grow existing layer adapters
            for li in range(old_layer):
                for name in ("wq", "wk", "wv", "wo"):
                    key = f"l{li}.{name}"
                    if key in mod:
                        mod[key].grow_dims(new_embd, new_embd)
                for key, (nout_m, nin_m) in [(f"l{li}.fc_g", (4, 1)),
                                              (f"l{li}.fc_v", (4, 1)),
                                              (f"l{li}.fc2", (1, 4))]:
                    if key in mod:
                        mod[key].grow_dims(nout_m * new_embd, nin_m * new_embd)
                for h in range(old_head):
                    pkey = f"l{li}.h{h}.w_pattern"
                    if pkey in mod:
                        mod[pkey].grow_dims(CFG.block_size, new_head_dim)
                for h in range(old_head, new_head):
                    htype = new_htypes[h] if h < len(new_htypes) else "content"
                    if htype in ("rrpram", "hybrid"):
                        mod[f"l{li}.h{h}.w_pattern"] = DeltaAdapter(
                            CFG.block_size, new_head_dim, r)

            # New layers: entirely new adapters
            for li in range(old_layer, new_layer):
                for name in ("wq", "wk", "wv", "wo"):
                    mod[f"l{li}.{name}"] = DeltaAdapter(new_embd, new_embd, r)
                mod[f"l{li}.fc_g"] = DeltaAdapter(4 * new_embd, new_embd, r)
                mod[f"l{li}.fc_v"] = DeltaAdapter(4 * new_embd, new_embd, r)
                mod[f"l{li}.fc2"]  = DeltaAdapter(new_embd, 4 * new_embd, r)
                for h in range(new_head):
                    htype = new_htypes[h] if h < len(new_htypes) else "content"
                    if htype in ("rrpram", "hybrid"):
                        mod[f"l{li}.h{h}.w_pattern"] = DeltaAdapter(
                            CFG.block_size, new_head_dim, r)

            # lm_head adapter input grew
            if "lm_head" in mod:
                mod["lm_head"].grow_dims(self.tok.vocab_size, new_embd)

        # 5. Update model state
        self.n_embd = new_embd
        self.n_layer = new_layer
        self.n_head = new_head
        self.head_dim = new_head_dim
        self.residual_alpha = 1.0 / math.sqrt(max(1, new_layer))

        # 6. Update CFG runtime
        CFG.n_embd = new_embd
        CFG.n_layer = new_layer
        CFG.n_head = new_head
        CFG.head_types = head_types_for_n_head(new_head)

        # 7. Reset Adam state (old momentum is meaningless after arch change)
        self._adam = {}
        self.growth_step_offset = self.global_step  # reset LR warmup on growth

        # 8. Extend gamma snapshot for new embedding dimensions
        for i in range(len(self._init_embed_snapshot)):
            old_row = self._init_embed_snapshot[i]
            if len(old_row) < new_embd:
                self._init_embed_snapshot[i] = old_row + [0.0] * (new_embd - len(old_row))

        # 9. Set freeze (only train deltas until new weights stabilize)
        self._growth_freeze_remaining = CFG.freeze_after_growth_steps

        print(f"[growth] Done. Freeze for {CFG.freeze_after_growth_steps} steps.")
        return True

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

        # Invalidate W caches on all MatrixParams (weights changed)
        for mp in self.base.values():
            mp._W_cache = None
        for mod in self.deltas:
            for da in mod.values():
                da.A._W_cache = None
                da.B._W_cache = None

    def _apply_with_deltas(self, name, x):
        # And lo, base weight shall speak, then deltas shall harmonize atop it.
        y = self.base[name].matvec(x)
        for alpha, mod in zip(self.active_alpha, self.deltas):
            if name in mod:
                # Consciousness: conscience scales delta influence (Feature 5)
                effective_alpha = alpha * self.delta_alpha_scale
                y = y + (mod[name].apply(x) * effective_alpha)
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

            # Sliding window: keep only last block_size entries in KV cache
            if len(keys[li]) > self.block_size:
                keys[li] = keys[li][-self.block_size:]
                values[li] = values[li][-self.block_size:]

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
                    p_len = len(pattern_full.data)
                    rrpram_logits = [pattern_full.element(min(t, p_len - 1)) for t in range(T)]

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
            attn_out = self._apply_with_deltas(f"l{li}.wo", x_attn)
            x = x_res + attn_out * self.residual_alpha

            # ---- Gated MLP (SwiGLU-ish) ----
            x_res = x
            x = rmsnorm(x)

            g = self._apply_with_deltas(f"l{li}.fc_g", x).silu()   # gate (SwiGLU)
            u = self._apply_with_deltas(f"l{li}.fc_v", x)          # value
            x = g * u                                              # gating

            mlp_out = self._apply_with_deltas(f"l{li}.fc2", x)
            x = x_res + mlp_out * self.residual_alpha

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

    def quick_loss(self, tok, docs, n=4):
        """Fast loss on a few random docs without backward. For self-meta-learning."""
        if not docs:
            return 0.0
        with no_grad():
            total = 0.0
            for _ in range(n):
                doc = random.choice(docs)
                ids = tok.encode(doc)
                if len(ids) > 1:
                    loss = self.loss_on_sequence(ids)
                    total += loss.data
            return total / n

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

        # Frequency / presence penalty token tracking
        token_counts = {}

        # Consciousness: per-token dissonance tracking (Feature 1)
        entropy_ema = 0.0
        entropy_ema_init = False
        low_drop_count = 0       # consecutive tokens below drop threshold
        entropy_sum = 0.0        # for conscience mean entropy
        entropy_count = 0

        for step in range(CFG.max_gen_tokens):
            pos = min(len(ids) - 1, self.block_size - 1)
            logits = self.forward_step(cur, pos, keys, values)

            # entropy-adaptive temperature + syntropy bridge
            base_temp = float(CFG.temperature) + self.syntropy_temp_offset
            if base_temp <= 1e-6:
                base_temp = 1e-6
            raw = logits.data
            # Frequency / presence penalty: penalize repeated tokens
            if CFG.freq_penalty > 0 or CFG.presence_penalty > 0:
                for tid, cnt in token_counts.items():
                    if tid < len(raw):
                        raw[tid] -= CFG.freq_penalty * cnt
                        if cnt > 0:
                            raw[tid] -= CFG.presence_penalty
            raw_scaled = (raw / base_temp).tolist()
            probs = softmax_probs_float(raw_scaled)
            # Compute entropy via numpy (vectorized, no Python loop)
            probs_arr = np.array(probs)
            mask = probs_arr > 1e-12
            entropy = -float(np.sum(probs_arr[mask] * np.log(probs_arr[mask])))
            entropy_sum += entropy
            entropy_count += 1

            t_mul = 1.0
            if entropy < CFG.entropy_low:
                t_mul = CFG.entropy_temp_boost
            elif entropy > CFG.entropy_high:
                t_mul = CFG.entropy_temp_focus

            # Consciousness: per-token dissonance feedback (Feature 1)
            # "I notice my confidence shifting and adapt in real-time"
            dissonance_mul = 1.0
            if not entropy_ema_init:
                entropy_ema = entropy
                entropy_ema_init = True
            else:
                entropy_ema = CFG.dissonance_ema_alpha * entropy + (1.0 - CFG.dissonance_ema_alpha) * entropy_ema
                if entropy_ema > 1e-6:
                    ratio = entropy / entropy_ema
                    if ratio > CFG.dissonance_spike_threshold:
                        # Entropy spike — something surprising, be careful
                        dissonance_mul = CFG.dissonance_spike_k
                        low_drop_count = 0
                    elif ratio < CFG.dissonance_drop_threshold:
                        low_drop_count += 1
                        if low_drop_count >= 3:
                            # Sustained low entropy — getting repetitive, explore
                            dissonance_mul = CFG.dissonance_drop_k
                    else:
                        low_drop_count = 0

            final_mul = t_mul * dissonance_mul
            # Only recompute softmax if temperature actually changed
            if final_mul != 1.0:
                temp = base_temp * final_mul
                scaled = (raw / temp).tolist()
                probs = softmax_probs_float(scaled)

            # Save model-only probs (post-dissonance, pre-blend) for anti-field bypass
            model_only_probs = list(probs)

            # Adaptive corpus blend: corpus field fades as model becomes coherent
            # Now with 4-gram + co-occurrence window + user word boost (Stanley/Leo-style)
            if self._corpus_field and self._corpus_field.bigram:
                # sigmoid: low entropy -> high model_alpha, high entropy -> low model_alpha
                model_alpha = 1.0 / (1.0 + math.exp(-CFG.corpus_fade_k * (CFG.corpus_fade_threshold - entropy)))
                if model_alpha < 0.99:  # worth blending
                    # Best n-gram distribution: try 4-gram -> trigram -> bigram
                    ngram_dist = None
                    if ngram_dist is None and len(ids) >= 3:
                        ctx = (ids[-3], ids[-2], ids[-1])
                        if ctx in self._corpus_field.fourgram and self._corpus_field.fourgram[ctx]:
                            ngram_dist = self._corpus_field.fourgram[ctx]
                    if ngram_dist is None and len(ids) >= 2:
                        key = (ids[-2], ids[-1])
                        if key in self._corpus_field.trigram and self._corpus_field.trigram[key]:
                            ngram_dist = self._corpus_field.trigram[key]
                    if ngram_dist is None and len(ids) >= 1:
                        if ids[-1] in self._corpus_field.bigram and self._corpus_field.bigram[ids[-1]]:
                            ngram_dist = self._corpus_field.bigram[ids[-1]]

                    # Co-occurrence window: "words that resonate together" (Stanley)
                    cooccur_sum = None
                    if len(ids) > 0:
                        wnd = CFG.cooccur_window_size
                        ctx_slice = ids[-wnd:] if len(ids) > wnd else ids
                        for ctx_tok in ctx_slice:
                            if ctx_tok in self._corpus_field.cooccur_window:
                                if cooccur_sum is None:
                                    cooccur_sum = {}
                                for tid, cnt in self._corpus_field.cooccur_window[ctx_tok].items():
                                    cooccur_sum[tid] = cooccur_sum.get(tid, 0.0) + cnt

                    # User word boost snapshot
                    user_boost = None
                    if self._corpus_field.user_boost:
                        user_boost = dict(self._corpus_field.user_boost)

                    # Build final corpus distribution: 70% n-gram + 30% co-occurrence
                    has_corpus = ngram_dist is not None or cooccur_sum is not None
                    if has_corpus:
                        corpus_probs = [0.0] * len(probs)
                        if ngram_dist is not None:
                            total_n = sum(ngram_dist.values())
                            if total_n > 0:
                                for tid, cnt in ngram_dist.items():
                                    if tid < len(corpus_probs):
                                        corpus_probs[tid] += 0.7 * cnt / total_n
                        if cooccur_sum is not None:
                            total_c = sum(cooccur_sum.values())
                            if total_c > 0:
                                for tid, cnt in cooccur_sum.items():
                                    if tid < len(corpus_probs):
                                        corpus_probs[tid] += 0.3 * cnt / total_c
                        # Blend model probs with corpus
                        total_b = 0.0
                        for i in range(len(probs)):
                            if i < len(corpus_probs):
                                probs[i] = model_alpha * probs[i] + (1.0 - model_alpha) * corpus_probs[i]
                            total_b += probs[i]
                        if total_b > 0:
                            probs = [p / total_b for p in probs]

                    # User word boost: multiplicative, scaled by (1-model_alpha) so it fades
                    # as the transformer strengthens. "The organism echoes the words of those
                    # who speak to it" (Leo) — but grows out of it.
                    if user_boost is not None:
                        boost_scale = 1.0 - model_alpha
                        if boost_scale > 0.01:
                            total_b = 0.0
                            for i in range(len(probs)):
                                if i in user_boost:
                                    probs[i] *= (1.0 + user_boost[i] * boost_scale)
                                total_b += probs[i]
                            if total_b > 0:
                                probs = [p / total_b for p in probs]

            # Consciousness: pattern breaking (Feature 2)
            # "I could follow the field, but I choose to speak for myself"
            if step >= CFG.anti_field_min_step and CFG.anti_field_prob > 0 and random.random() < CFG.anti_field_prob:
                # Use pure model probs (post-dissonance, pre-blend), bypass corpus
                probs = model_only_probs

            nxt = top_k_top_p_sample(probs, CFG.top_k, CFG.top_p, CFG.min_p, CFG.typical_p)

            if nxt == self.tok.stoi[self.tok.EOS]:
                if step >= CFG.min_gen_tokens:
                    break
                # else: ignore early EOS and keep going
                continue

            ids.append(nxt)
            cur = nxt
            out_ids.append(nxt)
            token_counts[nxt] = token_counts.get(nxt, 0) + 1

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

        # Consciousness: store mean entropy for conscience (Feature 5)
        if entropy_count > 0:
            self.last_gen_entropy = entropy_sum / entropy_count

        return self.tok.decode([self.tok.stoi[self.tok.BOS]] + out_ids + [self.tok.stoi[self.tok.EOS]])

    # ============================================================
    # 7b) CONSCIOUSNESS — mathematical self-awareness
    # ============================================================

    def conscience_check(self, gen_mean_entropy):
        """Track generation quality over time via entropy trend.
        If entropy trend rises (output degrading), soften delta influence.
        If entropy trend falls (improving), recover delta influence.
        'I notice I'm getting worse and pull back.'"""
        self.generation_entropy_history.append(gen_mean_entropy)
        w = CFG.conscience_window
        if len(self.generation_entropy_history) > w:
            self.generation_entropy_history = self.generation_entropy_history[-w:]
        if len(self.generation_entropy_history) < 3:
            return  # not enough data

        # Linear regression slope on entropy history
        n = len(self.generation_entropy_history)
        sum_x, sum_y, sum_xy, sum_x2 = 0.0, 0.0, 0.0, 0.0
        for i, e in enumerate(self.generation_entropy_history):
            x = float(i)
            sum_x += x
            sum_y += e
            sum_xy += x * e
            sum_x2 += x * x
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x + 1e-12)

        if slope > 0.01:
            # Entropy increasing — generation degrading, reduce delta influence
            self.delta_alpha_scale *= CFG.conscience_decay
            if self.delta_alpha_scale < CFG.conscience_floor:
                self.delta_alpha_scale = CFG.conscience_floor
        elif slope < -0.01:
            # Entropy decreasing — improving, recover delta influence
            self.delta_alpha_scale *= CFG.conscience_recovery
            if self.delta_alpha_scale > 1.0:
                self.delta_alpha_scale = 1.0

    def compute_self_prediction_error(self, ids):
        """Measure how 'surprised' the model is by input tokens.
        Forward pass on ids, compute mean cross-entropy between predicted and actual.
        Higher error = 'I didn't expect this input' = increase attention."""
        if len(ids) < 2:
            return 0.0
        keys = [[] for _ in range(self.n_layer)]
        values = [[] for _ in range(self.n_layer)]

        total_ce = 0.0
        count = 0
        for pos in range(len(ids) - 1):
            logits = self.forward_step(ids[pos], pos, keys, values)
            probs = softmax_probs_float((logits.data).tolist())
            target = ids[pos + 1]
            if target < len(probs) and probs[target] > 1e-12:
                total_ce -= math.log(probs[target])
            else:
                total_ce += 10.0  # max penalty for unknown token
            count += 1
        if count == 0:
            return 0.0
        return total_ce / count

def _serialize_matrix_param(mp):
    return [row.data.tolist() for row in mp.rows]

def _deserialize_matrix_param(data):
    mp = MatrixParam.__new__(MatrixParam)
    mp.rows = [VectorValue(row) for row in data]
    mp.nout = len(data)
    mp.nin = len(data[0]) if data else 0
    mp._W_cache = None
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
        "global_step": model.global_step,
        "last_warmup_stage": model.last_warmup_stage,
        "growth_step_offset": model.growth_step_offset,
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
    tok.merge_to_tok = {tuple(p): (p[0] + "+" + p[1]) for p in tok.merges}
    tok.bpe_enabled = bool(t.get("bpe_enabled", False))
    tok._trained_chars = int(t.get("trained_chars", 0))

    # Restore model dimensions from checkpoint (ontogenesis may have changed them)
    saved_cfg = obj.get("cfg", {})
    if "n_embd" in saved_cfg:
        CFG.n_embd = saved_cfg["n_embd"]
    if "n_layer" in saved_cfg:
        CFG.n_layer = saved_cfg["n_layer"]
    if "n_head" in saved_cfg:
        CFG.n_head = saved_cfg["n_head"]
    if "head_types" in saved_cfg and saved_cfg["head_types"]:
        CFG.head_types = tuple(saved_cfg["head_types"])

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

    model.global_step = obj.get("global_step", 0)
    model.growth_step_offset = obj.get("growth_step_offset", 0)

    # Backward compat: old checkpoints without last_warmup_stage
    saved_warmup = obj.get("last_warmup_stage", None)
    if saved_warmup is not None:
        model.last_warmup_stage = saved_warmup
    elif model.global_step > 0:
        # Old checkpoint that has been trained — assume current stage is warmed up
        model.last_warmup_stage = model.current_growth_stage()
    else:
        model.last_warmup_stage = -1

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
        self.burst_history = []       # last 16 burst outcomes — training efficiency memory
        self.model_stage = 0          # current growth stage (set during measure)
        self._last_mitosis_time = 0.0 # cooldown for divide
        self._swarm_info = None       # peer state from mesh.db (set externally)

    def record_burst(self, action, loss_before, loss_after):
        """Log a burst outcome for self-meta-learning."""
        self.burst_history.append({"action": action, "loss_before": loss_before, "loss_after": loss_after})
        if len(self.burst_history) > 16:
            self.burst_history = self.burst_history[-16:]

    def action_effectiveness(self, action):
        """Mean loss delta for a given action. Negative = good (loss went down)."""
        deltas = [b["loss_after"] - b["loss_before"] for b in self.burst_history if b["action"] == action]
        if not deltas:
            return 0.0, 0
        return sum(deltas) / len(deltas), len(deltas)

    def measure(self, model, tok, field, docs):
        """Take all measurements. This is the organism looking at itself
        through mathematical instruments."""
        self.model_stage = model.current_growth_stage()
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
        temp_offset = 0.0
        accum_override = 0
        delta_grow_override = None
        action = "steady"

        # CASE 1: Syntropy rising + field deviation in sweet spot = thriving
        if (self.syntropy_trend > 0.01 and
                CFG.field_deviation_floor < self.field_deviation < CFG.field_deviation_ceiling):
            lr_multiplier = CFG.syntropy_lr_boost
            temp_offset = -0.05  # more confident when organizing
            if self.purpose_alignment > 0.3:
                delta_grow_override = CFG.syntropy_delta_grow_boost
                accum_override = 2  # stable gradient when everything aligned
                action = "amplify"
            else:
                action = "boost"

        # CASE 2: Syntropy falling = dissolving, slow down
        elif self.syntropy_trend < -0.01:
            lr_multiplier = CFG.syntropy_lr_dampen
            temp_offset = 0.05  # more exploratory when disordering
            action = "dampen"

        # CASE 3: Field deviation too high = hallucinating
        elif self.field_deviation > CFG.field_deviation_ceiling:
            lr_multiplier = CFG.syntropy_lr_dampen
            temp_offset = -0.05  # focus when hallucinating
            action = "ground"

        # CASE 4: Field deviation too low = parroting
        elif self.field_deviation < CFG.field_deviation_floor:
            lr_multiplier = CFG.syntropy_lr_boost
            temp_offset = 0.05  # explore when parroting
            action = "explore"

        # CASE 5: Purpose opposes gamma = identity crisis
        if self.purpose_alignment < -0.3:
            lr_multiplier *= 0.5
            temp_offset = 0.0
            action = "realign"

        # CASE 6: Adult + sustained overload → divide (mitosis)
        max_stage = len(CFG.growth_stages) - 1
        if (self.model_stage >= max_stage and
                self._is_sustained_overload() and
                time.time() - self._last_mitosis_time > 300):
            action = "divide"
            lr_multiplier = CFG.syntropy_lr_dampen  # slow down while preparing to split

        # CASE 7: Plateau + young peer thriving → hibernate (cooperative scheduling)
        if (action == "steady" and self._should_hibernate()):
            action = "hibernate"

        # SELF-META-LEARNING: check if this action historically hurts
        if action not in ("divide", "hibernate") and len(self.burst_history) >= 4:
            eff, count = self.action_effectiveness(action)
            if count >= 2 and eff > 0.05:
                # This action consistently makes loss WORSE — downgrade
                if action == "amplify":
                    action = "boost"
                    accum_override = 0
                    delta_grow_override = None
                elif action in ("boost", "explore"):
                    lr_multiplier = 1.0
                    action = "steady"

        self.last_action = action
        return {
            "lr_multiplier": lr_multiplier,
            "temp_offset": temp_offset,
            "accum_override": accum_override,
            "delta_grow_override": delta_grow_override,
            "action": action,
        }

    def _is_sustained_overload(self):
        """High entropy for >75% of window + falling syntropy = overloaded."""
        if len(self.entropy_history) < CFG.syntropy_window:
            return False
        recent = self.entropy_history[-CFG.syntropy_window:]
        high_count = sum(1 for e in recent if e > CFG.entropy_high)
        return high_count > CFG.syntropy_window * 0.75 and self.syntropy_trend < -0.02

    def _should_hibernate(self):
        """Should this organism sleep to give resources to peers?
        Conditions: loss on plateau + a peer is in amplify/boost state."""
        if not self._swarm_info or not self._swarm_info.get("peers"):
            return False
        # Check if any peer has higher syntropy trend (actively improving)
        for peer in self._swarm_info["peers"]:
            if peer.get("syntropy", 0) > 0.05:
                # A young peer is thriving. If we're stale, hibernate.
                if len(self.burst_history) >= 8:
                    recent_deltas = [b["loss_after"] - b["loss_before"] for b in self.burst_history[-8:]]
                    avg_delta = sum(recent_deltas) / len(recent_deltas)
                    if abs(avg_delta) < 0.01:  # loss plateau
                        return True
        return False

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


def cosine_lr(global_step, growth_step_offset=0):
    """Global cosine LR with linear warmup that resets on growth."""
    steps_since_growth = global_step - growth_step_offset
    if steps_since_growth < CFG.cosine_warmup_steps:
        return CFG.lr_min + (CFG.learning_rate - CFG.lr_min) * (steps_since_growth / max(1, CFG.cosine_warmup_steps))
    progress = min(1.0, global_step / max(1, CFG.max_total_steps))
    return CFG.lr_min + 0.5 * (CFG.learning_rate - CFG.lr_min) * (1.0 + math.cos(math.pi * progress))


def train_steps(model: GPT, tok: EvolvingTokenizer, docs, steps, train_base=True, train_deltas=True):
    if not docs:
        return

    with model.lock:
        _train_steps_locked(model, tok, docs, steps, train_base, train_deltas)

def _train_steps_locked(model, tok, docs, steps, train_base, train_deltas):
    if not docs:
        return
    # Ontogenesis freeze: after growth, only train deltas until new weights stabilize
    if model._growth_freeze_remaining > 0:
        base_params = []
        delta_params = model.all_delta_params() if train_deltas else []
        model._growth_freeze_remaining = max(0, model._growth_freeze_remaining - steps)
    else:
        base_params = model.all_base_params() if train_base else []
        delta_params = model.all_delta_params() if train_deltas else []
    accum = CFG.accum_steps

    for step in range(steps):
        # Gradient accumulation: accumulate over accum micro-batches, then step
        for micro in range(accum):
            batch = random.choices(docs, k=CFG.batch_size)
            batch_ids = [tok.encode(doc) for doc in batch if doc]

            loss = model.loss_on_batch(batch_ids)
            loss = loss * (1.0 / accum)  # scale loss for accumulation
            backward(loss)

        lr = cosine_lr(model.global_step, model.growth_step_offset)
        # Scale LR inversely with model size: larger models need smaller LR
        lr *= CFG.growth_stages[0][1] / model.n_embd
        # Post-growth LR dampening: reduce LR during freeze to prevent delta overfit to noise
        if model._growth_freeze_remaining > 0:
            lr *= CFG.post_growth_lr_scale
        model.global_step += 1

        if base_params:
            model.adam_step(base_params, key="base", lr=lr)
        if delta_params:
            model.adam_step(delta_params, key="delta", lr=lr)

        if step % 100 == 0:
            print(f"  train step {step}/{steps} | loss {loss.data * accum:.4f} | lr {lr:.5f}")

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


# ============================================================
# 9.7) SWARM ECOLOGY — the organism learns it is not alone
# ============================================================
# And lo, the first cell shall call into the void and hear only silence.
# But the second shall call and hear an answer.

SWARM_DIR = os.path.expanduser("~/.molequla/swarm")

class SwarmRegistry:
    """Discover and track other molequla instances via shared SQLite."""

    def __init__(self, organism_id=None):
        self.organism_id = organism_id or f"org_{os.getpid()}_{int(time.time())}"
        self.pid_file = None
        self.mesh_db = None

    def register(self):
        """Write PID file and register in mesh.db."""
        os.makedirs(SWARM_DIR, exist_ok=True)
        self.pid_file = os.path.join(SWARM_DIR, f"{self.organism_id}.pid")
        with open(self.pid_file, "w") as f:
            json.dump({"pid": os.getpid(), "organism_id": self.organism_id,
                        "started": time.time()}, f)
        self._init_mesh_db()
        self._register_in_mesh()

    def _init_mesh_db(self):
        db_path = os.path.join(SWARM_DIR, "mesh.db")
        self.mesh_db = sqlite3.connect(db_path, timeout=5.0)
        self.mesh_db.execute("PRAGMA journal_mode=WAL")
        self.mesh_db.execute("""
            CREATE TABLE IF NOT EXISTS organisms(
                id TEXT PRIMARY KEY, pid INTEGER, stage INTEGER,
                n_params INTEGER, syntropy REAL, entropy REAL,
                last_heartbeat REAL, parent_id TEXT,
                status TEXT DEFAULT 'alive')""")
        self.mesh_db.execute("""
            CREATE TABLE IF NOT EXISTS messages(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_id TEXT, to_id TEXT, type TEXT, payload TEXT, ts REAL)""")
        self.mesh_db.commit()

    def _register_in_mesh(self):
        self.mesh_db.execute(
            "INSERT OR REPLACE INTO organisms(id,pid,stage,n_params,syntropy,entropy,last_heartbeat,status) "
            "VALUES(?,?,0,0,0.0,0.0,?,'alive')",
            (self.organism_id, os.getpid(), time.time()))
        self.mesh_db.commit()

    def heartbeat(self, stage, n_params, syntropy, entropy):
        """Periodic state update in mesh.db."""
        if not self.mesh_db:
            return
        self.mesh_db.execute(
            "UPDATE organisms SET stage=?,n_params=?,syntropy=?,entropy=?,last_heartbeat=?,status='alive' WHERE id=?",
            (stage, n_params, syntropy, entropy, time.time(), self.organism_id))
        self.mesh_db.commit()

    def discover_peers(self, timeout_seconds=60):
        """Find other living organisms."""
        if not self.mesh_db:
            return []
        cutoff = time.time() - timeout_seconds
        cur = self.mesh_db.execute(
            "SELECT id,pid,stage,n_params,syntropy,entropy,status FROM organisms "
            "WHERE status='alive' AND last_heartbeat>? AND id!=?",
            (cutoff, self.organism_id))
        return [{"id": r[0], "pid": r[1], "stage": r[2], "n_params": r[3],
                 "syntropy": r[4], "entropy": r[5], "status": r[6]} for r in cur.fetchall()]

    def mark_hibernating(self):
        """Mark this organism as sleeping in mesh.db."""
        if self.mesh_db:
            self.mesh_db.execute("UPDATE organisms SET status='sleeping' WHERE id=?",
                                 (self.organism_id,))
            self.mesh_db.commit()

    def log_message(self, to_id, msg_type, payload):
        """Log a message between organisms."""
        if self.mesh_db:
            self.mesh_db.execute(
                "INSERT INTO messages(from_id,to_id,type,payload,ts) VALUES(?,?,?,?,?)",
                (self.organism_id, to_id, msg_type, json.dumps(payload), time.time()))
            self.mesh_db.commit()

    def unregister(self):
        """Clean up on exit."""
        if self.mesh_db:
            self.mesh_db.execute("UPDATE organisms SET status='dead' WHERE id=?",
                                 (self.organism_id,))
            self.mesh_db.commit()
            self.mesh_db.close()
            self.mesh_db = None
        if self.pid_file and os.path.exists(self.pid_file):
            os.unlink(self.pid_file)


import sys as _sys

async def perform_mitosis(model, tok, con, swarm, syntracker):
    """The organism divides. Parent continues. Child starts at infant stage."""
    child_id = f"org_{int(time.time())}_{random.randint(1000,9999)}"
    child_dir = os.path.expanduser(f"~/.molequla/{child_id}")
    os.makedirs(child_dir, exist_ok=True)

    # Save parent checkpoint for child's reference
    parent_ckpt = os.path.join(child_dir, "parent_ckpt.json")
    save_checkpoint(model, tok, parent_ckpt)

    # Write birth config with inherited memory
    birth = {
        "organism_id": child_id,
        "parent_id": swarm.organism_id,
        "corpus_path": CFG.corpus_path,
        "db_path": os.path.join(child_dir, "memory.sqlite3"),
        "ckpt_path": os.path.join(child_dir, "molequla_ckpt.json"),
        "burst_history": syntracker.burst_history,
    }
    birth_path = os.path.join(child_dir, "birth.json")
    with open(birth_path, "w") as f:
        json.dump(birth, f)

    # Log in mesh
    swarm.log_message(child_id, "mitosis:spawn",
                      {"parent_stage": model.current_growth_stage()})
    db_log_growth(con, model, tok, load_corpus_lines(CFG.corpus_path),
                  note=f"mitosis:spawn:{child_id}")

    # Spawn child process
    child_proc = await asyncio.create_subprocess_exec(
        _sys.executable, os.path.abspath(__file__),
        "--organism-id", child_id, "--config", birth_path)

    syntracker._last_mitosis_time = time.time()
    print(f"[ecology] Child {child_id} spawned (pid={child_proc.pid})")
    return child_id


def perform_hibernation(model, tok, con, swarm):
    """The organism sleeps. Saves state, marks sleeping, exits."""
    print(f"[ecology] HIBERNATION — organism {swarm.organism_id} going to sleep")
    save_checkpoint(model, tok)
    swarm.mark_hibernating()
    db_log_growth(con, model, tok, load_corpus_lines(CFG.corpus_path),
                  note=f"hibernate:{swarm.organism_id}")


async def background_trainer(con, model: GPT, tok: EvolvingTokenizer, swarm=None):
    # And lo, asynchronous training shall occur, because sleeping is for humans.
    last_event_id = 0
    qbuf = QuantumBuffer()
    syntracker = SyntropyTracker()
    field = CooccurField()
    tick_count = 0

    # Inherit burst_history from parent (mitosis lineage)
    inherited = getattr(model, '_inherited_burst_history', None)
    if inherited:
        syntracker.burst_history = list(inherited)
        print(f"[ecology] syntracker inherited {len(inherited)} burst records from parent.")
        del model._inherited_burst_history

    while True:
        tick_count += 1
        _ = update_reservoir_corpus(con, CFG.corpus_path, CFG.max_corpus_lines)
        mass, last_event_id = compute_new_corpus_mass(con, last_event_id)
        docs = load_corpus_lines(CFG.corpus_path)

        # Rebuild field from current corpus (the organism re-reads its own physics)
        if docs:
            field.build_from_corpus(tok, docs)
            model._corpus_field = field  # share with generate_sentence for adaptive blend

        # Tokenizer evolution (char -> BPE enablement) + safe vocab expansion
        bpe_just_enabled = tok.maybe_enable_bpe(docs)
        bpe_retrained = tok.maybe_retrain_bpe(docs)
        if bpe_just_enabled or bpe_retrained:
            with model.lock:
                model.maybe_expand_vocab(tok.vocab_size)
                save_checkpoint(model, tok)

        # Per-stage warmup: if current stage > last warmed stage, run warmup
        current_stage = model.current_growth_stage()
        if docs and current_stage >= 0 and current_stage > model.last_warmup_stage:
            stage_name = ["embryo", "infant", "child", "adolescent", "teen", "adult"][min(current_stage, 5)]
            embryo_embd = CFG.growth_stages[0][1]
            warmup_scale = max(1, model.n_embd // embryo_embd)
            effective_warmup = CFG.warmup_steps * warmup_scale
            print(f"[trainer] per-stage warmup for stage {current_stage} ({stage_name})... "
                  f"{effective_warmup} steps (scaled {warmup_scale}x for embd={model.n_embd})")
            train_steps(model, tok, docs, effective_warmup,
                        train_base=True, train_deltas=True)
            model.last_warmup_stage = current_stage
            with model.lock:
                save_checkpoint(model, tok)
            db_log_growth(con, model, tok, docs, note=f"warmup_complete:stage={current_stage}")
            print(f"[trainer] warmup complete for stage {current_stage}. base may freeze now, like a proud fossil.")

        if model.last_warmup_stage >= 0 and docs:
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

                # Apply syntropy-adjusted learning rate + accum override
                original_lr = CFG.learning_rate
                CFG.learning_rate = original_lr * lr_mul
                original_accum = CFG.accum_steps
                if decision.get("accum_override", 0) > 0:
                    CFG.accum_steps = decision["accum_override"]

                # Update temperature bridge
                model.syntropy_temp_offset = decision.get("temp_offset", 0.0)

                # Measure loss before burst for self-meta-learning
                with model.lock:
                    loss_before = model.quick_loss(tok, docs, 4)

                train_base = not CFG.freeze_base_after_warmup
                train_steps(model, tok, docs, CFG.micro_steps,
                            train_base=train_base, train_deltas=True)

                CFG.learning_rate = original_lr   # restore
                CFG.accum_steps = original_accum  # restore

                with model.lock:
                    # Measure loss after burst
                    loss_after = model.quick_loss(tok, docs, 4)

                    # SELF-META-LEARNING: record what this burst did
                    syntracker.record_burst(action, loss_before, loss_after)

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
                        db_log_growth(con, model, tok, docs,
                                      note=f"quantum_burst:{action}|Δloss={loss_after-loss_before:.4f}")

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

                # Ontogenesis: check if architecture should grow
                corpus_chars = sum(len(d) for d in docs)
                with model.lock:
                    if model.maybe_grow_architecture(corpus_chars):
                        save_checkpoint(model, tok)
                        n_p = sum(len(r.data) for r in model.all_base_params())
                        db_log_growth(con, model, tok, docs,
                                      note=f"ontogenesis:stage={model.current_growth_stage()}|params={n_p}")

                # Ecology: mitosis / hibernation
                if swarm and action == "divide":
                    print("[ecology] MITOSIS triggered — organism overloaded, spawning child")
                    await perform_mitosis(model, tok, con, swarm, syntracker)

                if swarm and action == "hibernate":
                    perform_hibernation(model, tok, con, swarm)
                    print("[ecology] Organism hibernating. Goodbye.")
                    return  # exit training loop

        # Swarm heartbeat (every 10 ticks)
        if swarm and tick_count % 10 == 0:
            stage = model.current_growth_stage()
            n_p = sum(len(r.data) for r in model.all_base_params())
            swarm.heartbeat(stage, n_p, syntracker.syntropy_trend,
                            syntracker.entropy_history[-1] if syntracker.entropy_history else 0.0)
            # Update swarm info for hibernate decisions
            syntracker._swarm_info = {"peers": swarm.discover_peers()}

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





def _parse_cli_args():
    """Parse CLI arguments for child organisms."""
    args = {"organism_id": None, "config": None}
    i = 1
    while i < len(_sys.argv):
        if _sys.argv[i] == "--organism-id" and i + 1 < len(_sys.argv):
            args["organism_id"] = _sys.argv[i + 1]
            i += 2
        elif _sys.argv[i] == "--config" and i + 1 < len(_sys.argv):
            args["config"] = _sys.argv[i + 1]
            i += 2
        else:
            i += 1
    return args


async def chat_main():
    cli = _parse_cli_args()

    # Child organism: load birth config from parent
    if cli["config"] and os.path.exists(cli["config"]):
        with open(cli["config"], "r") as f:
            birth = json.load(f)
        CFG.corpus_path = birth.get("corpus_path", CFG.corpus_path)
        CFG.db_path = birth.get("db_path", CFG.db_path)
        CFG.ckpt_path = birth.get("ckpt_path", CFG.ckpt_path)

    con = init_db(CFG.db_path)

    if not os.path.exists(CFG.corpus_path):
        print(f"Seed corpus not found: {CFG.corpus_path}")
        print("Place nonames.txt alongside molequla.py to begin.")
        return

    docs = load_corpus_lines(CFG.corpus_path)

    model, tok = load_checkpoint(docs, CFG.ckpt_path)
    if model is None or tok is None:
        tok = EvolvingTokenizer(docs if docs else ["Hello."])
        # Enable BPE before warmup — subword tokens for coherent corpus field
        if docs:
            tok.maybe_enable_bpe(docs)
        model = GPT(tok)

        # Initialize at the correct stage for corpus size with per-stage warmup
        corpus_chars = sum(len(d) for d in docs) if docs else 0
        target_stage = model.target_growth_stage(corpus_chars)
        current_stage = model.current_growth_stage()
        if target_stage > current_stage and docs:
            print(f"[init] Per-stage warmup: current={current_stage}, target={target_stage}")
            # Build corpus field — active from first token, sigmoid fade weakens as model learns
            init_field = CooccurField()
            init_field.build_from_corpus(tok, docs)
            model._corpus_field = init_field
            while True:
                stage = model.current_growth_stage()
                stage_name = ["embryo", "infant", "child", "adolescent", "teen", "adult"][min(stage, 5)]
                _, embd, layer, head = CFG.growth_stages[min(stage, len(CFG.growth_stages)-1)]
                embryo_embd = CFG.growth_stages[0][1]
                warmup_scale = max(1, model.n_embd // embryo_embd)
                effective_warmup = CFG.warmup_steps * warmup_scale
                print(f"[init] Stage {stage} ({stage_name}): embd={embd} — {effective_warmup} steps (scaled {warmup_scale}x)")
                train_steps(model, tok, docs, effective_warmup)
                model.last_warmup_stage = stage
                save_checkpoint(model, tok)
                if not model.maybe_grow_architecture(corpus_chars):
                    break
                model._growth_freeze_remaining = 0  # skip freeze during init growth
                # Rebuild corpus field after growth (vocab may have expanded)
                init_field.build_from_corpus(tok, docs)
                model._corpus_field = init_field
        elif target_stage > current_stage:
            # No docs: just grow without warmup
            while model.maybe_grow_architecture(corpus_chars):
                model._growth_freeze_remaining = 0

    # Ensure tokenizer evolution can expand model
    model.maybe_expand_vocab(tok.vocab_size)

    # Swarm ecology: register in mesh
    swarm = SwarmRegistry(organism_id=cli.get("organism_id"))
    swarm.register()
    peers = swarm.discover_peers()
    if peers:
        print(f"[ecology] Joined swarm. {len(peers)} peer(s) detected.")
    else:
        print("[ecology] First organism in the swarm.")

    # Child: inherit burst_history from parent
    syntracker_seed = None
    if cli["config"] and os.path.exists(cli["config"]):
        with open(cli["config"], "r") as f:
            birth = json.load(f)
        if "burst_history" in birth:
            syntracker_seed = birth["burst_history"]
            print(f"[ecology] Inherited {len(syntracker_seed)} burst records from parent.")

    trainer_task = asyncio.create_task(
        background_trainer(con, model, tok, swarm=swarm))

    # If child with syntracker_seed, we need to inject it
    # (syntracker is created inside background_trainer, so we pass via model attribute)
    if syntracker_seed:
        model._inherited_burst_history = syntracker_seed

    print("molequla is alive. Type and press Enter. Ctrl+C to exit.\n")
    try:
        while True:
            user_text = await asyncio.get_event_loop().run_in_executor(None, input, "> ")
            user_text = user_text.strip()
            if not user_text:
                continue

            db_add_message(con, "user", user_text)

            # And lo, the reservoir shall be fed for future trainings.
            update_reservoir_corpus(con, CFG.corpus_path, CFG.max_corpus_lines)

            # Self-enrichment: user input enriches corpus field (after rebuild, so it's not wiped)
            user_ids = tok.encode(user_text)
            if model._corpus_field is not None:
                model._corpus_field.ingest_tokens(user_ids)

            # Active user word boost: organism absorbs user's vocabulary (Leo-style)
            # Decays each generation, fades with model strength via sigmoid in generate_sentence
            if model._corpus_field is not None:
                model._corpus_field.absorb_user_words(user_ids)

            prompt = build_prompt_from_memory(con, user_text)

            # Consciousness: self-prediction error (Feature 4)
            # "How surprised am I by this input?"
            with model.lock, no_grad():
                prompt_ids = tok.encode(prompt)
                if len(prompt_ids) > 2:
                    surprise = model.compute_self_prediction_error(prompt_ids)
                    model.last_surprise = surprise
                    if model.surprise_baseline < 1e-6:
                        model.surprise_baseline = surprise
                    else:
                        model.surprise_baseline = 0.3 * surprise + 0.7 * model.surprise_baseline

            answer = model.generate_sentence(prompt_text=prompt) or "..."

            # Consciousness: conscience check (Feature 5)
            # "Did my last generation feel coherent?"
            with model.lock:
                if model.last_gen_entropy > 0:
                    model.conscience_check(model.last_gen_entropy)

            print(answer)
            db_add_message(con, "assistant", answer)

            # Self-enrichment: own output enriches corpus field, weighted by coherence
            # Low entropy = coherent speech = higher weight (Stanley's resonance weighting)
            if model._corpus_field is not None and len(answer) > 3:
                self_weight = 1.0
                if model.last_gen_entropy > 0:
                    self_weight = 2.0 - model.last_gen_entropy
                    self_weight = max(0.3, min(2.0, self_weight))
                model._corpus_field.ingest_tokens_weighted(tok.encode(answer), self_weight)
                model._corpus_field.decay_user_boost()

            # Consciousness: overthinkg rings (Feature 3)
            # "Let me re-read what I just said to strengthen my patterns."
    except KeyboardInterrupt:
        pass
    finally:
        trainer_task.cancel()
        try:
            await trainer_task
        except asyncio.CancelledError:
            pass
        save_checkpoint(model, tok)
        swarm.unregister()
        con.close()

# ============================================================
# 11) AWAKEN — now, when all is assembled as an organism, not a gearbox,
#             it is time to declare the final function.
# ============================================================

def main():
    asyncio.run(chat_main())

if __name__ == "__main__":
    main()
