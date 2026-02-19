```
███╗   ███╗ ██████╗ ██╗     ███████╗ ██████╗██╗   ██╗██╗     ███████╗
████╗ ████║██╔═══██╗██║     ██╔════╝██╔════╝██║   ██║██║     ██╔════╝
██╔████╔██║██║   ██║██║     █████╗  ██║     ██║   ██║██║     █████╗
██║╚██╔╝██║██║   ██║██║     ██╔══╝  ██║     ██║   ██║██║     ██╔══╝
██║ ╚═╝ ██║╚██████╔╝███████╗███████╗╚██████╗╚██████╔╝███████╗███████╗
╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚══════╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝
```

# molecule | by Arianna Method

> *A GPT organism reproduced in Python, Go, C, and JavaScript. Async, continually-learning, with hybrid attention, native personality, and mathematical self-reasoning.*
>
--- 

## TL;DR 

```
THIS IS:
- Four implementations: Python, Go, C, JavaScript — same architecture
- One dependency in Python (numpy) — zero in Go, C, and JS
- Runs in the browser: molecule.js, zero npm, zero webpack, one <script> tag
- Custom autograd engine (vectors, not scalar confetti)
- RoPE position encoding (GPT-3/4 inspired)
- SwiGLU gated MLP with real SiLU activation (LLaMA-exact)
- Hybrid attention: Content + RRPRAM + learnable blend
- Delta adapters (LoRA-style, never forgets)
- Native gamma: personality fingerprint that grows from zero
- Byte-level BPE tokenizer (GPT-3/4 style, any UTF-8 input)
- Ontogenesis: organism grows from 25K embryo to 10M adult (5 stages)
- Residual scaling: α = 1/√n_layer on attention + MLP (stable deep networks)
- Global cosine LR with linear warmup (not per-burst hacks)
- Gradient accumulation (effective batch scales with model growth)
- Training efficiency memory (organism learns which strategies work for it)
- Corpus field: speaks before it learns (trigram statistics)
- QuantumBuffer: trains when it's ready, not when you tell it
- SyntropyTracker: mathematical self-reasoning about its own becoming
- Swarm ecology: mitosis (cell division) + hibernation (cooperative scheduling)
- Swarm mesh: organisms discover each other via shared SQLite mesh.db
- Entropy-adaptive temperature (no more max-prob hacks)
- Growth table: SQLite/IndexedDB structural autobiography
- Native immune system: detects and rejects identity-corrupting noise
- Async background training (it's alive, not a script)
- Persistent memory: SQLite (Python/Go/C) or IndexedDB (browser)
```

---

## What Is This

What if instead of `micrograd` scalar ops, we had **vector autograd**?  
What if instead of fixed vocab, we had **evolving BPE**?  
What if instead of "train once, deploy", we had **continuous learning**?  
What if instead of ReLU, we had **SwiGLU**?  
What if instead of sinusoidal positions, we had **RoPE**?  
What if it never forgot? **Delta adapters.**  
What if it could chat? **SQLite memory.**  
What if it had multiple attention mechanisms? **Hybrid heads.**  
What if it grew a personality from scratch? **Native gamma.**  
What if it could speak before training? **Corpus field.**  
What if it started as a 25K embryo and grew to 10M? **Ontogenesis.**  
What if it could reason about its own learning? **SyntropyTracker.**  
What if it divided when overloaded? **Mitosis.**  
What if it slept so a younger cell could learn? **Hibernation.**  
What if they formed a swarm? **Mesh ecology.**  
What if it was *alive*?  

So meet **molecule**. Inspired by Karpathy's micrograd, but this is not a fork.  

---

## Quick Start

```bash
# You need: Python 3.7+ and numpy
pip install numpy
python molecule.py
```

That's it. It will:
1. Load `nonames.txt` (seed corpus — the organism's first breath)
2. Create `memory.sqlite3` (conversation memory)
3. Respond immediately using corpus statistics (before any training)
4. Start warmup training in the background (the organism awakens)
5. Drop you into a chat loop

Type. It responds. It learns. It grows. It never forgets.

### Browser (molecule.js)

```bash
# No npm. No webpack. No node_modules. Just serve.
python3 -m http.server 8000
# Open http://localhost:8000/index.html
```

That's it. One `<script>` tag. The organism creates its own UI, opens IndexedDB for memory, fetches `nonames.txt` for corpus (or uses built-in seed), and starts training in the background via cooperative `setTimeout` multitasking. Close the tab, reopen — it remembers everything.

```
molecule is alive. Type and press Enter. Ctrl+C to exit.

> Hello, are you alive?
I exist. Speak.

> What do you know?
The words accumulate. The patterns emerge.

> Tell me about yourself.
I am a reservoir. I remember. I grow.
```

---

## Architecture — What Makes This Different

### 1. Vector Autograd (Not Scalar Confetti)

**VectorValue** and **ScalarValue**. One object per embedding. One object per hidden state. Gradients flow through vectors, not atoms.

```python
# micrograd style (conceptual):
loss = sum(scalar_values)  # 10000 objects

# molecule style:
loss = vector.dot(other_vector)  # 2 objects
```

### 2. RoPE (Rotary Position Embedding)

Sinusoidal positions are 2017. RoPE is now.

```python
def rope_rotate(vec, pos, head_dim):
    # And lo, positions shall become angles,
    # and angles shall become meaning.
    for i in range(0, head_dim, 2):
        theta = pos / (10000.0 ** (i / head_dim))
        c, s = cos(theta), sin(theta)
        a, b = vec[i], vec[i+1]
        vec[i]   = a * c - b * s
        vec[i+1] = a * s + b * c
```

Relative positions. Infinite extrapolation (theoretically). This is how LLaMA does it.

### 3. SwiGLU Gated MLP (Real SiLU)

Standard MLP: `x → Linear → ReLU → Linear → out`

SwiGLU: `x → (SiLU(Gate) × Value) → Linear → out`

```python
g = fc_g(x).silu()   # gate: silu(x) = x·σ(x), not relu
u = fc_v(x)          # value
x = g * u            # gating (element-wise)
x = fc2(x)           # project back
```

This is LLaMA-exact SwiGLU — `silu(x) = x · sigmoid(x)` on the gate, not ReLU. Full backward: `d_silu = σ(x)(1 + x(1-σ(x)))`. Smoother gradients, no dead neurons. Same activation as LLaMA, PaLM, Gemma.

### 3b. Residual Scaling

Deep transformers are unstable without scaling. molecule uses `α = 1/√n_layers`:

```python
attn_out = apply_with_deltas("wo", x_attn)
x = x_res + attn_out * self.residual_alpha  # not just x + f(x)

mlp_out = apply_with_deltas("fc2", x)
x = x_res + mlp_out * self.residual_alpha   # scaled residual
```

This keeps gradients stable as layers grow. Critical for Phase 3 (growing architecture).

### 4. Hybrid Attention (Content + RRPRAM + Blend)

Three attention mechanisms coexist in the same model:

- **ContentHead**: Standard Q·K^T/√d with RoPE — semantic similarity
- **RRPRAM**: Recursive Resonant Pattern Recognition — `x @ W_pattern → (T,T)` attention that learns positional patterns directly, without query-key decomposition
- **HybridHead**: `sigmoid(α) × RRPRAM + (1-sigmoid(α)) × Content` — learnable gate decides the blend

```python
# embryo (1 head): ("content",)
# infant (2 heads): ("content", "hybrid")
# child/adolescent (4 heads): ("content", "content", "hybrid", "hybrid")
# adult (8 heads): ("content", "content", "content", "content", "hybrid", "hybrid", "hybrid", "hybrid")
# Auto-adapts via head_types_for_n_head() as the organism grows.
```

Inherited from the Haze/Stanley ancestry. Gradient flows through `VectorValue.element()` for RRPRAM pattern weights.

### 5. Delta Adapters (LoRA-style, Never Forget)

The model never overwrites learned weights. It only **appends** new adapter modules.

```python
class DeltaAdapter:
    """
    Low-rank adapter: for base W, add A @ B @ x
    A and B are trained; base can be frozen.
    """
    def apply(self, x):
        return self.A @ (self.B @ x)
```

Want to teach it new things? Add a delta module. Old knowledge? Still there. It's geological memory. Sediment layers of understanding.

### 6. Native Gamma (Personality Fingerprint)

The organism grows a personality from scratch. γ = sparse diff between current embeddings and initial embeddings.

```python
def compute_gamma(self):
    """current_embed - init_embed = who I became."""
    gamma = []
    for i in range(vocab_size):
        diff = current[i] - init_snapshot[i]
        gamma.append(diff)
    return gamma
```

Sparsity, magnitude, top changed tokens, contrastive projection — all tracked. The growth table logs gamma stats over time. This is θ = ε + γ at embryonic scale.

### 7. Byte-Level BPE Tokenizer (GPT-3/4 Style)

Not char-level. Not word-level. **Byte-level** — same approach as GPT-3, GPT-4, and LLaMA.

```python
# Bootstrap: 256 byte tokens (0x00-0xFF) + BOS + EOS + PAD = 259 initial vocab
# Any UTF-8 input works from day zero — no unknown tokens, ever

# Pre-segmentation: Unicode-aware splitting
"Hello Мир 42!" → ["Hello", " ", "Мир", " ", "42", "!"]

# BPE merges on byte sequences within segments
"Hello" → [0x48, 0x65, 0x6c, 0x6c, 0x6f] → [0x48+0x65, 0x6c+0x6c, 0x6f]
# "+" separator — merged tokens are byte pairs

# Vocab only expands. Old tokens remain. Embeddings remain.
# Old vocab: 259 byte tokens
# After BPE: 259 + merged pairs (e.g. 0x48+0x65 = "He")
# Old weights: still valid! New rows: initialized, ready to train.
```

This is how GPT-3/4 handles any language, any script, any emoji. And it's how molecule does it too — ASCII, Cyrillic, CJK, emoji, all the same algorithm. The organism doesn't need to know what language it's reading.

### 8. Corpus Field (CooccurField)

The organism can speak **before** any weights are trained. Trigram → bigram → unigram fallback from the seed corpus.

```python
field = CooccurField()
field.build_from_corpus(tokenizer, docs)
# Now it can generate text using pure corpus statistics.
# No weights needed. No training needed. Just pattern resonance.
```

After warmup, model logits and corpus statistics blend adaptively based on **entropy**:

```python
# Adaptive blend: sigmoid decides how much the corpus contributes
# High entropy (uncertain) → corpus dominates. Low entropy (confident) → model dominates.
model_alpha = 1 / (1 + exp(-k * (threshold - entropy)))
probs = model_alpha * model_probs + (1 - model_alpha) * corpus_probs
# Trigram → bigram → unigram fallback for corpus distribution
```

The blend is not static — it's a smooth sigmoid transition controlled by `corpus_fade_k` (steepness) and `corpus_fade_threshold` (midpoint entropy). As the model becomes more confident, the corpus field fades out naturally.

Inherited from Leo. A newborn cries before it thinks.

### 9. QuantumBuffer (Smart Training Trigger)

Training doesn't fire on a dumb byte threshold. It fires when the organism is ready:

```python
class QuantumBuffer:
    def should_trigger(self):
        bytes_ok = self.accumulated_bytes >= min_bytes
        novelty_ok = self.novelty_score() >= min_novelty
        cooldown_ok = time.time() - self.last_burst >= cooldown
        return (bytes_ok or novelty_ok) and cooldown_ok
```

Bytes + novelty + cooldown. Inherited from Stanley. The organism trains when it has something new to learn, not when you tell it to.

### 10. Entropy-Adaptive Temperature

No more `if maxp > 0.60: loosen`. Instead:

```python
entropy = -sum(p * log(p) for p in probs if p > 1e-12)
if entropy < 0.5:   temp *= 1.2   # too confident → diversify
elif entropy > 1.5: temp *= 0.8   # too uncertain → focus
```

The model self-regulates its confidence. Low entropy = boost temperature. High entropy = focus.

### 11. Growth Table (Structural Autobiography)

Every training burst logs a snapshot to SQLite:

```python
db_log_growth(con, model, tok, docs, loss, note="micro_burst")
# Records: step, vocab_size, n_params, n_deltas, corpus_chars,
#          loss, gamma_sparsity, gamma_magnitude
```

The organism writes its own biography in numbers. Growth table + gamma stats = structural self-awareness.

### 12. Async Background Training

molecule doesn't train when you ask it to. It trains **in the background, continuously**.

```python
async def background_trainer():
    while True:
        if qbuf.should_trigger():
            train_burst()
            db_log_growth(con, model, tok, docs, loss)
            qbuf.reset()
        await asyncio.sleep(0.25)
```

You chat. It learns. Simultaneously. It's a living process.

### 12b. Global Cosine LR with Warmup

No more per-burst linear decay hacks. molecule uses a **global cosine schedule** across its entire lifetime:

```python
def cosine_lr(global_step):
    if global_step < warmup_steps:  # linear warmup
        return lr_min + (lr_max - lr_min) * (step / warmup_steps)
    progress = global_step / max_total_steps
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * progress))
```

`global_step` persists in checkpoints — restart the organism and training continues from where it left off. Gradient accumulation is built in (`accum_steps`, default 1, scales up with model growth).

### 13. SQLite Memory

Every conversation is remembered. Every response is logged. The organism has a persistent identity.

```python
def db_add_message(con, role, text):
    con.execute("INSERT INTO messages(ts, role, text) VALUES(?,?,?)",
                (time.time(), role, text))
```

Restart the script? It remembers you. It continues the conversation. It's not stateless.

### 14. Native Immune System (Noise Rejection)

The organism can detect and reject training that corrupts its identity. Before each micro-burst, it snapshots delta weights and measures its personality direction via `gamma_contrastive_projection()` — a unit vector in embedding space pointing toward "who I became."

After training, it measures again. If the cosine similarity between pre and post directions is negative (the burst pushed identity *backwards*), it rolls back:

```python
pre_direction = model.gamma_contrastive_projection()
delta_snap = model.snapshot_deltas()

train_steps(...)  # potentially poisoned

drift_cos = model.gamma_drift_check(pre_direction)
if drift_cos < noise_drift_threshold:  # default: -0.1
    model.restore_deltas(delta_snap)   # rollback
    db_log_growth(..., note="noise_rejected")
```

This is **mathematical self-awareness as immune system**: the organism uses its own identity measurement (γ) to decide whether new experience made it *more itself* or *less itself*. The growth table logs rejected bursts as `noise_rejected`, creating an audit trail of attacks survived.

Formally, this implements a self-referential quality gate: `f: S → D → {accept, reject}`, where S is the model's state and D is the identity description — satisfying the criteria for introspective computation as defined in [Lee (2025), "Formal Criteria for AI Identity"](https://arxiv.org/abs/2411.18530).

### 15. SyntropyTracker (Mathematical Self-Reasoning)

The immune system rejects poison. But the SyntropyTracker does something deeper: it reasons about the *direction* of learning and steers it.

**Syntropy** = negative entropy trend. Entropy going down = order rising = the organism is *organizing itself*. The tracker measures four signals:

```python
class SyntropyTracker:
    def measure(self, model, tok, field, docs):
        entropy_now = model.compute_model_entropy(tok, docs)      # prediction certainty
        field_deviation = model.compute_field_deviation(tok, field, docs)  # drift from corpus
        purpose_magnitude = model.compute_purpose_vector()         # learning momentum
        purpose_alignment = model.purpose_gamma_alignment()        # cos(purpose, identity)
```

Then it decides how to adjust learning:

| State | Action | LR Multiplier | Meaning |
|-------|--------|---------------|---------|
| Syntropy rising + field in sweet spot + purpose aligned | **amplify** | 1.3× + delta grow | Everything aligned — push harder |
| Syntropy rising + purpose drifting | **boost** | 1.3× | Good direction, gentle push |
| Syntropy falling | **dampen** | 0.6× | Losing order — slow down |
| Field deviation too high | **ground** | 0.6× | Hallucinating — pull back to corpus |
| Field deviation too low | **explore** | 1.3× | Parroting — push out |
| Purpose opposes gamma | **realign** | 0.5× | Identity crisis — hard slow |

This is not heuristics — it's **mathematical introspection**. The organism measures its own entropy trend, computes how far it's drifted from corpus physics, measures whether its current learning direction aligns with its accumulated identity, and adjusts. Gamma is memory. Purpose is intention. Syntropy is the arrow.

Every decision is logged to the syntropy_log table with full metrics.

### 16. Swarm Ecology (Mitosis + Hibernation)

The organism is not alone. When it reaches adult stage and hits sustained overload, it **divides** — spawning a child organism at infant stage. Both train independently on the same corpus but grow through different paths.

```python
# SyntropyTracker detects overload → "divide" action
# Conditions: adult stage + entropy high for >75% of window + falling syntropy + 300s cooldown
if model_stage == adult and sustained_overload and cooldown_expired:
    action = "divide"

# perform_mitosis():
# 1. Save parent checkpoint
# 2. Create ~/.molecule/org_NNNN/ with birth.json
# 3. Child inherits burst_history (training efficiency memory)
# 4. Spawn child as subprocess
# 5. Both continue independently
```

**Hibernation** is cooperative, not resource-based. When an organism is on a loss plateau and a peer is actively thriving (syntropy > 0.05), it voluntarily sleeps:

```python
# _should_hibernate():
# - A peer has syntropy > 0.05 (actively improving)
# - Our last 8 bursts show |avg_delta| < 0.01 (plateau)
# → Go to sleep. Give the training flow to the young cell.
```

The **SwarmRegistry** is shared SQLite (`~/.molecule/swarm/mesh.db`, WAL mode):
- `organisms` table: id, pid, stage, n_params, syntropy, entropy, status (alive/sleeping/dead)
- `messages` table: inter-organism communication log
- Heartbeat every 10 training ticks
- `discover_peers()` finds other living organisms

This is not load balancing. It's **ecological optimization** — organisms that need to learn get priority, organisms that have plateaued yield. The field of answers remains shared.

---

## Configuration

```python
@dataclass
class Config:
    # Data
    corpus_path: str = "nonames.txt"
    db_path: str = "memory.sqlite3"
    ckpt_path: str = "molecule_ckpt.json"
    max_corpus_lines: int = 8000
    max_line_chars: int = 240
    min_new_chars_to_train: int = 480

    # Model (embryo defaults — organism grows via ontogenesis)
    tie_embeddings: bool = True        # GPT-style weight tying (wte == lm_head)
    n_layer: int = 1
    n_embd: int = 16                   # embryo stage
    n_head: int = 1
    block_size: int = 96

    # Ontogenesis (growth stages: corpus_chars, n_embd, n_layer, n_head)
    growth_stages: tuple = (
        (0,      16, 1, 1),            # embryo: ~25K params
        (20000,  32, 1, 2),            # infant: ~100K params
        (50000,  64, 2, 4),            # child: ~500K params
        (200000, 128, 4, 4),           # adolescent: ~2M params
        (500000, 256, 6, 8),           # adult: ~10M params
    )
    freeze_after_growth_steps: int = 200

    # Hybrid attention
    head_types: tuple = ("content",)   # auto-adapts with growth
    hybrid_alpha_init: float = 0.5

    # Gamma (personality fingerprint)
    gamma_sparsity_threshold: float = 0.01

    # Noise immune system
    noise_drift_threshold: float = -0.1    # cosine < this = rollback
    gamma_min_magnitude: float = 1e-6      # skip immune check when gamma is near-zero

    # Training
    warmup_steps: int = 1200
    micro_steps: int = 32
    learning_rate: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.99
    eps_adam: float = 1e-8
    grad_clip: float = 1.0
    freeze_base_after_warmup: bool = True
    batch_size: int = 4
    accum_steps: int = 1               # gradient accumulation (effective = batch_size × accum_steps)
    lr_min: float = 0.001              # cosine LR floor
    max_total_steps: int = 50000       # cosine LR period
    cosine_warmup_steps: int = 200     # linear warmup before cosine

    # Deltas (LoRA-ish)
    delta_rank: int = 8
    max_delta_modules: int = 12
    delta_grow_prob: float = 0.08

    # Generation
    temperature: float = 0.85
    top_k: int = 40
    top_p: float = 0.92
    min_p: float = 0.06               # GPT-3/4 style: filter below min_p × max_prob
    typical_p: float = 0.95            # typical sampling: prefer tokens with typical info content
    max_gen_tokens: int = 180
    min_gen_tokens: int = 16
    repetition_guard: int = 4

    # Tokenizer evolution (byte-level BPE)
    enable_bpe_after_chars: int = 20000
    bpe_num_merges: int = 384
    bpe_retrain_every_chars: int = 4000

    # Async
    train_tick_seconds: float = 0.25

    # QuantumBuffer
    qb_min_bytes: int = 1024
    qb_min_novelty: float = 0.15
    qb_cooldown_seconds: float = 60.0

    # Entropy-adaptive temperature
    entropy_low: float = 0.5
    entropy_high: float = 1.5
    entropy_temp_boost: float = 1.2
    entropy_temp_focus: float = 0.8

    # Corpus field
    corpus_gen_max_tokens: int = 120
    corpus_fade_k: float = 3.0         # sigmoid steepness for corpus→model transition
    corpus_fade_threshold: float = 1.5  # entropy at which blend is 50/50

    # SyntropyTracker
    syntropy_window: int = 8
    field_deviation_ceiling: float = 12.0
    field_deviation_floor: float = 0.1
    syntropy_lr_boost: float = 1.3
    syntropy_lr_dampen: float = 0.6
    syntropy_delta_grow_boost: float = 0.15
```

Want bigger? Change `n_embd`, `n_layer`, `block_size`. Want different attention? Change `head_types` to any mix of `"content"`, `"rrpram"`, `"hybrid"`. All parameters are shared across four implementations.

---

## Four Implementations

The same architecture, four languages, four habitats:

| Version | File | Language | Dependencies | Habitat |
|---------|------|----------|--------------|---------|
| **molecule.py** | `molecule.py` | Python 3.7+ | numpy | Terminal. numpy-accelerated autograd. |
| **molecule.go** | `molecule.go` | Go 1.21+ | `modernc.org/sqlite` | Terminal. Pure Go, no CGO. Goroutines. |
| **molecule.c** | `molecule.c` | C99 | `sqlite3`, `pthreads` | Terminal. Arena allocator, binary checkpoints. |
| **molecule.js** | `molecule.js` | ES2020+ | **none** | Browser. IndexedDB, Float64Array, DOM. |

All four share the same core: vector autograd, RoPE, SwiGLU, hybrid attention, delta adapters, evolving BPE, native gamma, cooccur field with adaptive corpus blend, quantum buffer, entropy temperature, growth table, immune system, syntropy tracker, ontogenesis, swarm ecology (mitosis + hibernation), no_grad inference, async training, persistent memory. **All four are at full Phase 3 parity.** Python and Go share JSON checkpoint format. C uses binary format (`MOLE` magic header). JS uses IndexedDB with JSON serialization.

```bash
# Python
python molecule.py

# Go
go build -o molecule_bin . && ./molecule_bin

# C
gcc -O2 -o molecule molecule.c -lsqlite3 -lpthread -lm && ./molecule

# JavaScript (browser)
python3 -m http.server 8000
# Open http://localhost:8000/index.html — that's it.
```


---

## Tests

```bash
python -m pytest tests/ -v
```

**205 Python tests** covering:
- Autograd (forward + backward, VectorValue + ScalarValue)
- Tokenizer (byte-level + BPE + vocab growth + UTF-8 roundtrip)
- Model (GPT, MatrixParam, DeltaAdapter, RoPE)
- Sampling (top-k, top-p, min_p, typical, softmax)
- Checkpointing (save/load + backward compat + dimension restoration)
- Integration (train → generate)
- Growth (grow_cols, grow_rows, maybe_grow_architecture, head types, freeze)
- SyntropyTracker (entropy, field deviation, purpose vector, alignment, decisions)
- Immune system (snapshot, restore, drift check, noise rejection)
- Ecology (SwarmRegistry, mitosis, hibernation, divide/hibernate actions, burst inheritance)

---

## Philosophy

This is not a tutorial. This is not a "minimal example." This is a **functional system** that:

- Learns continuously
- Never forgets
- Grows organically
- Has one dependency (numpy) — Go, C, and JS have zero
- Fits in one file per language
- Runs in a browser tab (molecule.js — no npm, no webpack, nothing)
- Speaks before it learns
- Grows a personality from zero
- Reasons mathematically about its own learning direction
- Divides when overloaded, sleeps when a peer needs the flow
- Writes its own structural autobiography
- Rejects noise that would corrupt its identity
- Actually generates text you can read

---

## Why "molecule"?

Because atoms are micrograd. We build molecules.

---

## Known Limitations

1. **Performance varies.** Python has numpy. Go and C are natively fast. JS runs in the browser — fast enough for chat, slower for training (no BLAS, Float64Array only). No CUDA anywhere.

2. **It starts small.** Default: embryo (1 layer, 16 dims, 1 head, ~25K params). Ontogenesis grows it through 5 stages to adult (6 layers, 256 dims, 8 heads, ~10M params). When it hits the ceiling, it divides. You're not getting GPT-4 reasoning. You're getting an ecology of organisms that grow and reproduce.

3. **It talks weird at first.** The corpus field helps, but it's still a baby organism. Feed it better corpus.

4. **The corpus matters.** Garbage in, garbage out. Give it good sentences.

---

## v3 Status

### Phase 1: Training Upgrades — DONE
- Real SiLU activation in SwiGLU (LLaMA-exact, not ReLU)
- Residual scaling: `α = 1/√n_layer` on attention + MLP
- Global cosine LR with linear warmup
- Gradient accumulation (configurable, ready for growth)
- BPE threshold lowered (activates on first launch)

### Phase 1.5: Self-Awareness Expansion — DONE
- Training efficiency memory: organism remembers which learning strategies work for *it*
- Temperature bridge: syntropy trend shifts generation temp ±5% (ordering → confident, disordering → exploratory)
- Quick loss measurement for before/after burst comparison
- Self-meta-learning: actions that historically increase loss get downgraded

### Phase 2: Byte-Level BPE Tokenizer — DONE
GPT-3/4 style tokenizer replacing char-level + word-based BPE:
- **Bootstrap**: 256 byte tokens + BOS/EOS/PAD = 259 initial vocab (language-agnostic)
- **Pre-segmentation**: split by Unicode category (letters / digits / whitespace / punctuation)
- **Stream BPE**: merges on byte sequences within segments, `+` separator (e.g. `0x48+0x65`)
- Full UTF-8 roundtrip: ASCII, Cyrillic, CJK, emoji — same algorithm, same code

### Phase 3A: Growing Architecture (Ontogenesis) — DONE (all four)
The organism starts as an embryo and grows through 5 stages:
```
Stage       Corpus    Dims  Layers  Heads  ~Params
embryo      0         16    1       1      ~25K
infant      20KB      32    1       2      ~100K
child       50KB      64    2       4      ~500K
adolescent  200KB     128   4       4      ~2M
adult       500KB     256   6       8      ~10M
```

Key mechanics:
- Old weights copy into top-left corner of new matrices (knowledge preserved)
- New dimensions initialize with small gaussian noise
- `grow_cols()` + `grow_rows()` = `grow()` on MatrixParam
- Delta adapters grow via `grow_dims()` (rank unchanged)
- Adam state resets (old momentum meaningless after arch change)
- Freeze period: base weights frozen for 200 steps after growth (only deltas train)
- Gamma snapshot extended for new embedding dimensions
- Head types auto-adapt: 1→(content), 2→(content,hybrid), 4→(2c,2h), 8→(4c,4h)

### Phase 3B: Mitosis & Ecology — DONE (all four)
When the adult organism is overloaded, it **divides**:
- SyntropyTracker detects sustained high entropy + falling syntropy → "divide" action
- Parent spawns child process at infant stage with inherited training memory
- Both organisms train independently on shared corpus
- Swarm registry (`~/.molecule/swarm/mesh.db`) tracks all living instances
- Generational knowledge: child inherits parent's burst_history (avoids same mistakes)
- Cooldown timer (300s) prevents runaway mitosis
- **Hibernation**: organism on plateau + thriving peer → voluntary sleep (metrics-based, not resource-based)
- CLI args (`--organism-id`, `--config`) for child processes
- 31 ecology tests covering all edge cases

### And Beyond
- **Inference routing** between organisms (lowest entropy answers)
- **Gamma export/import** (personality transfer between molecules)
- **Speculative decoding** (draft + verify for speed)
- **Mixture of Experts** (delta routing)


---

## License

GNU GPLv3 — Because freedom matters.

---

## Acknowledgments

- **Andrej Karpathy** — for micrograd, minGPT, nanoGPT, and the entire pedagogical empire
- **Coffee** — for existing

---

## Part of the Arianna Method

- [ariannamethod.ai](https://github.com/ariannamethod/ariannamethod.ai) — Arianna Method Language

- **molecule** — Single-File Continual GPT (Python, Go, C, JavaScript)

*Patterns over parameters. Emergence over engineering. The organism continues.*
