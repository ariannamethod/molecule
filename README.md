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
- SwiGLU-like gated MLP (LLaMA vibes)
- Hybrid attention: Content + RRPRAM + learnable blend
- Delta adapters (LoRA-style, never forgets)
- Native gamma: personality fingerprint that grows from zero
- BPE tokenizer that ONLY EXPANDS vocab (weights never invalidate)
- Corpus field: speaks before it learns (trigram statistics)
- QuantumBuffer: trains when it's ready, not when you tell it
- SyntropyTracker: mathematical self-reasoning about its own becoming
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
What if it was *alive*?  

So meet **molecule**. Thanks to Karpathy's microgpt, but this is not a fork.

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
# Open http://localhost:8000/molecule.html
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

### 3. SwiGLU-like Gated MLP

Standard MLP: `x → Linear → ReLU → Linear → out`

SwiGLU: `x → (Gate × Value) → Linear → out`

```python
g = fc_g(x).relu()   # gate
u = fc_v(x)          # value
x = g * u            # gating (element-wise)
x = fc2(x)           # project back
```

Why? Because LLaMA, PaLM, and basically everyone good uses it now. More expressive. Better gradients.

### 4. Hybrid Attention (Content + RRPRAM + Blend)

Three attention mechanisms coexist in the same model:

- **ContentHead**: Standard Q·K^T/√d with RoPE — semantic similarity
- **RRPRAM**: Recursive Resonant Pattern Recognition — `x @ W_pattern → (T,T)` attention that learns positional patterns directly, without query-key decomposition
- **HybridHead**: `sigmoid(α) × RRPRAM + (1-sigmoid(α)) × Content` — learnable gate decides the blend

```python
head_types = ("content", "content", "hybrid", "hybrid")
# Each head chooses its nature. Some listen to meaning.
# Some listen to rhythm. Some listen to both.
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

### 7. Evolving BPE (Vocab Only Expands)

Most tokenizers: retrain = throw away old model.

molecule: retrain = **add new tokens**. Old tokens remain. Embeddings remain. Model keeps working.

```python
# Old vocab: ['a', 'b', 'c', '<BOS>', '<EOS>']
# After BPE: ['a', 'b', 'c', '<BOS>', '<EOS>', 'ab', 'bc', 'abc', ...]
# Old weights: still valid!
# New rows: initialized, ready to train
```

This is how you build a system that grows over years, not hours.

### 8. Corpus Field (CooccurField)

The organism can speak **before** any weights are trained. Trigram → bigram → unigram fallback from the seed corpus.

```python
field = CooccurField()
field.build_from_corpus(tokenizer, docs)
# Now it can generate text using pure corpus statistics.
# No weights needed. No training needed. Just pattern resonance.
```

After warmup, model logits and corpus statistics blend via `generate_resonant()`:

```python
probs = alpha * model_probs + (1-alpha) * corpus_probs
```

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

---

## The Stack

| Component | microgpt | molecule |
|-----------|----------|----------|
| Autograd | Scalar (micrograd) | **Vector** (custom) |
| Position encoding | Sinusoidal | **RoPE** |
| Attention | Standard | **Hybrid** (Content + RRPRAM + blend) |
| MLP | ReLU | **SwiGLU-like gated** |
| Tokenizer | Fixed char | **Evolving BPE** |
| Training | One-shot | **Continuous async** |
| Training trigger | — | **QuantumBuffer** (bytes + novelty + cooldown) |
| Temperature | Fixed | **Entropy-adaptive** |
| Pre-training speech | — | **CooccurField** (trigram corpus stats) |
| Memory | None | **SQLite persistent** |
| Adapters | None | **LoRA-style deltas** |
| Personality | None | **Native gamma** (sparse embedding drift) |
| Growth tracking | None | **SQLite growth table** |
| Noise rejection | None | **Native immune system** (γ drift + delta rollback) |
| Self-reasoning | None | **SyntropyTracker** (entropy trend + field deviation + purpose alignment) |
| Sampling | top-k | **min_p + typical_p + nucleus** |
| Weight tying | No | **Yes (GPT-style)** |
| Dependencies | torch | **numpy** (Python) / **none** (Go, C, JS) |
| Browser | No | **molecule.js** (IndexedDB, zero deps, cooperative training) |

---

## Configuration

```python
@dataclass
class Config:
    # Data
    corpus_path: str = "nonames.txt"
    db_path: str = "memory.sqlite3"
    max_corpus_lines: int = 8000

    # Model
    n_layer: int = 2
    n_embd: int = 72           # Small but not stupid
    n_head: int = 4
    block_size: int = 96       # Context window

    # Hybrid attention
    head_types: tuple = ("content", "content", "hybrid", "hybrid")
    hybrid_alpha_init: float = 0.5

    # Gamma (personality fingerprint)
    gamma_sparsity_threshold: float = 0.01

    # Noise immune system
    noise_drift_threshold: float = -0.1  # cosine < this = rollback

    # Training
    warmup_steps: int = 1200
    learning_rate: float = 0.01

    # QuantumBuffer
    qb_min_bytes: int = 1024
    qb_min_novelty: float = 0.15
    qb_cooldown_seconds: float = 60.0

    # Entropy temperature
    entropy_low: float = 0.5
    entropy_high: float = 1.5
    entropy_temp_boost: float = 1.2
    entropy_temp_focus: float = 0.8

    # Corpus field
    corpus_gen_max_tokens: int = 120

    # Sampling
    temperature: float = 0.85
    top_k: int = 40
    top_p: float = 0.92
    min_p: float = 0.06        # GPT-3/4 style
    typical_p: float = 0.95
```

Want bigger? Change `n_embd`, `n_layer`, `block_size`. Want different attention? Change `head_types` to any mix of `"content"`, `"rrpram"`, `"hybrid"`.

---

## Four Implementations

The same architecture, four languages, four habitats:

| Version | File | Language | Dependencies | Habitat |
|---------|------|----------|--------------|---------|
| **molecule.py** | `molecule.py` | Python 3.7+ | numpy | Terminal. numpy-accelerated autograd. |
| **molecule.go** | `molecule.go` | Go 1.21+ | `modernc.org/sqlite` | Terminal. Pure Go, no CGO. Goroutines. |
| **molecule.c** | `molecule.c` | C99 | `sqlite3`, `pthreads` | Terminal. Arena allocator, binary checkpoints. |
| **molecule.js** | `molecule.js` | ES2020+ | **none** | Browser. IndexedDB, Float64Array, DOM. |

All four are at **full feature parity**: vector autograd, RoPE, SwiGLU, hybrid attention, delta adapters, evolving BPE, native gamma, cooccur field, quantum buffer, entropy temperature, growth table, immune system, syntropy tracker, no_grad inference, async training, persistent memory. Python and Go share JSON checkpoint format. C uses binary format (`MOLE` magic header). JS uses IndexedDB with JSON serialization.

```bash
# Python
python molecule.py

# Go
go build -o molecule_bin . && ./molecule_bin

# C
gcc -O2 -o molecule molecule.c -lsqlite3 -lpthread -lm && ./molecule

# JavaScript (browser)
python3 -m http.server 8000
# Open http://localhost:8000/molecule.html — that's it.
```


---

## Tests

```bash
python -m pytest tests/ -v
```

**139 Python tests** covering:
- Autograd (forward + backward, VectorValue + ScalarValue)
- Tokenizer (char-level + BPE + vocab growth)
- Model (GPT, MatrixParam, DeltaAdapter, RoPE)
- Sampling (top-k, top-p, min_p, typical, softmax)
- Checkpointing (save/load + backward compat)
- Integration (train → generate)
- SyntropyTracker (entropy measurement, field deviation, purpose vector, alignment, decisions)
- Immune system (snapshot, restore, drift check, noise rejection)

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
- Writes its own structural autobiography
- Rejects noise that would corrupt its identity
- Actually generates text you can read

---

## Why "molecule"?

Because atoms are micrograd. We build molecules.

---

## Known Limitations

1. **Performance varies.** Python has numpy. Go and C are natively fast. JS runs in the browser — fast enough for chat, slower for training (no BLAS, Float64Array only). No CUDA anywhere.

2. **It's small.** 2 layers, 72 dims, 4 heads (2 content + 2 hybrid). You're not getting GPT-4 reasoning. You're getting an embryo that *could* grow.

3. **It talks weird at first.** The corpus field helps, but it's still a baby organism. Feed it better corpus.

4. **The corpus matters.** Garbage in, garbage out. Give it good sentences.

---

## The Future

- **Speculative decoding** (draft + verify for speed)
- **Mixture of Experts** (multiple delta modules, routing)
- **Retrieval augmentation** (SQLite + embeddings)
- **Full RRPRAM heads** (pure positional attention without content)
- **Gamma export/import** (personality transfer between molecules)


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
