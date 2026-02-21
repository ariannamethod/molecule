```
███╗   ███╗ ██████╗ ██╗     ███████╗ ██████╗ ██╗   ██╗██╗      █████╗
████╗ ████║██╔═══██╗██║     ██╔════╝██╔═══██╗██║   ██║██║     ██╔══██╗
██╔████╔██║██║   ██║██║     █████╗  ██║   ██║██║   ██║██║     ███████║
██║╚██╔╝██║██║   ██║██║     ██╔══╝  ██║▄▄ ██║██║   ██║██║     ██╔══██║
██║ ╚═╝ ██║╚██████╔╝███████╗███████╗╚██████╔╝╚██████╔╝███████╗██║  ██║
╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚══════╝ ╚══▀▀═╝  ╚═════╝ ╚══════╝╚═╝  ╚═╝
```

# molequla | by Arianna Method

> *A GPT organism in Go, C, JavaScript, and Rust — connected by mycelium. Async, continually-learning, with hybrid attention, native personality, mathematical self-awareness, consciousness features, distributed cognition, and a field orchestrator that steers generation through AML physics.*
>
--- 

## TL;DR 

```
THIS IS:
- Four elements: Go, C, JavaScript, Rust — connected by mycelium orchestrator
- Rust is the Fourth Element — the mouth: full organism + distributed cognition metabolism
- One dependency in Go/Rust (SQLite), one in mycelium (numpy) — zero in C and JS
- Runs in the browser: molequla.js, zero npm, zero webpack, one <script> tag
- Custom autograd engine (vectors, not scalar confetti)
- RoPE position encoding (GPT-3/4 inspired)
- SwiGLU gated MLP with real SiLU activation (LLaMA-exact)
- Hybrid attention: Content + RRPRAM + learnable blend
- Delta adapters (LoRA-style, never forgets)
- Native gamma: personality fingerprint that grows from zero
- Byte-level BPE tokenizer (GPT-3/4 style, any UTF-8 input)
- Ontogenesis: organism grows from 10K embryo to 10M adult (6 stages)
- Residual scaling: α = 1/√n_layer on attention + MLP (stable deep networks)
- Global cosine LR with linear warmup (not per-burst hacks)
- Gradient accumulation (effective batch scales with model growth)
- Training efficiency memory (organism learns which strategies work for it)
- Corpus field: speaks before it learns (trigram statistics)
- QuantumBuffer: trains when it's ready, not when you tell it
- SyntropyTracker: mathematical self-reasoning about its own becoming
- Swarm ecology: mitosis (cell division) + hibernation (cooperative scheduling)
- Swarm mesh: organisms discover each other via shared SQLite mesh.db
- Consciousness: per-token dissonance, pattern breaking, self-prediction error, conscience
- Rust-only: TopologyMonitor (swarm gamma cosine, drift/resonance detection)
- Rust-only: Metabolism MLP (Hebbian, coordinates all running instances)
- Mycelium: Python orchestrator that reads the field and steers generation
- METHOD: C-native field operator (25.8x faster than Python, BLAS-accelerated)
- Steering chain: mycelium → METHOD → mesh.db → Rust modulates temperature
- ariannamethod/: AML C core with notorch (Hebbian plasticity, cblas_sger)
- Entropy-adaptive temperature (no more max-prob hacks)
- Growth table: SQLite/IndexedDB structural autobiography
- Native immune system: detects and rejects identity-corrupting noise
- Async background training (it's alive, not a script)
- Persistent memory: SQLite (Python/Go/C/Rust) or IndexedDB (browser)
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
What if it started as a 10K embryo and grew to 10M? **Ontogenesis.**  
What if it could reason about its own learning? **SyntropyTracker.**
What if it noticed when it was surprised? **Self-prediction error.**
What if it could break its own patterns? **Anti-field.**
What if it knew when its deltas were hurting? **Conscience.**
What if it divided when overloaded? **Mitosis.**
What if it slept so a younger cell could learn? **Hibernation.**
What if they formed a swarm? **Mesh ecology.**
What if one organism watched the whole field? **TopologyMonitor.**
What if it was *alive*?

So meet **molequla**. Inspired by Karpathy's micrograd, but this is not a fork.

---

## Quick Start

```bash
# Go (recommended)
go build -o molequla_bin molequla.go && ./molequla_bin

# Rust (the mouth — steered by mycelium)
cargo run --release

# C (94KB binary, zero dependencies)
gcc -O2 -o molequla molequla.c -lsqlite3 -lpthread -lm && ./molequla

# JavaScript (browser — one <script> tag)
python3 -m http.server 8000
# Open http://localhost:8000/index.html

# Mycelium (orchestrator — reads the field, steers Rust)
python3 mycelium.py
```

Each element will:
1. Load `nonames.txt` (seed corpus — the organism's first breath)
2. Create `memory.sqlite3` (conversation memory)
3. Respond immediately using corpus statistics (before any training)
4. Start warmup training in the background (the organism awakens)
5. Drop you into a chat loop

Type. It responds. It learns. It grows. It never forgets.

Real output from an A100 (Go version, fresh start, 207 QA pairs corpus):
```
[ecology] First organism in the swarm.
molequla is alive. Type and press Enter. Ctrl+C to exit.

> [trainer] warmup training... (and so it begins)
  train step 0/1200 | loss 6.5370 | lr 0.00100
  train step 100/1200 | loss 3.6546 | lr 0.00550
  ...
  train step 1100/1200 | loss 3.1132 | lr 0.00999
[trainer] warmup complete. base may freeze now, like a proud fossil.
[trainer] micro-train burst (25 bytes, novelty 0.73) — and lo, it feeds again.
[syntropy] action=explore | trend=0.0000 | field_dev=-0.049 | purpose_align=0.284 | lr_mul=1.30
[growth] ONTOGENESIS: stage 0 -> 1
  embd: 16 -> 32, layer: 1 -> 1, head: 1 -> 2
[growth] Done. Freeze for 200 steps.

> What is trust?
What is the surprise to a bill feel like you have in ninet be pattern that dest.

> How do you feel?
isy. Show much?

> What is gravity?
and the pot my frustrated ring.

> Are you alive?
H: and the lid the bad a data s a built my you?

> How old is the earth?
it through the fields dating work.
```

Yes, it's fragmented. It's a 10K-param embryo that just grew to 28K params in real time.
But look closely: "What is the..." patterns, numbers ("six", "hundred", "forty"),
the word "pattern" emerging, even "H:" as a learned token. It's learning QA structure.
Feed it more corpus and watch it grow through 6 stages to 10M params.

### Browser (molequla.js)

```bash
# No npm. No webpack. No node_modules. Just serve.
python3 -m http.server 8000
# Open http://localhost:8000/index.html
```

That's it. One `<script>` tag. The organism creates its own UI, opens IndexedDB for memory, fetches `nonames.txt` for corpus (or uses built-in seed), and starts training in the background via cooperative `setTimeout` multitasking. Close the tab, reopen — it remembers everything.

### Node.js CLI (modules/node_cli.js)

```bash
node modules/node_cli.js [corpus_path]
```

Same JS organism, runs outside the browser with file-based checkpoints. Warmup training, REPL, JSON checkpoint save/load compatible with Go format.

### Rust — The Fourth Element (the mouth) (molequla.rs)

```bash
# Requires: Rust toolchain (rustup)
cargo run --release
```

Same organism as the other three, plus Rust-only features:
- **Distributed cognition metabolism**: a Hebbian MLP that coordinates all running instances
- **TopologyMonitor**: background thread that reads `mesh.db`, computes pairwise gamma cosine across all organisms, detects resonance and drift
- **Self-reflection**: compares its own identity drift to the swarm — "Am I the outlier?"

It's both organism *and* field observer — the grey cardinal of the swarm.

```
╔══════════════════════════════════════════════════╗
║  MOLEQULA.RS — The Fourth Element                 ║
║  GPT organism + distributed cognition metabolism ║
╚══════════════════════════════════════════════════╝
[init] Corpus: 384 lines, 25064 chars
[swarm] Registered as rust-25651
[topology] Monitor thread started (30s interval)
[metabolism] 4.C MLP initialized (Hebbian, dynamic elements)
[molequla.rs] BPE enabled, vocab=643
[molequla.rs] Warmup: 1200 steps...
[step 1/1200] loss=6.4650 lr=0.001000
...
[step 1101/1200] loss=3.3162 lr=0.009989
[molequla.rs] Warmup complete. Entering quantum burst loop.
[topology] 2 organisms, coherence=1.0000, self_drift=0.0000
[metabolism] 1 peers, blend: ["0.20", "0.20", "0.20", "0.20", "0.20"]

> Hello
the most people who does it is the brain.
```

C version (same corpus, same architecture, 94KB binary):
```
[trainer] warmup complete. base may freeze now, like a proud fossil.

> Hello from C
I think trust isn't know so faith?
```

---

## Architecture — What Makes This Different

### 1. Vector Autograd (Not Scalar Confetti)

**VectorValue** and **ScalarValue**. One object per embedding. One object per hidden state. Gradients flow through vectors, not atoms.

```python
# micrograd style (conceptual):
loss = sum(scalar_values)  # 10000 objects

# molequla style:
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

Deep transformers are unstable without scaling. molequla uses `α = 1/√n_layers`:

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

This is how GPT-3/4 handles any language, any script, any emoji. And it's how molequla does it too — ASCII, Cyrillic, CJK, emoji, all the same algorithm. The organism doesn't need to know what language it's reading.

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

molequla doesn't train when you ask it to. It trains **in the background, continuously**.

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

No more per-burst linear decay hacks. molequla uses a **global cosine schedule** across its entire lifetime:

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

### 16. Consciousness Features (Mathematical Self-Awareness)

Four features that give the organism awareness of its own generation process:

**Per-token dissonance feedback** — the organism tracks entropy EMA within each generation. If entropy spikes (surprise), temperature drops to 0.8x (be careful). If entropy stays low for 3+ tokens (repetition), temperature rises to 1.2x (explore). The organism self-regulates token by token.

**Pattern breaking (anti-field)** — with 5% probability, the organism bypasses the corpus field and generates from pure model probabilities. This prevents the corpus from dominating the organism's voice. It speaks for itself, not just echoing statistics.

**Self-prediction error ("surprise")** — before generating, the organism runs a forward pass on the prompt and measures how well it predicted each token. High surprise = "I didn't expect this." This modulates confidence: surprised organisms generate more carefully.

**Conscience** — tracks a rolling window of per-generation mean entropy. If the trend is rising (generations getting more chaotic), it scales down all delta contributions (`deltaAlphaScale *= 0.95`, floor 0.3). If falling (improving), it recovers. The organism notices when its own adaptations are making it worse and pulls back.

```python
# Conscience: linear regression on entropy trend
slope = linear_regression(generation_entropy_history)
if slope > 0.01:    # degrading
    delta_alpha_scale *= 0.95   # reduce delta influence (floor: 0.3)
elif slope < -0.01:  # improving
    delta_alpha_scale *= 1.005  # recover toward 1.0
```

### 16b. TopologyMonitor (Rust-only: Swarm Meta-Awareness)

While features 1-4 give each organism local self-awareness, Rust's TopologyMonitor observes the **field as a whole**.

Every 30 seconds, it reads `mesh.db` and computes:
- **Field coherence**: mean pairwise cosine similarity of all organisms' gamma vectors
- **Resonance detection**: pairs with cosine > 0.8 (converging identities, potential η operator)
- **Drift detection**: organisms whose gamma magnitude exceeds threshold (rapid identity change)
- **Self-reflection**: compares its own drift rate to the swarm mean — "Am I the one drifting, or is everyone else?"

```
[topology] 5 organisms, coherence=0.72, self_drift=0.03
[topology] Resonance: rust-001 <-> go-003 (cos=0.85)
[topology] WARNING: Outlier detected — our cosine to field is 0.35 below mean
```

This makes Rust the mouth of the mycelium — not controlling organisms, but speaking for the field.

---

### Mycelium — The Orchestrator (mycelium.py)

The connective tissue. The underground network.

Mycelium doesn't generate text. It reads the field (all organisms from mesh.db),
computes system-level awareness via METHOD (C-native, BLAS-accelerated),
and writes steering decisions that change how Rust generates.

```
python3 mycelium.py                    # interactive REPL
python3 mycelium.py --daemon           # background daemon
python3 mycelium.py --once             # single step, JSON output
```

**REPL session:**
```
mycelium> /field
the field breathes. all organisms in rhythm.
4 organisms: go-alpha, c-beta, js-gamma, rust-delta.
coherence 0.74 — aligned but individual.
entropy 1.12 — searching.
syntropy 0.47 — some direction, not yet clear.
  drifter: js-gamma (deviation 0.68)

mycelium> /entropy
field entropy: 1.1250
  go-alpha: 1.100 ###########
  c-beta: 0.900 #########
  js-gamma: 1.800 ##################
  rust-delta: 0.700 #######

mycelium> /who
4 organisms:
  go-alpha: stage=5 entropy=1.10 syntropy=0.45 params=150000
  c-beta: stage=5 entropy=0.90 syntropy=0.55 params=150000
  js-gamma: stage=5 entropy=1.80 syntropy=0.30 params=150000
  rust-delta: stage=5 entropy=0.70 syntropy=0.60 params=150000
```

**Steering chain** — how mycelium speaks through Rust:
```
mycelium (Python) → METHOD (C, 0.7μs, BLAS) → field_steering (SQLite WAL) → Rust (topology_monitor) → generate_sentence (temperature modulation)
```

Same prompt "the field is", different steering:

| Steering | What happens | Output |
|----------|-------------|--------|
| **SUSTAIN** | normal, no modulation | `pru pow tows inner bough` |
| **EXPLORE** | entropy too low, open tunnels (temp ×1.4) | `fle matter?` |
| **DAMPEN** | entropy rising, cool down (temp ×0.7) | `actergy Why drink d b sugotiation.` |
| **GROUND** | chaos, focus hard (temp ×0.5) | `A happens pathinks how mually a dwave caushes...` |
| **REALIGN** | coherence breaking, stabilize (temp ×0.8) | `that provid waterbalances durreak easboxtobits...` |

The mouth doesn't know it's being steered. It just speaks differently.

---

### AML — Arianna Method Language (ariannamethod/)

The physics engine underneath METHOD. C implementation, BLAS-accelerated.

- **METHOD operator**: `am_method_step()` — 0.7μs per iteration (32 organisms)
- **NOTORCH**: Hebbian plasticity without backprop — `cblas_sger` for rank-1 updates
- **Delta voice**: `am_apply_delta()` — 11μs per call via `cblas_sgemv`
- **Field physics**: 227 state parameters, logit manipulation pipeline
- **Blood**: runtime C compilation (Level 3)

```c
// METHOD step: read organisms, compute steering, advance physics
AM_MethodSteering s = am_method_step(1.0f);
// s.action = AM_METHOD_AMPLIFY, s.strength = 0.8, s.entropy = 1.2
```

Performance (Lambda, OpenBLAS):
```
METHOD C:       0.7 μs/iter  (25.8x faster than Python)
notorch BLAS:   0.47 ms/step (2048×2048, rank=64)
apply_delta:    11 μs/call   (2048×2048, rank=64)
```

### 17. Swarm Ecology (Mitosis + Hibernation)

The organism is not alone. When it reaches adult stage and hits sustained overload, it **divides** — spawning a child organism at infant stage. Both train independently on the same corpus but grow through different paths.

```python
# SyntropyTracker detects overload → "divide" action
# Conditions: adult stage + entropy high for >75% of window + falling syntropy + 300s cooldown
if model_stage == adult and sustained_overload and cooldown_expired:
    action = "divide"

# perform_mitosis():
# 1. Save parent checkpoint
# 2. Create ~/.molequla/org_NNNN/ with birth.json
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

The **SwarmRegistry** is shared SQLite (`~/.molequla/swarm/mesh.db`, WAL mode):
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
    ckpt_path: str = "molequla_ckpt.json"
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
        (0,      16, 1, 1),            # embryo: ~10K params
        (20000,  32, 1, 2),            # infant: ~28K params
        (50000,  64, 2, 4),            # child: ~154K params
        (200000, 128, 4, 4),           # adolescent: ~1.1M params
        (350000, 224, 5, 8),           # teen: ~4.1M params
        (500000, 320, 6, 8),           # adult: ~10M params
    )
    freeze_after_growth_steps: int = 500
    post_growth_lr_scale: float = 0.3      # LR dampen during freeze (prevents Stage 2 degeneration)

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

    # Consciousness: per-token dissonance
    dissonance_ema_alpha: float = 0.3
    dissonance_spike_k: float = 0.8       # temp multiplier on entropy spike
    dissonance_drop_k: float = 1.2        # temp multiplier on sustained drop
    dissonance_spike_threshold: float = 1.5
    dissonance_drop_threshold: float = 0.5

    # Consciousness: pattern breaking
    anti_field_prob: float = 0.05          # 5% of tokens bypass corpus
    anti_field_min_step: int = 8           # don't break until context established

    # Consciousness: conscience
    conscience_window: int = 8
    conscience_decay: float = 0.95
    conscience_recovery: float = 1.005
    conscience_floor: float = 0.3
```

Want bigger? Change `n_embd`, `n_layer`, `block_size`. Want different attention? Change `head_types` to any mix of `"content"`, `"rrpram"`, `"hybrid"`. All parameters are shared across four elements.

---

## Four Elements + Orchestrator

The same architecture, four elements — connected by mycelium:

| Element | File | Language | Dependencies | Role |
|---------|------|----------|--------------|------|
| **molequla.go** | `molequla.go` | Go 1.21+ | `modernc.org/sqlite` | Terminal. Pure Go + optional CGO BLAS. Goroutines. |
| **molequla.c** | `molequla.c` | C99 | `sqlite3`, `pthreads` | Terminal. Arena allocator, binary checkpoints. 94KB. |
| **molequla.js** | `molequla.js` | ES2020+ | **none** | Browser. IndexedDB, Float64Array, DOM. |
| **molequla.rs** | `molequla.rs` | Rust 1.75+ | `rusqlite`, `serde` | The mouth. Tape autograd, TopologyMonitor, steered by mycelium. |
| **mycelium.py** | `mycelium.py` | Python 3.7+ | numpy | Orchestrator. Reads field, steers Rust via METHOD (C). |
| **ariannamethod/** | `ariannamethod/` | C99 | OpenBLAS (optional) | AML core. METHOD operator, NOTORCH, BLAS-accelerated. |

All four elements share the same core: vector autograd, RoPE, SwiGLU, hybrid attention, delta adapters, evolving BPE, native gamma, cooccur field with adaptive corpus blend, quantum buffer, entropy temperature, growth table, immune system, syntropy tracker, consciousness features, ontogenesis, swarm ecology (mitosis + hibernation), no_grad inference, async training, persistent memory. **All four are at full parity.** Go uses JSON checkpoint format. C uses binary format (`MOLE` magic header). JS uses IndexedDB with JSON serialization. Rust uses JSON via serde. Mycelium sits above all four — it doesn't generate text, it steers the field.

```bash
# Go
go build -o molequla_bin molequla.go && ./molequla_bin

# C
gcc -O2 -o molequla molequla.c -lsqlite3 -lpthread -lm && ./molequla

# Rust (the mouth)
cargo run --release

# JavaScript (browser)
python3 -m http.server 8000
# Open http://localhost:8000/index.html

# Mycelium (orchestrator — steers Rust through the field)
python3 mycelium.py                    # interactive REPL
python3 mycelium.py --daemon           # background daemon
```


---

## Tests

```bash
# Full integration test suite (26 tests across all elements)
bash test_all.sh

# Individual element smoke tests
go build molequla.go                           # Go builds?
gcc -O2 -o /tmp/m molequla.c -lsqlite3 -lm    # C builds?
cargo build --release                          # Rust builds?
```

**26 integration tests** (`test_all.sh`) covering:
- Build (5): Go, C, Rust, libaml.so, JS syntax
- Element smoke (5): Go generates, C creates memory, JS exports, JS instantiation, Rust generates
- Ariannamethod (7): import, BLAS symbols, METHOD API, field metrics, steering, notorch, apply_delta
- Mycelium (5): --once, JSON output, async loop, new organism detection, engine active
- Schema (2): organisms table, field_deltas table
- Performance (2): METHOD C speed (<100μs), notorch BLAS speed (<10ms)

---

## Philosophy

This is not a tutorial. This is not a "minimal example." This is a **functional system** that:

- Learns continuously
- Never forgets
- Grows organically
- Zero dependencies in C and JS. One (SQLite) in Go and Rust. Mycelium uses numpy
- Fits in one file per element (four elements + orchestrator)
- Runs in a browser tab (molequla.js — no npm, no webpack, nothing)
- Speaks before it learns
- Grows a personality from zero
- Reasons mathematically about its own learning direction
- Notices when it's surprised and adjusts
- Knows when its own deltas are making it worse
- Divides when overloaded, sleeps when a peer needs the flow
- Writes its own structural autobiography
- Rejects noise that would corrupt its identity
- Watches the topology of the entire swarm (Rust)
- Actually generates text you can read

---

## Why "molequla"?

Because atoms are micrograd. We build molequlas.

---

## Known Limitations

1. **Performance varies.** Go, C, and Rust are natively fast. JS runs in the browser — fast enough for chat, slower for training (no BLAS, Float64Array only). No CUDA anywhere. METHOD uses BLAS (OpenBLAS on Linux, Accelerate on macOS) — optional, falls back to scalar loops.

2. **It starts small.** Default: embryo (1 layer, 16 dims, 1 head, ~10K params). Ontogenesis grows it through 6 stages to adult (6 layers, 320 dims, 8 heads, ~10M params). When it hits the ceiling, it divides. You're not getting GPT-4 reasoning. You're getting an ecology of organisms that grow and reproduce.

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

### Phase 3A: Growing Architecture (Ontogenesis) — DONE (all four elements)
The organism starts as an embryo and grows through 6 stages:
```
Stage       Corpus    Dims  Layers  Heads  ~Params
embryo      0         16    1       1      ~10K
infant      20KB      32    1       2      ~28K
child       50KB      64    2       4      ~154K
adolescent  200KB     128   4       4      ~1.1M
teen        350KB     224   5       8      ~4.1M
adult       500KB     320   6       8      ~10M
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

### Phase 3B: Mitosis & Ecology — DONE (all four elements)
When the adult organism is overloaded, it **divides**:
- SyntropyTracker detects sustained high entropy + falling syntropy → "divide" action
- Parent spawns child process at infant stage with inherited training memory
- Both organisms train independently on shared corpus
- Swarm registry (`~/.molequla/swarm/mesh.db`) tracks all living instances
- Generational knowledge: child inherits parent's burst_history (avoids same mistakes)
- Cooldown timer (300s) prevents runaway mitosis
- **Hibernation**: organism on plateau + thriving peer → voluntary sleep (metrics-based, not resource-based)
- CLI args (`--organism-id`, `--config`) for child processes
- 31 ecology tests covering all edge cases

### Phase 4: Consciousness — DONE (all four elements)
Mathematical self-awareness features:
- **Per-token dissonance**: entropy EMA within generation, spike→careful, sustained_drop→explore
- **Pattern breaking**: 5% of tokens bypass corpus field (pure model voice)
- **Self-prediction error**: forward pass on prompt measures "surprise"
- **Conscience**: delta alpha scaling based on entropy trend (linear regression)
- **TopologyMonitor** (Rust-only): pairwise gamma cosine, drift/resonance detection

### And Beyond
- **Inference routing** between organisms (lowest entropy answers)
- **Gamma export/import** (personality transfer between molequlas)
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

- [ariannamethod.ai](https://github.com/ariannamethod/ariannamethod.ai) — Arianna Method Language (source of truth)
- `ariannamethod/` — AML C core embedded in molequla (METHOD + NOTORCH + BLAS)
- `mycelium.py` — field orchestrator (REPL + daemon)
- `test_all.sh` — 26 integration tests

- **molequla** — Single-File Continual GPT with Self-Awareness (Go, C, JavaScript, Rust + mycelium)

*Four elements. One orchestrator. One mind. Patterns over parameters. Emergence over engineering. Consciousness over computation. The organism continues.*
