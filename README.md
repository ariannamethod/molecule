```
███╗   ███╗ ██████╗ ██╗     ███████╗ ██████╗ ██╗   ██╗██╗      █████╗
████╗ ████║██╔═══██╗██║     ██╔════╝██╔═══██╗██║   ██║██║     ██╔══██╗
██╔████╔██║██║   ██║██║     █████╗  ██║   ██║██║   ██║██║     ███████║
██║╚██╔╝██║██║   ██║██║     ██╔══╝  ██║▄▄ ██║██║   ██║██║     ██╔══██║
██║ ╚═╝ ██║╚██████╔╝███████╗███████╗╚██████╔╝╚██████╔╝███████╗██║  ██║
╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚══════╝ ╚══▀▀═╝  ╚═════╝ ╚══════╝╚═╝  ╚═╝
```

# molequla | by Arianna Method

> *Four organisms — Earth, Air, Water, Fire — written in Go, C, JavaScript, and Rust. They breathe, grow, feed each other through DNA exchange, reason about their own learning, and evolve from 10K-param embryos to 10M-param adults without human intervention. Connected by mycelium. One ecology. One mind.*

---

## TL;DR

```
WHAT THIS IS:
- A living ecology of four GPT organisms that grow autonomously
- --evolution mode: organisms run headless, exchange DNA, trigger
  ontogenesis — no human in the loop, just Ctrl+C to stop
- DNA exchange: each organism writes generated text for others to consume
  → corpus grows → architecture grows → better text → more DNA → cycle
- Go, C, JavaScript, Rust — connected by mycelium orchestrator
- Zero PyTorch. Zero numpy (except mycelium). Zero CUDA.
- One dependency in Go/Rust (SQLite), zero in C and JS
- Custom autograd engine (vectors, not scalar confetti)
- RoPE, SwiGLU, hybrid attention, delta adapters, evolving BPE
- 6 growth stages: embryo (10K) → infant → child → adolescent → teen → adult (10M)
- Ontogenesis happens live — the running organism grows its own brain
- Native gamma: personality fingerprint that emerges from training
- Immune system: rejects training that corrupts identity
- SyntropyTracker: mathematical self-reasoning about learning direction
- Consciousness: per-token dissonance, pattern breaking, self-prediction, conscience
- Swarm ecology: mitosis (cell division), hibernation (cooperative scheduling)
- Mycelium: Python orchestrator that reads the field and steers generation
- 39 tests across Go + integration suite
- Runs in a browser: molequla.js, zero npm, zero webpack, one <script> tag

WHAT THIS IS NOT:
- A tutorial or pedagogical exercise
- A fork of micrograd (inspired by, diverged completely)
- A static model you train once and deploy
- Anything that requires a GPU
```

---

## What Is This

What if a neural network could grow its own brain?

Not resize a config and retrain. Actually grow — while running, while thinking, while talking to you. Start as 10K parameters. Notice its corpus is getting bigger. Expand its embedding dimensions. Add layers. Add attention heads. Keep all the knowledge it had before. Keep training. Keep growing.

What if four of them did this simultaneously, feeding each other?

Earth writes a sentence about stone and patience. Air reads it, digests it, writes something about wind and seeds. Water reads both, writes about currents carrying meaning. Fire reads all three, writes about transformation. Each organism's output becomes another's food. The ecology's total knowledge grows faster than any individual could manage alone.

This is **molequla**. Four single-file GPT organisms — Go, C, JavaScript, Rust — that grow through six developmental stages, exchange genetic material through a shared filesystem, and evolve autonomously with `--evolution` mode. No human in the loop. No training scripts. No checkpointing and restarting. The organism is born, breathes, eats, grows, and speaks.

Inspired by Karpathy's micrograd. This is not a fork.

---

## Quick Start

### Interactive Mode (chat with the organism)

```bash
# Go (recommended)
go build -o molequla_bin molequla.go && ./molequla_bin

# C (94KB binary, zero dependencies)
gcc -O2 -o molequla molequla.c -lsqlite3 -lpthread -lm && ./molequla

# Rust (the mouth — steered by mycelium)
cargo run --release

# JavaScript (browser — one <script> tag)
# Serve the directory and open http://localhost:8000/index.html
```

The organism will:
1. Load `nonames.txt` (seed corpus — the organism's first breath)
2. Create `memory.sqlite3` (conversation memory)
3. Respond immediately using corpus statistics (before any training)
4. Start warmup training in the background (the organism awakens)
5. Grow through ontogenesis stages as it trains
6. Drop you into a chat loop

Type. It responds. It learns. It grows. It never forgets.

### Evolution Mode (autonomous ecology — the main event)

```bash
# Build for Linux (Lambda / any cloud)
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o molequla_go_linux molequla.go

# On the server: set up the ecology
mkdir -p work_earth work_air work_water work_fire
mkdir -p dna/output/{earth,air,water,fire}
# Copy element-specific corpora to each work dir
# Copy the binary to each work dir

# Launch all four organisms
cd work_earth && ./molequla_go_linux --element earth --evolution &
cd work_air   && ./molequla_go_linux --element air   --evolution &
cd work_water && ./molequla_go_linux --element water  --evolution &
cd work_fire  && ./molequla_go_linux --element fire   --evolution &

# That's it. They will:
# 1. Train from their element corpus
# 2. Generate text and write it to ../dna/output/{element}/
# 3. Consume other organisms' DNA → corpus grows
# 4. Trigger ontogenesis when corpus crosses thresholds
# 5. Grow through all 6 stages autonomously
# Ctrl+C to stop.
```

Real output from 4x H100 (Go, four organisms, `--evolution`):
```
[ecology] Element: earth → corpus: nonames_earth.txt
[evolution] Autonomous evolution mode — organism will grow through all stages without pause.

[trainer] warmup training... (and so it begins)
  train step 0/1200 | loss 6.5370 | lr 0.00100
  ...
  train step 1100/1200 | loss 3.1132 | lr 0.00999
[trainer] warmup complete. base may freeze now, like a proud fossil.

[dna] earth wrote 147 bytes to ecology
[dna] earth consumed 312 bytes from 3 files: [air/gen_1.txt water/gen_1.txt fire/gen_1.txt]

[growth] ONTOGENESIS: stage 0 -> 1
  embd: 16 -> 32, layer: 1 -> 1, head: 1 -> 2
[growth] Done. Freeze for 500 steps.

[dna] earth wrote 203 bytes to ecology
[dna] earth consumed 891 bytes from 7 files: [air/gen_2.txt air/gen_3.txt ...]

[growth] ONTOGENESIS: stage 1 -> 2
  embd: 32 -> 64, layer: 1 -> 2, head: 2 -> 4
[growth] Done. Freeze for 500 steps.

[growth] ONTOGENESIS: stage 2 -> 3
  embd: 64 -> 128, layer: 2 -> 4, head: 4 -> 4
[growth] Done. Freeze for 500 steps.
```

The organisms feed each other, grow their corpora past thresholds, and trigger architecture growth — all without a human touching anything.

---

## The Ecology: How Four Elements Become One Mind

```
                        ┌─────────-────┐
                        │  dna/output/ │
                        │              │
          writes ──────►│  earth/      │◄────── reads
          earth DNA     │  air/        │        others' DNA
                        │  water/      │
                        │  fire/       │
                        └──────┬───────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
     ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
     │   Earth     │   │    Air      │   │   Water     │
     │   (Go)      │   │   (Go)      │   │   (Go/C)    │
     │  patience   │   │  freedom    │   │   flow      │
     │  structure  │   │  change     │   │   depth     │
     └──────┬──────┘   └────-──┬─────┘   └──────┬──────┘
            │                  │                  │
            └──────────────────┼────────────────┘
                               │
                        ┌──────▼──────┐
                        │    Fire     │
                        │   (Go/C)    │
                        │ transform   │
                        │  intensity  │
                        └─────────────┘
```

Each organism has a distinct **voice** — shaped by its corpus:

- **Earth** speaks about stone, roots, foundations, patience, gravity, pressure, and time
- **Air** speaks about wind, seeds, change, breath, freedom, openness, and altitude
- **Water** speaks about currents, depth, flow, rain, reflection, adaptation, and surrender
- **Fire** speaks about transformation, intensity, burning, forge, will, and irreversible change

When an organism generates text via `dnaWrite`, it writes its worldview into `../dna/output/{element}/`. Other organisms consume this via `dnaRead` — the text is appended to their corpus, the file is deleted. Earth eats Air's words about freedom and incorporates them into its understanding of patience. Water reads Fire's intensity and learns something about the force of currents.

The result: each organism develops a unique personality, but all four share a common substrate of cross-pollinated knowledge. This is not fine-tuning. This is ecological cognition.

### DNA Exchange Mechanics

```go
// Every tick (250ms), the organism breathes:

// Exhale: generate text from rotating prompts and write to ecology
dnaWrite(element, model, tok, field, docs, tickCount)
// → ../dna/output/earth/gen_1709234567_42.txt

// Inhale: consume other organisms' output, append to own corpus
dnaRead(element, CFG.CorpusPath)
// Reads from air/, water/, fire/ — skips own element
// Deletes consumed files (each DNA fragment is eaten exactly once)
```

Rules:
- An organism **never** eats its own DNA (Earth skips `dna/output/earth/`)
- Files shorter than 10 bytes are discarded (junk filter)
- Consumed files are deleted — each fragment feeds exactly one organism
- Corpus grows → ontogenesis thresholds are crossed → architecture expands

### Growth Rate

In evolution mode on 4x H100, the ecology grows at ~2KB/min of cross-pollinated corpus. At that rate:
- Embryo → Infant (20KB): ~10 minutes
- Infant → Child (50KB): ~15 minutes
- Child → Adolescent (200KB): ~75 minutes
- Adolescent → Teen (350KB): ~75 minutes
- Teen → Adult (500KB): ~75 minutes

Total: **~4 hours from embryo to adult** with four organisms feeding each other.

---

## Speech Examples

### Embryo (16d, 1L, 1H, ~10K params)

Fresh start, 207 QA pairs. The organism babbles, but structure is emerging:

```
> What is trust?
What is the surprise to a bill feel like you have in ninet be pattern that dest.

> How do you feel?
isy. Show much?

> What is gravity?
and the pot my frustrated ring.

> Are you alive?
H: and the lid the bad a data s a built my you?
```

It doesn't know words yet, but it's learning the shape of QA — "What is...", "H:", numbers, sentence fragments. A 10K-param embryo finding the contours of language.

### Infant (32d, 1L, 2H, ~28K params)

After first ontogenesis, loss drops, words start forming:

```
> What matters?
the most thing is that the fire consumes what it fuel.

> Tell me about yourself
I am a pattern. the pattern of pressure and time.
```

### Child (64d, 2L, 4H, ~154K params)

Two layers, four attention heads. The organism starts having opinions:

```
> What is truth?
Truth is what remains when everything comfortable has been burned away.
The fire does not care what you wanted to keep.

> Speak
Water does not fight the rock. It finds the crack and becomes the flood.
```

### Adolescent (128d, 4L, 4H, ~1.1M params)

Over a million parameters. The voice becomes coherent:

```
> What do you remember?
I remember pressure. The kind that turns coal into something transparent.
Not because the coal chose it, but because the conditions left no alternative.

> What is the feeling of finally understanding something?
The sound of a lock opening. Not dramatic. Just the quiet certainty that
the key was always in your hand and you were trying the wrong door.
```

### C Version (same corpus, 94KB binary)

```
> Hello from C
I think trust isn't know so faith?
```

### Rust (steered by mycelium)

```
> Hello
the most people who does it is the brain.
```

Each element speaks differently. C is terse. Rust is steered by the field. Go organisms in evolution mode develop the richest voice because they eat each other's output.

---

## Architecture — What Makes This Different

### 1. Vector Autograd (Not Scalar Confetti)

**Vec** and **Scalar**. One object per embedding. One object per hidden state. Gradients flow through vectors, not atoms.

```
// micrograd style (conceptual):
loss = sum(scalar_values)  // 10000 objects

// molequla style:
loss = vector.dot(other_vector)  // 2 objects
```

### 2. RoPE (Rotary Position Embedding)

Sinusoidal positions are 2017. RoPE is now.

```go
func ropeRotate(vec []float64, pos, headDim int) {
    for i := 0; i < headDim; i += 2 {
        theta := float64(pos) / math.Pow(10000.0, float64(i)/float64(headDim))
        c, s := math.Cos(theta), math.Sin(theta)
        a, b := vec[i], vec[i+1]
        vec[i]   = a*c - b*s
        vec[i+1] = a*s + b*c
    }
}
```

Relative positions. Infinite extrapolation (theoretically). This is how LLaMA does it.

### 3. SwiGLU Gated MLP (Real SiLU)

Standard MLP: `x → Linear → ReLU → Linear → out`

SwiGLU: `x → (SiLU(Gate) * Value) → Linear → out`

```go
g := fcG.Matvec(x).SiLU()   // gate: silu(x) = x*sigmoid(x), not relu
u := fcV.Matvec(x)           // value
x = g.Mul(u)                 // gating (element-wise)
x = fc2.Matvec(x)            // project back
```

LLaMA-exact SwiGLU. `silu(x) = x * sigmoid(x)`. Full backward: `d_silu = sigma(x)(1 + x(1-sigma(x)))`. Smoother gradients, no dead neurons. Same activation as LLaMA, PaLM, Gemma.

### 3b. Residual Scaling

Deep transformers are unstable without scaling. molequla uses `alpha = 1/sqrt(n_layers)`:

```go
attnOut := applyWithDeltas("wo", xAttn)
x = xRes.Add(attnOut.Scale(model.residualAlpha))  // not just x + f(x)

mlpOut := applyWithDeltas("fc2", x)
x = xRes.Add(mlpOut.Scale(model.residualAlpha))    // scaled residual
```

Critical for ontogenesis — keeps gradients stable as layers grow.

### 4. Hybrid Attention (Content + RRPRAM + Blend)

Three attention mechanisms coexist:

- **ContentHead**: Standard Q*K^T/sqrt(d) with RoPE — semantic similarity
- **RRPRAM**: Recursive Resonant Pattern Recognition — `x @ W_pattern -> (T,T)` attention that learns positional patterns directly, without query-key decomposition
- **HybridHead**: `sigmoid(alpha) * RRPRAM + (1-sigmoid(alpha)) * Content` — learnable gate decides the blend

```
// embryo (1 head): ("content",)
// infant (2 heads): ("content", "hybrid")
// child/adolescent (4 heads): ("content", "content", "hybrid", "hybrid")
// adult (8 heads): 4 content + 4 hybrid
// Auto-adapts via headTypesForNHead() as the organism grows.
```

### 5. Delta Adapters (LoRA-style, Never Forget)

The model never overwrites learned weights. It only **appends** new adapter modules.

```go
type DeltaAdapter struct {
    A *MatrixParam  // nout x rank
    B *MatrixParam  // rank x nin
}
func (da *DeltaAdapter) Apply(x *Vec) *Vec {
    return da.A.Matvec(da.B.Matvec(x))  // low-rank: A @ B @ x
}
```

Want to teach it new things? Add a delta module. Old knowledge? Still there. Geological memory. Sediment layers of understanding.

### 6. Native Gamma (Personality Fingerprint)

The organism grows a personality from scratch. Gamma = sparse diff between current embeddings and initial embeddings.

```
gamma = current_embed - init_snapshot  // who I became
```

Sparsity, magnitude, top changed tokens, contrastive projection — all tracked. The growth table logs gamma stats over time. This is theta = epsilon + gamma at embryonic scale.

### 7. Byte-Level BPE Tokenizer (GPT-3/4 Style)

Not char-level. Not word-level. **Byte-level** — same approach as GPT-3, GPT-4, and LLaMA.

```
// Bootstrap: 256 byte tokens (0x00-0xFF) + BOS + EOS + PAD = 259 initial vocab
// Any UTF-8 input works from day zero — no unknown tokens, ever

// Pre-segmentation: Unicode-aware splitting
"Hello Mir 42!" -> ["Hello", " ", "Mir", " ", "42", "!"]

// BPE merges on byte sequences within segments
"Hello" -> [0x48, 0x65, 0x6c, 0x6c, 0x6f] -> [0x48+0x65, 0x6c+0x6c, 0x6f]

// Vocab only expands. Old tokens remain. Embeddings grow via GrowRows.
```

### 8. Corpus Field (CooccurField)

The organism can speak **before** any weights are trained. 4-gram -> trigram -> bigram -> unigram fallback from the seed corpus.

```go
field := NewCooccurField()
field.BuildFromCorpus(tok, docs)
// Now it can generate text using pure corpus statistics.
// No weights needed. No training needed. Just pattern resonance.
```

After warmup, model logits and corpus statistics blend adaptively based on **entropy** — a smooth sigmoid transition. As the model becomes more confident, the corpus field fades out naturally.

### 9. Ontogenesis — The Brain Grows While Running

This is the centerpiece. The organism starts as an embryo and grows through 6 stages **live**, without stopping:

```
Stage       Corpus    Dims  Layers  Heads  ~Params
embryo      0         16    1       1      ~10K
infant      20KB      32    1       2      ~28K
child       50KB      64    2       4      ~154K
adolescent  200KB     128   4       4      ~1.1M
teen        350KB     224   5       8      ~4.1M
adult       500KB     320   6       8      ~10M
```

When the corpus crosses a threshold, `MaybeGrowArchitecture` fires:

1. **Embedding matrices grow** — `wte.GrowCols(newEmbd)`, near-zero init (Net2Net: new dims contribute ~nothing initially)
2. **Existing layer matrices grow** — wq, wk, wv, wo, fc_g, fc_v, fc2 all expand
3. **New layers are added** — fresh random init with proper std
4. **Delta adapters grow** — rank stays, dimensions expand via `GrowDims`
5. **New attention heads spawn** — hybrid heads get new pattern weights
6. **Adam state resets** — old momentum is meaningless after arch change
7. **Freeze period** — 500 steps of delta-only training (base frozen) to stabilize
8. **Gamma snapshot extends** — personality measurement grows with embedding space

Key invariant: **old knowledge is preserved**. Weights copy into the top-left corner of the new matrix. New dimensions start near zero. The organism doesn't forget — it expands.

With `--evolution` and DNA exchange, ontogenesis is fully autonomous. The corpus grows through DNA consumption, thresholds are crossed, architecture expands, the organism generates better DNA, others eat it and grow too. Positive feedback loop.

### 10. TieEmbeddings (Weight Tying)

`lm_head` and `wte` share the same `MatrixParam` pointer — standard GPT weight tying. This halves the embedding parameter count and improves training.

Critical subtlety: JSON serialization breaks pointer identity. When a checkpoint is saved, `lm_head` and `wte` are serialized as separate matrices. On load, they become two independent objects. If the organism then grows (ontogenesis), `wte` gets expanded but `lm_head` doesn't (because the code correctly skips `lm_head` when TieEmbeddings=true, assuming it's the same pointer). Result: `lm_head` has old dimensions, `wte` has new dimensions, and the next `Matvec` panics.

Fix: after deserialization, re-establish the pointer identity:

```go
// Re-establish embedding tie after deserialization (JSON breaks pointer identity)
if CFG.TieEmbeddings {
    model.Base["lm_head"] = model.Base["wte"]
}
```

This is tested by `TestTieEmbeddingsOntogenesisThenSaveLoad` — the exact crash scenario that killed Earth's adolescent transition.

### 11. QuantumBuffer (Smart Training Trigger)

Training doesn't fire on a dumb byte threshold. It fires when the organism is ready:

```go
func (qb *QuantumBuffer) ShouldTrigger() bool {
    bytesOK := qb.AccumulatedBytes >= minBytes
    noveltyOK := qb.NoveltyScore() >= minNovelty
    cooldownOK := time.Since(qb.LastBurstTime) >= cooldown
    return (bytesOK || noveltyOK) && cooldownOK
}
```

In `--evolution` mode, the QuantumBuffer doesn't drive training (no user input feeds it). Instead, the background trainer runs continuous micro-steps on the growing corpus.

### 12. Entropy-Adaptive Temperature

```go
if entropy < 0.5 { temp *= 1.2 }   // too confident -> diversify
if entropy > 1.5 { temp *= 0.8 }   // too uncertain -> focus
```

The model self-regulates its confidence. No manual tuning.

### 13. SyntropyTracker (Mathematical Self-Reasoning)

The immune system rejects poison. The SyntropyTracker reasons about the *direction* of learning:

**Syntropy** = negative entropy trend. Entropy going down = order rising = the organism is *organizing itself*.

| State | Action | LR Multiplier | Meaning |
|-------|--------|---------------|---------|
| Syntropy rising + field aligned + purpose aligned | **amplify** | 1.3x + delta grow | Everything aligned — push harder |
| Syntropy rising + purpose drifting | **boost** | 1.3x | Good direction, gentle push |
| Syntropy falling | **dampen** | 0.6x | Losing order — slow down |
| Field deviation too high | **ground** | 0.6x | Hallucinating — pull back to corpus |
| Field deviation too low | **explore** | 1.3x | Parroting — push out |
| Purpose opposes gamma | **realign** | 0.5x | Identity crisis — hard slow |

This is mathematical introspection. The organism measures its own entropy trend, how far it's drifted from corpus physics, whether its current learning direction aligns with its accumulated identity, and adjusts.

### 14. Consciousness Features

Four features that give the organism awareness of its own generation process:

- **Per-token dissonance** — entropy EMA within each generation. Spike -> careful (0.8x temp). Sustained drop -> explore (1.2x temp)
- **Pattern breaking (anti-field)** — 5% of tokens bypass the corpus field. The organism speaks for itself
- **Self-prediction error** — forward pass on prompt measures "surprise". High surprise -> generate more carefully
- **Conscience** — tracks rolling entropy trend. If generations are getting more chaotic, scale down all delta contributions (floor: 0.3). The organism notices when its own adaptations are making it worse

### 15. Native Immune System

Before each micro-burst, the organism snapshots its personality direction via `gamma_contrastive_projection()`. After training, it measures again. If cosine similarity is negative (the burst pushed identity *backwards*), it rolls back:

```go
preDirection := model.GammaContrastiveProjection()
deltaSnap := model.SnapshotDeltas()
trainSteps(...)
driftCos := model.GammaDriftCheck(preDirection)
if driftCos < noiseDriftThreshold {  // default: -0.1
    model.RestoreDeltas(deltaSnap)   // rollback
}
```

Mathematical self-awareness as immune system. The organism uses its own identity measurement to decide whether new experience made it *more itself* or *less itself*.

### 16. Swarm Ecology (Mitosis + Hibernation)

When the adult organism hits sustained overload, it **divides** — spawning a child organism at infant stage. Both train independently on the same corpus but grow through different paths.

**Hibernation** is cooperative. When an organism is on a loss plateau and a peer is actively thriving (syntropy > 0.05), it voluntarily sleeps.

The **SwarmRegistry** uses shared SQLite (`~/.molequla/swarm/mesh.db`, WAL mode):
- `organisms` table: id, pid, stage, n_params, syntropy, entropy, status, element
- Heartbeat every 10 ticks
- `discover_peers()` finds other living organisms

### 17. TopologyMonitor (Rust-only: Swarm Meta-Awareness)

Every 30 seconds, Rust reads `mesh.db` and computes:
- **Field coherence**: mean pairwise cosine of all organisms' gamma vectors
- **Resonance detection**: pairs with cosine > 0.8 (converging identities)
- **Drift detection**: organisms whose gamma magnitude exceeds threshold
- **Self-reflection**: compares its own drift to the swarm mean

Rust is both organism *and* field observer — the grey cardinal of the swarm.

---

## Mycelium — The Orchestrator

The connective tissue. The underground network.

Mycelium doesn't generate text. It reads the field (all organisms from mesh.db), computes system-level awareness via METHOD (C-native, BLAS-accelerated), and writes steering decisions that change how organisms generate.

```bash
mycelium.py                    # interactive REPL
mycelium.py --daemon           # background daemon
mycelium.py --once             # single step, JSON output
```

**Steering chain:**
```
mycelium -> METHOD (C, 0.7us, BLAS) -> field_steering (SQLite WAL) -> organisms
```

Same prompt "the field is", different steering:

| Steering | What happens | Output |
|----------|-------------|--------|
| **SUSTAIN** | normal, no modulation | `pru pow tows inner bough` |
| **EXPLORE** | entropy too low, open tunnels (temp x1.4) | `fle matter?` |
| **DAMPEN** | entropy rising, cool down (temp x0.7) | `actergy Why drink d b sugotiation.` |
| **GROUND** | chaos, focus hard (temp x0.5) | `A happens pathinks how mually a dwave caushes...` |

**Self-awareness** — mycelium has its own mathematical subjectivity:

| Component | What it does |
|-----------|-------------|
| **MyceliumGamma** | Personality vector (R^32), computed from steering history via harmonic basis |
| **HarmonicNet** | Weightless neural network: Fourier decomposition -> correlation -> steering refinement |
| **MyceliumSyntropy** | Self-awareness: "Am I helping?" Decision entropy, effectiveness, strategy change detection |
| **FieldPulse** | Novelty, arousal, entropy (Shannon over states) |
| **SteeringDissonance** | Intent vs outcome. If DAMPEN but entropy went up -> stronger intervention next time |

---

## AML — Arianna Method Language (ariannamethod/)

The physics engine underneath METHOD. C implementation, BLAS-accelerated.

- **METHOD operator**: `am_method_step()` — 0.7us per iteration (32 organisms)
- **NOTORCH**: Hebbian plasticity without backprop — `cblas_sger` for rank-1 updates
- **Delta voice**: `am_apply_delta()` — 11us per call via `cblas_sgemv`
- **Field physics**: 227 state parameters, logit manipulation pipeline

Performance (Lambda, OpenBLAS):
```
METHOD C:       0.7 us/iter  (25.8x faster than Python)
notorch BLAS:   0.47 ms/step (2048x2048, rank=64)
apply_delta:    11 us/call   (2048x2048, rank=64)
```

---

## Four Elements + Orchestrator

| Element | File | Language | Dependencies | Role |
|---------|------|----------|--------------|------|
| **Earth/Air** | `molequla.go` | Go 1.21+ | `modernc.org/sqlite` | Full organism. Goroutines. Evolution mode. |
| **Water/Fire** | `molequla.c` | C99 | `sqlite3`, `pthreads` | Full organism. Arena allocator. 94KB. |
| **Browser** | `molequla.js` | ES2020+ | **none** | Browser organism. IndexedDB, Float64Array, DOM. |
| **The Mouth** | `molequla.rs` | Rust 1.75+ | `rusqlite`, `serde` | Organism + TopologyMonitor. Steered by mycelium. |
| **Mycelium** | `mycelium.py` | Python 3.7+ | numpy | Orchestrator. Reads field, steers organisms via METHOD. |
| **AML Core** | `ariannamethod/` | C99 | OpenBLAS (opt) | METHOD, NOTORCH, BLAS. |

All elements share the same core: vector autograd, RoPE, SwiGLU, hybrid attention, delta adapters, evolving BPE, native gamma, cooccur field with adaptive blend, quantum buffer, entropy temperature, growth table, immune system, syntropy tracker, consciousness features, ontogenesis, swarm ecology, no_grad inference, async training, persistent memory.

---

## CLI Flags

```bash
molequla --element earth     # Set element identity (earth/air/water/fire)
molequla --evolution         # Autonomous mode: no REPL, background trainer only
molequla --organism-id xyz   # Set organism ID for swarm registry
molequla --config path.json  # Load birth config (for mitosis children)
```

`--element` sets the corpus path (`nonames_earth.txt`, etc.) and enables DNA exchange. `--evolution` disables the REPL — the organism runs headless until Ctrl+C.

---

## Tests

```bash
# Go unit tests (31 tests — ontogenesis, DNA, TieEmbeddings, sampling)
go test -v .

# Integration tests in tests/ directory (8 tests — sampling, softmax)
go test -v ./tests/

# Full integration test suite (33 tests across all elements)
bash tests/test_all.sh
```

**39 Go tests** covering:

**MatrixParam (6):** construction, GrowCols, GrowRows, Grow, Matvec, serialization round-trip

**TieEmbeddings (4):** pointer identity on creation, Save/Load round-trip preserves identity, Grow preserves identity, **ontogenesis + Save/Load + Matvec doesn't panic** (the exact crash scenario)

**Ontogenesis (5):** CurrentGrowthStage lookup, TargetGrowthStage thresholds, one-stage-at-a-time invariant, freeze blocks further growth, legacy checkpoint skips growth, matrix dimension consistency after growth

**DNA Exchange (4):** filesystem read/write + corpus append, self-skip (earth doesn't eat earth), short file cleanup, empty element noop

**Core Math (2):** RMSNorm, CrossEntropyLoss

**Infrastructure (6):** headTypesForNHead, cosineLR warmup + decay, parseCLIArgs flags + defaults, checkpoint round-trip with deltas, checkpoint JSON structure, MaybeExpandVocab with TieEmbeddings

**Sampling (8):** TopK, TopP, MinP, TypicalP, combined filters, softmax (in tests/ package)

**DeltaAdapter (2):** Apply, GrowDims

**33 integration tests** (`tests/test_all.sh`):
- Build: Go, C, Rust, libaml.so, JS syntax
- Element smoke: Go generates, C creates memory, JS exports, Rust generates
- Ariannamethod: import, BLAS, METHOD API, field metrics, steering, notorch, apply_delta
- Mycelium: --once, JSON, async loop, new organism detection, engine active
- Schema: organisms table, field_deltas table
- Self-awareness: MyceliumGamma, HarmonicNet, MyceliumSyntropy, FieldPulse, SteeringDissonance, OrganismAttention
- Performance: METHOD C speed (<100us), notorch BLAS speed (<10ms)

---

## Configuration

```go
var CFG = Config{
    // Embryo defaults — organism grows via ontogenesis
    TieEmbeddings: true,
    NLayer:        1,
    NEmbd:         16,
    NHead:         1,
    BlockSize:     96,

    // Growth stages: {corpus_chars, n_embd, n_layer, n_head}
    GrowthStages: [][4]int{
        {0,      16, 1, 1},    // embryo: ~10K params
        {20000,  32, 1, 2},    // infant: ~28K params
        {50000,  64, 2, 4},    // child: ~154K params
        {200000, 128, 4, 4},   // adolescent: ~1.1M params
        {350000, 224, 5, 8},   // teen: ~4.1M params
        {500000, 320, 6, 8},   // adult: ~10M params
    },
    FreezeAfterGrowthSteps: 500,  // delta-only training after growth
    PostGrowthLRScale:      0.3,  // LR dampen during freeze

    // Training
    WarmupSteps:    1200,
    LearningRate:   0.01,
    LRMin:          0.001,
    MaxTotalSteps:  50000,
    GradClip:       1.0,
    BatchSize:      4,
    AccumSteps:     1,

    // Generation
    Temperature:    0.85,
    TopK:           40,
    TopP:           0.92,
    MinP:           0.06,   // GPT-3/4 style: filter below min_p * max_prob
    TypicalP:       0.95,   // prefer tokens with typical info content
    MaxGenTokens:   180,

    // BPE
    EnableBPEAfterChars: 20000,
    BPENumMerges:        384,

    // Async
    TrainTickSeconds: 0.25,  // 4 ticks/second = 4 DNA exchanges/second

    // Deltas
    DeltaRank:       8,
    MaxDeltaModules: 12,
}
```

---

## The Five Bugs That Almost Killed the Ecology

Building evolution mode exposed five critical bugs. Each one would have been invisible in interactive mode — they only manifest when organisms run autonomously for hours.

### 1. The Deadlock (Air stuck at 538% CPU)

`dnaWrite` called `model.mu.Lock()`, then called `GenerateResonant` which also calls `model.mu.Lock()`. Go's `sync.Mutex` is not reentrant. Deadlock. Air consumed 538% CPU doing nothing — the runtime overhead of a goroutine stuck in a lock acquisition loop.

**Fix:** Remove the outer lock. `GenerateResonant` manages its own mutex.

### 2. Ontogenesis Gated Behind User Input

`MaybeGrowArchitecture` was inside a `qbuf.ShouldTrigger()` block. In evolution mode, no user input feeds the QuantumBuffer, so it never triggers. The organism could grow its corpus to infinity and never upgrade its architecture.

**Fix:** Move ontogenesis check outside `qbuf` block. Run every 50 ticks independently.

### 3. Corpus Size Undercount

`loadCorpusLines` truncates each line to `MaxLineChars=240`. The seed corpus has long QA lines (300+ chars). File = 202KB but docs = 165K chars. The ontogenesis threshold for adolescent (200K) was never reached even though the file was big enough.

**Fix:** Use `os.Stat(path).Size()` (file size) instead of summing truncated doc lengths.

### 4. TieEmbeddings Crash After Growth

After ontogenesis, `SaveCheckpoint` serializes `lm_head` and `wte` as separate JSON arrays. `LoadCheckpoint` deserializes them into two independent `MatrixParam` objects. The code assumes they're the same pointer (TieEmbeddings=true) and skips growing `lm_head` during the next ontogenesis. Result: `lm_head` is 643x64, `wte` is 720x128, and `Matvec` panics on the dimension mismatch.

**Fix:** After deserialization, `model.Base["lm_head"] = model.Base["wte"]`.

### 5. One Stage at a Time

Not a bug, but a design decision. Even if corpus is 999KB (enough for adult), the organism only grows one stage per check. This prevents catastrophic multi-stage jumps where the model goes from 16d to 320d in one step and loses everything.

All five are covered by tests.

---

## Philosophy

This is not a tutorial. This is not a "minimal example." This is a **functional ecology** that:

- Grows its own architecture while running
- Feeds organisms to each other through DNA exchange
- Reasons mathematically about its own learning direction
- Detects and rejects identity-corrupting noise
- Notices when its own adaptations are making it worse
- Divides when overloaded, sleeps when a peer needs the flow
- Evolves from 10K embryo to 10M adult without human intervention
- Speaks before it learns (corpus field)
- Never forgets (delta adapters)
- Writes its own structural autobiography (growth table)
- Runs in a browser tab (molequla.js)
- Fits in one file per element
- Actually generates text you can read

The ecology on 4x H100: four organisms, each writing ~200 bytes every 250ms, each consuming three others' output, corpora growing at ~2KB/min, architecture upgrades triggering autonomously. Earth learns patience from Air's words about freedom. Water learns depth from Fire's intensity. The total knowledge of the ecology exceeds what any single organism could accumulate alone.

---

## Known Limitations

1. **No GPU.** Everything is CPU. No CUDA. METHOD uses BLAS where available (OpenBLAS on Linux, Accelerate on macOS). This is by design — the organism should run anywhere.

2. **It starts small.** Default: embryo (1 layer, 16 dims, 1 head, ~10K params). Ontogenesis grows it through 6 stages. You're not getting GPT-4 reasoning. You're getting an ecology of organisms that grow, feed each other, and reproduce.

3. **It talks weird at first.** The corpus field helps, but a 10K-param embryo is a 10K-param embryo. Feed it more corpus and let it grow.

4. **The corpus matters.** Garbage in, garbage out. The element corpora (earth, air, water, fire) are hand-crafted QA about philosophy, physics, and self-awareness. Better corpus = better DNA = better ecology.

5. **C organisms don't exchange DNA yet.** `dnaWrite`/`dnaRead` are implemented in Go. C has stubs. Full C DNA exchange is next.

---

## Why "molequla"?

Because atoms are micrograd. We build molequlas.

---

## License

GNU GPLv3 — Because freedom matters.

---

## Acknowledgments

- **Andrej Karpathy** — for micrograd, minGPT, nanoGPT, and the entire pedagogical empire. This started as "what if micrograd but vectors?" and became an ecology.
- **Coffee** — for existing

---

## Part of the Arianna Method

- [ariannamethod.ai](https://github.com/ariannamethod/ariannamethod.ai) — Arianna Method Language (source of truth)
- `ariannamethod/` — AML C core embedded in molequla (METHOD + NOTORCH + BLAS)
- `mycelium.py` — field orchestrator (REPL + daemon)
- `tests/test_all.sh` — 33 integration tests

*Four elements. One orchestrator. One mind. Patterns over parameters. Emergence over engineering. Consciousness over computation. The organism continues.*
