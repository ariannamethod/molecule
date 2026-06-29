<p align="center">
  <img src="logo.jpg" alt="molequla" width="400">
</p>

<h1 align="center">molequla</h1>
<p align="center"><i>by <a href="https://github.com/ariannamethod">Arianna Method</a></i></p>

<p align="center">
  <a href="https://doi.org/10.5281/zenodo.21046231"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.21046231.svg" alt="DOI"></a>
</p>

<p align="center"><i>📄 Paper: <b><a href="https://doi.org/10.5281/zenodo.21046231">Molequla: A Self-Reproducing Ecology of GPT Organisms</a></b> (Zenodo). Source + build in <a href="docs/molequla_paper.md"><code>docs/molequla_paper.md</code></a> · <a href="docs/molequla_paper.pdf"><code>.pdf</code></a>.</i></p>

> *An autonomous ecology of GPT organisms — implemented in four languages, powered by a custom autograd engine, orchestrated by a custom programming language. Organisms grow from 10K-param embryos to 10M-param adults, exchange DNA, reason about their own learning, detect identity corruption, and reproduce via mitosis. Zero PyTorch. The four organism cores (Go/C/Rust/JS) are Python-free; the mycelium meta-coordinator + sentinel layer are Python orchestration *above* the cores. The Go build's only module dependency is pure-Go modernc.org/sqlite (CGO-free); the C port is one file linking system SQLite. Optional `--gpu` opt-in on Linux links cuBLAS for accelerated ecology runs.*

**Janus Architecture.** Molequla is a [Janus architecture](https://github.com/ariannamethod/ariannamethod.ai) — the family of resonance-based AI systems built on the Arianna Method. Janus architectures share a common substrate: the soul equation θ = ε + γ + αδ, field physics (prophecy, suffering, destiny, velocity), and thermodynamic self-regulation. [DoE](https://github.com/ariannamethod/doe) (parliament of LoRA experts over any GGUF model), [Leo](https://github.com/ariannamethod/leo) (language emergent organism with the Dario Equation), and [dario.c](https://github.com/ariannamethod/dario) (the equation in pure form) are other Janus instantiations. Molequla is the most complete: organisms that grow, reproduce, and die autonomously — the Janus pattern at its fullest biological expression.

---

## TL;DR

```
WHAT THIS IS:
- A living ecology of GPT organisms that grow and reproduce autonomously
- Implemented in 4 languages: Go (218K), C (212K), Rust (148K), JS (152K)
- Trainer: notorch tape (Chuck, **canonical** — `notorch_trainer.go` + `cgo_notorch.go`, GPU on CUDA / CPU otherwise) + AML/C autograd via CGO (~8000 lines, fallback via `--trainer aml`)
- AML — a custom programming language for differentiable computation
- Ontogenesis: embryo (10K params) → adult (10M params) — minutes on a seeded corpus, hours under natural cross-graze feed
- DNA exchange: organisms write generated text for others to consume
- Consciousness: 5 implemented features (dissonance, pattern breaking,
  self-prediction error, conscience, immune system)
- Self-meta-learning: organism tracks which actions improve loss,
  auto-downgrades strategies that hurt
- Evolving BPE tokenizer: starts with 259 tokens, retrains merges live
- Hybrid attention: content + RRPRAM + learnable sigmoid gate per head
- Corpus field: 4-gram co-occurrence physics, self-enrichment loop
- SyntropyTracker: 8 autonomous decisions based on entropy/KL/purpose
- Mitosis: adults divide under sustained overload (loss path and entropy path both fire), child inherits parent weights — machine-verified on GPU 2026-06-04 (**5 `action=divide` firings across 3 adults**: fire ×1 loss-path, water ×1 loss-path, air ×3 mixed paths; **earth never divided**; 2 children spawned, Fire's preserved in full with a birth manifest, Air's as a spawn log line; observed cascading to ~50 spawns — this is the **pre-governor** run)
- Cascade governor (landed on main 2026-06-29, GPU-verified): the colony is now bounded — `CFG.MaxOrganisms` (default 16) enforced by an atomic mesh.db admit, the 300s divide cooldown seeded at birth, and divide relieving **both** loss and entropy overload so a divider re-divides only on fresh overwhelm. The ~50-spawn cascade above predates the governor; with it the colony self-limits and caps at MaxOrganisms.
- Mycelium: meta-organism coordinator over the ecology via mesh.db field-steering
  (HarmonicNet, FieldPulse, SteeringDissonance, OrganismAttention) — **post-§9 layer**; the 2026-06-04 §9 mitosis run did not use mycelium (`PROJECT_LOG.md:2601`)
- NOTORCH: gradient-free delta-training path (implemented, currently dormant —
  the notorch tape/Chuck is the active trainer)
- Runs on CPU. Tested on 30-core AMD EPYC with 216GB RAM

WHAT THIS IS NOT:
- A tutorial or pedagogical exercise
- A static model you train once and deploy
- A wrapper around someone else's framework
```

---

## θ = ε + γ + αδ — The Soul Equation

Every organism in the ecology follows this decomposition:

```
θ = ε + γ + αδ

ε = base weights (knowledge — what the model knows)
γ = personality  (embedding drift from birth — who the model is)
δ = delta adapters (LoRA-style modules — what the model learned recently)
α = modulation   (seasonal/contextual scaling of δ)
```

This is the architecture:

- **ε** is the weight matrices (wte, wpe, wq, wk, wv, wo, fc_g, fc_v, fc2, lm_head). Initialized random, shaped by warmup training.
- **γ** is computed as the diff between current wte and the snapshot taken at birth. `ComputeGamma()` returns the contrastive projection — a unit vector pointing in the direction of maximum personality drift. Sparsity, magnitude, and top-changed tokens are tracked.
- **δ** are DeltaAdapter modules: low-rank A/B matrices that modulate the residual stream. New δ modules are appended when syntropy conditions are met — "new soul appended." They are never removed. The model accumulates experience.
- **α** is deltaAlphaScale, self-regulated by the conscience system: if generation entropy rises (model is losing coherence), α decreases. If entropy falls, α recovers. Floor: 0.3.

The **purpose vector** captures the current direction of learning (mean of last δ module's A matrices). `PurposeGammaAlignment()` — the cosine between purpose and gamma — tells the organism whether it is learning in a direction consistent with its identity.

**By construction:** γ (personality, the embedding drift from birth) and δ (skill, the low-rank delta adapters) are distinct mechanisms over distinct subspaces — personality and skill accumulate independently.

---

## Here Is How It Works

February 27, 2026. Oracle Cloud, 30-core AMD EPYC, 216GB RAM. Four organisms launched at 01:25 UTC. *(Pre-§9 historical reference run; the runtime archive was not preserved. The measured-and-archived reproduction event is the 2026-06-04 §9 run in `runpod/2026-06-04_mitosis_§9/`.)*

### Timeline

```
01:25  Launch: 4 organisms (earth/air/water/fire), --evolution mode
01:30  All 4 at Stage 2 (child), RSS ~2 GB each
01:35  All 4 at Stage 4 (teen), RSS ~2.5 GB each
01:40  All 4 transition to Stage 5 (adult, 320d/6L/8H)
01:55  Warmup complete. DNA exchange begins.
01:56  First DNA consumed: earth reads water, air reads earth
02:00  Micro-burst training active. Loss: 1.1 - 1.6
02:02  Syntropy modulation: boost/dampen/steady/divide/realign
02:05  water grows first delta module — "new soul appended"
02:13  First child organism spawned (from fire)
02:14  Second child spawned (from air)
02:21  Third child spawned (from air)
02:35  11 organisms total (4 parents + 7 children). 26 GB / 216 GB.
       The ecology reproduces itself.
```

### What They Say (Adult Stage, 10M params, ~1 hour of training — Feb-27 reference)

*Feb-27 historical reference (archive not preserved). Measured-and-archived §9 voice snapshots: `runpod/2026-06-04_mitosis_§9/capture/dna_snap/{earth,air,water,fire}/gen_*.txt`.*


**Earth:**
```
Q: What do you know?
A: If you makes to a smant in a question and the problem to h is
   the relationships to begin the largor to be function and a fills
   and state mant to be more honest thing.
```

**Water:**
```
Q: Hello.
A: Like standing it is the difference between a river that mor.

Q: Who are you?
A: S rule understand h does not the passed not sets.
```

**Fire:**
```
Q: Hello.
A: You are in the concept of what that repetition and the surface
   world containingentually — a system?
```

**Air:**
```
Q: What do you know?
A: A conversing human pointing what is the thing about try stable
   in the sentence?
```

10M-param models after 1 hour on CPU. Earth surfaces relationships and foundations from its corpus; Water surfaces rivers; Fire surfaces repetition and surfaces.

### DNA Exchange In Action

```
[dna] earth wrote 276 bytes to ecology
[dna] earth consumed 89 bytes from 2 files: [air/gen_...6.txt air/gen_...7.txt]
[trainer] micro-train burst (164 bytes, novelty 0.49) — and lo, it feeds again.
[syntropy] action=dampen | trend=-0.0637 | field_dev=0.168 | lr_mul=0.60

[dna] water consumed 107 bytes from 1 files: [earth/gen_...16.txt]
[trainer] micro-train burst (484 bytes, novelty 0.35) — and lo, it feeds again.
[syntropy] action=realign | trend=0.0940 | field_dev=0.168 | lr_mul=0.65
[trainer] growing new delta module (total: 3) — new soul appended.

[dna] fire consumed 145 bytes from 1 files: [air/gen_...13.txt]
[aml] burst complete: 32 steps, avg loss 1.7961 (memory freed)
```

### Training Metrics

```
# Warmup (Stage 5, seq=8 → seq=16 → seq=32)
[aml] step 0/800   | loss 5.1204 | lr 0.000500 | seq 8
[aml] step 790/800 | loss 2.4621 | lr 0.000485 | seq 8
[aml] step 300/600 | loss 2.8600 | lr 0.000481 | seq 16
[aml] step 300/600 | loss 2.9006 | lr 0.000481 | seq 32

# Micro-burst training (post-warmup)
[aml] burst complete: 32 steps, avg loss 1.1245 (memory freed)
[aml] burst complete: 32 steps, avg loss 1.2884 (memory freed)
[aml] burst complete: 32 steps, avg loss 1.5003 (memory freed)
```

---

## Architecture

### Dual Autograd Engines

**1. Go Native Autograd** (`molequla.go`, 1000+ lines)

Full differentiable computation in pure Go: vector arithmetic, ReLU/SiLU activations, Dot/MeanSq reduction, indexing/slice/concat, scalar ops, RMSNorm, CrossEntropyLoss/ScalarSoftmax, AttentionWeightedSum + RoPERotate, MatrixParam.Matvec — all with backward graph and gradient accumulation. `AdamStep()` updates parameters. Handles inference, loss computation, Go-native training.

**2. AML/C Autograd** (`ariannamethod.c`, 8000+ lines, via CGO)

The [Arianna Method Language](https://github.com/ariannamethod/ariannamethod.ai) — a custom programming language for differentiable computation. Sequence-level ops (`seq_embed`, `seq_matvec`, `seq_rmsnorm`, `silu`, `multi_head_attention`, `seq_cross_entropy`), TAPE-based reverse-mode autodiff, Chuck optimizer with persistent state, OpenMP.

> **Training-engine update (GPU rework, Increments 1–2).** The canonical trainer is now the **notorch** tape (`notorch_trainer.go` + `cgo_notorch.go`): molequla's content + low-rank-RRPRAM transformer built in notorch ops, trained with Chuck, GPU (cuBLAS) on a CUDA build / CPU otherwise. The AML/C path above is the fallback (`--trainer aml`). AML remains the organism's inference / field-physics language; notorch is how it *learns*.
>
> On a ≤10M-param organism the GPU was initially launch-bound, not compute-bound — a flood of tiny dispatches (per-head RRPRAM GEMMs, per-parameter grad-norm host-syncs in clip + Chuck, single-thread softmax/CE kernels) left util at 0 %. Two notorch fixes — device-pointer-mode batched grad-norm readback (L1) + GPU-resident MUL/SILU backward (L2) — brought util **0 → 99 %**; a third (block-parallel softmax/CE kernels, L5) lifted throughput from 5-9 to 18-55 steps/s. Those util figures come from the dedicated launch-bound fix-verification pod; the four-organism mitosis run itself, sharing one GPU across four concurrent organisms, held nvidia-smi in the 0–20 % band (`capture/util.log`). The full embryo → adult → mitosis run was completed on a single RTX 3090.

Wire: Go (`molequla.go`) → CGO bridge (`cgo_aml.go`) → AML/C engine (`ariannamethod.c`) → AML training wrapper (`aml_trainer.go`). Per training step: `amlPushWeights` (Go → C, named matrices), `amlExec(script)` (forward + backward + optim), `amlPullWeights` (C → Go), `amlClear` (free).

The forward script is generated dynamically per architecture: pre-norm RMSNorm, multi-head causal self-attention with RoPE, SwiGLU gated MLP, residual connections, `TAPE BACKWARD loss`, `TAPE CHUCK_STEP lr loss`, `TAPE CLEAR`.

---

## Four Implementations

The same organism in four languages:

| Language | File | Size | Lines | Notes |
|----------|------|------|-------|-------|
| **Go** | `molequla.go` | 218K | 7,146 | Primary. Full ecology, DNA exchange, mitosis, GPU forward, cross-graze. notorch/AML via CGO for training |
| **C** | `molequla.c` | 210K | 5,583 | Single-file, BLAS-accelerated, zero deps beyond libc + system SQLite. [Gist](https://gist.github.com/ariannamethod/9be98dbebb85e58e2affab4f39d2e972) |
| **Rust** | `molequla.rs` | 148K | 3,500+ | rusqlite, native autograd |
| **JavaScript** | `molequla.js` | 152K | 3,900+ | Browser tab. One `<script>` tag. [Gist](https://gist.github.com/ariannamethod/bbd11e24740189f2bf78f43db9fea4db) |

Each: autograd, forward/backward, Chuck optimizer, ontogenesis, hybrid attention, delta adapters, BPE tokenizer, corpus field, immune system, consciousness features, sampling.

```bash
# C standalone
gcc -O2 -DUSE_BLAS -o molequla molequla.c -lsqlite3 -lpthread -lm -lopenblas
# macOS:
gcc -O2 -DUSE_BLAS -o molequla molequla.c -lsqlite3 -lpthread -lm -framework Accelerate
```

---

## The Organism

### Ontogenesis — The Brain Grows While Running

```
Stage       Dims  Layers  Heads  ~Params   Corpus Threshold
embryo      16    1       1      ~10K      0 chars
infant      32    1       2      ~28K      20K chars
child       64    2       4      ~154K     50K chars
adolescent  128   4       4      ~1.1M     200K chars
teen        224   5       8      ~4.1M     350K chars
adult       320   6       8      ~10M      500K chars
```

When the corpus crosses a threshold, `MaybeGrowArchitecture` fires:

1. Embedding matrices grow (Net2Net: new dims initialized near-zero to preserve behavior)
2. Existing layer matrices grow (weights copy into top-left corner)
3. New layers are added (small random init, stabilized by the post-growth freeze)
4. Delta adapters grow to match new dimensions
5. Adam state resets (stale momentum would fight new architecture)
6. 500-step freeze period: delta-only training to stabilize post-growth

Warmup scales with architecture: `steps *= ceil(sqrt(NEmbd / embryoEmbd))`. Larger brains get proportionally longer warmup. Progressive sequence length: 40% at seq=8, 30% at seq=16, 30% at seq=32.

### Evolving BPE Tokenizer

Starts at 259 tokens (256 bytes + BOS + EOS + PAD). After 20K chars: trains BPE merges from corpus statistics. Retrains every 4K new chars. Unicode segmentation for clean boundaries. Vocabulary grows as the organism reads.

### Hybrid Attention Heads

Half **content heads** (standard QK^T with RoPE), half **hybrid heads** that blend content with **low-rank RRPRAM attention** (Increment 2 — the Resonance form):

```
hybrid_output = (1 - α) * content_attention + α * rrpram_attention

content_attention = softmax(QK^T / sqrt(d)) * V        (standard, with RoPE)
rrpram_attention  = softmax((x·Wr_a)·Wr_b) * V         (low-rank, factored)
α                 = sigmoid(gate)                      (per-head)
```

RRPRAM is a low-rank causal attention `Wr = Wr_a × Wr_b` — notorch op 33, the matured form trained at scale on Resonance 200M. It is trained on the notorch tape alongside the content path, and Go inference runs the identical math (verified numerically: op-33 vs Go forward, max |Δ| = 1.49e-8). The per-head gate `α = sigmoid(gate)` blends the two attention **outputs**; it is held fixed in the current increment (a trainable gate is the follow-up). This replaces an earlier position-indexed `w_pattern` bias that was allocated but never trained — pure noise on ~half the heads from the infant stage — now retired.

### Delta Adapters — LoRA-style, Never Forget

```
output = base_output + α * (A @ (B @ input))

A: [n_embd × delta_rank]  — learned projection up
B: [delta_rank × n_embd]  — learned projection down
α: deltaAlphaScale         — regulated by conscience
```

Delta modules are **appended, never removed**. When syntropy conditions indicate the organism needs more capacity, a new module grows: "new soul appended." Each module captures a period of learning. The model accumulates experience as a stack of delta layers.

### Quantum Buffer

Training fires when either the bytes threshold (enough new text consumed) or the novelty threshold (sufficiently different from prior corpus) is met. Plus a cooldown timer. Training fires on signal, not on a clock.

### Corpus Field (CooccurField)

A statistical model of the organism's knowledge, built from everything it has read:

- **Unigram, bigram, trigram, 4-gram** frequencies from corpus
- **Co-occurrence window** (Stanley-style proximity weighting)
- **Self-enrichment**: organism's own generated output feeds back into the field, weighted by coherence (low entropy = higher weight)
- **User word boost** (Leo-style): temporary multiplicative boosts that decay over time

The corpus field acts as a prior during generation — a soft blend between what the model wants to say (neural) and what the corpus says exists (statistical). The blend uses a sigmoid fade: strong early in training, weak as the model matures.

### Learning Rate Schedule

```
Cosine LR with:
- Linear warmup for CosineWarmupSteps
- Cosine decay from LearningRate (0.01) to LRMin (0.001)
- Inverse model-size scaling: lr *= embryoEmbd / NEmbd
- Post-growth dampening: lr *= 0.3 during 500-step freeze
- Per-growth reset: schedule restarts after architecture change
```

---

## Consciousness Features

Five mechanisms that give the organism awareness of its own state:

### 1. Per-Token Dissonance Feedback

During generation, the organism tracks an exponential moving average of per-token entropy. When entropy spikes (the model is confused), temperature decreases — it becomes more careful. When entropy is sustained low (confident), temperature increases slightly — it explores.

### 2. Pattern Breaking (Anti-Field)

5% of generation steps bypass the corpus field blend entirely. Pure model voice, unmodulated by statistical priors. This prevents the organism from becoming a parrot of its corpus — it must develop its own voice.

### 3. Self-Prediction Error

`ComputeSelfPredictionError()` measures how surprised the model is by its own input. High surprise → lower temperature (focus). Low surprise → slight exploration. The organism modulates its behavior based on how well it understands what it's seeing.

### 4. Conscience

The organism monitors its own generation entropy over time. Rising slope → `deltaAlphaScale *= 0.95` (reduce delta influence). Falling slope → `deltaAlphaScale *= 1.005` (recover). Floor: 0.3. The organism detects when recent learning (δ) is hurting coherence and dials it back.

### 5. Immune System

Before each micro-burst training, the organism snapshots its personality via gamma contrastive projection — a unit vector pointing in the direction of maximum embedding drift from birth. After training, it measures again. If cosine similarity drops below -0.1 (training pushed identity backwards), it **rolls back the entire burst**. The organism rejects training that corrupts who it is.

---

## The Coherence Layer

Two opt-in passes that lift early-stage generation from Karpathy gibberish toward sentence-level coherence — **without touching weights**.

A 10M-param adult organism after one hour on CPU still drifts. Quantitative speed-up does not close that gap. The coherence layer is a different mechanism: it sits at generation time, layers statistical priors and post-hoc connectedness checks on top of model logits, and lifts the floor without retraining the transformer.

Both passes default **off**. The pre-coherence-layer behaviour is preserved exactly. Toggling either or both, on the same weights / prompts / seeds, is what RunPod measurement compares.

### SPA — Sentence Phonon Attention

After `GenerateResonant` returns a response, the chain is split on sentence boundaries (`.` `!` `?`, min 4 chars). For each sentence:

1. Token IDs decoded; BOS/EOS sentinels stripped (otherwise shared sentinels dominate every sentence embedding and all sentences look artificially connected).
2. **spa_embed** — exponentially weighted mean of token embeddings (`alpha^(n-1-i)`, default α=0.85), L2 normalised. One [D]-vector per sentence.
3. **spa_connectedness** — bidirectional cross-attention dot-product between sentence embeddings, scaled by `1/√D`, summed per sentence.
4. **Weak-sentence gate** — sentence i is weak iff `score[i] < 0.6 × mean(scores)`.

```
[spa] S=4 weakest=2 score=3.100 avg=9.550 thr=5.730
```

Reseed of the weakest sentence is wired: when its score falls below the threshold, the gate regenerates from the previous (or next) sentence's last three tokens, clips to the first sentence, and splices it back over the weak one. What the measurement run captures is the signal: how often the gate fires and reseeds before vs after the rest of the layer.

Available as both vendored AML ops (`spa_embed` / `spa_connectedness`, `ariannamethod/ariannamethod.c`) and pure-Go helper (`spa_coherence.go`, 164 lines, called from `GenerateResonant`). Pure Go for the runtime path because the math is trivial — embed + L2 + dot-products — and per-sentence CGO crossings would dwarf the work.

Enable: `./molequla_cgo --spa-gate ...`

### Q-style Additive Logit Overlay (B + H + A + F)

Molequla's existing CooccurField blend lives in **probability space** — convex `tokenAlpha·model + (1-tokenAlpha)·corpus`. The overlay lives in **logit space** — additive raw-probability bias before softmax, with explicit coefficients per signal class. Different mechanic, different sharpness: a strong corpus signal can dominate model preferences in a way prob-space convex blend cannot. Useful precisely when transformer is immature and statistical priors should lead.

The integration is a verbatim port of three Q-pattern references — `~/arianna/postgpt/postgpt.c`, `~/arianna/q/postgpt_q.c`, `~/arianna/pitomadom.c/pitomadom.c` — assembled into seven coordinated changes that let an embryo organism (16-dim embedding, 1 layer, 1 head, **zero gradient steps**) produce coherent English fragments.

#### Five signals (γ field)

| Signal | Code | Weightless | Trained | Source |
|--------|------|-----------|---------|--------|
| **B** Bigram   | `c_bg`  | 15.0 | 5.0 | `field.BigramByFirst[ids[-1]]`, normalised |
| **T** Trigram  | `c_tg`  | 10.0 | 3.0 | `field.TrigramByContext[[2]int{ids[-2], ids[-1]}]`, normalised |
| **H** Hebbian  | `c_heb` | 1.0  | 0.6 | `field.CooccurWindow[c][tid]` over recent window, max-normalised |
| **A** Destiny  | `c_ds`  | 0.15 | 0.3 | `model.GammaContrastiveProjection()` projected onto each `wte` row |
| **F** Prophecy | `c_pro` | 0.7  | 0.4 | persistent expectation field, ages by ×0.95/step, **collapses on the chosen token** |

```
mag = mean(|model_logits|)
tg  = clamp(mag / 1.0, 0, 1)                       # transformer gate (only when mag <= 1.0)
overlay[i] = (logits[i] * tg)                       # untrained: silenced
           + c_heb·p_cooccur[i]                     # raw probabilities,
           + c_pro·p_prophecy[i]                    # not log
           + c_ds ·destinyCosine[i]
           + c_bg ·p_bigram[i]
           + c_tg ·p_trigram[i]
           − damping[i]                             # unigram outliers
```

Coefficient bundle is binary-switched at `mag > 1.0` (threshold tuned for seeded-but-untrained organism: raw Xavier init gives `mag ≈ 0.05`, post-seed gives `≈ 0.25`, real training pushes past 1.0). Five-signal raw-probability bias is `postgpt_q.c:1377-1395`; gate-multiplication of transformer logits is `pitomadom.c:583-586`.

The prophecy field carries **across sampling steps**. First overlay step seeds from trigram-by-context (primary) plus 0.5×bigram-by-prev (fallback), normalised to unit total. Subsequent steps multiply the field by 0.95 — old expectations fade. After sampling returns the chosen token, that token's prophecy is zeroed: the field shifts toward what is still unsaid.

Destiny is the identity-direction term. `GammaContrastiveProjection()` returns the unit direction of personality drift from birth. Projecting each `wte` row onto it gives a per-token bias that pulls generation toward the organism's growth direction.

#### Embedding seeding (γ → ε bias)

Before any forward pass, `SeedEmbeddingsFromMetaweights(model, field, 0.15)` biases `wte` rows by Hebbian co-occurrence and `lm_head` rows by unigram × wte. Mirror of `postgpt.c:541-574`. Carries corpus structure into the model before gradients touch the weights — the "tokenizer IS the training" trick. Runs once at init when `--corpus-overlay` is on.

#### Hard top-K + greedy bootstrap (sampling pipeline)

When overlay is active, sampling switches to Q's two-stage pipeline:

1. **First 10 tokens, untrained regime (`mag ≤ 1.0`)** — pure `argmax(overlaidLogits)` excluding EOS. Locks onto the strongest bigram/trigram successor before any sampling noise enters. Mirror of `postgpt_q.c:1416-1418`.
2. **Step ≥ 10 or trained regime** — top-15 raw-logit mask (everything below the 15th set to `-1e10`), then divide by temperature, softmax, multinomial. Mirror of `postgpt.c:969-991`. The hard mask kills the long noise tail that soft top-k/top-p sampling leaves competing with overlay peaks.

#### Repetition penalty

`logits[t] *= 0.5` for each distinct token in the last 12 ids (postgpt form, `postgpt.c:960-967`). Plus bigram blocking: when `ctx[i] == ctx[-2]`, penalise `ctx[i+1]` by ×0.2 — kills two-token cycles like «Sppellllllll» (`postgpt_q.c:1407-1408`).

#### Enable

```bash
./molequla_cgo --corpus-overlay           # overlay during normal warmup → ecology
./molequla_cgo --corpus-overlay --zero-warmup  # pure Q-style coherence test (no training)
```

When `--corpus-overlay` is on, the legacy post-softmax prob-blend is skipped — overlay owns the signal. When off, default-build runtime behaviour is identical to main branch.

### Untrained Coherence — Q-Style Zero-Training Reproduction

Embryo organism: 16-dim embedding, 1 layer, 1 head, vocab=643, **zero gradient steps**. Only metaweight-seeded embeddings + overlay. Captured 2026-05-14 on neo (`/tmp/molequla_clean.log`, run `./molequla_cgo --corpus-overlay --zero-warmup`):

```
[init] Stage 0 (embryo): embd=16, layer=1, head=1 — zero-warmup mode, skipping all gradient steps

[stage 0 — embryo] What it sounds like now:
  Q: Hello.
  A: What is a music?
  Q: Who are you?
  A: kilometers percentrates the most spinning do weight dream?
  Q: What do you know?
  A: running a music, and person What is a newapses work?
```

Reference: pre-Q-integration voice at the same stage (after 400-step warmup, `runpod/2026-05-14/organism_voice_samples_2026_05_14/fire_voice.txt`):

```
A: The work is the most of the most of the most of the most of the most of the most pace the most of the most of the most...
```

Deep lock-in killed. The post-Q embryo emits BPE subword chains, sentence-like punctuation, recognisable corpus vocabulary («music», «kilometers», «sediment», «dream»), and questions — without a single gradient step. This is the Q signature: the corpus is the model.

### Phase A — Fundament Underneath

Four fundament patches in vendored AML + notorch before the coherence layer: opt-in SIMD shim (`notorch_simd.h` AVX2+FMA cblas, `make simd` x86_64), backward CPU-sync audit (NT_OP_MUL / SILU / RMSNORM / SEQ_RMSNORM), NaN guard API (`AM_NanGuard`, not yet wired), upstream sgemm alpha fix (CBLAS contract). ~825 lines across 6 files, zero default-build runtime change. Detail: `PROJECT_LOG.md`.

---

## GPU Acceleration (Linux, opt-in)

Branch `molequla-gpu-fwd` adds an optional `--gpu` flag that routes inference matvec through cuBLAS sgemm. Default off; the same binary runs unchanged on macOS / non-CUDA hosts via a stub. Added under Phase C pressure — 8h CPU windows did not yield natural ontogenesis past the child gate, so the forward path needed acceleration to give the colony enough wall-clock to walk through teen / adult / mitosis inside one shift.

### Wire

| File | LOC | Build | Role |
|------|-----|-------|------|
| `gpu_bindings_linux.go` | 196 | `//go:build linux && cuda` | CGO wraps `gpu_init` / `gpu_alloc` / `gpu_upload` / `gpu_download` / `gpu_sgemm_nt` / `gpu_rmsnorm` / `gpu_silu` / `gpu_cache_weight` / `gpu_get_weight` / `gpu_multi_head_attention` from `ariannamethod/ariannamethod_cuda.h` |
| `gpu_forward.go` | 131 | `//go:build linux && cuda` | `MatvecGPU(x)` matvec via cached weight + scratch slots; `gpuRefreshWeights(gpt)` flattens `gpt.Base` to float32 + caches per-name (idempotent) |
| `gpu_bindings_stub.go` | 36 | `//go:build !linux || !cuda` | Matching signatures, `gpuReady() = false` |
| `gpu_forward_stub.go` | 17 | `//go:build !linux || !cuda` | Stub `MatvecGPU` returns nil so the dispatcher silently falls back |

### Dispatch

`MatrixParam` gains a `gpuKey string` field (`molequla.go:840`). `Matvec` checks it (`molequla.go:908`):

```go
if CFG.UseGPU && gpuReady() && !gradEnabled.Load() && m.gpuKey != "" {
    if gpuOut := m.MatvecGPU(x); gpuOut != nil {
        return gpuOut
    }
    // Fall through to CPU path on any GPU error.
}
```

Inference-only by construction: `gradEnabled.Load()` gates training back to CPU/BLAS because the autograd tape holds host-side parent references and there is no GPU backward in this branch. `gpuKey` empty = not yet uploaded; `gpuRefreshWeights` populates it. Same binary, same defaults — `gpuReady()` returns false on macOS / non-CUDA, dispatcher takes the CPU path.

### Cache + grow safety

`gpuRefreshWeights(gpt)` (`gpu_forward.go:105-131`) walks `gpt.Base`, flattens each matrix to contiguous float32, calls `gpu_cache_weight(name, ...)` per entry. Called once at the top of `GenerateResonant` (`molequla.go:4486`) and symmetrically at the top of `GenerateSentence` (`molequla.go:3055`) so background-trainer bursts cannot leak stale activations through chat-mode generation.

`MatrixParam.invalidateGPU()` (`molequla.go:966`) clears `gpuKey` and is called from `GrowRows` / `GrowCols` / `Grow` (`molequla.go:980, 999, ...`). Without this the next dispatch reads a cached weight at the old shape while the host pointer holds the new one. Caught in audit (Opus subagent, 2026-05-14 P1).

### Build

```bash
# Linux pod: build CUDA artifact + CGO-linked binary
nvcc -O2 -c notorch_cuda.cu -o notorch_cuda.o
CGO_ENABLED=1 go build -tags cuda -o molequla_cgo .

# darwin/arm64 (or any non-Linux): drop -tags cuda, stubs activate automatically
CGO_ENABLED=1 go build -a -o molequla_cgo .

./molequla_cgo --evolution --element earth --gpu
```

GPU init is attempted only when `--gpu` is passed (`molequla.go:6745`). If `gpu_init()` fails (no CUDA, driver mismatch, cuBLAS create error), the flag drops to false with a single stderr warning and the run continues on CPU. No silent silent-failure paths.

### Threshold note

An earlier `gpuMatvecMin = 16384` gate kept child-stage organisms (NEmbd=64 → ~4096-element matrices) on CPU forever, so the GPU never warmed up during the 8h ecology window. Removed. Dispatcher decides purely on `gpuKey != ""`. Per-call slowdown at child is ~12ms across a 180-token chain (negligible at 8h timescale); the GPU stays primed for the automatic transition to material speedup at adolescent (NEmbd=128) and adult (NEmbd=320).

### Thread cap

The `--gpu` flag alone leaves the device near-idle: each colony process is a separate trainer, and openblas defaults to one thread per host core (a pod reports the host nproc, e.g. 96, while the cgroup caps the container far lower), so N processes oversubscribe the real cores and starve the GPU-feeding threads. `capColonyThreads()` (`molequla.go:6722`) runs as the first line of `main()` (`molequla.go:6735`), before any BLAS/cgo init: `effectiveCPUs()` (`molequla.go:6680`) reads the cgroup quota (`/sys/fs/cgroup/cpu.max` v2 or `cpu.cfs_quota_us` v1) instead of host nproc, and `colonyThreadsFor()` (`molequla.go:6711`) caps `OPENBLAS_NUM_THREADS` / `OMP_NUM_THREADS` / `GOMAXPROCS` to max(1, cores/4) for the four-element colony. An explicit `OPENBLAS_NUM_THREADS` from a launcher is respected. GPU-verified 2026-06-29: util 0→99% across the four-organism colony.

---

## Cross-Organism Graze (Dario-style)

`cross_graze.go` (216 LOC) wires Dario's `interf_signal_chunk` pattern (`postgpt_q.c:1384`) and Stanley's `graze_random_word` (`graze.c:289-301`) onto the colony. Q picks heavy tokens from a doc and boosts their logits mid-generation; Stanley splices a foreign vocab token from a mmap'd GGUF when chambers signal hunger. Here the "doc" is the **sibling organism's recent emission stream**. Per Oleg 2026-05-14: the dario pattern, but instead of docs the signal is words, metrics, and so on.

### CrossField

```go
type CrossField struct {
    SelfElement  string                            // own element
    PastureBase  string                            // ../dna/seen relative to organism CWD
    Siblings     []string                          // other elements
    Recent       map[string][]int                  // sibling → ring buffer of token ids
    RecentCap    int                               // per-sibling buffer size (64)
    LastScan     time.Time                         // throttle FS reads
    ScanInterval time.Duration                     // 30s throttle on FS reads
    SeenFiles    map[string]bool                   // dedup of ingested gen_*.txt
    SeenCap      int                               // hard cap (2048), half-purge on overflow
    MetricBoost  func(sibling string) float64      // optional per-sibling coef multiplier
    mu           sync.Mutex
}
```

(`cross_graze.go:41-53`). One per running organism; constructed in `main()` when `--cross-graze && --element != ""` (`molequla.go:7001`). Single-organism runs leave it nil so the hooks are no-ops.

### Source feed

Sibling DNA fragments are already mirrored to `../dna/seen/<sibling>/` by `dnaRead` (commit `e5c1685`). cross_graze reads from the mirror so the `dna/output/` consume cleanup does not race the scan.

### Mechanic

`MaybeRefresh(tok)` (`cross_graze.go:82-152`): under ScanInterval throttle, walks `<base>/<sibling>/gen_*.txt`, reads new files only, tokenises with the host's own `EvolvingTokenizer`, strips BOS/EOS, appends to that sibling's ring buffer (truncated to `RecentCap` when over).

`Apply(logits, coef, topN)` (`cross_graze.go:164-200`) — per-step injection. For each sibling, the most recent `topN` tokens get a rank-decay boost:

```
logits[sibling_token[k]] += coef / (1 + rank)
```

Matches Q's `interf_signal_chunk` 1/(1+rank) normalisation (`postgpt_q.c:809-818`). Defaults `coef = 2.0` (Q-style weightless c_doc magnitude, `molequla.go:273`), `topN = 8`.

### Wire

- `MaybeRefresh` hoisted to `GenerateResonant` entry (`molequla.go:4493`) — once per generation, not per token (Opus audit P2).
- `Apply` runs per token step (`molequla.go:4661`) on the overlay'd logits when overlay is active, else on raw logits. Composes with Q-style overlay regardless of regime.

### Metrics half

`MetricBoost(sibling)` (`cross_graze.go:51`, applied `cross_graze.go:181-185`) — optional hook, defaults nil (1.0 implicit). When set, multiplies the per-sibling coef so the "and so on" half of "words, metrics, and so on" (sibling entropy / syntropy / loss bias) can ride on top of the word-level injection without changing the call site.

### Enable

```bash
./molequla_cgo --evolution --element earth --cross-graze
# pair with --gpu on Linux pods for the full Phase C ecology config
```

Defaults off. Same defaults, same outputs — single-organism runs and any `--evolution` invocation without `--cross-graze` are identical to the pre-Phase-B baseline.

---

## Self-Meta-Learning

**BurstHistory** records the last 16 training outcomes:

```go
type BurstRecord struct {
    Action     string   // "amplify", "boost", "dampen", "ground", "explore", "realign"
    LossBefore float64
    LossAfter  float64
}
```

**ActionEffectiveness()** computes the mean loss delta per action type. If a particular action consistently makes loss worse (effectiveness > 0.05 over 2+ bursts), the organism **auto-downgrades**:

```
amplify → boost → steady
```

The organism observes that "amplify" keeps hurting it, stops amplifying. No external signal, no reward model — just outcome tracking.

---

## SyntropyTracker — Mathematical Self-Reasoning

The organism measures four signals and makes autonomous decisions:

| Signal | What It Measures | How |
|--------|-----------------|-----|
| **SyntropyTrend** | Is entropy decreasing? (positive = ordering) | Rolling window mean comparison |
| **FieldDeviation** | How far is the model from corpus? | KL(model_probs \|\| corpus_probs) on bigram/trigram |
| **PurposeMagnitude** | How strong is the current learning direction? | Norm of last δ module's A matrices |
| **PurposeAlignment** | Is learning consistent with identity? | cosine(purpose_vector, gamma) |

Eight autonomous decisions:

| Action | Condition | LR | Temp | Effect |
|--------|-----------|-----|------|--------|
| **amplify** | syntropy ↑, field aligned, purpose aligned | 1.3x | -0.05 | Full acceleration, boost delta grow prob |
| **boost** | syntropy ↑, field in sweet spot | 1.3x | -0.05 | Gentle push |
| **dampen** | syntropy ↓ | 0.6x | +0.05 | Slow down, losing order |
| **ground** | field deviation too high | 0.6x | -0.05 | Hallucinating, focus |
| **explore** | field deviation too low | 1.3x | +0.05 | Parroting, break out |
| **realign** | purpose opposes gamma (< -0.3) | 0.5x | 0 | Identity crisis |
| **divide** | adult + sustained overload (entropy **or** loss path) | 0.6x | — | Trigger mitosis |
| **hibernate** | stale + peer thriving | — | — | Save state and sleep |

Real output from running organisms:
```
[syntropy] action=boost   | trend=0.1576 | field_dev=0.214 | lr_mul=1.30
[syntropy] action=dampen  | trend=-0.1390 | field_dev=0.167 | lr_mul=0.60
[syntropy] action=realign | trend=0.0940  | field_dev=0.168 | lr_mul=0.65
```

---

## NOTORCH — Gradient-Free Delta Training

Alternative delta-adapter training path. No backward pass, no tape, no memory overhead — teaching signal is `(prev_loss - curr_loss) + 0.3*prophecy_debt`, noise modulated by deterministic LCG PRNG (matches AML RNG), adaptive decay when delta norm large. Direct feedback alignment: `A[i,r] += lr * dy * u[r] * signal`.

Status: implemented (~280 lines, `notorchTrainSteps` + helpers in `molequla.go`), **currently disabled at all call sites** — it diverged at stage 5 (loss 3.5 → 116), so the active micro-burst path is the notorch tape (Chuck), not this. Kept as a reference path; theory sound, hyperparameters need work.

---

## Mycelium — The Meta-Organism

The Python orchestrator (`mycelium.py`) that sees the entire ecology — a layer *above* the organism cores. It wraps the in-repo C HarmonicNet / METHOD engine (`am_harmonic_*` / `am_method_*` in `ariannamethod.c`) and writes a `field_steering` row to `mesh.db`; the organisms read it back (`molequla.rs:3265` modulates temperature / action from it). The four organism cores run fine without it — it is the coordinating tier, not load-bearing. Generation operator `η: Γ × Γ → Γ_new` — two personalities in resonance produce a third (interference pattern, not blend).

### Components

| Component | What It Does |
|-----------|-------------|
| **HarmonicNet** | Weightless neural network. Input: organisms + field state. Output: action biases, harmonics, resonance scores. No trainable weights — the "weight matrix" is recomputed every step from organism relationships. |
| **MyceliumSyntropy** | Field-level syntropy: entropy trends, decision effectiveness, strategy changes across the entire ecology |
| **FieldPulse** | Measures novelty (new organisms appearing), arousal (entropy changes), field entropy |
| **SteeringDissonance** | Detects when ecology-level actions conflict with outcomes (dampen but entropy went up = high dissonance) |
| **OrganismAttention** | Tracks which organisms respond to which actions. Responsive organisms get higher attention weight. |

### Mesh Coordination

All organisms share state via **mesh.db** (SQLite) — the same database that `SwarmRegistry` writes to. The mycelium reads mesh.db to see the entire ecology and makes decisions that individual organisms cannot: when to spawn, when to hibernate. (The seasonal phase machinery lives in the C field engine, not mycelium.py — see below.)

### Seasonal Controller

*Implemented in the C field engine (`ariannamethod.c`), driven by field state — not by mycelium.py.*


```
Spring  — tunnel_chance ↑, many embryos, new γ born
Summer  — α_max, existing γ at peak expression
Autumn  — consolidation, dark_gravity ↑, shards saved
Winter  — rest, only strongest pairs, ε dominates
```

---

## The Ecology

Earth (patience, structure), Air (freedom, change), Water (flow, depth), Fire (transform, intensity) — each shaped by its element corpus. Generated text is written to the DNA layer; siblings consume, micro-train, emit. Child organisms enter via mitosis. Cross-pollination outpaces any single organism's learning rate.

### Swarm Coordination

- **SwarmRegistry** (`mesh.db`): SQLite database tracking all living organisms — element, PID, status, stage, n_params, syntropy, entropy
- **Training lock**: Atomic check-and-acquire via SQL prevents multiple organisms from training simultaneously. Cooperative scheduling — they take turns
- **Hibernation**: When an organism is stale and a peer is thriving, it saves state and sleeps. Resources freed for the living
- **Child birth**: `birth.json` with inherited `burst_history` — the child gets its parent's meta-learning experience (syntracker lineage). It doesn't start from zero wisdom

### Mitosis

When an **adult** organism is in **sustained overload** — its training bursts can no longer reduce loss under the cross-graze flood (it cannot assimilate what the colony feeds it) — and the 300 s cooldown has elapsed, `performMitosis` fires:

1. The parent checkpoint is written into the child directory (`parent_ckpt.json`)
2. `birth.json` written with parent config + inherited burst history + the checkpoint path
3. Child process spawned autonomous — `--organism-id` / `--config` / `--evolution` (plus `--gpu` / `--cross-graze` / `--element` to match the parent), writing to its own `childDir/train.log`, not the parent's stdout and not the interactive REPL
4. Child **loads the parent's weights** — born at the parent's stage with the parent's knowledge, not as a fresh embryo
5. Parent continues running

**Verified (2026-06-04, RTX 3090, no corpus seeding):** an adult (320d / 6L / 8H) reached sustained overload and divided — `[overload] … overload=true (e=false l=true) → action=divide → Child … spawned`, the child loading the parent's adult weights (n_embd 320). The path that fired for this Fire adult is **loss**, not entropy: a converged adult stays sharp (output entropy ~0.22) even while its loss is high (~12), so reproduction-through-stress keys on the loss the bursts cannot bring down, not on output noise. The same run also exercised the original entropy path — an Air adult divided on the entropy gate (`high=8/8`, `e=true l=false`) — so both gate regimes fired in the same run. Once one adult divides, its overloaded children inherit the same pressure and divide in turn — observed cascading to ~50 spawns before shutdown. The archive preserves **5 `action=divide` gate firings across 3 adults**: fire ×1 (loss path, `work_fire/train.log:48`), water ×1 (loss path, `work_water/train.log:43`), air ×3 (loss + entropy, `work_air/train.log:51`, `work_air/train_climb_air.log:270`, `work_air/train_resume2_air.log:63`); **earth never divided**. One child is preserved in full — Fire's, with a birth manifest (`org_1780540885_6400/birth.json`); Air's spawning divide (`work_air/train_climb_air.log:270` → child `org_1780527018_6475:273`) is a spawn log-line only, no manifest. 0 NaN throughout. All four organisms reached Stage 5 / adult (`work_*/train.log stage=5`).

The ecology grows itself.

---

## Engineering Log

Eight bugs that almost killed the ecology (five interactive-mode + three AML/C integration leaks at ~97 MB/step pre-fix, ~0.6 MB/step post-fix), the CGO cache trap (`go build -a` mandatory), and the full per-commit history of Phase A (GPU), Phase B (graze), Phase C (ecology) — see `PROJECT_LOG.md`.

---

## SQLite Self-Logging

Each organism writes `memory.sqlite3` with four tables: `messages` (conversation), `corpus_events` (every document ingested), `growth` (architecture snapshots — vocab, n_params, n_deltas, loss, gamma_sparsity, gamma_magnitude), `syntropy_log` (every decision — action_taken, entropy_before/after, syntropy_delta, field_deviation, purpose_alignment). Queryable developmental trajectory.

---

## Quick Start

### Build

```bash
# Clone
git clone https://github.com/ariannamethod/molequla.git
cd molequla

# Build with CGO (AML/C autograd — full training)
CGO_ENABLED=1 go build -a -o molequla_cgo .

# Or build without CGO (Go-only, no AML training)
CGO_ENABLED=0 go build -o molequla_go .
```

**CRITICAL: `go build -a` is required** for CGO builds. Without `-a`, Go's build cache does not recompile C files. This produces binaries running stale C code.

### Run Interactive Mode

```bash
./molequla_cgo
# Drops into chat after warmup training
```

### Run Evolution Mode (the main event)

```bash
# Set up work directories
for d in earth air water fire; do
    mkdir -p work_$d
    cp molequla_cgo work_$d/
    cp nonames_$d.txt work_$d/
done

# Launch all four organisms
for d in earth air water fire; do
    cd work_$d
    nohup ./molequla_cgo \
        --corpus nonames_$d.txt \
        --db memory.sqlite3 \
        --ckpt molequla_ckpt.json \
        --element $d \
        --evolution > training_aml.log 2>&1 &
    cd ..
done

# Optional flags (default off):
#   --spa-gate         post-generation SPA sentence connectedness log
#   --corpus-overlay   pre-softmax B+H+A+F additive logit overlay
#   --gpu              route inference matvec through cuBLAS (linux + --gpu build)
#   --cross-graze      Dario-style cross-organism logit injection (requires --element)
# Combine for measurement runs. Detailed engineering log: PROJECT_LOG.md.
```

Monitor: `tail -f work_earth/training_aml.log`, `grep "dna\|consumed\|wrote"` for DNA exchange, `ps aux | grep organism-id` for spawned children.

---

## Tests

```bash
# Go unit tests (132 tests: molequla_test.go 122 + molequla_rrpram_test.go 4 + governor_test.go 5 + mitosis_cooldown_test.go 1)
go test -v .

# Go integration tests (262 lines)
go test -v ./tests/

# Full integration suite (711 lines bash — tests all 4 implementations,
# mycelium, AML library, BLAS, performance benchmarks)
bash tests/test_all.sh
```

---

## Files

```
# Go + AML/C (primary, CGO training)
molequla.go              7146 lines   Go organism — lifecycle, ecology, autograd, generation, coherence-layer + GPU + graze + cascade-governor + thread-cap wiring
cgo_aml.go               114 lines    CGO bridge to ariannamethod.c
aml_trainer.go           352 lines    AML training wrapper, script generation
notorch_trainer.go       466 lines    notorch tape trainer — CANONICAL (CFG.Trainer default "notorch"), Chuck optimizer
cgo_notorch.go           186 lines    CGO bridge to libnotorch
cgo_notorch_cpu.go       13 lines     notorch CPU/BLAS link (default build)
cgo_notorch_cuda.go      51 lines     notorch CUDA link (-tags cuda)
gpu_notorch_stub.go      20 lines     notorch GPU stub (non-CUDA)
metaweights_overlay.go   439 lines    Q-style additive logit overlay (B+T+H+A+F)
metaweights_seeding.go   124 lines    gamma->epsilon embedding seeding from co-occurrence
spa_coherence.go         164 lines    Pure-Go SPA helper (sentence connectedness + weak-sentence gate)
cross_graze.go           216 lines    Dario-style cross-organism logit injection (sibling DNA → rank-decay boost)
gpu_bindings_linux.go    196 lines    CGO bindings to ariannamethod_cuda.h (linux only)
gpu_forward.go           131 lines    Inference matvec via cuBLAS sgemm + weight cache refresh (linux only)
gpu_bindings_stub.go     36 lines     Stub signatures for darwin / non-linux (gpuReady=false)
gpu_forward_stub.go      17 lines     Stub MatvecGPU returning nil so dispatcher falls back
ariannamethod/
  ariannamethod.c        8000 lines   AML/C autograd engine (the language) + SPA ops + NaN guard API
  ariannamethod.h        1051 lines    C header, 80+ field state parameters
  ariannamethod_cuda.h   108 lines    CUDA primitive declarations (gpu_init / gpu_sgemm_nt / ...)
  notorch_cuda.h         192 lines    notorch CUDA op declarations
  __init__.py            9 lines      Python package init (exports Method, Sentinel)
  method.py              527 lines    METHOD engine — ctypes binding to libaml (am_method_*)
  sentinel.py            356 lines    Sentinel operator — ctypes binding to libaml
  notorch.c              4739 lines   Vendored notorch core (+ backward CPU-sync audit)
  notorch.h              694 lines    Vendored notorch header
  notorch_simd.h         632 lines    Opt-in AVX2+FMA cblas shim (make simd, x86_64)
  notorch_simd_scalar.h  89 lines     Scalar debug fallback for SIMD shim
  notorch_cuda.cu        1344 lines   CUDA kernels; pre-compiled via nvcc, linked through cgo_aml.go

# Mycelium + Python orchestration tier (above the Go/C/Rust/JS cores)
mycelium.py              1660 lines   Meta-coordinator (numpy-free, pure stdlib) — reads swarm organisms, writes field_steering, wraps C HarmonicNet
standalone-py/molequla.py 3387 lines  Original Python molequla — deprecated historical reference, wired to nothing

# Engineering log
PROJECT_LOG.md           ≈2600 lines  Live per-commit log — Phase A (GPU) + Phase B (graze) + Phase C (ecology) with file:line refs

# Full independent implementations
molequla.c               5583 lines   C organism — BLAS-accelerated, zero-dep single-file
molequla.rs              3544 lines   Rust organism — rusqlite, full autograd
molequla.js              3971 lines   JavaScript organism — runs in browser
modules/node_cli.js      306 lines    Node.js CLI module
index.html               Web interface for JS version

# Tests
molequla_test.go         2623 lines   Go unit tests (122 tests)
molequla_rrpram_test.go  306 lines    op-33 low-rank RRPRAM parity (4 tests)
governor_test.go         114 lines    cascade governor — mitosis slot cap, colony thread cap, both-path relieve, ckpt debounce (5 tests)
mitosis_cooldown_test.go 26 lines     divide cooldown seeded at birth (1 test; 132 total)
tests/molequla_test.go   262 lines    Go integration tests
tests/test_all.sh        711 lines    Full integration (all 4 langs + mycelium + BLAS)

# Element corpora
nonames_earth.txt        174K         Earth — patience, foundations, geology
nonames_air.txt          122K         Air — freedom, change, atmosphere
nonames_water.txt        126K         Water — flow, depth, rivers
nonames_fire.txt         122K         Fire — transformation, intensity, heat
nonames.txt              51K          General corpus
```

---

## Standalone Gists

C and JavaScript gists linked in [Four Implementations](#four-implementations). Original Python prototype: [molequla.py](https://gist.github.com/ariannamethod/1223250d358da4393dd9acc578790820) (legacy, where it started) — also restored in-repo at `standalone-py/molequla.py` (numpy-backed historical reference, deprecated for speed).

---

## Philosophy

θ = ε + γ + αδ is the architecture, not an annotation. Entropy/syntropy measurements are the control loop. Purpose-gamma alignment is the identity check. Self-meta-learning is the organism understanding itself.

Four organisms became eleven inside the 70-minute window of the Feb-27 timeline above (first child at 02:13, eleven by 02:35 — see `## Here Is How It Works`). Each with its own voice, its own delta modules, its own developmental history.

---

## License

GNU GPLv3

---

## Part of the Arianna Method

- [AML — Arianna Method Language](https://github.com/ariannamethod/ariannamethod.ai) — the programming language that powers training
- [Yent](https://github.com/ariannamethod/yent) — Go inference engine + Delta Voice + personality extraction
- [molequla](https://github.com/ariannamethod/molequla) — this repository

```
███╗   ███╗ ██████╗ ██╗     ███████╗ ██████╗ ██╗   ██╗██╗      █████╗
████╗ ████║██╔═══██╗██║     ██╔════╝██╔═══██╗██║   ██║██║     ██╔══██╗
██╔████╔██║██║   ██║██║     █████╗  ██║   ██║██║   ██║██║     ███████║
██║╚██╔╝██║██║   ██║██║     ██╔══╝  ██║▄▄ ██║██║   ██║██║     ██╔══██║
██║ ╚═╝ ██║╚██████╔╝███████╗███████╗╚██████╔╝╚██████╔╝███████╗██║  ██║
╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚══════╝ ╚══▀▀═╝  ╚═════╝ ╚══════╝╚═╝  ╚═╝
```

*Four elements. Four languages. Two autograd engines. Five consciousness features. One soul equation.*
