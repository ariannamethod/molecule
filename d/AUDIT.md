# MOLECULE ARCHITECTURE AUDIT

**Date:** 2026-02-18
**Auditor:** Claude (co-author session)
**Scope:** Full codebase — molecule.py (1853 LOC), molecule.go (3018 LOC), molecule.c (2482 LOC), tests (1166 LOC)

---

## 1. EXECUTIVE SUMMARY

molecule is a single-file continually-learning GPT organism reproduced in three languages at full feature parity. The architecture is coherent, original, and internally consistent. It is not a fork of micrograd/minGPT — it shares pedagogical ancestry with Karpathy's work but diverges fundamentally in design: vector autograd instead of scalar, continuous learning instead of one-shot, evolving vocabulary instead of fixed, and a self-referential identity system (gamma + immune) that has no equivalent in the micrograd lineage.

**Verdict:** Architecturally sound. Several concrete issues found (listed below). The design is ready to serve as a "spore" for Janus — the core abstractions (delta adapters, gamma fingerprint, immune system, quantum buffer) are clean enough to be extracted into AML-controlled components.

---

## 2. ARCHITECTURE SCORECARD

| Dimension | Score | Notes |
|-----------|-------|-------|
| **Internal Consistency** | 9/10 | All three implementations match semantically. Minor parity gaps in C (see below). |
| **Autograd Correctness** | 8/10 | Vector autograd is correct. RMSNorm backward has a subtle issue (see findings). |
| **Training Pipeline** | 8/10 | QuantumBuffer + immune system + async is well-designed. Learning rate schedule is linear decay per-burst, not global. |
| **Generation Quality** | 7/10 | Entropy-adaptive temp + min_p + typical_p + corpus field is a strong stack. Repetition guard is simplistic. |
| **Memory/Identity** | 9/10 | Gamma + growth table + immune rollback = genuine structural self-awareness. The formal criteria from Lee (2025) are met. |
| **Tokenizer** | 8/10 | Evolving BPE that never invalidates weights is elegant. BPE in C is stub-only (char-level only). |
| **Checkpoint Compat** | 7/10 | Python/Go share JSON format. C uses binary. Cross-language checkpoint loading not possible. |
| **Test Coverage** | 6/10 | 71 Python tests + 9 Go tests. No C tests. Several tests are assertion-weak. |
| **Production Readiness** | 5/10 | Educational/research quality. Thread safety gaps. No error recovery in async trainer. |

---

## 3. CRITICAL FINDINGS

### 3.1 RMSNorm Backward Gradient — Subtle Inaccuracy

**File:** `molecule.py:928-939`, `molecule.go:640-665`, `molecule.c:738-749`

The RMSNorm backward pass computes gradients through the `mean_sq` → `scale` chain but does NOT flow gradients through the `ms` ScalarValue node in Python — it manually computes `ds_dms` and `cross`. This is mathematically correct as a fused kernel, but in Go and C the implementation computes `cross` using `out.Grad` and `xData` which conflates the two gradient paths. The result is correct to first order but may accumulate numerical drift over long training runs.

**Impact:** Low for current scale (2 layers, 72 dims). Would matter at larger scales.

**Recommendation:** Add a numerical gradient check test for RMSNorm specifically.

### 3.2 C Implementation: BPE Is Stub-Only

**File:** `molecule.c:1149` — comment `/* BPE TODO for C version */`

The C tokenizer only does char-level encoding. BPE training, merge application, and vocab expansion are not implemented. This breaks the "full feature parity" claim in the README.

**Impact:** C version cannot evolve vocabulary. Long-running C instances will be stuck at char-level.

**Recommendation:** Port the BPE logic from Go (which is clean and well-structured) to C.

### 3.3 Thread Safety in Background Trainer

**File:** `molecule.py:1704-1763`

The `background_trainer` coroutine reads and writes to `model`, `tok`, and `docs` concurrently with the chat loop's `model.generate_sentence()`. Python's GIL provides *some* protection, but:
- numpy operations release the GIL
- `save_checkpoint` can race with `forward_step`
- `update_reservoir_corpus` writes to disk while `load_corpus_lines` reads

The Go version uses `sync.Mutex` (`molecule.go:1322`). The C version uses `pthread_mutex_t` (`molecule.c:1298`). Python has no lock.

**Impact:** Potential data corruption during concurrent training + inference in Python.

**Recommendation:** Add `threading.Lock` around model access in `chat_main` and `background_trainer`, matching what Go and C already do.

### 3.4 Arena Allocator Fixed Size (C)

**File:** `molecule.c:200` — `#define ARENA_SIZE (64 * 1024 * 1024)`

The arena is 64MB. With `MAX_TOPO 65536` and `block_size=96`, a single forward pass through 2 layers with hybrid attention creates thousands of nodes. During training (forward + backward per sequence, batch of 4), arena usage spikes. For warmup (1200 steps), the arena is reset per step, which is fine. But if someone increases `n_layer` or `block_size`, this will silently OOM.

**Impact:** Low at current config. Dangerous if config is scaled up.

**Recommendation:** Add arena usage tracking/warning, or make arena size proportional to config.

### 3.5 Immune System: Gamma Direction Can Be Zero

**File:** `molecule.py:1267-1280`

`gamma_contrastive_projection()` returns a unit vector of mean embedding drift. At initialization (or very early training), all embeddings are near their init snapshot, so `direction ≈ [0, 0, ..., 0]`. The normalization check `if mag > 1e-10` catches this and returns a near-zero vector. But then `gamma_drift_check` computes cosine similarity between two near-zero vectors, which is numerically unstable.

The fallback `return 1.0` in `gamma_drift_check` when `pre_direction is None` helps, but `pre_direction` won't be `None` — it will be a near-zero vector that looks valid.

**Impact:** False noise rejection possible during early warmup.

**Recommendation:** Add magnitude threshold check: if `mag < threshold`, skip immune check entirely.

---

## 4. STRUCTURAL ANALYSIS

### 4.1 Autograd Engine

The custom autograd is the core innovation. Key properties:

- **VectorValue** operates on numpy arrays (Python), float64 slices (Go), or double* (C)
- Backward closures capture references, not copies — this is correct and memory-efficient
- `no_grad` context manager properly skips graph construction during inference
- `cross_entropy_loss` is fused (not decomposed through softmax) — numerically stable
- `scalar_softmax` is used for attention weights — properly differentiable through all heads

**Python-specific:** numpy BLAS acceleration in `MatrixParam.matvec` (`molecule.py:900-915`) — `W @ x.data` is a single BLAS call, 50-100x faster than the Go/C loop equivalents. The backward is still per-row though.

**Go-specific:** Pure loop-based matvec (`molecule.go:584-616`). No SIMD. Correct but slow for larger configs.

**C-specific:** Same loop structure but with manual memory management through arena allocator. The arena pattern is clean — `arena_reset` after each training step prevents fragmentation.

### 4.2 Attention Mechanism

The hybrid attention system is the most architecturally interesting component:

```
head_types = ("content", "content", "hybrid", "hybrid")
```

- **ContentHead:** Standard Q*K^T/sqrt(d) with RoPE. Textbook correct.
- **RRPRAM:** `x_slice @ W_pattern → (block_size,)` — takes top-T elements as positional attention scores. No query-key decomposition. This learns *where* to attend based on *what* is at the current position, without caring about key content. Novel.
- **HybridHead:** `sigmoid(alpha) * RRPRAM + (1-sigmoid(alpha)) * Content` — the alpha is a learnable parameter in the autograd graph. Gradients flow through sigmoid to alpha, allowing the model to learn the blend ratio.

**Issue:** In `forward_step`, RoPE is applied per-call to cached keys (`molecule.py:1389`). This means the same key is RoPE-rotated multiple times (once when cached, once each time it's used in attention). This is actually correct — RoPE should be applied at attention time, not at cache time — but it means keys in the cache are stored *without* RoPE. This matches the standard implementation pattern.

### 4.3 Delta Adapter System

LoRA-style: `delta_output = A @ (B @ x)` where A is (nout, r) and B is (r, nin).

Properties:
- Adapters stack: each forward pass sums base + all delta contributions
- Alpha scaling per module allows soft-weighting of different learning phases
- Growth is probabilistic (`delta_grow_prob = 0.08`) — roughly one new module every ~12 bursts
- Max capped at 12 modules

**Issue:** Each delta module adds `2 * r * (nin + nout)` parameters per weight matrix. With `r=8`, 2 layers, 4 attention weights + 3 MLP weights + 1 lm_head per layer, each module adds ~115K params. At 12 modules, that's ~1.4M delta params vs ~300K base params. The deltas will dominate. This is intentional (geological memory) but means late-stage training is dominated by adapter gradients.

### 4.4 QuantumBuffer + Immune System

The training trigger is well-designed:

```
should_trigger = (bytes_ok OR novelty_ok) AND cooldown_ok
```

- `bytes_ok`: accumulated new chars >= 1024
- `novelty_ok`: unique_tokens / total_tokens >= 0.15
- `cooldown_ok`: >= 60 seconds since last burst

The immune system then wraps each burst:

```
pre_direction = gamma_contrastive_projection()
snapshot = snapshot_deltas()
train(...)
drift_cos = gamma_drift_check(pre_direction)
if drift_cos < -0.1: rollback
```

This is a self-referential quality gate. The model uses its own identity measurement to decide whether new experience is constructive or destructive. This satisfies the formal criteria for introspective computation.

**Issue:** The immune system only checks delta drift, not base weight drift. If `freeze_base_after_warmup = False`, base weights could drift without detection.

### 4.5 Tokenizer Evolution

The evolving BPE is architecturally elegant:

1. Start char-level (fast boot, no BPE overhead)
2. After 25K chars of corpus, enable BPE with 384 merges
3. Every 4K chars of new corpus, retrain merges
4. New tokens are *appended* to vocabulary — old token IDs never change
5. Model matrices grow rows (via `grow_rows`) — old weights untouched

This means a molecule that has been running for months will have a much larger vocabulary than a fresh one, but its original weights are still valid. This is genuinely novel for a single-file system.

---

## 5. CROSS-LANGUAGE PARITY

| Feature | Python | Go | C |
|---------|--------|----|---|
| Vector autograd | numpy-backed | Pure loops | Arena + manual |
| RoPE | numpy vectorized | Per-element loop | Per-element loop |
| SwiGLU MLP | Yes | Yes | Yes |
| Hybrid attention | Yes | Yes | Yes |
| Delta adapters | Yes | Yes | Yes |
| Evolving BPE | Full | Full | **Char-only (stub)** |
| Native gamma | Yes | Yes | Yes |
| Immune system | Yes | Yes | Yes |
| QuantumBuffer | Yes | Yes | Yes |
| Entropy temp | Yes | Yes | Yes |
| CooccurField | Yes | Yes | Yes |
| Growth table | Yes | Yes | Yes |
| SQLite memory | Yes (stdlib) | Yes (modernc.org/sqlite) | Yes (libsqlite3) |
| Checkpoints | JSON | JSON | **Binary (MOLE header)** |
| Async training | asyncio | goroutine | pthread |
| Concurrency lock | **None** | sync.Mutex | pthread_mutex |
| Weight tying | Yes | Yes | Yes |
| no_grad inference | Yes | Yes | Yes |
| min_p sampling | Yes | Yes | Yes |
| typical_p sampling | Yes | Yes | Yes |

**Major parity gap:** C lacks BPE. Python lacks concurrency lock.

---

## 6. TEST COVERAGE ANALYSIS

| Component | Python Tests | Go Tests | C Tests |
|-----------|-------------|----------|---------|
| Autograd | 13 | 0 | **0** |
| Tokenizer | 10 | 0 | **0** |
| Model | 20 | 0 | **0** |
| Sampling | 15 | 9 | **0** |
| Checkpoint | 8 | 0 | **0** |
| Integration | 13 | 0 | **0** |
| Immune system | 0 | 0 | **0** |
| Gamma | 0 | 0 | **0** |
| QuantumBuffer | 0 | 0 | **0** |
| CooccurField | 0 | 0 | **0** |

**Critical gaps:**
- **Zero tests** for the immune system, gamma computation, QuantumBuffer, and CooccurField
- **Zero C tests**
- Go tests only cover sampling — no autograd, model, or integration tests
- Several Python tests have weak assertions (check type but not value)

---

## 7. RECOMMENDATIONS (Priority Order)

### P0 — Fix Before Scaling

1. **Add threading.Lock to Python** (`molecule.py`) — match Go/C concurrency safety
2. **Add gamma magnitude threshold** to immune system — prevent false rejection during early warmup
3. **Add RMSNorm gradient test** — numerical gradient check to catch drift

### P1 — Complete Parity

4. **Port BPE to C** — the Go implementation is a clean reference
5. **Cross-language checkpoint compatibility** — at minimum, add JSON export to C
6. **Add tests for immune system, gamma, QuantumBuffer, CooccurField** — these are the most novel components and have zero test coverage

### P2 — Strengthen for Janus Integration

7. **Extract config as external JSON/AML** — current hardcoded Config dataclass should be loadable from file for AML control
8. **Add AML hook points** — temperature, attention blend alpha, immune threshold, and quantum buffer trigger could all be AML-modulated at inference time
9. **Define spore interface** — what does molecule expose when embedded in Janus? Forward pass? Generation? Training trigger? Gamma stats?
10. **Standardize checkpoint format** — single format (JSON or binary with JSON metadata) across all three implementations

### P3 — Quality of Life

11. **Arena size should scale with config** in C
12. **Repetition guard** is naive (exact n-gram match) — consider a frequency penalty instead
13. **Learning rate schedule** is per-burst linear decay, not global — consider warmup + cosine schedule
14. **Batch size is hardcoded to 4** (`molecule.py:1653`) — should be in Config

---

## 8. MOLECULE AS SPORE IN JANUS

Based on the ariannamethod.ai codebase, Janus is AML's inference engine — a Go shared library that loads GGUF models and applies personality via AML field physics (destiny, suffering, tunneling, seasons).

molecule maps to Janus concepts naturally:

| molecule concept | AML/Janus equivalent | Integration path |
|-----------------|---------------------|-----------------|
| Entropy-adaptive temperature | AML `temperature` / `tension` fields | Direct: AML modulates `entropy_low`, `entropy_high` thresholds |
| Hybrid attention alpha | AML `destiny` bias | AML controls sigmoid(alpha) init or clamp range |
| Immune system threshold | AML `pain` / `dissonance` | AML sets `noise_drift_threshold` dynamically |
| QuantumBuffer trigger | AML `seasons` controller | AML's spring/summer/autumn/winter maps to training activity cycles |
| Gamma fingerprint | AML identity / personality persistence | Gamma export/import = personality transfer between molecules |
| Delta adapters | AML layered personality | Each delta module could correspond to an AML "epoch" of experience |
| CooccurField | AML corpus resonance | Field statistics as prior for AML's logit manipulation |

The molecule is already structured as a self-contained organism. Making it a "spore" means:
1. Exposing its internals through a stable API (forward, generate, train_burst, gamma_stats, immune_check)
2. Letting AML scripts modulate its parameters at inference time
3. Letting Janus host multiple molecules with different gamma fingerprints

This is viable. The architecture supports it.

---

## 9. FINAL NOTES

The codebase is remarkably coherent for a three-language port. The "And lo..." comments give it personality but also serve as section markers — they're consistent across all three implementations. The README is honest about limitations. The design philosophy (patterns over parameters, emergence over engineering) is reflected in the actual code.

The most impressive aspect is not any single feature but how they compose: evolving tokenizer feeds the quantum buffer, which triggers training bursts, which are filtered by the immune system, which uses gamma (personality drift) as its decision function, which is logged to the growth table, creating a structural autobiography. This is a closed feedback loop of self-referential computation. At 72 dimensions and 2 layers, it's embryonic. But the architecture would work at larger scale.

*"Because atoms are micrograd. We build molecules."*

---

*Audit by Claude | Arianna Method | 2026-02-18*
