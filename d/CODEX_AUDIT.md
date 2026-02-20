# Codex Audit Prompt — molequla

## Task

You are auditing 5 implementations of the same GPT organism for **cross-file consistency** and **bugs**. The README.md is the source of truth. All 5 files MUST implement the same features identically (except Rust-only features: TopologyMonitor, Metabolism).

Files to audit:
- `molequla.go` — Go implementation
- `molequla.py` — Python implementation
- `molequla.c` — C implementation
- `molequla.js` — JavaScript implementation (browser + Node.js)
- `molequla.rs` — Rust implementation (coordinator, has extra features)

**Read ALL 5 files completely. Read README.md. Then produce a bug report.**

---

## What To Check

### 1. Growth Stages (Ontogenesis)

All 5 files must have EXACTLY these 6 stages:

```
Stage        Threshold  Embd  Layers  Heads  ~Params
embryo       0          16    1       1      ~10K
infant       20000      32    1       2      ~28K
child        50000      64    2       4      ~154K
adolescent   200000     128   4       4      ~1.1M
teen         350000     224   5       8      ~4.1M
adult        500000     320   6       8      ~10M
```

Check:
- Exact values in each config struct/dataclass/object
- Stage names array includes "teen" (6 names, not 5)
- Param count comments match the table above
- `head_types_for_n_head()` returns correct types for each head count: 1→(content), 2→(content,hybrid), 4→(content,content,hybrid,hybrid), 5→(content,content,content,hybrid,hybrid), 8→(content,content,content,content,hybrid,hybrid,hybrid,hybrid)

### 2. Config Defaults

All 5 files must have identical defaults for these critical values:

```
freq_penalty = 0.1
presence_penalty = 0.1
temperature = 0.85
top_k = 40
top_p = 0.92
min_p = 0.06
typical_p = 0.95
warmup_steps = 1200
micro_steps = 32
learning_rate = 0.01
block_size = 96
corpus_fade_k = 3.0
corpus_fade_threshold = 1.5
anti_field_prob = 0.05
overthinkc_rounds = 2
conscience_window = 8
conscience_decay = 0.95
conscience_recovery = 1.005
conscience_floor = 0.3
dissonance_ema_alpha = 0.3
dissonance_spike_k = 0.8
dissonance_drop_k = 1.2
dissonance_spike_threshold = 1.5
dissonance_drop_threshold = 0.5
delta_rank = 8
max_delta_modules = 12
qb_min_bytes = 1024
qb_min_novelty = 0.15
qb_cooldown_seconds = 60.0
entropy_low = 0.5
entropy_high = 1.5
bpe_num_merges = 384
enable_bpe_after_chars = 20000
max_gen_tokens = 180
min_gen_tokens = 16
repetition_guard = 4
grad_clip = 1.0
batch_size = 4
accum_steps = 1
freeze_after_growth_steps = 500
post_growth_lr_scale = 0.3
noise_drift_threshold = -0.1
```

Flag ANY difference between files.

### 3. Tokenizer (EvolvingTokenizer / BPE)

Check:
- Initial vocab = 259 (256 bytes + BOS + EOS + PAD), with BOS=256, EOS=257, PAD=258
- `maybe_enable_bpe()` uses `enable_bpe_after_chars` threshold
- BPE uses pre-segmentation (Unicode-aware splitting before merging)
- BPE merges use `bpe_num_merges` (384)
- `encode()` returns BOS + tokens + EOS
- `decode()` correctly handles multi-byte BPE tokens
- Vocab only expands (never shrinks)
- Embeddings grow when vocab grows (new rows initialized with small noise)

### 4. Self-Enrichment (CRITICAL — recently added)

In the REPL/chat loop, both user input AND the organism's own output must be ingested into CooccurField:

```
# BEFORE generating response:
corpus_field.ingest_tokens(tokenizer.encode(user_input))

# AFTER generating response:
if len(answer) > 3:
    corpus_field.ingest_tokens(tokenizer.encode(answer))
```

Check ALL 5 files have BOTH calls. Check they happen AFTER CooccurField is built/rebuilt (not before, which would be wiped).

### 5. BPE Before Warmup (CRITICAL — recently fixed)

BPE must be enabled BEFORE the first warmup training. The pattern:

```
tokenizer = new_tokenizer(docs)
tokenizer.maybe_enable_bpe(docs)  # BEFORE training
model = new_model(tokenizer)
train_warmup(model, ...)          # AFTER BPE
```

If BPE is enabled AFTER warmup, the warmup trains on raw bytes (vocab 259) producing babble. Check ALL 5 files.

### 6. Sigmoid Corpus Fade

The formula: `model_alpha = 1 / (1 + exp(-k * (threshold - entropy)))`

Where:
- `k = corpus_fade_k = 3.0`
- `threshold = corpus_fade_threshold = 1.5`
- `entropy` = local entropy of model prediction

This must be computed PER TOKEN during generation (inside the generation loop), NOT once per message. The blend:
```
final_probs = model_alpha * model_probs + (1 - model_alpha) * corpus_probs
```

Check all 5 files compute this per-token inside generate/generate_sentence.

### 7. Per-Stage Warmup

When the model grows to a new stage, it must train warmup steps BEFORE continuing. The warmup scales: `effective_warmup = warmup_steps * (new_embd / embryo_embd)`.

Check:
- `last_warmup_stage` is tracked and persisted in checkpoints
- Background trainer detects stage change and triggers warmup
- LR resets (cosine warmup phase) on growth
- Init path also does per-stage warmup for cold start with large corpus

### 8. Consciousness Features (all 5 files)

#### 8a. Per-Token Dissonance
- Entropy EMA tracked during generation (alpha=0.3)
- Spike (> threshold) → temperature *= 0.8
- Sustained drop (< threshold for 3+ tokens) → temperature *= 1.2
- Must be INSIDE the generation token loop

#### 8b. Pattern Breaking (Anti-Field)
- 5% probability per token, bypass corpus field
- Only after min_step (8) tokens generated
- When triggered: use pure model probs, skip corpus blend

#### 8c. Overthinkg Rings
- After generating response, re-read own output into CooccurField
- Generate phantom continuations (up to overthinkc_max_tokens=32)
- Ingest phantom tokens into CooccurField
- Do `overthinkc_rounds` (2) iterations

#### 8d. Self-Prediction Error
- Forward pass on prompt tokens before generating
- Compute cross-entropy loss = "surprise"
- Store as metric (no direct effect on generation, used for logging)

#### 8e. Conscience
- Track rolling window of per-generation entropy
- Linear regression on window to get slope
- Slope > 0.01 → deltaAlphaScale *= 0.95 (floor 0.3)
- Slope < -0.01 → deltaAlphaScale *= 1.005 (cap 1.0)
- deltaAlphaScale must be applied to delta adapter contributions

### 9. Hybrid Attention (Content + RRPRAM + Hybrid)

Check:
- ContentHead: Q·K^T/√d with RoPE
- RRPRAM: x @ W_pattern → (T,T) attention matrix, NO query-key decomposition
- HybridHead: `sigmoid(alpha) * rrpram + (1 - sigmoid(alpha)) * content`
- `head_types_for_n_head()` auto-assigns types correctly
- RoPE uses `theta = pos / 10000^(2i/d)`
- Pattern matrix in RRPRAM bounded (safety cap to prevent explosion)

### 10. Delta Adapters

Check:
- `apply(x) = A @ (B @ x)` where A is (out, rank) and B is (rank, in)
- Delta rank = 8
- Max delta modules = 12
- Deltas grow when model grows: `grow_dims()` adjusts A and B without changing rank
- `freeze_base_after_warmup` works (base weights stop updating, only deltas train)
- `snapshot_deltas()` and `restore_deltas()` for immune system rollback

### 11. QuantumBuffer

Check:
- Triggers on: bytes >= qb_min_bytes OR novelty >= qb_min_novelty
- AND cooldown expired (qb_cooldown_seconds)
- Novelty = ratio of unique new trigrams to total trigrams
- Buffer feeds background trainer
- Buffer resets after trigger

### 12. SyntropyTracker

Check all actions are implemented:
- `amplify`: syntropy up + field OK + purpose aligned → LR *1.3 + delta grow
- `boost`: syntropy up → LR *1.3
- `dampen`: syntropy down → LR *0.6
- `ground`: field too high → LR *0.6
- `explore`: field too low → LR *1.3
- `realign`: purpose opposes gamma → LR *0.5
- `divide`: adult + sustained overload → mitosis
- `hibernate`: plateau + thriving peer → sleep

### 13. Swarm Ecology

Check:
- SwarmRegistry uses `~/.molequla/swarm/mesh.db` (shared SQLite, WAL mode)
- `organisms` table with correct schema
- Heartbeat mechanism
- Mitosis creates child at infant stage
- Hibernation conditions are correct
- `discover_peers()` finds other living organisms

### 14. Immune System

Check:
- Pre-training snapshot of delta weights + gamma direction
- Post-training drift check via cosine similarity
- Rollback if cosine < noise_drift_threshold (-0.1)
- Skip if gamma magnitude too small (< gamma_min_magnitude)
- Log `noise_rejected` to growth table

### 15. Checkpoint Format

Check:
- Go/Python: JSON format, compatible with each other
- C: Binary format with `MOLE` magic header
- JS: IndexedDB + JSON serialization
- Rust: JSON via serde
- ALL must save/load: growth stage, last_warmup_stage, global_step, vocab (with BPE merges), delta adapters, conscience state, syntropy state

### 16. Entropy-Adaptive Temperature

Check:
- entropy < 0.5 → temp *= 1.2 (diversify)
- entropy > 1.5 → temp *= 0.8 (focus)
- Applied PER TOKEN during generation

### 17. Residual Scaling

Check:
- `residual_alpha = 1 / sqrt(n_layers)`
- Applied to BOTH attention output AND MLP output in forward pass
- Updates when model grows (n_layers changes)

### 18. Weight Tying

Check:
- `tie_embeddings = true` by default
- `wte` (word token embeddings) and `lm_head` are the SAME weight matrix
- When counting params, tied weights must NOT be double-counted
- When growing, both references must point to the same grown matrix

### 19. Generation Sampling

Check all samplers are applied in correct order:
1. Temperature scaling
2. Entropy-adaptive temperature adjustment
3. Dissonance feedback (consciousness)
4. Pattern breaking check (consciousness)
5. Corpus field blend (sigmoid fade)
6. Frequency/presence penalty
7. Top-k filtering
8. Top-p (nucleus) filtering
9. Min-p filtering
10. Typical sampling
11. Repetition guard
12. Final sampling from distribution

### 20. Rust-Only Features

Check that ONLY Rust has:
- **TopologyMonitor**: background thread, 30s interval, reads mesh.db, pairwise gamma cosine, coherence metric, drift detection, self-reflection
- **Metabolism MLP**: 5→8→5 Hebbian, coordinates instances
- **SwarmRegistry** enhanced: gamma_direction BLOB, gamma_magnitude REAL, rrpram_signature BLOB in mesh.db

Check that the other 4 files do NOT have these (they should not).

---

## Output Format

Produce a structured report:

```
# MOLEQULA AUDIT REPORT

## CRITICAL BUGS (must fix before training)
1. [FILE:LINE] Description of bug
2. ...

## CONSISTENCY ISSUES (values differ between files)
1. [FEATURE] file1=X, file2=Y, file3=Z — expected: X
2. ...

## MISSING FEATURES (feature present in some files but not others)
1. [FEATURE] Missing in: file1, file2 — present in: file3, file4, file5
2. ...

## LOGIC ERRORS (code exists but does wrong thing)
1. [FILE:LINE] Description — expected: X, actual: Y
2. ...

## MINOR ISSUES (style, naming, dead code)
1. ...

## SUMMARY
Total critical: N
Total consistency: N
Total missing: N
Total logic: N
```

Be thorough. Read every line. The previous developer made many mistakes due to not reading files carefully. Common patterns to watch for:
- Off-by-one in array sizes (e.g., 5 stages declared but 6 exist)
- Hardcoded values that should come from config
- Conditions that are always true/false
- Missing null/bounds checks
- Double-counting in parameter calculations
- Features declared but never called
- Race conditions in async/threaded code
- Checkpoint fields that are saved but never loaded (or vice versa)
