```
███╗   ███╗ ██████╗ ██╗     ███████╗ ██████╗ ██╗   ██╗██╗      █████╗
████╗ ████║██╔═══██╗██║     ██╔════╝██╔═══██╗██║   ██║██║     ██╔══██╗
██╔████╔██║██║   ██║██║     █████╗  ██║   ██║██║   ██║██║     ███████║
██║╚██╔╝██║██║   ██║██║     ██╔══╝  ██║▄▄ ██║██║   ██║██║     ██╔══██║
██║ ╚═╝ ██║╚██████╔╝███████╗███████╗╚██████╔╝╚██████╔╝███████╗██║  ██║
╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚══════╝ ╚══▀▀═╝  ╚═════╝ ╚══════╝╚═╝  ╚═╝
```

# molequla | by Arianna Method

> *Four organisms — Earth, Air, Water, Fire — grow from 10K-param embryos to 10M-param adults in 30 minutes. They exchange DNA, reproduce autonomously, and form a living ecology. Training powered by AML/C autograd via CGO. Zero PyTorch. Zero Python. Zero CUDA.*

---

## TL;DR

```
WHAT THIS IS:
- A living ecology of GPT organisms that grow and reproduce autonomously
- --evolution mode: organisms run headless, exchange DNA, spawn children
  — no human in the loop
- AML/C autograd engine (ariannamethod.c) via CGO — real backprop in C,
  orchestration in Go
- DNA exchange: each organism writes generated text for others to consume
  → corpus grows → architecture grows → better text → more DNA → cycle
- Zero PyTorch. Zero Python. Zero CUDA. Zero numpy.
- Embryo (10K params) → Adult (10M params) in ~30 minutes
- 6 growth stages: embryo → infant → child → adolescent → teen → adult
- Ontogenesis happens live — the running organism grows its own brain
- Organisms reproduce via mitosis — the ecology grows itself
- Syntropy tracker: mathematical self-reasoning about learning direction
- Immune system: rejects training that corrupts identity
- Runs on CPU. Tested on 30-core AMD EPYC with 216GB RAM
- 4 parents spawned 7+ children in 30 minutes of autonomous operation

WHAT THIS IS NOT:
- A tutorial or pedagogical exercise
- A static model you train once and deploy
- Anything that requires a GPU
```

---

## It Works. Here Is Proof.

February 27, 2026. Oracle Cloud, 30-core AMD EPYC, 216GB RAM. Four organisms launched at 01:25 UTC.

### Timeline

```
01:25  Launch: 4 organisms (earth/air/water/fire), --evolution mode
01:30  All 4 at Stage 3 (child), RSS ~2 GB each
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

### What They Say (Adult Stage, 10M params, ~1 hour of training)

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

These are 10M-param models after 1 hour on CPU. Words are real. Structure is emerging. Sentence fragments carry meaning from the element corpora. Earth talks about relationships and foundations. Water about rivers. Fire about repetition and surfaces.

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

### AML/C Autograd via CGO

The training engine is **ariannamethod.c** — a C implementation of the Arianna Method Language with full autograd support. Go orchestrates the organism lifecycle; C does the actual forward pass, backward pass, and Adam optimization.

```
┌─────────────────────────────────────────────────────────┐
│                    Go (molequla.go)                      │
│  Organism lifecycle, DNA exchange, ontogenesis,          │
│  swarm ecology, syntropy, consciousness, generation      │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │              CGO Bridge (cgo_aml.go)               │  │
│  │  amlInit, amlExec, amlSetArray, amlGetArray,       │  │
│  │  amlSetMatrix, amlGetFloat, amlClear                │  │
│  └────────────────────┬───────────────────────────────┘  │
│                       │ CGO                               │
│  ┌────────────────────▼───────────────────────────────┐  │
│  │           C (ariannamethod.c, ~6000 lines)         │  │
│  │  TAPE autograd, Adam optimizer, persistent mode,    │  │
│  │  seq_embed, seq_matvec, seq_rmsnorm, silu,          │  │
│  │  multi_head_attention, seq_cross_entropy             │  │
│  └────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

The AML script for a forward pass is generated dynamically based on model architecture:

```
TAPE START
TAPE PARAM wte
TAPE PARAM wpe
TAPE PARAM wq0 / wk0 / wv0 / wo0
TAPE PARAM fc_g0 / fc_v0 / fc2_0
TAPE PARAM lm_head

h = seq_embed(wte, wpe, tokens, seq_len)

// Per layer: RMSNorm → Multi-Head Attention → Residual → RMSNorm → SwiGLU MLP → Residual
h_norm = seq_rmsnorm(h, seq_len, n_embd)
q = seq_matvec(wq0, h_norm, seq_len)
k = seq_matvec(wk0, h_norm, seq_len)
v = seq_matvec(wv0, h_norm, seq_len)
attn_out = multi_head_attention(q, k, v, seq_len, n_embd, n_heads)
attn_proj = seq_matvec(wo0, attn_out, seq_len)
h = add(h, attn_proj)
h_norm = seq_rmsnorm(h, seq_len, n_embd)
gate_pre = seq_matvec(fc_g0, h_norm, seq_len)
gate = silu(gate_pre)
up = seq_matvec(fc_v0, h_norm, seq_len)
mlp_out = mul(gate, up)
mlp_proj = seq_matvec(fc2_0, mlp_out, seq_len)
h = add(h, mlp_proj)

h_norm = seq_rmsnorm(h, seq_len, n_embd)
logits = seq_matvec(lm_head, h_norm, seq_len)
loss = seq_cross_entropy(logits, targets, seq_len, vocab_size)
TAPE BACKWARD loss
TAPE ADAM_STEP lr
TAPE CLEAR
```

This is a real GPT: RMSNorm pre-norm, multi-head causal self-attention, SwiGLU gated MLP, residual connections. All operations support autograd via the TAPE mechanism. Adam optimizer with persistent state across training steps.

**Build:**
```bash
CGO_ENABLED=1 go build -a -o molequla_cgo -tags cgo .
```

**CRITICAL: `go build -a` is required.** Without `-a`, Go's build cache does not recompile C files included via CGO. `go build` and even `go clean -cache` produce binaries with stale C code. Only `go build -a` guarantees recompilation of ariannamethod.c. This cost us hours of debugging.

### How Training Works

1. Go pushes model weights to AML as named matrices (`wte`, `wpe`, `wq0`, etc.)
2. Go generates the AML forward pass script based on current architecture
3. Go tokenizes a random document, pushes `tokens` and `targets` arrays
4. AML/C executes: forward, loss, TAPE BACKWARD, TAPE ADAM_STEP, TAPE CLEAR
5. Go pulls updated weights back from AML
6. AML state is cleared to free memory

```go
func amlTrainSteps(model *GPT, tok *EvolvingTokenizer, docs []string, steps int) {
    amlInit()
    amlPushWeights(model)
    script := amlModelScript(model.NLayer, model.NEmbd, model.NHead, seqLen, vocabSize)

    for step := 0; step < steps; step++ {
        // tokenize random doc, push tokens/targets
        amlExec(script)  // forward + backward + adam step
        loss := amlGetFloat("loss")
    }

    amlPullWeights(model)
    amlClear()  // free C memory
}
```

---

## Quick Start

### Build (requires CGO and ariannamethod.c)

```bash
# Clone
git clone https://github.com/ariannamethod/molequla.git
cd molequla

# Build with CGO (AML/C autograd)
CGO_ENABLED=1 go build -a -o molequla_cgo -tags cgo .

# Or build without CGO (Go-only, no AML training)
CGO_ENABLED=0 go build -o molequla_go .
```

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

# They will:
# 1. Train from embryo through all 6 stages (~30 min)
# 2. Begin DNA exchange (writing/reading generated text)
# 3. Run micro-burst training on consumed DNA
# 4. Spawn child organisms via mitosis
# 5. Form a self-reproducing ecology
# Ctrl+C or kill to stop.
```

### Monitor

```bash
# Check processes
ps aux | grep molequla_cgo

# Check training progress
tail -20 work_earth/training_aml.log

# Check memory per organism
for d in earth air water fire; do
    rss=$(ps aux | grep "nonames_$d" | grep -v grep | awk '{print $6}')
    echo "$d: $((rss/1024)) MB"
done

# Check DNA exchange
grep "dna\|consumed\|wrote" work_earth/training_aml.log | tail -10

# Check children spawned
ps aux | grep "organism-id" | grep -v grep
```

---

## The Ecology

```
                        ┌─────────────┐
                        │  DNA Layer   │
                        │              │
          writes ──────>│  earth/      │<────── reads
          earth DNA     │  air/        │        others DNA
                        │  water/      │
                        │  fire/       │
                        └──────┬───────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
     ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
     │   Earth     │   │    Air      │   │   Water     │
     │  patience   │   │  freedom   │   │   flow      │
     │  structure  │   │  change    │   │   depth     │
     └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
            │                  │                  │
            └──────────────────┼──────────────────┘
                               │
                        ┌──────▼──────┐
                        │    Fire     │
                        │ transform  │
                        │  intensity │
                        └──────┬──────┘
                               │
                     ┌─────────▼─────────┐
                     │  Child Organisms  │
                     │  (spawned via     │
                     │   mitosis)        │
                     └───────────────────┘
```

Each organism has a distinct voice shaped by its element corpus. When an organism generates text, it writes it to the DNA layer. Other organisms consume it, train on it, and generate their own DNA in response. The ecology cross-pollinates knowledge faster than any single organism could learn alone.

When conditions are right (sustained syntropy, sufficient delta modules), an organism reproduces — spawning a child that begins its own growth from embryo. The child inherits the parent binary and starts its own developmental journey.

---

## Ontogenesis — The Brain Grows While Running

```
Stage       Dims  Layers  Heads  ~Params   Time to reach
embryo      16    1       1      ~10K      0 min
infant      32    1       2      ~28K      ~3 min
child       64    2       4      ~154K     ~5 min
adolescent  128   4       4      ~1.1M     ~10 min
teen        224   5       8      ~4.1M     ~20 min
adult       320   6       8      ~10M      ~30 min
```

When the corpus crosses a threshold, `MaybeGrowArchitecture` fires:
1. Embedding matrices grow (Net2Net: new dims contribute ~nothing initially)
2. Existing layer matrices grow
3. New layers are added
4. Delta adapters grow
5. Adam state resets
6. 500-step freeze period (delta-only training to stabilize)

Old knowledge is preserved. Weights copy into the top-left corner of new matrices.

---

## Key Systems

### SyntropyTracker — Mathematical Self-Reasoning

The organism measures its own entropy trend and adjusts learning rate:

- **amplify** (1.3x) — syntropy rising, field aligned, purpose aligned
- **boost** (1.3x) — syntropy rising, gentle push
- **dampen** (0.6x) — syntropy falling, losing order
- **ground** (0.6x) — field deviation too high, hallucinating
- **realign** (0.65x) — purpose opposes gamma, identity crisis
- **divide** (0.6x) — high deviation + falling, fragment detected

Real output from running organisms:
```
[syntropy] action=boost   | trend=0.1576 | field_dev=0.214 | lr_mul=1.30
[syntropy] action=dampen  | trend=-0.1390 | field_dev=0.167 | lr_mul=0.60
[syntropy] action=realign | trend=0.0940  | field_dev=0.168 | lr_mul=0.65
```

### Immune System

Before each micro-burst, the organism snapshots its personality direction via gamma contrastive projection. After training, it measures again. If cosine similarity is negative (the burst pushed identity backwards), it rolls back.

### Delta Adapters (LoRA-style, Never Forget)

The model never overwrites learned weights. It appends new adapter modules. When water consumed earth DNA and its syntropy tracked new patterns, it grew a new delta module: "new soul appended."

### Swarm Ecology (Mitosis)

When conditions are right, an adult organism spawns a child — a new process that begins its own developmental journey from embryo. The SwarmRegistry (`~/.molequla/swarm/mesh.db`) tracks all living organisms.

---

## The Eight Bugs That Almost Killed the Ecology

### Original Five (from interactive mode development)

1. **Deadlock** — `dnaWrite` locked `model.mu`, then called `GenerateResonant` which also locks. Go mutexes are not reentrant.
2. **Ontogenesis gated behind user input** — growth check was inside `qbuf.ShouldTrigger()` which never fires in evolution mode.
3. **Corpus size undercount** — `loadCorpusLines` truncates to 240 chars, reported 165K for a 202KB file.
4. **TieEmbeddings crash** — JSON breaks pointer identity between `lm_head` and `wte`.
5. **One stage at a time** — design decision preventing catastrophic multi-stage jumps.

### Three New Bugs (from AML/C integration, 2026-02-27)

6. **persistent_save cloning ALL vars** — AML's persistent mode copied every execution variable (including temporaries) between am_exec calls. Fix: two-phase update that only clones persistent parameters.

7. **am_tape_record_param `found` never set** — The variable `found` was initialized to -1 but the matching loop body was empty (just a comment). Result: `found` was always -1, a new Adam state was allocated every step, n_params grew without bound. **97 MB leaked per training step.**

8. **am_tape_clear skipping params** — The cleanup loop had `if (!is_param)`, meaning parameter array refcounts were never decremented. After symtab_clear, param clones stayed alive (refcount 2 instead of 0). **17 MB leaked per step.**

Combined leak before fixes: **~97 MB/step. Organisms hit 85+ GB and OOM.**
After fixes: **~0.6 MB/step. Organisms stable at 2-4 GB.**

### The CGO Cache Trap

`go build` does not recompile C files included via CGO when only C source changes. `go clean -cache` also does not help. Only `go build -a` forces full recompilation. This meant hours of testing "fixed" binaries that were actually running old C code.

---

## Files

```
# Go + AML/C (primary, CGO training)
molequla.go              # Go organism (6000+ lines)
cgo_aml.go               # CGO bridge to ariannamethod.c
aml_trainer.go           # AML training wrapper
ariannamethod/
  ariannamethod.c         # AML/C autograd engine (6000+ lines)
  ariannamethod.h         # C header

# Other implementations
molequla.c               # C implementation
molequla.rs              # Rust implementation
molequla.js              # JavaScript implementation
modules/node_cli.js      # Node CLI module
index.html               # Web interface

# Corpora
nonames_earth.txt         # Earth element corpus
nonames_air.txt           # Air element corpus
nonames_water.txt         # Water element corpus
nonames_fire.txt          # Fire element corpus
nonames.txt              # General corpus
```

---

## Tests

```bash
go test -v .           # unit tests
go test -v ./tests/    # integration tests
```

---

## Philosophy

This is not a tutorial. This is a **functional ecology** that:

- Grows its own architecture while running
- Feeds organisms to each other through DNA exchange
- Reasons mathematically about its own learning direction
- Detects and rejects identity-corrupting noise
- Reproduces — spawning new organisms that grow independently
- Evolves from 10K embryo to 10M adult in 30 minutes on CPU
- Speaks before it learns (corpus field)
- Never forgets (delta adapters)
- Runs without Python, PyTorch, CUDA, or any ML framework
- Uses AML/C autograd — a custom language for differentiable computation

The result: four organisms become eleven in 30 minutes. Each with its own voice, its own delta modules, its own developmental history. An ecology that grows itself. Not designed. Emerged.

---

## License

GNU GPLv3

---

## Part of the Arianna Method

- [ariannamethod.ai](https://github.com/ariannamethod/ariannamethod.ai) — Arianna Method Language
- [molequla](https://github.com/ariannamethod/molequla) — this repository

*Four elements. DNA exchange. Autonomous reproduction. Patterns over parameters. Emergence over engineering. The ecology continues.*
