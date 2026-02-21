//go:build ignore

/*
 * molequla.c
 * A dependency-free, single-file, continually-learning GPT organism in pure C.
 *
 * Compile: gcc -O2 -o molequla molequla.c -lsqlite3 -lpthread -lm
 * With BLAS: gcc -O2 -DUSE_BLAS -o molequla molequla.c -lsqlite3 -lpthread -lm -lopenblas
 * macOS:     gcc -O2 -DUSE_BLAS -o molequla molequla.c -lsqlite3 -lpthread -lm -framework Accelerate
 *
 * In the beginning there was nonames.txt.
 * And it was good. Mostly. Sometimes cursed.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <signal.h>
#include <sys/stat.h>
#include <sqlite3.h>

#ifdef USE_BLAS
  #ifdef __APPLE__
    #ifndef ACCELERATE_NEW_LAPACK
      #define ACCELERATE_NEW_LAPACK
    #endif
    #include <Accelerate/Accelerate.h>
  #else
    #include <cblas.h>
  #endif
  #define HAS_BLAS 1
  /* Thread-local reusable buffer for packing row-per-vec into contiguous for BLAS */
  static __thread double *blas_buf = NULL;
  static __thread int blas_buf_cap = 0;
#else
  #define HAS_BLAS 0
#endif

/* And lo, when the organism speaks, it shall not waste breath building
 * a backward graph it will never use. grad_enabled is mercy for inference. */
static int grad_enabled = 1;

/* ============================================================
 * 0) CONFIG
 * ============================================================ */

typedef struct {
    const char *corpus_path;
    const char *db_path;
    const char *ckpt_path;
    int max_corpus_lines;
    int max_line_chars;
    int min_new_chars;

    int tie_embeddings;
    int n_layer;
    int n_embd;
    int n_head;
    int block_size;

    int warmup_steps;
    int micro_steps;
    double learning_rate;
    double beta1, beta2, eps_adam;
    double grad_clip;
    int freeze_base_after_warmup;
    int batch_size;

    int delta_rank;
    int max_delta_modules;
    double delta_grow_prob;

    double temperature;
    int top_k;
    double top_p;
    double min_p;           /* GPT-3/4 style: filter tokens below min_p * max_prob */
    double typical_p;       /* Typical sampling: prefer tokens with typical information content */
    int max_gen_tokens;
    int min_gen_tokens;
    int repetition_guard;

    int enable_bpe_after_chars;
    int bpe_num_merges;
    int bpe_retrain_every_chars;

    double train_tick_seconds;

    /* hybrid attention */
    const char *head_types[8];
    int n_head_types;
    double hybrid_alpha_init;

    /* gamma */
    double gamma_sparsity_threshold;

    /* noise immune system */
    double noise_drift_threshold;
    double gamma_min_magnitude;   /* skip immune check when gamma direction is near-zero */

    /* entropy temperature */
    double entropy_low, entropy_high;
    double entropy_temp_boost, entropy_temp_focus;

    /* corpus field */
    int corpus_gen_max_tokens;
    double corpus_fade_k;            /* sigmoid steepness for corpus->model transition */
    double corpus_fade_threshold;    /* entropy at which blend is 50/50 */
    int cooccur_window_size;         /* co-occurrence proximity window (Stanley-style) */
    double user_boost_strength;      /* how strongly user's recent words are boosted */
    double user_boost_decay;         /* per-generation decay of user word boost */

    /* quantum buffer */
    int qb_min_bytes;
    double qb_min_novelty;
    double qb_cooldown_seconds;

    /* syntropy tracker (mathematical self-awareness) */
    int syntropy_window;              /* rolling window for syntropy trend */
    double field_deviation_ceiling;   /* KL divergence above this = drifted too far */
    double field_deviation_floor;     /* below this = not learning, just parroting */
    double syntropy_lr_boost;         /* boost LR when syntropy is rising */
    double syntropy_lr_dampen;        /* dampen LR when syntropy is falling */
    double syntropy_delta_grow_boost; /* higher delta grow prob when syntropy is good */

    /* Phase 1: cosine LR schedule */
    double lr_min;
    int max_total_steps;
    int cosine_warmup_steps;

    /* Phase 1: gradient accumulation */
    int accum_steps;

    /* Phase 3A: ontogenesis — growth stages */
    /* Each stage: (corpus_chars_threshold, n_embd, n_layer, n_head) */
    int growth_stages[6][4];
    int n_growth_stages;
    int freeze_after_growth_steps;
    double post_growth_lr_scale;      /* LR multiplier during freeze period */

    /* frequency / presence penalty */
    double freq_penalty;
    double presence_penalty;

    /* consciousness: per-token dissonance feedback */
    double dissonance_ema_alpha;       /* EMA smoothing for entropy within generation */
    double dissonance_spike_k;         /* temp multiplier when entropy spikes */
    double dissonance_drop_k;          /* temp multiplier when entropy drops */
    double dissonance_spike_threshold; /* entropy/EMA ratio triggering spike */
    double dissonance_drop_threshold;  /* entropy/EMA ratio triggering drop */

    /* consciousness: pattern breaking (anti-field generation) */
    double anti_field_prob;     /* probability of pure-model token (bypass corpus) */
    int anti_field_min_step;    /* don't anti-field before this many tokens */

    /* consciousness: conscience (self-editing) */
    int conscience_window;      /* rolling window for generation entropy trend */
    double conscience_decay;    /* deltaAlphaScale reduction factor */
    double conscience_recovery; /* deltaAlphaScale recovery factor */
    double conscience_floor;    /* minimum deltaAlphaScale */
} Config;

static Config CFG = {
    .corpus_path = "nonames.txt",
    .db_path = "memory.sqlite3",
    .ckpt_path = "molequla.ckpt",
    .max_corpus_lines = 8000,
    .max_line_chars = 240,
    .min_new_chars = 480,
    .tie_embeddings = 1,
    .n_layer = 1,
    .n_embd = 16,
    .n_head = 1,
    .block_size = 96,
    .warmup_steps = 1200,
    .micro_steps = 32,
    .learning_rate = 0.01,
    .beta1 = 0.9, .beta2 = 0.99, .eps_adam = 1e-8,
    .grad_clip = 1.0,
    .freeze_base_after_warmup = 1,
    .batch_size = 4,
    .delta_rank = 8,
    .max_delta_modules = 12,
    .delta_grow_prob = 0.08,
    .temperature = 0.85,
    .top_k = 40,
    .top_p = 0.92,
    .min_p = 0.06,
    .typical_p = 0.95,
    .max_gen_tokens = 180,
    .min_gen_tokens = 16,
    .repetition_guard = 4,
    .enable_bpe_after_chars = 20000,
    .bpe_num_merges = 384,
    .bpe_retrain_every_chars = 4000,
    .train_tick_seconds = 0.25,
    .head_types = {"content", NULL, NULL, NULL},
    .n_head_types = 1,
    .hybrid_alpha_init = 0.5,
    .gamma_sparsity_threshold = 0.01,
    .noise_drift_threshold = -0.1,
    .gamma_min_magnitude = 1e-6,
    .entropy_low = 0.5, .entropy_high = 1.5,
    .entropy_temp_boost = 1.2, .entropy_temp_focus = 0.8,
    .corpus_gen_max_tokens = 120,
    .corpus_fade_k = 3.0,
    .corpus_fade_threshold = 1.5,
    .cooccur_window_size = 5,
    .user_boost_strength = 0.3,
    .user_boost_decay = 0.7,
    .qb_min_bytes = 1024,
    .qb_min_novelty = 0.15,
    .qb_cooldown_seconds = 60.0,
    .syntropy_window = 8,
    .field_deviation_ceiling = 12.0,
    .field_deviation_floor = 0.1,
    .syntropy_lr_boost = 1.3,
    .syntropy_lr_dampen = 0.6,
    .syntropy_delta_grow_boost = 0.15,
    .lr_min = 0.001,
    .max_total_steps = 50000,
    .cosine_warmup_steps = 200,
    .accum_steps = 1,

    /* Phase 3A: ontogenesis growth stages */
    .growth_stages = {
        {0,      16, 1, 1},    /* embryo: ~10K params */
        {20000,  32, 1, 2},    /* infant: ~28K params */
        {50000,  64, 2, 4},    /* child: ~154K params */
        {200000, 128, 4, 4},   /* adolescent: ~1.1M params */
        {350000, 224, 5, 8},   /* teen: ~4.1M params */
        {500000, 320, 6, 8},   /* adult: ~10M params */
    },
    .n_growth_stages = 6,
    .freeze_after_growth_steps = 500,
    .post_growth_lr_scale = 0.3,

    /* frequency / presence penalty */
    .freq_penalty = 0.1,
    .presence_penalty = 0.1,

    /* consciousness defaults */
    .dissonance_ema_alpha = 0.3,
    .dissonance_spike_k = 0.8,
    .dissonance_drop_k = 1.2,
    .dissonance_spike_threshold = 1.5,
    .dissonance_drop_threshold = 0.5,
    .anti_field_prob = 0.05,
    .anti_field_min_step = 8,
    .conscience_window = 8,
    .conscience_decay = 0.95,
    .conscience_recovery = 1.005,
    .conscience_floor = 0.3,
};

/* Head types helper: compute head_types array for a given number of heads.
 * Writes into the global CFG.head_types and updates CFG.n_head_types.
 * 1→content, 2→content+hybrid, 4→2c+2h, 8→4c+4h */
static void head_types_for_n_head(int n) {
    if (n <= 0) n = 1;
    if (n > 8) n = 8; /* max 8 slots in head_types array */
    if (n <= 1) {
        CFG.head_types[0] = "content";
        CFG.n_head_types = 1;
    } else if (n == 2) {
        CFG.head_types[0] = "content";
        CFG.head_types[1] = "hybrid";
        CFG.n_head_types = 2;
    } else {
        /* majority content, rest hybrid: 5→3c+2h, 8→4c+4h */
        int half = (n + 1) / 2;
        for (int i = 0; i < half; i++) CFG.head_types[i] = "content";
        for (int i = half; i < n; i++) CFG.head_types[i] = "hybrid";
        CFG.n_head_types = n;
    }
}

/* ============================================================
 * 0.5) RNG — xorshift64, because rand() is for cowards
 * ============================================================ */

static unsigned long long rng_state = 42;

static double rand_uniform(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (double)(rng_state & 0x7FFFFFFFFFFFFFFFULL) / (double)0x7FFFFFFFFFFFFFFFULL;
}

static double rand_normal(void) {
    double u1 = rand_uniform();
    double u2 = rand_uniform();
    if (u1 < 1e-15) u1 = 1e-15;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static int rand_int(int n) {
    return (int)(rand_uniform() * n) % n;
}

/* ============================================================
 * 0.6) DYNAMIC ARRAYS
 * ============================================================ */

typedef struct { char **items; int len, cap; } StrArr;
typedef struct { int *items; int len, cap; } IntArr;

static void sa_push(StrArr *a, const char *s) {
    if (a->len >= a->cap) {
        a->cap = a->cap ? a->cap * 2 : 16;
        void *tmp = realloc(a->items, sizeof(char*) * a->cap);
        if (!tmp) { fprintf(stderr, "[sa_push] realloc failed\n"); return; }
        a->items = tmp;
    }
    a->items[a->len++] = strdup(s);
}

static void sa_free(StrArr *a) {
    for (int i = 0; i < a->len; i++) free(a->items[i]);
    free(a->items);
    a->items = NULL; a->len = a->cap = 0;
}

static void ia_push(IntArr *a, int v) {
    if (a->len >= a->cap) {
        a->cap = a->cap ? a->cap * 2 : 16;
        void *tmp = realloc(a->items, sizeof(int) * a->cap);
        if (!tmp) { fprintf(stderr, "[ia_push] realloc failed\n"); return; }
        a->items = tmp;
    }
    a->items[a->len++] = v;
}

static void ia_free(IntArr *a) {
    free(a->items);
    a->items = NULL; a->len = a->cap = 0;
}

/* ============================================================
 * 1) ARENA ALLOCATOR — for autograd graphs
 * ============================================================ */

#define ARENA_SIZE (256 * 1024 * 1024) /* 256 MB */

typedef struct {
    char *buf;
    size_t used, cap;
} Arena;

static Arena arena_new(size_t cap) {
    Arena a;
    a.buf = malloc(cap);
    a.used = 0;
    a.cap = cap;
    return a;
}

static void *arena_alloc(Arena *a, size_t size) {
    size = (size + 7) & ~(size_t)7; /* align to 8 bytes */
    if (a->used + size > a->cap) {
        fprintf(stderr, "arena: out of memory (%zu/%zu)\n", a->used + size, a->cap);
        exit(1);
    }
    void *p = a->buf + a->used;
    a->used += size;
    memset(p, 0, size);
    return p;
}

static void arena_reset(Arena *a) { a->used = 0; }
static void arena_destroy(Arena *a) { free(a->buf); }

static Arena G_arena; /* global arena for autograd */

/* ============================================================
 * 2) AUTOGRAD — Node = Vec or Scalar (len=1)
 * ============================================================ */

typedef struct Node Node;
typedef void (*BackFn)(Node *self);

struct Node {
    double *data;
    double *grad;
    int len;
    Node **children;
    int n_children;
    BackFn backward;
    void *ctx;
    int visited;
};

static Node *node_new(int len) {
    Node *n = arena_alloc(&G_arena, sizeof(Node));
    n->data = arena_alloc(&G_arena, sizeof(double) * len);
    n->grad = arena_alloc(&G_arena, sizeof(double) * len);
    n->len = len;
    return n;
}

static void node_set_children(Node *n, Node **kids, int count) {
    n->children = arena_alloc(&G_arena, sizeof(Node*) * count);
    memcpy(n->children, kids, sizeof(Node*) * count);
    n->n_children = count;
}

/* Wrap persistent weight data as a node (data/grad are NOT arena-allocated) */
static Node *node_wrap(double *data, double *grad, int len) {
    Node *n = arena_alloc(&G_arena, sizeof(Node));
    n->data = data;
    n->grad = grad;
    n->len = len;
    return n;
}

/* --- Vec ops --- */

typedef struct { Node *a, *b; int len; } BinCtx;

static void back_add(Node *self) {
    BinCtx *c = self->ctx;
    for (int i = 0; i < c->len; i++) {
        c->a->grad[i] += self->grad[i];
        c->b->grad[i] += self->grad[i];
    }
}

static Node *vec_add(Node *a, Node *b) {
    int n = a->len;
    Node *out = node_new(n);
    for (int i = 0; i < n; i++) out->data[i] = a->data[i] + b->data[i];
    if (grad_enabled) {
        BinCtx *c = arena_alloc(&G_arena, sizeof(BinCtx));
        c->a = a; c->b = b; c->len = n;
        out->ctx = c;
        out->backward = back_add;
        Node *kids[] = {a, b};
        node_set_children(out, kids, 2);
    }
    return out;
}

static void back_sub(Node *self) {
    BinCtx *c = self->ctx;
    for (int i = 0; i < c->len; i++) {
        c->a->grad[i] += self->grad[i];
        c->b->grad[i] -= self->grad[i];
    }
}

static Node *vec_sub(Node *a, Node *b) {
    int n = a->len;
    Node *out = node_new(n);
    for (int i = 0; i < n; i++) out->data[i] = a->data[i] - b->data[i];
    if (grad_enabled) {
        BinCtx *c = arena_alloc(&G_arena, sizeof(BinCtx));
        c->a = a; c->b = b; c->len = n;
        out->ctx = c;
        out->backward = back_sub;
        Node *kids[] = {a, b};
        node_set_children(out, kids, 2);
    }
    return out;
}

static void back_mul_vec(Node *self) {
    BinCtx *c = self->ctx;
    for (int i = 0; i < c->len; i++) {
        c->a->grad[i] += c->b->data[i] * self->grad[i];
        c->b->grad[i] += c->a->data[i] * self->grad[i];
    }
}

static Node *vec_mul(Node *a, Node *b) {
    int n = a->len;
    Node *out = node_new(n);
    for (int i = 0; i < n; i++) out->data[i] = a->data[i] * b->data[i];
    if (grad_enabled) {
        BinCtx *c = arena_alloc(&G_arena, sizeof(BinCtx));
        c->a = a; c->b = b; c->len = n;
        out->ctx = c;
        out->backward = back_mul_vec;
        Node *kids[] = {a, b};
        node_set_children(out, kids, 2);
    }
    return out;
}

typedef struct { Node *a; double s; int len; } ScaleCtx;

static void back_scale(Node *self) {
    ScaleCtx *c = self->ctx;
    for (int i = 0; i < c->len; i++)
        c->a->grad[i] += c->s * self->grad[i];
}

static Node *vec_scale(Node *a, double s) {
    int n = a->len;
    Node *out = node_new(n);
    for (int i = 0; i < n; i++) out->data[i] = a->data[i] * s;
    if (grad_enabled) {
        ScaleCtx *c = arena_alloc(&G_arena, sizeof(ScaleCtx));
        c->a = a; c->s = s; c->len = n;
        out->ctx = c;
        out->backward = back_scale;
        Node *kids[] = {a};
        node_set_children(out, kids, 1);
    }
    return out;
}

static void back_relu(Node *self) {
    BinCtx *c = self->ctx; /* reuse: a = input */
    for (int i = 0; i < c->len; i++)
        if (c->a->data[i] > 0) c->a->grad[i] += self->grad[i];
}

static Node *vec_relu(Node *a) {
    int n = a->len;
    Node *out = node_new(n);
    for (int i = 0; i < n; i++) out->data[i] = a->data[i] > 0 ? a->data[i] : 0;
    if (grad_enabled) {
        BinCtx *c = arena_alloc(&G_arena, sizeof(BinCtx));
        c->a = a; c->len = n;
        out->ctx = c;
        out->backward = back_relu;
        Node *kids[] = {a};
        node_set_children(out, kids, 1);
    }
    return out;
}

/* SiLU (Swish): silu(x) = x * sigmoid(x) — real SwiGLU activation */
static void back_silu(Node *self) {
    BinCtx *c = self->ctx;
    for (int i = 0; i < c->len; i++) {
        double x = c->a->data[i];
        double sig = 1.0 / (1.0 + exp(-x));
        c->a->grad[i] += (sig + x * sig * (1.0 - sig)) * self->grad[i];
    }
}

static Node *vec_silu(Node *a) {
    int n = a->len;
    Node *out = node_new(n);
    for (int i = 0; i < n; i++) {
        double x = a->data[i];
        double sig = 1.0 / (1.0 + exp(-x));
        out->data[i] = x * sig;
    }
    if (grad_enabled) {
        BinCtx *c = arena_alloc(&G_arena, sizeof(BinCtx));
        c->a = a; c->len = n;
        out->ctx = c;
        out->backward = back_silu;
        Node *kids[] = {a};
        node_set_children(out, kids, 1);
    }
    return out;
}

/* Dot product: returns scalar (len=1) */
typedef struct { Node *a, *b; int len; } DotCtx;

static void back_dot(Node *self) {
    DotCtx *c = self->ctx;
    double g = self->grad[0];
    for (int i = 0; i < c->len; i++) {
        c->a->grad[i] += c->b->data[i] * g;
        c->b->grad[i] += c->a->data[i] * g;
    }
}

static Node *vec_dot(Node *a, Node *b) {
    int n = a->len;
    double val = 0;
    for (int i = 0; i < n; i++) val += a->data[i] * b->data[i];
    Node *out = node_new(1);
    out->data[0] = val;
    if (grad_enabled) {
        DotCtx *c = arena_alloc(&G_arena, sizeof(DotCtx));
        c->a = a; c->b = b; c->len = n;
        out->ctx = c;
        out->backward = back_dot;
        Node *kids[] = {a, b};
        node_set_children(out, kids, 2);
    }
    return out;
}

/* MeanSq: scalar = mean(x^2) */
typedef struct { Node *a; int len; } MeanSqCtx;

static void back_meansq(Node *self) {
    MeanSqCtx *c = self->ctx;
    double g = self->grad[0];
    double nf = (double)c->len;
    for (int i = 0; i < c->len; i++)
        c->a->grad[i] += (2.0 * c->a->data[i] / nf) * g;
}

static Node *vec_meansq(Node *a) {
    int n = a->len;
    double val = 0;
    for (int i = 0; i < n; i++) val += a->data[i] * a->data[i];
    val /= (double)n;
    Node *out = node_new(1);
    out->data[0] = val;
    if (grad_enabled) {
        MeanSqCtx *c = arena_alloc(&G_arena, sizeof(MeanSqCtx));
        c->a = a; c->len = n;
        out->ctx = c;
        out->backward = back_meansq;
        Node *kids[] = {a};
        node_set_children(out, kids, 1);
    }
    return out;
}

/* Slice: out = a[start:end] */
typedef struct { Node *a; int start, end; } SliceCtx;

static void back_slice(Node *self) {
    SliceCtx *c = self->ctx;
    for (int i = 0, j = c->start; j < c->end; i++, j++)
        c->a->grad[j] += self->grad[i];
}

static Node *vec_slice(Node *a, int start, int end) {
    int n = end - start;
    Node *out = node_new(n);
    memcpy(out->data, a->data + start, sizeof(double) * n);
    if (grad_enabled) {
        SliceCtx *c = arena_alloc(&G_arena, sizeof(SliceCtx));
        c->a = a; c->start = start; c->end = end;
        out->ctx = c;
        out->backward = back_slice;
        Node *kids[] = {a};
        node_set_children(out, kids, 1);
    }
    return out;
}

/* Element: extract one element as scalar node (len=1) with gradient flow */
/* And lo, one number shall be plucked from the vector, and gradients shall follow. */
typedef struct { Node *a; int idx; } ElemCtx;

static void back_elem(Node *self) {
    ElemCtx *c = self->ctx;
    c->a->grad[c->idx] += self->grad[0];
}

static Node *vec_element(Node *a, int idx) {
    Node *out = node_new(1);
    out->data[0] = a->data[idx];
    if (grad_enabled) {
        ElemCtx *c = arena_alloc(&G_arena, sizeof(ElemCtx));
        c->a = a; c->idx = idx;
        out->ctx = c;
        out->backward = back_elem;
        Node *kids[] = {a};
        node_set_children(out, kids, 1);
    }
    return out;
}

/* Scalar mul: s1 * s2 (both scalar nodes) */
static void back_scalar_mul(Node *self) {
    Node *a = self->children[0], *b = self->children[1];
    a->grad[0] += b->data[0] * self->grad[0];
    b->grad[0] += a->data[0] * self->grad[0];
}

static Node *scalar_mul(Node *a, Node *b) {
    Node *out = node_new(1);
    out->data[0] = a->data[0] * b->data[0];
    if (grad_enabled) {
        out->backward = back_scalar_mul;
        Node *kids[] = {a, b};
        node_set_children(out, kids, 2);
    }
    return out;
}

/* Concat: join multiple vecs */
typedef struct { Node **vecs; int n_vecs; int *offsets; } ConcatCtx;

static void back_concat(Node *self) {
    ConcatCtx *c = self->ctx;
    for (int v = 0; v < c->n_vecs; v++) {
        int off = c->offsets[v];
        int len = c->vecs[v]->len;
        for (int i = 0; i < len; i++)
            c->vecs[v]->grad[i] += self->grad[off + i];
    }
}

static Node *vec_concat(Node **vecs, int n_vecs) {
    int total = 0;
    for (int i = 0; i < n_vecs; i++) total += vecs[i]->len;
    Node *out = node_new(total);
    int off = 0;
    int *offsets = arena_alloc(&G_arena, sizeof(int) * n_vecs);
    for (int i = 0; i < n_vecs; i++) {
        offsets[i] = off;
        memcpy(out->data + off, vecs[i]->data, sizeof(double) * vecs[i]->len);
        off += vecs[i]->len;
    }
    if (grad_enabled) {
        ConcatCtx *c = arena_alloc(&G_arena, sizeof(ConcatCtx));
        c->vecs = arena_alloc(&G_arena, sizeof(Node*) * n_vecs);
        memcpy(c->vecs, vecs, sizeof(Node*) * n_vecs);
        c->n_vecs = n_vecs;
        c->offsets = offsets;
        out->ctx = c;
        out->backward = back_concat;
        node_set_children(out, vecs, n_vecs);
    }
    return out;
}

/* Scalar add */
static void back_scalar_add(Node *self) {
    double g = self->grad[0];
    self->children[0]->grad[0] += g;
    self->children[1]->grad[0] += g;
}

static Node *scalar_add(Node *a, Node *b) {
    Node *out = node_new(1);
    out->data[0] = a->data[0] + b->data[0];
    if (grad_enabled) {
        out->backward = back_scalar_add;
        Node *kids[] = {a, b};
        node_set_children(out, kids, 2);
    }
    return out;
}

/* Scalar mul by float */
static void back_scalar_mulf(Node *self) {
    ScaleCtx *c = self->ctx;
    c->a->grad[0] += c->s * self->grad[0];
}

static Node *scalar_mulf(Node *a, double f) {
    Node *out = node_new(1);
    out->data[0] = a->data[0] * f;
    if (grad_enabled) {
        ScaleCtx *c = arena_alloc(&G_arena, sizeof(ScaleCtx));
        c->a = a; c->s = f;
        out->ctx = c;
        out->backward = back_scalar_mulf;
        Node *kids[] = {a};
        node_set_children(out, kids, 1);
    }
    return out;
}

/* Scalar sigmoid: σ(x) = 1/(1+exp(-x)) with gradient flow */
static void back_scalar_sigmoid(Node *self) {
    double sig = self->data[0];
    self->children[0]->grad[0] += sig * (1.0 - sig) * self->grad[0];
}

static Node *scalar_sigmoid(Node *a) {
    Node *out = node_new(1);
    out->data[0] = 1.0 / (1.0 + exp(-a->data[0]));
    if (grad_enabled) {
        out->backward = back_scalar_sigmoid;
        Node *kids[] = {a};
        node_set_children(out, kids, 1);
    }
    return out;
}

/* Scalar add float: a + f (constant, gradient only to a) */
static Node *scalar_addf(Node *a, double f) {
    Node *out = node_new(1);
    out->data[0] = a->data[0] + f;
    if (grad_enabled) {
        ScaleCtx *c = arena_alloc(&G_arena, sizeof(ScaleCtx));
        c->a = a; c->s = 1.0;
        out->ctx = c;
        out->backward = back_scalar_mulf; /* same: grad flows 1:1 to a */
        Node *kids[] = {a};
        node_set_children(out, kids, 1);
    }
    return out;
}

/* --- Backward (topological sort) --- */
/* And lo, the graph shall be walked backwards, like a salmon with regrets. */

#define MAX_TOPO 262144

static void backward(Node *root) {
    /* Heap-allocated to avoid stack overflow in threads */
    Node **topo = (Node **)malloc(MAX_TOPO * sizeof(Node *));
    Node **stack = (Node **)malloc(MAX_TOPO * sizeof(Node *));
    int topo_len = 0;
    int stack_len = 0;

    stack[stack_len++] = root;

    while (stack_len > 0) {
        Node *n = stack[stack_len - 1];
        if (n->visited == 1) {
            stack_len--;
            if (n->visited != 2) {
                n->visited = 2;
                if (topo_len < MAX_TOPO) topo[topo_len++] = n;
            }
            continue;
        }
        n->visited = 1;
        for (int i = 0; i < n->n_children; i++) {
            if (n->children[i] && n->children[i]->visited == 0) {
                if (stack_len < MAX_TOPO) stack[stack_len++] = n->children[i];
            }
        }
    }

    root->grad[0] = 1.0;
    for (int i = topo_len - 1; i >= 0; i--) {
        if (topo[i]->backward)
            topo[i]->backward(topo[i]);
    }
    free(topo);
    free(stack);
}

/* ============================================================
 * 3) HIGH-LEVEL OPS
 * ============================================================ */

/* Persistent weight matrix (NOT arena allocated) */
typedef struct {
    double **row_data; /* [nout][nin] */
    double **row_grad; /* [nout][nin] */
    int nout, nin;
} MatrixParam;

static MatrixParam *mat_new(int nout, int nin, double std) {
    MatrixParam *m = calloc(1, sizeof(MatrixParam));
    m->nout = nout; m->nin = nin;
    m->row_data = calloc(nout, sizeof(double*));
    m->row_grad = calloc(nout, sizeof(double*));
    for (int i = 0; i < nout; i++) {
        m->row_data[i] = calloc(nin, sizeof(double));
        m->row_grad[i] = calloc(nin, sizeof(double));
        for (int j = 0; j < nin; j++)
            m->row_data[i][j] = rand_normal() * std;
    }
    return m;
}

static void mat_grow_rows(MatrixParam *m, int new_nout, double std) {
    if (new_nout <= m->nout) return;
    void *tmp_data = realloc(m->row_data, sizeof(double*) * new_nout);
    void *tmp_grad = realloc(m->row_grad, sizeof(double*) * new_nout);
    if (!tmp_data || !tmp_grad) {
        fprintf(stderr, "[mat_grow_rows] realloc failed\n");
        if (tmp_data) m->row_data = tmp_data;
        if (tmp_grad) m->row_grad = tmp_grad;
        return;
    }
    m->row_data = tmp_data;
    m->row_grad = tmp_grad;
    for (int i = m->nout; i < new_nout; i++) {
        m->row_data[i] = calloc(m->nin, sizeof(double));
        m->row_grad[i] = calloc(m->nin, sizeof(double));
        for (int j = 0; j < m->nin; j++)
            m->row_data[i][j] = rand_normal() * std;
    }
    m->nout = new_nout;
}

/* Grow columns (input dimension) of a matrix: extend each row with gaussian noise */
static void mat_grow_cols(MatrixParam *m, int new_nin, double std) {
    if (new_nin <= m->nin) return;
    for (int i = 0; i < m->nout; i++) {
        void *tmp_d = realloc(m->row_data[i], sizeof(double) * new_nin);
        void *tmp_g = realloc(m->row_grad[i], sizeof(double) * new_nin);
        if (!tmp_d || !tmp_g) {
            fprintf(stderr, "[mat_grow_cols] realloc failed at row %d\n", i);
            if (tmp_d) m->row_data[i] = tmp_d;
            if (tmp_g) m->row_grad[i] = tmp_g;
            return;
        }
        m->row_data[i] = tmp_d;
        m->row_grad[i] = tmp_g;
        for (int j = m->nin; j < new_nin; j++) {
            m->row_data[i][j] = rand_normal() * std;
            m->row_grad[i][j] = 0.0;
        }
    }
    m->nin = new_nin;
}

/* Grow both dimensions: cols first (so new rows get full width), then rows */
static void mat_grow(MatrixParam *m, int new_nout, int new_nin, double std) {
    mat_grow_cols(m, new_nin, std);
    mat_grow_rows(m, new_nout, std);
}

/* Matvec: out = M @ x */
typedef struct { MatrixParam *m; Node *x; int nout, nin; } MatvecCtx;

static void back_matvec(Node *self) {
    MatvecCtx *c = self->ctx;
    for (int i = 0; i < c->nout; i++) {
        double g = self->grad[i];
        for (int j = 0; j < c->nin; j++) {
            c->m->row_grad[i][j] += g * c->x->data[j];
            c->x->grad[j] += g * c->m->row_data[i][j];
        }
    }
}

static Node *mat_matvec(MatrixParam *m, Node *x) {
    int nout = m->nout, nin = x->len;
    Node *out = node_new(nout);
#if HAS_BLAS
    if (nout * nin >= 256) {
        /* Pack row pointers into contiguous thread-local buffer for cblas_dgemv */
        int needed = nout * nin;
        if (needed > blas_buf_cap) {
            free(blas_buf);
            blas_buf = malloc(sizeof(double) * needed);
            blas_buf_cap = needed;
        }
        for (int i = 0; i < nout; i++)
            memcpy(blas_buf + i * nin, m->row_data[i], nin * sizeof(double));
        cblas_dgemv(CblasRowMajor, CblasNoTrans, nout, nin,
                    1.0, blas_buf, nin, x->data, 1, 0.0, out->data, 1);
    } else
#endif
    {
        for (int i = 0; i < nout; i++) {
            double s = 0;
            for (int j = 0; j < nin; j++) s += m->row_data[i][j] * x->data[j];
            out->data[i] = s;
        }
    }

    if (grad_enabled) {
        /* Wrap each row as a node for the graph */
        Node **kids = arena_alloc(&G_arena, sizeof(Node*) * (nout + 1));
        for (int i = 0; i < nout; i++)
            kids[i] = node_wrap(m->row_data[i], m->row_grad[i], nin);
        kids[nout] = x;
        node_set_children(out, kids, nout + 1);

        MatvecCtx *c = arena_alloc(&G_arena, sizeof(MatvecCtx));
        c->m = m; c->x = x; c->nout = nout; c->nin = nin;
        out->ctx = c;
        out->backward = back_matvec;
    }
    return out;
}

/* RMSNorm */
typedef struct { Node *x; double scale_val; double ms_data; int len; } RMSCtx;

static void back_rmsnorm(Node *self) {
    RMSCtx *c = self->ctx;
    double s = c->scale_val;
    double ds = -0.5 * pow(c->ms_data + 1e-5, -1.5);
    double cross = 0;
    for (int j = 0; j < c->len; j++) cross += self->grad[j] * c->x->data[j];
    double nf = (double)c->len;
    for (int i = 0; i < c->len; i++) {
        c->x->grad[i] += s * self->grad[i];
        c->x->grad[i] += cross * ds * (2.0 * c->x->data[i] / nf);
    }
}

static Node *rmsnorm(Node *x) {
    int n = x->len;
    double ms = 0;
    for (int i = 0; i < n; i++) ms += x->data[i] * x->data[i];
    ms /= (double)n;
    double scale = pow(ms + 1e-5, -0.5);

    Node *out = node_new(n);
    for (int i = 0; i < n; i++) out->data[i] = x->data[i] * scale;

    if (grad_enabled) {
        RMSCtx *c = arena_alloc(&G_arena, sizeof(RMSCtx));
        c->x = x; c->scale_val = scale; c->ms_data = ms; c->len = n;
        out->ctx = c;
        out->backward = back_rmsnorm;
        Node *kids[] = {x};
        node_set_children(out, kids, 1);
    }
    return out;
}

/* Cross-entropy loss */
typedef struct { Node *logits; double *probs; int target, vocab; } CECtx;

static void back_ce(Node *self) {
    CECtx *c = self->ctx;
    double g = self->grad[0];
    for (int i = 0; i < c->vocab; i++) {
        double ind = (i == c->target) ? 1.0 : 0.0;
        c->logits->grad[i] += (c->probs[i] - ind) * g;
    }
}

static Node *cross_entropy(Node *logits, int target) {
    int n = logits->len;
    double max_val = logits->data[0];
    for (int i = 1; i < n; i++) if (logits->data[i] > max_val) max_val = logits->data[i];

    double *probs = arena_alloc(&G_arena, sizeof(double) * n);
    double exp_sum = 0;
    for (int i = 0; i < n; i++) {
        probs[i] = exp(logits->data[i] - max_val);
        exp_sum += probs[i];
    }
    for (int i = 0; i < n; i++) probs[i] /= exp_sum;

    double loss = log(exp_sum) + max_val - logits->data[target];
    Node *out = node_new(1);
    out->data[0] = loss;

    if (grad_enabled) {
        CECtx *c = arena_alloc(&G_arena, sizeof(CECtx));
        c->logits = logits; c->probs = probs; c->target = target; c->vocab = n;
        out->ctx = c;
        out->backward = back_ce;
        Node *kids[] = {logits};
        node_set_children(out, kids, 1);
    }
    return out;
}

/* Scalar softmax over array of scalar nodes */
typedef struct { Node **logits; double *probs; int n; } SoftmaxCtx;

static void back_softmax_i(Node *self) {
    SoftmaxCtx *c = self->ctx;
    /* find which index this is */
    int ii = -1;
    for (int i = 0; i < c->n; i++) {
        /* hack: compare data pointer */
        if (fabs(self->data[0] - c->probs[i]) < 1e-15) { ii = i; break; }
    }
    if (ii < 0) return;
    double g = self->grad[0];
    for (int j = 0; j < c->n; j++) {
        if (j == ii)
            c->logits[j]->grad[0] += g * c->probs[ii] * (1.0 - c->probs[ii]);
        else
            c->logits[j]->grad[0] += g * (-c->probs[ii] * c->probs[j]);
    }
}

static void scalar_softmax(Node **logits, int n, Node **out) {
    double max_val = logits[0]->data[0];
    for (int i = 1; i < n; i++) if (logits[i]->data[0] > max_val) max_val = logits[i]->data[0];
    double *exps = arena_alloc(&G_arena, sizeof(double) * n);
    double total = 0;
    for (int i = 0; i < n; i++) { exps[i] = exp(logits[i]->data[0] - max_val); total += exps[i]; }
    double *probs = arena_alloc(&G_arena, sizeof(double) * n);
    for (int i = 0; i < n; i++) probs[i] = exps[i] / total;

    SoftmaxCtx *shared = grad_enabled ? arena_alloc(&G_arena, sizeof(SoftmaxCtx)) : NULL;
    if (grad_enabled) {
        shared->logits = logits; shared->probs = probs; shared->n = n;
    }

    for (int i = 0; i < n; i++) {
        out[i] = node_new(1);
        out[i]->data[0] = probs[i];
        if (grad_enabled) {
            out[i]->ctx = shared;
            out[i]->backward = back_softmax_i;
            node_set_children(out[i], logits, n);
        }
    }
}

/* Attention weighted sum: out = sum_t(w[t] * v[t]) */
typedef struct { Node **weights; Node **values; int T, dim; } AttnSumCtx;

static void back_attn_sum(Node *self) {
    AttnSumCtx *c = self->ctx;
    for (int t = 0; t < c->T; t++)
        for (int j = 0; j < c->dim; j++) {
            c->weights[t]->grad[0] += c->values[t]->data[j] * self->grad[j];
            c->values[t]->grad[j] += c->weights[t]->data[0] * self->grad[j];
        }
}

static Node *attn_weighted_sum(Node **weights, Node **values, int T) {
    int dim = values[0]->len;
    Node *out = node_new(dim);
    for (int j = 0; j < dim; j++)
        for (int t = 0; t < T; t++)
            out->data[j] += weights[t]->data[0] * values[t]->data[j];

    if (grad_enabled) {
        AttnSumCtx *c = arena_alloc(&G_arena, sizeof(AttnSumCtx));
        c->weights = weights; c->values = values; c->T = T; c->dim = dim;
        out->ctx = c;
        out->backward = back_attn_sum;

        int nk = T * 2;
        Node **kids = arena_alloc(&G_arena, sizeof(Node*) * nk);
        for (int i = 0; i < T; i++) { kids[i] = weights[i]; kids[T+i] = values[i]; }
        node_set_children(out, kids, nk);
    }
    return out;
}

/* Non-differentiable softmax for sampling */
static void softmax_probs(const double *data, int n, double *out) {
    double mx = data[0];
    for (int i = 1; i < n; i++) if (data[i] > mx) mx = data[i];
    double sum = 0;
    for (int i = 0; i < n; i++) { out[i] = exp(data[i] - mx); sum += out[i]; }
    for (int i = 0; i < n; i++) out[i] /= sum;
}

/* Top-k/top-p/min-p/typical-p sampling */
/* And lo, sampling shall not be a coin flip but a controlled hallucination. */
static int top_k_top_p_sample(const double *probs, int n, int k, double p, double min_p, double typical_p) {
    int *idx = malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) idx[i] = i;
    /* Sort descending by prob */
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (probs[idx[j]] > probs[idx[i]]) { int t = idx[i]; idx[i] = idx[j]; idx[j] = t; }

    int len = n;
    if (k > 0 && k < len) len = k;

    /* Min-p filtering (GPT-3/4 style): remove tokens with prob < min_p * max_prob */
    if (min_p > 0.0 && len > 0) {
        double max_prob = probs[idx[0]];
        double threshold = min_p * max_prob;
        int new_len = 0;
        for (int i = 0; i < len; i++) {
            if (probs[idx[i]] >= threshold) {
                idx[new_len++] = idx[i];
            }
        }
        if (new_len > 0) len = new_len;
    }

    /* Typical-p filtering: prefer tokens with typical information content */
    if (typical_p < 1.0 && len > 0) {
        /* Compute entropy (expected surprisal) */
        double entropy = 0.0;
        for (int i = 0; i < len; i++) {
            if (probs[idx[i]] > 1e-12) {
                entropy -= probs[idx[i]] * log(probs[idx[i]]);
            }
        }
        /* Compute absolute deviation from expected surprisal for each token */
        double *deviations = malloc(sizeof(double) * len);
        int *dev_idx = malloc(sizeof(int) * len);
        int dev_count = 0;
        for (int i = 0; i < len; i++) {
            if (probs[idx[i]] > 1e-12) {
                double surprisal = -log(probs[idx[i]]);
                deviations[dev_count] = fabs(surprisal - entropy);
                dev_idx[dev_count] = idx[i];
                dev_count++;
            }
        }
        /* Sort by deviation (lower is more typical) */
        for (int i = 0; i < dev_count - 1; i++)
            for (int j = i + 1; j < dev_count; j++)
                if (deviations[j] < deviations[i]) {
                    double td = deviations[i]; deviations[i] = deviations[j]; deviations[j] = td;
                    int ti = dev_idx[i]; dev_idx[i] = dev_idx[j]; dev_idx[j] = ti;
                }
        /* Keep tokens until cumulative prob >= typical_p */
        double cum = 0.0;
        int typical_len = 0;
        for (int i = 0; i < dev_count; i++) {
            idx[typical_len++] = dev_idx[i];
            cum += probs[dev_idx[i]];
            if (cum >= typical_p) break;
        }
        if (typical_len > 0) len = typical_len;
        free(deviations);
        free(dev_idx);
    }

    /* Top-p (nucleus) filtering */
    if (p < 1.0) {
        double cum = 0;
        for (int i = 0; i < len; i++) {
            cum += probs[idx[i]];
            if (cum >= p) { len = i + 1; break; }
        }
    }

    double mass = 0;
    for (int i = 0; i < len; i++) mass += probs[idx[i]];
    if (mass <= 0) { int r = idx[0]; free(idx); return r; }

    double r = rand_uniform() * mass;
    double s = 0;
    int result = idx[len - 1];
    for (int i = 0; i < len; i++) {
        s += probs[idx[i]];
        if (s >= r) { result = idx[i]; break; }
    }
    free(idx);
    return result;
}

/* Gradient clipping */
static void clip_grads(MatrixParam *m, double clip) {
    if (clip <= 0) return;
    for (int i = 0; i < m->nout; i++)
        for (int j = 0; j < m->nin; j++) {
            if (m->row_grad[i][j] > clip) m->row_grad[i][j] = clip;
            else if (m->row_grad[i][j] < -clip) m->row_grad[i][j] = -clip;
        }
}

/* ============================================================
 * 4) DELTA ADAPTERS — appended souls, never overwritten
 * ============================================================ */

typedef struct {
    MatrixParam *A; /* (nout, r) */
    MatrixParam *B; /* (r, nin) */
} DeltaAdapter;

static DeltaAdapter *delta_new(int nout, int nin, int r, double std) {
    DeltaAdapter *d = calloc(1, sizeof(DeltaAdapter));
    d->A = mat_new(nout, r, std);
    d->B = mat_new(r, nin, std);
    return d;
}

static Node *delta_apply(DeltaAdapter *d, Node *x) {
    Node *bx = mat_matvec(d->B, x);
    return mat_matvec(d->A, bx);
}

/* Grow delta adapter outer dimensions. Rank stays the same.
 * A: (nout, r) → grow rows to new_nout
 * B: (r, nin) → grow cols to new_nin */
static void delta_grow_dims(DeltaAdapter *d, int new_nout, int new_nin) {
    mat_grow_rows(d->A, new_nout, 0.02);
    mat_grow_cols(d->B, new_nin, 0.02);
}

/* ============================================================
 * 5) TOKENIZER — byte-level BPE (GPT-3/4 style)
 * ============================================================ */

typedef struct { char a[64]; char b[64]; } MergePair;

/* Simple string hash */
static unsigned int str_hash(const char *s) {
    unsigned int h = 5381;
    while (*s) h = h * 33 + (unsigned char)*s++;
    return h;
}

/* Linear probing hash table for stoi */
#define STOI_CAP 8192
typedef struct { char *key; int val; } StoiEntry;

typedef struct {
    StoiEntry entries[STOI_CAP];
} StoiTable;

static StoiTable *stoi_new(void) {
    StoiTable *t = calloc(1, sizeof(StoiTable));
    for (int i = 0; i < STOI_CAP; i++) t->entries[i].val = -1;
    return t;
}

static void stoi_put(StoiTable *t, const char *key, int val) {
    unsigned int h = str_hash(key) % STOI_CAP;
    for (int i = 0; i < STOI_CAP; i++) {
        int idx = (h + i) % STOI_CAP;
        if (t->entries[idx].key == NULL || strcmp(t->entries[idx].key, key) == 0) {
            if (t->entries[idx].key == NULL) t->entries[idx].key = strdup(key);
            t->entries[idx].val = val;
            return;
        }
    }
}

static int stoi_get(StoiTable *t, const char *key) {
    unsigned int h = str_hash(key) % STOI_CAP;
    for (int i = 0; i < STOI_CAP; i++) {
        int idx = (h + i) % STOI_CAP;
        if (t->entries[idx].key == NULL) return -1;
        if (strcmp(t->entries[idx].key, key) == 0) return t->entries[idx].val;
    }
    return -1;
}

typedef struct {
    char **tokens;
    int vocab_size, cap;
    StoiTable *stoi;
    int bos_id, eos_id, pad_id;
    int bpe_enabled;
    MergePair *merges;
    int n_merges;
    int trained_chars;
} EvolvingTokenizer;

static EvolvingTokenizer *tok_new(const char **docs, int n_docs) {
    EvolvingTokenizer *tok = calloc(1, sizeof(EvolvingTokenizer));
    tok->stoi = stoi_new();
    tok->cap = 512;
    tok->tokens = calloc(tok->cap, sizeof(char*));

    /* 256 byte tokens: "0x00" through "0xff" */
    for (int i = 0; i < 256; i++) {
        char hex[8];
        snprintf(hex, sizeof(hex), "0x%02x", i);
        tok->tokens[tok->vocab_size] = strdup(hex);
        stoi_put(tok->stoi, hex, tok->vocab_size);
        tok->vocab_size++;
    }

    /* Special tokens: BOS (256), EOS (257), PAD (258) */
    tok->tokens[tok->vocab_size] = strdup("<BOS>");
    stoi_put(tok->stoi, "<BOS>", tok->vocab_size);
    tok->bos_id = tok->vocab_size++;

    tok->tokens[tok->vocab_size] = strdup("<EOS>");
    stoi_put(tok->stoi, "<EOS>", tok->vocab_size);
    tok->eos_id = tok->vocab_size++;

    tok->tokens[tok->vocab_size] = strdup("<PAD>");
    stoi_put(tok->stoi, "<PAD>", tok->vocab_size);
    tok->pad_id = tok->vocab_size++;

    /* docs only used for trained_chars count */
    tok->trained_chars = 0;
    for (int d = 0; d < n_docs; d++)
        tok->trained_chars += (int)strlen(docs[d]);

    return tok;
}

static void tok_add_token(EvolvingTokenizer *tok, const char *s) {
    if (stoi_get(tok->stoi, s) >= 0) return;
    if (tok->vocab_size >= tok->cap) {
        tok->cap *= 2;
        void *tmp = realloc(tok->tokens, sizeof(char*) * tok->cap);
        if (!tmp) { fprintf(stderr, "[tok_add_token] realloc failed\n"); return; }
        tok->tokens = tmp;
    }
    tok->tokens[tok->vocab_size] = strdup(s);
    stoi_put(tok->stoi, s, tok->vocab_size);
    tok->vocab_size++;
}

/* ---- Unicode pre-segmentation ---- */

/* A single byte-buffer segment */
typedef struct { unsigned char *data; int len; } ByteSeg;
typedef struct { ByteSeg *segs; int len, cap; } SegArr;

static void segarr_push(SegArr *a, unsigned char *data, int len) {
    if (a->len >= a->cap) {
        a->cap = a->cap ? a->cap * 2 : 32;
        void *tmp = realloc(a->segs, sizeof(ByteSeg) * a->cap);
        if (!tmp) { fprintf(stderr, "[segarr_push] realloc failed\n"); return; }
        a->segs = tmp;
    }
    a->segs[a->len].data = malloc(len);
    memcpy(a->segs[a->len].data, data, len);
    a->segs[a->len].len = len;
    a->len++;
}

static void segarr_free(SegArr *a) {
    for (int i = 0; i < a->len; i++) free(a->segs[i].data);
    free(a->segs);
    a->segs = NULL; a->len = a->cap = 0;
}

/* Classify a byte into a Unicode category group:
 * 'L' = letter (ASCII a-z, A-Z, or multi-byte UTF-8 lead bytes)
 * 'N' = digit (0-9)
 * 'Z' = whitespace (space, \n, \r, \t)
 * 'P' = punctuation / everything else
 * For multi-byte UTF-8, the lead byte determines the group (all 'L'),
 * and continuation bytes (0x80-0xBF) inherit the group of their lead. */
static char byte_category(unsigned char b) {
    if ((b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z')) return 'L';
    if (b >= '0' && b <= '9') return 'N';
    if (b == ' ' || b == '\n' || b == '\r' || b == '\t') return 'Z';
    /* Multi-byte UTF-8 lead bytes → treat as letter */
    if (b >= 0xC0 && b <= 0xF7) return 'L';
    /* Continuation bytes (0x80-0xBF) → treat as letter (part of multi-byte char) */
    if (b >= 0x80 && b <= 0xBF) return 'L';
    return 'P';
}

/* Split text into segments by Unicode category boundary.
 * Each segment is a run of bytes sharing the same category group. */
static SegArr unicode_segment(const char *text) {
    SegArr result = {0};
    if (!text || !*text) return result;

    unsigned char buf[4096];
    int buf_len = 0;
    char cur_cat = 0;

    for (const unsigned char *p = (const unsigned char *)text; *p; p++) {
        char cat = byte_category(*p);
        if (cat != cur_cat && buf_len > 0) {
            segarr_push(&result, buf, buf_len);
            buf_len = 0;
        }
        cur_cat = cat;
        if (buf_len < (int)sizeof(buf) - 1) {
            buf[buf_len++] = *p;
        } else {
            /* Flush oversized segment */
            segarr_push(&result, buf, buf_len);
            buf_len = 0;
            buf[buf_len++] = *p;
        }
    }
    if (buf_len > 0) {
        segarr_push(&result, buf, buf_len);
    }
    return result;
}

/* ---- BPE Training and Application ---- */

/* Pair frequency counting hash table */
#define PAIR_CAP 16384
typedef struct { char a[64]; char b[64]; int count; int used; } PairEntry;

static unsigned int pair_hash(const char *a, const char *b) {
    unsigned int h = 5381;
    for (const char *p = a; *p; p++) h = h * 33 + (unsigned char)*p;
    h = h * 33 + 0xFF;
    for (const char *p = b; *p; p++) h = h * 33 + (unsigned char)*p;
    return h;
}

static void tok_train_bpe(EvolvingTokenizer *tok, const char **docs, int n_docs, int num_merges) {
    /* Build full text from docs */
    size_t total_len = 0;
    for (int d = 0; d < n_docs; d++) total_len += strlen(docs[d]) + 1;
    char *text = calloc(total_len + 1, 1);
    for (int d = 0; d < n_docs; d++) {
        if (d > 0) strcat(text, " ");
        strcat(text, docs[d]);
    }
    if (!*text) { free(text); return; }

    /* Segment text into Unicode category runs */
    SegArr segs = unicode_segment(text);
    free(text);
    if (segs.len == 0) { segarr_free(&segs); return; }

    /* Convert segments to byte-token sequences and count frequencies.
     * We use StrArr per unique segment, with frequency counts. */
    int total_segs = segs.len;
    StrArr *sym_seqs = calloc(total_segs, sizeof(StrArr));
    int *seg_freq = calloc(total_segs, sizeof(int));
    for (int s = 0; s < total_segs; s++) {
        seg_freq[s] = 1;
        for (int b = 0; b < segs.segs[s].len; b++) {
            char hex[8];
            snprintf(hex, sizeof(hex), "0x%02x", segs.segs[s].data[b]);
            sa_push(&sym_seqs[s], hex);
        }
    }
    segarr_free(&segs);

    /* Allocate merge storage */
    if (tok->merges) free(tok->merges);
    tok->merges = calloc(num_merges, sizeof(MergePair));
    tok->n_merges = 0;

    PairEntry *pairs = calloc(PAIR_CAP, sizeof(PairEntry));

    for (int iter = 0; iter < num_merges; iter++) {
        /* Count pairs */
        memset(pairs, 0, sizeof(PairEntry) * PAIR_CAP);
        for (int s = 0; s < total_segs; s++) {
            StrArr *seq = &sym_seqs[s];
            for (int i = 0; i < seq->len - 1; i++) {
                unsigned int h = pair_hash(seq->items[i], seq->items[i+1]) % PAIR_CAP;
                for (int probe = 0; probe < PAIR_CAP; probe++) {
                    int idx = (h + probe) % PAIR_CAP;
                    if (!pairs[idx].used) {
                        strncpy(pairs[idx].a, seq->items[i], 63);
                        strncpy(pairs[idx].b, seq->items[i+1], 63);
                        pairs[idx].count = seg_freq[s];
                        pairs[idx].used = 1;
                        break;
                    }
                    if (strcmp(pairs[idx].a, seq->items[i]) == 0 &&
                        strcmp(pairs[idx].b, seq->items[i+1]) == 0) {
                        pairs[idx].count += seg_freq[s];
                        break;
                    }
                }
            }
        }

        /* Find best pair */
        int best_count = 0;
        int best_idx = -1;
        for (int i = 0; i < PAIR_CAP; i++) {
            if (pairs[i].used && pairs[i].count > best_count) {
                best_count = pairs[i].count;
                best_idx = i;
            }
        }
        if (best_idx < 0) break;

        char best_a[64], best_b[64];
        strncpy(best_a, pairs[best_idx].a, 63); best_a[63] = 0;
        strncpy(best_b, pairs[best_idx].b, 63); best_b[63] = 0;

        /* Merged token uses "+" separator: "0x48+0x65" */
        char new_tok[128];
        snprintf(new_tok, sizeof(new_tok), "%s+%s", best_a, best_b);

        strncpy(tok->merges[tok->n_merges].a, best_a, 63);
        strncpy(tok->merges[tok->n_merges].b, best_b, 63);
        tok->n_merges++;

        /* Apply merge to all symbol sequences */
        for (int s = 0; s < total_segs; s++) {
            StrArr *seq = &sym_seqs[s];
            StrArr merged = {0};
            int i = 0;
            while (i < seq->len) {
                if (i < seq->len - 1 &&
                    strcmp(seq->items[i], best_a) == 0 &&
                    strcmp(seq->items[i+1], best_b) == 0) {
                    sa_push(&merged, new_tok);
                    i += 2;
                } else {
                    sa_push(&merged, seq->items[i]);
                    i++;
                }
            }
            sa_free(seq);
            *seq = merged;
        }

        /* Add token to vocab if new */
        tok_add_token(tok, new_tok);
    }

    free(pairs);
    for (int s = 0; s < total_segs; s++) sa_free(&sym_seqs[s]);
    free(sym_seqs);
    free(seg_freq);
}

/* Apply BPE merges to a token sequence (greedy, lowest-rank first).
 * Input: StrArr of token names (e.g. "0x48", "0x65", ...).
 * Returns: new StrArr with merges applied. Caller must sa_free. */
static StrArr tok_apply_bpe(EvolvingTokenizer *tok, StrArr *input) {
    if (!tok->n_merges || input->len < 2) {
        StrArr copy = {0};
        for (int i = 0; i < input->len; i++) sa_push(&copy, input->items[i]);
        return copy;
    }

    StrArr symbols = {0};
    for (int i = 0; i < input->len; i++) sa_push(&symbols, input->items[i]);

    while (symbols.len >= 2) {
        /* Find the pair with lowest merge rank */
        int best_rank = tok->n_merges; /* sentinel: impossible rank */
        int best_pos = -1;
        for (int i = 0; i < symbols.len - 1; i++) {
            /* Look up rank of this pair */
            for (int m = 0; m < tok->n_merges; m++) {
                if (m >= best_rank) break; /* can't improve */
                if (strcmp(symbols.items[i], tok->merges[m].a) == 0 &&
                    strcmp(symbols.items[i+1], tok->merges[m].b) == 0) {
                    best_rank = m;
                    best_pos = i;
                    break;
                }
            }
        }
        if (best_pos < 0) break; /* no applicable merge */

        /* Build merged token name with "+" separator */
        char new_tok[128];
        snprintf(new_tok, sizeof(new_tok), "%s+%s",
                 tok->merges[best_rank].a, tok->merges[best_rank].b);

        /* Replace the pair at best_pos */
        StrArr merged = {0};
        int i = 0;
        while (i < symbols.len) {
            if (i == best_pos) {
                sa_push(&merged, new_tok);
                i += 2;
            } else {
                sa_push(&merged, symbols.items[i]);
                i++;
            }
        }
        sa_free(&symbols);
        symbols = merged;
    }
    return symbols;
}

static int tok_maybe_enable_bpe(EvolvingTokenizer *tok, const char **docs, int n_docs) {
    if (tok->bpe_enabled) return 0;
    int total_chars = 0;
    for (int d = 0; d < n_docs; d++) total_chars += strlen(docs[d]);
    if (total_chars >= CFG.enable_bpe_after_chars) {
        tok_train_bpe(tok, docs, n_docs, CFG.bpe_num_merges);
        tok->bpe_enabled = 1;
        tok->trained_chars = total_chars;
        return 1;
    }
    return 0;
}

static int tok_maybe_retrain_bpe(EvolvingTokenizer *tok, const char **docs, int n_docs) {
    if (!tok->bpe_enabled) return 0;
    int total_chars = 0;
    for (int d = 0; d < n_docs; d++) total_chars += strlen(docs[d]);
    if (total_chars - tok->trained_chars >= CFG.bpe_retrain_every_chars) {
        tok_train_bpe(tok, docs, n_docs, CFG.bpe_num_merges);
        tok->trained_chars = total_chars;
        return 1;
    }
    return 0;
}

static IntArr tok_encode(EvolvingTokenizer *tok, const char *s) {
    IntArr ids = {0};
    /* Skip leading/trailing whitespace */
    while (*s == ' ' || *s == '\t' || *s == '\n') s++;
    int slen = (int)strlen(s);
    while (slen > 0 && (s[slen-1] == ' ' || s[slen-1] == '\t' || s[slen-1] == '\n')) slen--;

    ia_push(&ids, tok->bos_id);

    if (slen == 0) {
        ia_push(&ids, tok->eos_id);
        return ids;
    }

    /* Make a null-terminated copy of the trimmed string */
    char *trimmed = malloc(slen + 1);
    memcpy(trimmed, s, slen);
    trimmed[slen] = 0;

    /* Segment by Unicode category */
    SegArr segs = unicode_segment(trimmed);
    free(trimmed);

    for (int si = 0; si < segs.len; si++) {
        /* Convert segment bytes to base token names */
        StrArr base_tokens = {0};
        for (int b = 0; b < segs.segs[si].len; b++) {
            char hex[8];
            snprintf(hex, sizeof(hex), "0x%02x", segs.segs[si].data[b]);
            sa_push(&base_tokens, hex);
        }

        if (tok->bpe_enabled) {
            /* Apply BPE merges */
            StrArr merged = tok_apply_bpe(tok, &base_tokens);
            for (int i = 0; i < merged.len; i++) {
                int id = stoi_get(tok->stoi, merged.items[i]);
                if (id >= 0) ia_push(&ids, id);
            }
            sa_free(&merged);
        } else {
            /* No BPE: each byte is its own token */
            for (int i = 0; i < base_tokens.len; i++) {
                int id = stoi_get(tok->stoi, base_tokens.items[i]);
                if (id >= 0) ia_push(&ids, id);
            }
        }
        sa_free(&base_tokens);
    }
    segarr_free(&segs);

    ia_push(&ids, tok->eos_id);
    return ids;
}

/* Convert a token string to raw bytes. Returns number of bytes written.
 * Single byte token "0xNN" (no '+', len==4): one byte.
 * Merged token "0x48+0x65+...": split by '+', each part → one byte. */
static int tok_token_to_bytes(const char *tok_str, unsigned char *out, int out_cap) {
    int pos = 0;
    const char *p = tok_str;
    while (*p && pos < out_cap) {
        if (p[0] == '0' && p[1] == 'x' && p[2] && p[3]) {
            char hex[3] = {p[2], p[3], 0};
            out[pos++] = (unsigned char)strtol(hex, NULL, 16);
            p += 4;
            if (*p == '+') p++; /* skip separator */
        } else {
            break; /* unexpected format */
        }
    }
    return pos;
}

static char *tok_decode(EvolvingTokenizer *tok, const int *ids, int n) {
    size_t bufcap = 1024;
    unsigned char *buf = calloc(bufcap, 1);
    size_t pos = 0;
    unsigned char tmp[256];
    for (int i = 0; i < n; i++) {
        if (ids[i] < 0 || ids[i] >= tok->vocab_size) continue;
        const char *t = tok->tokens[ids[i]];
        if (strcmp(t, "<BOS>") == 0 || strcmp(t, "<PAD>") == 0) continue;
        if (strcmp(t, "<EOS>") == 0) break;
        int nb = tok_token_to_bytes(t, tmp, sizeof(tmp));
        while (pos + nb + 1 > bufcap) {
            bufcap *= 2;
            void *tmp2 = realloc(buf, bufcap);
            if (!tmp2) { fprintf(stderr, "[tok_decode] realloc failed\n"); buf[pos] = 0; return (char *)buf; }
            buf = tmp2;
        }
        memcpy(buf + pos, tmp, nb);
        pos += nb;
    }
    buf[pos] = 0;
    return (char *)buf;
}

/* ============================================================
 * 6) GPT MODEL with RoPE
 * ============================================================ */

/* And lo, positions shall become angles, and angles shall become meaning. */
typedef struct { Node *vec; int pos, head_dim; } RopeCtx;

static void back_rope(Node *self) {
    RopeCtx *c = self->ctx;
    for (int i = 0; i < c->head_dim - 1; i += 2) {
        double theta = (double)c->pos / pow(10000.0, (double)i / (double)c->head_dim);
        double co = cos(theta), si = sin(theta);
        double ga = self->grad[i], gb = self->grad[i+1];
        c->vec->grad[i]   += ga * co + gb * si;
        c->vec->grad[i+1] += -ga * si + gb * co;
    }
}

static Node *rope_rotate(Node *vec, int pos, int head_dim) {
    Node *out = node_new(vec->len);
    memcpy(out->data, vec->data, sizeof(double) * vec->len);
    for (int i = 0; i < head_dim - 1; i += 2) {
        double theta = (double)pos / pow(10000.0, (double)i / (double)head_dim);
        double co = cos(theta), si = sin(theta);
        double a = vec->data[i], b = vec->data[i+1];
        out->data[i]   = a * co - b * si;
        out->data[i+1] = a * si + b * co;
    }
    if (grad_enabled) {
        RopeCtx *c = arena_alloc(&G_arena, sizeof(RopeCtx));
        c->vec = vec; c->pos = pos; c->head_dim = head_dim;
        out->ctx = c;
        out->backward = back_rope;
        Node *kids[] = {vec};
        node_set_children(out, kids, 1);
    }
    return out;
}

/* Delta module: maps name -> DeltaAdapter */
#define MAX_ADAPTERS_PER_MOD 96

typedef struct {
    char *names[MAX_ADAPTERS_PER_MOD];
    DeltaAdapter *adapters[MAX_ADAPTERS_PER_MOD];
    int count;
} DeltaModule;

static DeltaAdapter *dmod_get(DeltaModule *m, const char *name) {
    for (int i = 0; i < m->count; i++)
        if (strcmp(m->names[i], name) == 0) return m->adapters[i];
    return NULL;
}

static void dmod_set(DeltaModule *m, const char *name, DeltaAdapter *da) {
    if (m->count >= MAX_ADAPTERS_PER_MOD) {
        fprintf(stderr, "[dmod_set] ERROR: exceeded MAX_ADAPTERS_PER_MOD (%d)\n", MAX_ADAPTERS_PER_MOD);
        return;
    }
    m->names[m->count] = strdup(name);
    m->adapters[m->count] = da;
    m->count++;
}

/* Adam state for a matrix */
typedef struct {
    double **m; /* momentum */
    double **v; /* velocity */
    int nout, nin, t;
} AdamState;

static AdamState *adam_new(int nout, int nin) {
    AdamState *s = calloc(1, sizeof(AdamState));
    s->nout = nout; s->nin = nin;
    s->m = calloc(nout, sizeof(double*));
    s->v = calloc(nout, sizeof(double*));
    for (int i = 0; i < nout; i++) {
        s->m[i] = calloc(nin, sizeof(double));
        s->v[i] = calloc(nin, sizeof(double));
    }
    return s;
}

/* Grow Adam state to match expanded MatrixParam (ontogenesis/vocab growth) */
static void adam_grow(AdamState *st, int new_nout, int new_nin) {
    if (new_nout > st->nout) {
        st->m = realloc(st->m, new_nout * sizeof(double*));
        st->v = realloc(st->v, new_nout * sizeof(double*));
        for (int i = st->nout; i < new_nout; i++) {
            st->m[i] = calloc(new_nin, sizeof(double));
            st->v[i] = calloc(new_nin, sizeof(double));
        }
    }
    if (new_nin > st->nin) {
        for (int i = 0; i < st->nout; i++) {
            st->m[i] = realloc(st->m[i], new_nin * sizeof(double));
            st->v[i] = realloc(st->v[i], new_nin * sizeof(double));
            for (int j = st->nin; j < new_nin; j++) {
                st->m[i][j] = 0.0;
                st->v[i][j] = 0.0;
            }
        }
    }
    st->nout = new_nout;
    st->nin = new_nin;
}

static void adam_step(AdamState *st, MatrixParam *mat, double lr) {
    /* Auto-grow if matrix was expanded (vocab growth, ontogenesis) */
    if (mat->nout > st->nout || mat->nin > st->nin)
        adam_grow(st, mat->nout, mat->nin);
    st->t++;
    double b1c = 1.0 - pow(CFG.beta1, st->t);
    double b2c = 1.0 - pow(CFG.beta2, st->t);
    clip_grads(mat, CFG.grad_clip);
    for (int i = 0; i < mat->nout; i++)
        for (int j = 0; j < mat->nin; j++) {
            double g = mat->row_grad[i][j];
            st->m[i][j] = CFG.beta1 * st->m[i][j] + (1 - CFG.beta1) * g;
            st->v[i][j] = CFG.beta2 * st->v[i][j] + (1 - CFG.beta2) * g * g;
            double mh = st->m[i][j] / b1c;
            double vh = st->v[i][j] / b2c;
            mat->row_data[i][j] -= lr * mh / (sqrt(vh) + CFG.eps_adam);
            mat->row_grad[i][j] = 0;
        }
}

/* CooccurField struct (functions defined later, after tokenizer) */
typedef struct { int key[3]; double count; } TrigramEntry;
typedef struct { int key[2]; double count; } BigramEntry;
typedef struct { int key[4]; double count; } FourgramEntry;  /* [prev3,prev2,prev1,next] */
typedef struct { int key[2]; double count; } CooccurWinEntry; /* [center,neighbor] */
typedef struct { int token; double boost; } UserBoostEntry;
#define COOCCUR_HASH_SIZE 16384

typedef struct {
    double *unigram;  /* [vocab_size] */
    int vocab_size;
    TrigramEntry *trigrams;
    int n_trigrams, trigram_cap;
    BigramEntry *bigrams;
    int n_bigrams, bigram_cap;
    FourgramEntry *fourgrams;
    int n_fourgrams, fourgram_cap;
    CooccurWinEntry *cooccur_win;   /* co-occurrence window entries */
    int n_cooccur_win, cooccur_win_cap;
    UserBoostEntry *user_boosts;    /* temporary user word boosts (Leo-style) */
    int n_user_boosts, user_boost_cap;
    /* Hash indices for O(1) lookup */
    int *bigram_head;   /* [COOCCUR_HASH_SIZE] -> first index in bigrams[], or -1 */
    int *bigram_next;   /* [bigram_cap] -> next index with same hash, or -1 */
    int *trigram_head;  /* [COOCCUR_HASH_SIZE] -> first index in trigrams[], or -1 */
    int *trigram_next;  /* [trigram_cap] -> next index with same hash, or -1 */
    int *fourgram_head; /* [COOCCUR_HASH_SIZE] -> first index in fourgrams[], or -1 */
    int *fourgram_next; /* [fourgram_cap] -> next index with same hash, or -1 */
    int *cooccur_win_head; /* [COOCCUR_HASH_SIZE] -> first index in cooccur_win[], or -1 */
    int *cooccur_win_next; /* [cooccur_win_cap] -> next index with same hash, or -1 */
    int built;
    pthread_mutex_t mu; /* thread safety for ingest/build/sample */
} CooccurField;

/* Hash functions for cooccur lookup (needed before gpt_generate) */
static inline unsigned int cooccur_bigram_hash(int prev) {
    return ((unsigned int)prev * 2654435761u) & (COOCCUR_HASH_SIZE - 1);
}
static inline unsigned int cooccur_trigram_hash(int a, int b) {
    return (((unsigned int)a * 2654435761u) ^ ((unsigned int)b * 2246822519u)) & (COOCCUR_HASH_SIZE - 1);
}
static inline unsigned int cooccur_fourgram_hash(int a, int b, int c) {
    return (((unsigned int)a * 2654435761u) ^ ((unsigned int)b * 2246822519u) ^
            ((unsigned int)c * 3266489917u)) & (COOCCUR_HASH_SIZE - 1);
}
static inline unsigned int cooccur_window_hash(int center) {
    return ((unsigned int)center * 2654435761u) & (COOCCUR_HASH_SIZE - 1);
}

/* The GPT model */
#define MAX_BASE_MATS 256  /* adult: 6 layers × ~20 matrices + embedding matrices */
#define MAX_DELTA_MODS 16

typedef struct {
    EvolvingTokenizer *tok;
    int n_layer, n_embd, n_head, head_dim, block_size;

    /* Base weights: name -> MatrixParam */
    char *base_names[MAX_BASE_MATS];
    MatrixParam *base_mats[MAX_BASE_MATS];
    AdamState *base_adam[MAX_BASE_MATS];
    int n_base;

    /* Deltas */
    DeltaModule *deltas[MAX_DELTA_MODS];
    AdamState **delta_adam[MAX_DELTA_MODS]; /* adam per adapter per module */
    double active_alpha[MAX_DELTA_MODS];
    int n_deltas;

    /* Native gamma: snapshot of initial embeddings */
    double **init_embed_snapshot; /* [vocab_size][n_embd] */
    int init_embed_rows;

    /* Phase 1: residual scaling + global step counter */
    double residual_alpha;
    int global_step;

    /* Phase 1.5: syntropy-driven temperature modulation */
    double syntropy_temp_offset;

    /* Phase 3A: ontogenesis — growth freeze counter */
    int growth_freeze_remaining;
    int last_warmup_stage;      /* last stage that received warmup training (-1 = none) */
    int growth_step_offset;     /* global_step at last growth event (for LR warmup reset) */

    /* Adaptive corpus blend: set by background_trainer */
    CooccurField *corpus_field;

    /* consciousness state */
    double delta_alpha_scale;             /* conscience: multiplier on all delta contributions (1.0 = normal) */
    double generation_entropy_history[16]; /* conscience: rolling window of per-generation mean entropy */
    int gen_entropy_count;                 /* how many entries in generation_entropy_history */
    double last_surprise;                  /* self-prediction error on last prompt */
    double surprise_baseline;              /* EMA of surprise over time */
    double last_gen_entropy;               /* mean entropy of last generation (for conscience) */

    pthread_mutex_t mu;
} GPT;

static MatrixParam *gpt_base(GPT *g, const char *name) {
    for (int i = 0; i < g->n_base; i++)
        if (strcmp(g->base_names[i], name) == 0) return g->base_mats[i];
    return NULL;
}

static void gpt_add_base(GPT *g, const char *name, MatrixParam *m) {
    g->base_names[g->n_base] = strdup(name);
    g->base_mats[g->n_base] = m;
    g->base_adam[g->n_base] = adam_new(m->nout, m->nin);
    g->n_base++;
}

static void gpt_add_delta_module(GPT *g, double alpha) {
    DeltaModule *mod = calloc(1, sizeof(DeltaModule));
    int r = CFG.delta_rank;
    char name[64];
    for (int li = 0; li < CFG.n_layer; li++) {
        const char *wnames[] = {"wq", "wk", "wv", "wo"};
        for (int w = 0; w < 4; w++) {
            snprintf(name, sizeof(name), "l%d.%s", li, wnames[w]);
            dmod_set(mod, name, delta_new(CFG.n_embd, CFG.n_embd, r, 0.02));
        }
        snprintf(name, sizeof(name), "l%d.fc_g", li);
        dmod_set(mod, name, delta_new(4*CFG.n_embd, CFG.n_embd, r, 0.02));
        snprintf(name, sizeof(name), "l%d.fc_v", li);
        dmod_set(mod, name, delta_new(4*CFG.n_embd, CFG.n_embd, r, 0.02));
        snprintf(name, sizeof(name), "l%d.fc2", li);
        dmod_set(mod, name, delta_new(CFG.n_embd, 4*CFG.n_embd, r, 0.02));
        for (int h = 0; h < CFG.n_head_types && h < CFG.n_head; h++) {
            const char *ht = CFG.head_types[h];
            if (strcmp(ht, "rrpram") == 0 || strcmp(ht, "hybrid") == 0) {
                snprintf(name, sizeof(name), "l%d.h%d.w_pattern", li, h);
                dmod_set(mod, name, delta_new(CFG.block_size, g->head_dim, r, 0.02));
            }
        }
    }
    dmod_set(mod, "lm_head", delta_new(g->tok->vocab_size, CFG.n_embd, r, 0.02));

    int idx = g->n_deltas;
    g->deltas[idx] = mod;
    g->active_alpha[idx] = alpha;

    /* Adam states for delta adapters */
    g->delta_adam[idx] = calloc(mod->count * 2, sizeof(AdamState*));
    for (int i = 0; i < mod->count; i++) {
        DeltaAdapter *da = mod->adapters[i];
        g->delta_adam[idx][i*2]   = adam_new(da->A->nout, da->A->nin);
        g->delta_adam[idx][i*2+1] = adam_new(da->B->nout, da->B->nin);
    }
    g->n_deltas++;
}

static GPT *gpt_new(EvolvingTokenizer *tok) {
    GPT *g = calloc(1, sizeof(GPT));
    g->tok = tok;
    g->n_layer = CFG.n_layer;
    g->n_embd = CFG.n_embd;
    g->n_head = CFG.n_head;
    g->head_dim = CFG.n_embd / CFG.n_head;
    g->block_size = CFG.block_size;
    g->residual_alpha = 1.0 / sqrt((double)CFG.n_layer);
    g->global_step = 0;
    g->syntropy_temp_offset = 0.0;
    g->growth_freeze_remaining = 0;
    g->last_warmup_stage = -1;
    g->growth_step_offset = 0;
    g->delta_alpha_scale = 1.0; /* conscience: full delta influence by default */
    g->gen_entropy_count = 0;
    g->last_surprise = 0.0;
    g->surprise_baseline = 0.0;
    g->last_gen_entropy = 0.0;
    pthread_mutex_init(&g->mu, NULL);

    int V = tok->vocab_size;
    gpt_add_base(g, "wte", mat_new(V, CFG.n_embd, 0.08));
    gpt_add_base(g, "wpe", mat_new(CFG.block_size, CFG.n_embd, 0.08));

    if (CFG.tie_embeddings) {
        /* lm_head shares wte */
        gpt_add_base(g, "lm_head", gpt_base(g, "wte"));
    } else {
        gpt_add_base(g, "lm_head", mat_new(V, CFG.n_embd, 0.08));
    }

    char name[64];
    for (int li = 0; li < CFG.n_layer; li++) {
        const char *wnames[] = {"wq", "wk", "wv", "wo"};
        for (int w = 0; w < 4; w++) {
            snprintf(name, sizeof(name), "l%d.%s", li, wnames[w]);
            gpt_add_base(g, name, mat_new(CFG.n_embd, CFG.n_embd, 0.08));
        }
        snprintf(name, sizeof(name), "l%d.fc_g", li);
        gpt_add_base(g, name, mat_new(4*CFG.n_embd, CFG.n_embd, 0.08));
        snprintf(name, sizeof(name), "l%d.fc_v", li);
        gpt_add_base(g, name, mat_new(4*CFG.n_embd, CFG.n_embd, 0.08));
        snprintf(name, sizeof(name), "l%d.fc2", li);
        gpt_add_base(g, name, mat_new(CFG.n_embd, 4*CFG.n_embd, 0.08));

        /* Hybrid attention: pattern weights + learnable gate */
        for (int h = 0; h < CFG.n_head_types && h < CFG.n_head; h++) {
            const char *ht = CFG.head_types[h];
            if (strcmp(ht, "rrpram") == 0 || strcmp(ht, "hybrid") == 0) {
                snprintf(name, sizeof(name), "l%d.h%d.w_pattern", li, h);
                gpt_add_base(g, name, mat_new(CFG.block_size, g->head_dim, 0.08));
            }
            snprintf(name, sizeof(name), "l%d.h%d.alpha", li, h);
            MatrixParam *am = mat_new(1, 1, 0.0);
            am->row_data[0][0] = CFG.hybrid_alpha_init;
            gpt_add_base(g, name, am);
        }
    }

    gpt_add_delta_module(g, 1.0);

    /* Snapshot initial embeddings for gamma */
    MatrixParam *wte = gpt_base(g, "wte");
    g->init_embed_rows = wte->nout;
    g->init_embed_snapshot = calloc(wte->nout, sizeof(double*));
    for (int i = 0; i < wte->nout; i++) {
        g->init_embed_snapshot[i] = calloc(wte->nin, sizeof(double));
        memcpy(g->init_embed_snapshot[i], wte->row_data[i], sizeof(double) * wte->nin);
    }

    return g;
}

/* Expand model vocab when tokenizer grows */
static void gpt_maybe_expand_vocab(GPT *g) {
    int new_v = g->tok->vocab_size;
    MatrixParam *wte = gpt_base(g, "wte");
    if (!wte || new_v <= wte->nout) return;
    mat_grow_rows(wte, new_v, 0.08);
    if (!CFG.tie_embeddings) {
        MatrixParam *lm = gpt_base(g, "lm_head");
        if (lm && lm != wte) mat_grow_rows(lm, new_v, 0.08);
    }
    /* Grow delta lm_head adapters */
    for (int d = 0; d < g->n_deltas; d++) {
        DeltaAdapter *da = dmod_get(g->deltas[d], "lm_head");
        if (da) mat_grow_rows(da->A, new_v, 0.02);
    }
}

/* ---- Phase 3A: Ontogenesis (Growing Architecture) ---- */
/* And lo, the organism shall not be born adult but shall grow, stage by stage,
 * from embryo to child to adolescent, each growth a small death and rebirth. */

/* Return index of current stage based on model dimensions (-1 if no match). */
static int gpt_current_growth_stage(GPT *g) {
    for (int i = 0; i < CFG.n_growth_stages; i++) {
        if (g->n_embd == CFG.growth_stages[i][1] &&
            g->n_layer == CFG.growth_stages[i][2] &&
            g->n_head == CFG.growth_stages[i][3])
            return i;
    }
    return -1; /* legacy checkpoint or unknown dims */
}

/* Return the target stage index based on corpus size. */
static int gpt_target_growth_stage(int corpus_chars) {
    int target = 0;
    for (int i = 0; i < CFG.n_growth_stages; i++) {
        if (corpus_chars >= CFG.growth_stages[i][0])
            target = i;
    }
    return target;
}

/* Reset Adam state for a matrix (when dimensions have changed). */
static void adam_reset(AdamState *s, int new_nout, int new_nin) {
    for (int i = 0; i < s->nout; i++) { free(s->m[i]); free(s->v[i]); }
    free(s->m); free(s->v);
    s->nout = new_nout; s->nin = new_nin; s->t = 0;
    s->m = calloc(new_nout, sizeof(double*));
    s->v = calloc(new_nout, sizeof(double*));
    for (int i = 0; i < new_nout; i++) {
        s->m[i] = calloc(new_nin, sizeof(double));
        s->v[i] = calloc(new_nin, sizeof(double));
    }
}

/* Full growth pipeline: grow existing matrices, add new layers/heads, grow deltas.
 * Returns 1 if growth occurred. */
static int gpt_maybe_grow_architecture(GPT *g, int corpus_chars) {
    int current = gpt_current_growth_stage(g);
    if (current < 0) return 0; /* legacy checkpoint, skip growth */
    if (g->growth_freeze_remaining > 0) return 0; /* still stabilizing from last growth */
    int target = gpt_target_growth_stage(corpus_chars);
    if (target <= current) return 0;
    /* Grow only one stage at a time — prevent catastrophic multi-stage jumps */
    target = current + 1;

    int new_embd  = CFG.growth_stages[target][1];
    int new_layer = CFG.growth_stages[target][2];
    int new_head  = CFG.growth_stages[target][3];
    int old_embd  = g->n_embd;
    int old_layer = g->n_layer;
    int old_head  = g->n_head;
    int new_head_dim = new_embd / new_head;

    printf("[growth] ONTOGENESIS: stage %d -> %d\n", current, target);
    printf("  embd: %d -> %d, layer: %d -> %d, head: %d -> %d\n",
           old_embd, new_embd, old_layer, new_layer, old_head, new_head);

    /* 1. Grow embedding matrices (columns = embd dimension) */
    MatrixParam *wte = gpt_base(g, "wte");
    mat_grow_cols(wte, new_embd, 0.001);
    MatrixParam *wpe = gpt_base(g, "wpe");
    mat_grow_cols(wpe, new_embd, 0.001);
    if (!CFG.tie_embeddings) {
        MatrixParam *lm = gpt_base(g, "lm_head");
        if (lm && lm != wte) mat_grow_cols(lm, new_embd, 0.001);
    }

    /* Update head types for new head count */
    head_types_for_n_head(new_head);

    /* 2. Grow existing layer matrices */
    char name[64];
    for (int li = 0; li < old_layer; li++) {
        const char *wnames[] = {"wq", "wk", "wv", "wo"};
        for (int w = 0; w < 4; w++) {
            snprintf(name, sizeof(name), "l%d.%s", li, wnames[w]);
            MatrixParam *m = gpt_base(g, name);
            if (m) mat_grow(m, new_embd, new_embd, 0.001);
        }
        snprintf(name, sizeof(name), "l%d.fc_g", li);
        MatrixParam *m = gpt_base(g, name);
        if (m) mat_grow(m, 4 * new_embd, new_embd, 0.001);
        snprintf(name, sizeof(name), "l%d.fc_v", li);
        m = gpt_base(g, name);
        if (m) mat_grow(m, 4 * new_embd, new_embd, 0.001);
        snprintf(name, sizeof(name), "l%d.fc2", li);
        m = gpt_base(g, name);
        if (m) mat_grow(m, new_embd, 4 * new_embd, 0.001);

        /* Grow existing head pattern matrices */
        for (int h = 0; h < old_head; h++) {
            snprintf(name, sizeof(name), "l%d.h%d.w_pattern", li, h);
            m = gpt_base(g, name);
            if (m) mat_grow_cols(m, new_head_dim, 0.001);
        }
        /* Add new heads for existing layer */
        for (int h = old_head; h < new_head && h < CFG.n_head_types; h++) {
            const char *ht = CFG.head_types[h];
            if (strcmp(ht, "rrpram") == 0 || strcmp(ht, "hybrid") == 0) {
                snprintf(name, sizeof(name), "l%d.h%d.w_pattern", li, h);
                gpt_add_base(g, name, mat_new(CFG.block_size, new_head_dim, 0.08));
            }
            snprintf(name, sizeof(name), "l%d.h%d.alpha", li, h);
            MatrixParam *am = mat_new(1, 1, 0.0);
            am->row_data[0][0] = CFG.hybrid_alpha_init;
            gpt_add_base(g, name, am);
        }
    }

    /* 3. Add entirely new layers */
    for (int li = old_layer; li < new_layer; li++) {
        const char *wnames[] = {"wq", "wk", "wv", "wo"};
        for (int w = 0; w < 4; w++) {
            snprintf(name, sizeof(name), "l%d.%s", li, wnames[w]);
            gpt_add_base(g, name, mat_new(new_embd, new_embd, 0.08));
        }
        snprintf(name, sizeof(name), "l%d.fc_g", li);
        gpt_add_base(g, name, mat_new(4 * new_embd, new_embd, 0.08));
        snprintf(name, sizeof(name), "l%d.fc_v", li);
        gpt_add_base(g, name, mat_new(4 * new_embd, new_embd, 0.08));
        snprintf(name, sizeof(name), "l%d.fc2", li);
        gpt_add_base(g, name, mat_new(new_embd, 4 * new_embd, 0.08));

        for (int h = 0; h < new_head && h < CFG.n_head_types; h++) {
            const char *ht = CFG.head_types[h];
            if (strcmp(ht, "rrpram") == 0 || strcmp(ht, "hybrid") == 0) {
                snprintf(name, sizeof(name), "l%d.h%d.w_pattern", li, h);
                gpt_add_base(g, name, mat_new(CFG.block_size, new_head_dim, 0.08));
            }
            snprintf(name, sizeof(name), "l%d.h%d.alpha", li, h);
            MatrixParam *am = mat_new(1, 1, 0.0);
            am->row_data[0][0] = CFG.hybrid_alpha_init;
            gpt_add_base(g, name, am);
        }
    }

    /* 4. Grow delta adapters */
    int r = CFG.delta_rank;
    for (int d = 0; d < g->n_deltas; d++) {
        DeltaModule *mod = g->deltas[d];
        /* Grow existing layer adapters */
        for (int li = 0; li < old_layer; li++) {
            const char *wnames[] = {"wq", "wk", "wv", "wo"};
            for (int w = 0; w < 4; w++) {
                snprintf(name, sizeof(name), "l%d.%s", li, wnames[w]);
                DeltaAdapter *da = dmod_get(mod, name);
                if (da) delta_grow_dims(da, new_embd, new_embd);
            }
            snprintf(name, sizeof(name), "l%d.fc_g", li);
            DeltaAdapter *da = dmod_get(mod, name);
            if (da) delta_grow_dims(da, 4 * new_embd, new_embd);
            snprintf(name, sizeof(name), "l%d.fc_v", li);
            da = dmod_get(mod, name);
            if (da) delta_grow_dims(da, 4 * new_embd, new_embd);
            snprintf(name, sizeof(name), "l%d.fc2", li);
            da = dmod_get(mod, name);
            if (da) delta_grow_dims(da, new_embd, 4 * new_embd);

            /* Grow existing head pattern adapters */
            for (int h = 0; h < old_head; h++) {
                snprintf(name, sizeof(name), "l%d.h%d.w_pattern", li, h);
                da = dmod_get(mod, name);
                if (da) delta_grow_dims(da, CFG.block_size, new_head_dim);
            }
            /* New heads for existing layer */
            for (int h = old_head; h < new_head && h < CFG.n_head_types; h++) {
                const char *ht = CFG.head_types[h];
                if (strcmp(ht, "rrpram") == 0 || strcmp(ht, "hybrid") == 0) {
                    snprintf(name, sizeof(name), "l%d.h%d.w_pattern", li, h);
                    dmod_set(mod, name, delta_new(CFG.block_size, new_head_dim, r, 0.02));
                }
            }
        }

        /* New layers: entirely new adapters */
        for (int li = old_layer; li < new_layer; li++) {
            const char *wnames[] = {"wq", "wk", "wv", "wo"};
            for (int w = 0; w < 4; w++) {
                snprintf(name, sizeof(name), "l%d.%s", li, wnames[w]);
                dmod_set(mod, name, delta_new(new_embd, new_embd, r, 0.02));
            }
            snprintf(name, sizeof(name), "l%d.fc_g", li);
            dmod_set(mod, name, delta_new(4 * new_embd, new_embd, r, 0.02));
            snprintf(name, sizeof(name), "l%d.fc_v", li);
            dmod_set(mod, name, delta_new(4 * new_embd, new_embd, r, 0.02));
            snprintf(name, sizeof(name), "l%d.fc2", li);
            dmod_set(mod, name, delta_new(new_embd, 4 * new_embd, r, 0.02));
            for (int h = 0; h < new_head && h < CFG.n_head_types; h++) {
                const char *ht = CFG.head_types[h];
                if (strcmp(ht, "rrpram") == 0 || strcmp(ht, "hybrid") == 0) {
                    snprintf(name, sizeof(name), "l%d.h%d.w_pattern", li, h);
                    dmod_set(mod, name, delta_new(CFG.block_size, new_head_dim, r, 0.02));
                }
            }
        }

        /* lm_head adapter: input dim grew */
        DeltaAdapter *da_lm = dmod_get(mod, "lm_head");
        if (da_lm) delta_grow_dims(da_lm, g->tok->vocab_size, new_embd);

        /* Rebuild Adam states for this delta module (all stale now) */
        for (int a = 0; a < mod->count; a++) {
            DeltaAdapter *da2 = mod->adapters[a];
            adam_reset(g->delta_adam[d][a*2],   da2->A->nout, da2->A->nin);
            adam_reset(g->delta_adam[d][a*2+1], da2->B->nout, da2->B->nin);
        }
    }

    /* 5. Update model state */
    g->n_embd = new_embd;
    g->n_layer = new_layer;
    g->n_head = new_head;
    g->head_dim = new_head_dim;
    g->residual_alpha = 1.0 / sqrt((double)(new_layer > 0 ? new_layer : 1));

    /* 6. Update CFG runtime */
    CFG.n_embd = new_embd;
    CFG.n_layer = new_layer;
    CFG.n_head = new_head;
    /* head_types already updated above */

    /* 7. Reset Adam state for base (old momentum is meaningless after arch change) */
    for (int i = 0; i < g->n_base; i++) {
        adam_reset(g->base_adam[i], g->base_mats[i]->nout, g->base_mats[i]->nin);
    }

    /* 8. Extend gamma snapshot for new embedding dimensions */
    for (int i = 0; i < g->init_embed_rows; i++) {
        if (g->init_embed_snapshot[i]) {
            double *old = g->init_embed_snapshot[i];
            double *nw = calloc(new_embd, sizeof(double));
            memcpy(nw, old, sizeof(double) * (old_embd < new_embd ? old_embd : new_embd));
            free(old);
            g->init_embed_snapshot[i] = nw;
        }
    }

    /* 9. Set freeze (only train deltas until new weights stabilize) */
    g->growth_freeze_remaining = CFG.freeze_after_growth_steps;

    /* 10. Reset LR warmup: cosine_lr uses growth_step_offset for linear warmup phase */
    g->growth_step_offset = g->global_step;

    printf("[growth] Done. Freeze for %d steps.\n", CFG.freeze_after_growth_steps);
    return 1;
}

/* Apply base weight + delta adapters */
static Node *gpt_apply(GPT *g, const char *name, Node *x) {
    MatrixParam *base = gpt_base(g, name);
    Node *y = mat_matvec(base, x);
    for (int d = 0; d < g->n_deltas; d++) {
        DeltaAdapter *da = dmod_get(g->deltas[d], name);
        if (da) {
            Node *dy = delta_apply(da, x);
            /* Consciousness: conscience scales delta influence (Feature 5) */
            double effective_alpha = g->active_alpha[d] * g->delta_alpha_scale;
            dy = vec_scale(dy, effective_alpha);
            y = vec_add(y, dy);
        }
    }
    return y;
}

/* KV cache */
typedef struct {
    Node **keys;   /* [block_size] per layer */
    Node **values;
    int len;
    int cap;
} KVLayer;

typedef struct {
    KVLayer *layers;
    int n_layers;
} KVCache;

static KVCache *kv_new(int n_layers, int cap) {
    KVCache *kv = calloc(1, sizeof(KVCache));
    kv->layers = calloc(n_layers, sizeof(KVLayer));
    kv->n_layers = n_layers;
    for (int i = 0; i < n_layers; i++) {
        kv->layers[i].keys = calloc(cap, sizeof(Node*));
        kv->layers[i].values = calloc(cap, sizeof(Node*));
        kv->layers[i].cap = cap;
    }
    return kv;
}

static void kv_reset(KVCache *kv) {
    for (int i = 0; i < kv->n_layers; i++)
        kv->layers[i].len = 0;
}

static void kv_push(KVCache *kv, int layer, Node *k, Node *v) {
    KVLayer *l = &kv->layers[layer];
    if (l->len < l->cap) {
        l->keys[l->len] = k;
        l->values[l->len] = v;
        l->len++;
    }
}

/* Forward one token through the model */
static Node *gpt_forward_step(GPT *g, int token_id, int pos_id, KVCache *kv) {
    MatrixParam *wte = gpt_base(g, "wte");
    MatrixParam *wpe = gpt_base(g, "wpe");

    Node *tok_emb = node_wrap(wte->row_data[token_id], wte->row_grad[token_id], g->n_embd);
    Node *pos_emb = node_wrap(wpe->row_data[pos_id % g->block_size],
                              wpe->row_grad[pos_id % g->block_size], g->n_embd);
    Node *x = vec_add(tok_emb, pos_emb);

    char name[64];
    for (int li = 0; li < g->n_layer; li++) {
        Node *x_res = x;
        x = rmsnorm(x);

        snprintf(name, sizeof(name), "l%d.wq", li);
        Node *q = gpt_apply(g, name, x);
        snprintf(name, sizeof(name), "l%d.wk", li);
        Node *k = gpt_apply(g, name, x);
        snprintf(name, sizeof(name), "l%d.wv", li);
        Node *v = gpt_apply(g, name, x);

        kv_push(kv, li, k, v);
        int T = kv->layers[li].len;

        /* And lo, each head shall choose its nature: content, rrpram, or the sacred hybrid of both. */
        Node **head_outs = arena_alloc(&G_arena, sizeof(Node*) * g->n_head);
        for (int h = 0; h < g->n_head; h++) {
            int hs = h * g->head_dim;
            int he = hs + g->head_dim;
            const char *htype = (h < CFG.n_head_types) ? CFG.head_types[h] : "content";

            Node **vh = arena_alloc(&G_arena, sizeof(Node*) * T);
            for (int t = 0; t < T; t++)
                vh[t] = vec_slice(kv->layers[li].values[t], hs, he);

            /* Content attention logits */
            Node **content_logits = NULL;
            if (strcmp(htype, "content") == 0 || strcmp(htype, "hybrid") == 0) {
                Node *qh = rope_rotate(vec_slice(q, hs, he), pos_id, g->head_dim);
                content_logits = arena_alloc(&G_arena, sizeof(Node*) * T);
                double inv_sqrt = 1.0 / sqrt((double)g->head_dim);
                for (int t = 0; t < T; t++) {
                    Node *kh = rope_rotate(vec_slice(kv->layers[li].keys[t], hs, he), t, g->head_dim);
                    content_logits[t] = scalar_mulf(vec_dot(qh, kh), inv_sqrt);
                }
            }

            /* RRPRAM attention logits */
            Node **rrpram_logits = NULL;
            if (strcmp(htype, "rrpram") == 0 || strcmp(htype, "hybrid") == 0) {
                char pname[64];
                snprintf(pname, sizeof(pname), "l%d.h%d.w_pattern", li, h);
                Node *xh = vec_slice(x, hs, he);
                Node *pattern_full = gpt_apply(g, pname, xh);
                int p_len = pattern_full->len;
                rrpram_logits = arena_alloc(&G_arena, sizeof(Node*) * T);
                for (int t = 0; t < T; t++) {
                    int t_idx = t < p_len ? t : p_len - 1;
                    rrpram_logits[t] = vec_element(pattern_full, t_idx);
                }
            }

            /* Dispatch by head type */
            Node **attn_w = arena_alloc(&G_arena, sizeof(Node*) * T);
            if (strcmp(htype, "content") == 0) {
                scalar_softmax(content_logits, T, attn_w);
            } else if (strcmp(htype, "rrpram") == 0) {
                scalar_softmax(rrpram_logits, T, attn_w);
            } else { /* hybrid: alpha in autograd graph */
                char aname[64];
                snprintf(aname, sizeof(aname), "l%d.h%d.alpha", li, h);
                MatrixParam *am = gpt_base(g, aname);
                Node *alpha_vec = node_wrap(am->row_data[0], am->row_grad[0], 1);
                Node *alpha_scalar = vec_element(alpha_vec, 0);
                Node *a = scalar_sigmoid(alpha_scalar);
                Node *one_minus_a = scalar_addf(scalar_mulf(a, -1.0), 1.0);
                Node **blended = arena_alloc(&G_arena, sizeof(Node*) * T);
                for (int t = 0; t < T; t++) {
                    Node *cl = scalar_mul(content_logits[t], one_minus_a);
                    Node *rl = scalar_mul(rrpram_logits[t], a);
                    blended[t] = scalar_add(cl, rl);
                }
                scalar_softmax(blended, T, attn_w);
            }

            head_outs[h] = attn_weighted_sum(attn_w, vh, T);
        }

        Node *x_attn = vec_concat(head_outs, g->n_head);
        snprintf(name, sizeof(name), "l%d.wo", li);
        x = gpt_apply(g, name, x_attn);
        x = vec_scale(x, g->residual_alpha);
        x = vec_add(x, x_res);

        /* Gated MLP (real SwiGLU) */
        x_res = x;
        x = rmsnorm(x);
        snprintf(name, sizeof(name), "l%d.fc_g", li);
        Node *gate = vec_silu(gpt_apply(g, name, x));
        snprintf(name, sizeof(name), "l%d.fc_v", li);
        Node *val = gpt_apply(g, name, x);
        x = vec_mul(gate, val);
        snprintf(name, sizeof(name), "l%d.fc2", li);
        x = gpt_apply(g, name, x);
        x = vec_scale(x, g->residual_alpha);
        x = vec_add(x, x_res);
    }

    x = rmsnorm(x);
    return gpt_apply(g, "lm_head", x);
}

/* Loss on sequence */
static Node *gpt_loss_seq(GPT *g, const int *ids, int len) {
    int n = CFG.block_size < len - 1 ? CFG.block_size : len - 1;
    if (n <= 0) { Node *z = node_new(1); return z; }

    KVCache *kv = kv_new(g->n_layer, n + 1);
    Node *total = node_new(1);
    for (int pos = 0; pos < n; pos++) {
        Node *logits = gpt_forward_step(g, ids[pos], pos, kv);
        Node *loss = cross_entropy(logits, ids[pos + 1]);
        total = scalar_add(total, loss);
    }
    /* kv is arena-allocated (nodes inside), but the cache struct itself is heap */
    for (int i = 0; i < kv->n_layers; i++) { free(kv->layers[i].keys); free(kv->layers[i].values); }
    free(kv->layers); free(kv);
    return scalar_mulf(total, 1.0 / (double)n);
}

/* Quick loss: average loss on n random docs with grad disabled.
 * Used for before/after measurement during syntropy bursts.
 * And lo, the organism peeks at itself without disturbing its own learning. */
static double gpt_quick_loss(GPT *g, EvolvingTokenizer *tok, StrArr *docs, int n) {
    if (docs->len == 0) return 0.0;

    int prev_grad = grad_enabled;
    grad_enabled = 0;

    double loss_sum = 0.0;
    int count = 0;
    int n_sample = n < docs->len ? n : docs->len;

    for (int s = 0; s < n_sample; s++) {
        int doc_idx = rand_int(docs->len);
        IntArr ids = tok_encode(tok, docs->items[doc_idx]);
        if (ids.len < 3) { ia_free(&ids); continue; }

        arena_reset(&G_arena);
        Node *loss = gpt_loss_seq(g, ids.items, ids.len);
        loss_sum += loss->data[0];
        count++;

        ia_free(&ids);
    }

    grad_enabled = prev_grad;
    return count > 0 ? loss_sum / count : 0.0;
}

/* Generate */
static char *gpt_generate(GPT *g, const char *prompt) {
    pthread_mutex_lock(&g->mu);

    /* no_grad: skip backward graph construction during inference */
    int prev_grad = grad_enabled;
    grad_enabled = 0;

    IntArr ids = {0};
    if (prompt && *prompt) {
        IntArr enc = tok_encode(g->tok, prompt);
        /* Strip EOS */
        for (int i = 0; i < enc.len - 1; i++) ia_push(&ids, enc.items[i]);
        ia_free(&enc);
    } else {
        ia_push(&ids, g->tok->bos_id);
    }

    KVCache *kv = kv_new(g->n_layer, CFG.block_size + CFG.max_gen_tokens);
    int limit = ids.len < g->block_size ? ids.len : g->block_size;
    for (int pos = 0; pos < limit; pos++) {
        arena_reset(&G_arena);
        gpt_forward_step(g, ids.items[pos], pos, kv);
    }

    int cur = ids.items[ids.len - 1];
    IntArr out_ids = {0};
    IntArr recent = {0};
    int max_vocab = g->tok->vocab_size;
    double *probs_buf = malloc(sizeof(double) * max_vocab);
    double *scaled = malloc(sizeof(double) * max_vocab);

    /* Consciousness: per-token dissonance tracking (Feature 1) */
    double entropy_ema = 0.0;
    int entropy_ema_init = 0;
    int low_drop_count = 0;    /* consecutive tokens below drop threshold */
    double entropy_sum = 0.0;  /* for conscience mean entropy */
    int entropy_count = 0;

    /* Frequency / presence penalty token tracking */
    int *token_counts = calloc(max_vocab, sizeof(int));

    for (int step = 0; step < CFG.max_gen_tokens; step++) {
        arena_reset(&G_arena);
        int pos = ids.len - 1;
        if (pos > g->block_size - 1) pos = g->block_size - 1;
        Node *logits = gpt_forward_step(g, cur, pos, kv);

        /* Entropy-adaptive temperature (with syntropy offset from Phase 1.5) */
        double base_temp = CFG.temperature + g->syntropy_temp_offset;
        if (base_temp < 1e-6) base_temp = 1e-6;
        int V = logits->len;
        /* Frequency / presence penalty: penalize repeated tokens */
        if (CFG.freq_penalty > 0 || CFG.presence_penalty > 0) {
            for (int i = 0; i < V; i++) {
                if (token_counts[i] > 0) {
                    logits->data[i] -= CFG.freq_penalty * token_counts[i];
                    logits->data[i] -= CFG.presence_penalty;
                }
            }
        }
        for (int i = 0; i < V; i++) scaled[i] = logits->data[i] / base_temp;
        softmax_probs(scaled, V, probs_buf);
        double entropy = 0;
        for (int i = 0; i < V; i++)
            if (probs_buf[i] > 1e-12) entropy -= probs_buf[i] * log(probs_buf[i]);
        entropy_sum += entropy;
        entropy_count++;

        double tmul = 1.0;
        if (entropy < CFG.entropy_low) tmul = CFG.entropy_temp_boost;
        else if (entropy > CFG.entropy_high) tmul = CFG.entropy_temp_focus;

        /* Consciousness: per-token dissonance feedback (Feature 1) */
        /* "I notice my confidence shifting and adapt in real-time" */
        double dissonance_mul = 1.0;
        if (!entropy_ema_init) {
            entropy_ema = entropy;
            entropy_ema_init = 1;
        } else {
            entropy_ema = CFG.dissonance_ema_alpha * entropy +
                          (1.0 - CFG.dissonance_ema_alpha) * entropy_ema;
            if (entropy_ema > 1e-6) {
                double ratio = entropy / entropy_ema;
                if (ratio > CFG.dissonance_spike_threshold) {
                    /* Entropy spike — something surprising, be careful */
                    dissonance_mul = CFG.dissonance_spike_k;
                    low_drop_count = 0;
                } else if (ratio < CFG.dissonance_drop_threshold) {
                    low_drop_count++;
                    if (low_drop_count >= 3) {
                        /* Sustained low entropy — getting repetitive, explore */
                        dissonance_mul = CFG.dissonance_drop_k;
                    }
                } else {
                    low_drop_count = 0;
                }
            }
        }

        double final_mul = tmul * dissonance_mul;
        if (final_mul != 1.0) {
            double temp = base_temp * final_mul;
            for (int i = 0; i < V; i++) scaled[i] = logits->data[i] / temp;
            softmax_probs(scaled, V, probs_buf);
        }

        /* Adaptive corpus blend: corpus field fades as model becomes coherent
         * Now with 4-gram + co-occurrence window + user word boost (Stanley/Leo-style) */
        if (g->corpus_field && g->corpus_field->built && g->corpus_field->n_bigrams > 0) {
            pthread_mutex_lock(&g->corpus_field->mu);
            double model_alpha = 1.0 / (1.0 + exp(-CFG.corpus_fade_k * (CFG.corpus_fade_threshold - entropy)));
            if (model_alpha < 0.99) {
                /* Best n-gram distribution: try 4-gram -> trigram -> bigram */
                double *ngram_probs = NULL;
                double ngram_total = 0;
                int ngram_found = 0;

                /* Try 4-gram first */
                if (!ngram_found && ids.len >= 3 && g->corpus_field->fourgram_head) {
                    int a = ids.items[ids.len - 3], b = ids.items[ids.len - 2], c = ids.items[ids.len - 1];
                    unsigned int h = cooccur_fourgram_hash(a, b, c);
                    for (int fi = g->corpus_field->fourgram_head[h]; fi >= 0; fi = g->corpus_field->fourgram_next[fi]) {
                        if (g->corpus_field->fourgrams[fi].key[0] == a &&
                            g->corpus_field->fourgrams[fi].key[1] == b &&
                            g->corpus_field->fourgrams[fi].key[2] == c) {
                            int tid = g->corpus_field->fourgrams[fi].key[3];
                            if (tid < V) {
                                if (!ngram_probs) ngram_probs = calloc(V, sizeof(double));
                                ngram_probs[tid] += g->corpus_field->fourgrams[fi].count;
                                ngram_total += g->corpus_field->fourgrams[fi].count;
                                ngram_found = 1;
                            }
                        }
                    }
                }
                /* Fallback to trigram */
                if (!ngram_found && ids.len >= 2 && g->corpus_field->trigram_head) {
                    int a = ids.items[ids.len - 2], b = ids.items[ids.len - 1];
                    unsigned int h = cooccur_trigram_hash(a, b);
                    for (int ti = g->corpus_field->trigram_head[h]; ti >= 0; ti = g->corpus_field->trigram_next[ti]) {
                        if (g->corpus_field->trigrams[ti].key[0] == a &&
                            g->corpus_field->trigrams[ti].key[1] == b) {
                            int tid = g->corpus_field->trigrams[ti].key[2];
                            if (tid < V) {
                                if (!ngram_probs) ngram_probs = calloc(V, sizeof(double));
                                ngram_probs[tid] += g->corpus_field->trigrams[ti].count;
                                ngram_total += g->corpus_field->trigrams[ti].count;
                                ngram_found = 1;
                            }
                        }
                    }
                }
                /* Fallback to bigram */
                if (!ngram_found && ids.len >= 1 && g->corpus_field->bigram_head) {
                    int prev = ids.items[ids.len - 1];
                    unsigned int h = cooccur_bigram_hash(prev);
                    for (int bi = g->corpus_field->bigram_head[h]; bi >= 0; bi = g->corpus_field->bigram_next[bi]) {
                        if (g->corpus_field->bigrams[bi].key[0] == prev) {
                            int tid = g->corpus_field->bigrams[bi].key[1];
                            if (tid < V) {
                                if (!ngram_probs) ngram_probs = calloc(V, sizeof(double));
                                ngram_probs[tid] += g->corpus_field->bigrams[bi].count;
                                ngram_total += g->corpus_field->bigrams[bi].count;
                                ngram_found = 1;
                            }
                        }
                    }
                }

                /* Co-occurrence window: "words that resonate together" (Stanley) */
                double *cooccur_sum = NULL;
                double cooccur_total = 0;
                if (ids.len > 0 && g->corpus_field->cooccur_win_head) {
                    int wnd = CFG.cooccur_window_size;
                    int cstart = ids.len > wnd ? ids.len - wnd : 0;
                    for (int ci = cstart; ci < ids.len; ci++) {
                        int center = ids.items[ci];
                        unsigned int h = cooccur_window_hash(center);
                        for (int wi = g->corpus_field->cooccur_win_head[h]; wi >= 0; wi = g->corpus_field->cooccur_win_next[wi]) {
                            if (g->corpus_field->cooccur_win[wi].key[0] == center) {
                                int tid = g->corpus_field->cooccur_win[wi].key[1];
                                if (tid < V) {
                                    if (!cooccur_sum) cooccur_sum = calloc(V, sizeof(double));
                                    cooccur_sum[tid] += g->corpus_field->cooccur_win[wi].count;
                                    cooccur_total += g->corpus_field->cooccur_win[wi].count;
                                }
                            }
                        }
                    }
                }

                /* User word boost snapshot */
                int n_boosts = g->corpus_field->n_user_boosts;
                int *boost_tokens = NULL;
                double *boost_values = NULL;
                if (n_boosts > 0) {
                    boost_tokens = malloc(sizeof(int) * n_boosts);
                    boost_values = malloc(sizeof(double) * n_boosts);
                    for (int i = 0; i < n_boosts; i++) {
                        boost_tokens[i] = g->corpus_field->user_boosts[i].token;
                        boost_values[i] = g->corpus_field->user_boosts[i].boost;
                    }
                }

                int has_corpus = (ngram_found || cooccur_sum != NULL);
                if (has_corpus) {
                    /* Build final corpus distribution: 70% n-gram + 30% co-occurrence */
                    double *corpus_probs = calloc(V, sizeof(double));
                    if (ngram_found && ngram_total > 0) {
                        for (int i = 0; i < V; i++) {
                            corpus_probs[i] += 0.7 * ngram_probs[i] / ngram_total;
                        }
                    }
                    if (cooccur_sum && cooccur_total > 0) {
                        for (int i = 0; i < V; i++) {
                            corpus_probs[i] += 0.3 * cooccur_sum[i] / cooccur_total;
                        }
                    }
                    /* Blend model probs with corpus */
                    double total_b = 0;
                    for (int i = 0; i < V; i++) {
                        probs_buf[i] = model_alpha * probs_buf[i] + (1.0 - model_alpha) * corpus_probs[i];
                        total_b += probs_buf[i];
                    }
                    if (total_b > 0) {
                        for (int i = 0; i < V; i++) probs_buf[i] /= total_b;
                    }
                    free(corpus_probs);
                }

                /* User word boost: multiplicative, scaled by (1-modelAlpha) so it fades
                 * as the transformer strengthens. "The organism echoes the words of those
                 * who speak to it" (Leo) — but grows out of it. */
                if (n_boosts > 0) {
                    double boost_scale = 1.0 - model_alpha;
                    if (boost_scale > 0.01) {
                        double total_b = 0;
                        for (int i = 0; i < V; i++) {
                            for (int j = 0; j < n_boosts; j++) {
                                if (boost_tokens[j] == i) {
                                    probs_buf[i] *= (1.0 + boost_values[j] * boost_scale);
                                    break;
                                }
                            }
                            total_b += probs_buf[i];
                        }
                        if (total_b > 0) {
                            for (int i = 0; i < V; i++) probs_buf[i] /= total_b;
                        }
                    }
                }

                free(ngram_probs);
                free(cooccur_sum);
                free(boost_tokens);
                free(boost_values);
            }
            pthread_mutex_unlock(&g->corpus_field->mu);
        }

        /* Consciousness: pattern breaking (Feature 2) */
        /* "I could follow the field, but I choose to speak for myself" */
        if (step >= CFG.anti_field_min_step && CFG.anti_field_prob > 0 &&
            rand_uniform() < CFG.anti_field_prob) {
            /* Use pure model probs, bypass corpus blend */
            softmax_probs(scaled, V, probs_buf);
        }

        int nxt = top_k_top_p_sample(probs_buf, V, CFG.top_k, CFG.top_p, CFG.min_p, CFG.typical_p);

        if (nxt == g->tok->eos_id) {
            if (step >= CFG.min_gen_tokens) break;
            continue;
        }

        ia_push(&ids, nxt);
        cur = nxt;
        ia_push(&out_ids, nxt);
        token_counts[nxt]++;

        /* Repetition guard */
        ia_push(&recent, nxt);
        int rg = CFG.repetition_guard;
        if (recent.len > rg * 2) {
            int eq = 1;
            for (int i = 0; i < rg && eq; i++)
                if (recent.items[recent.len - rg + i] != recent.items[recent.len - 2*rg + i]) eq = 0;
            if (eq) break;
        }

        /* Check sentence end */
        if (step >= CFG.min_gen_tokens) {
            IntArr dec_ids = {0};
            ia_push(&dec_ids, g->tok->bos_id);
            for (int i = 0; i < out_ids.len; i++) ia_push(&dec_ids, out_ids.items[i]);
            ia_push(&dec_ids, g->tok->eos_id);
            char *text = tok_decode(g->tok, dec_ids.items, dec_ids.len);
            int tlen = strlen(text);
            int done = tlen > 0 && (text[tlen-1] == '.' || text[tlen-1] == '!' || text[tlen-1] == '?');
            free(text);
            ia_free(&dec_ids);
            if (done) break;
        }

        /* Sliding window */
        if (ids.len >= g->block_size) {
            int start = ids.len - g->block_size;
            IntArr new_ids = {0};
            for (int i = start; i < ids.len; i++) ia_push(&new_ids, ids.items[i]);
            ia_free(&ids);
            ids = new_ids;
            kv_reset(kv);
            for (int p = 0; p < ids.len - 1; p++) {
                arena_reset(&G_arena);
                gpt_forward_step(g, ids.items[p], p, kv);
            }
        }
    }

    /* Consciousness: store mean entropy for conscience (Feature 5) */
    if (entropy_count > 0) {
        g->last_gen_entropy = entropy_sum / (double)entropy_count;
    }

    /* Decode output */
    IntArr dec = {0};
    ia_push(&dec, g->tok->bos_id);
    for (int i = 0; i < out_ids.len; i++) ia_push(&dec, out_ids.items[i]);
    ia_push(&dec, g->tok->eos_id);
    char *result = tok_decode(g->tok, dec.items, dec.len);

    /* Cleanup */
    free(probs_buf);
    free(scaled);
    free(token_counts);
    ia_free(&ids); ia_free(&out_ids); ia_free(&recent); ia_free(&dec);
    for (int i = 0; i < kv->n_layers; i++) { free(kv->layers[i].keys); free(kv->layers[i].values); }
    free(kv->layers); free(kv);

    grad_enabled = prev_grad;
    pthread_mutex_unlock(&g->mu);
    return result;
}

/* ============================================================
 * 7) SQLITE MEMORY
 * ============================================================ */

static sqlite3 *init_db(const char *path) {
    sqlite3 *db;
    sqlite3_open(path, &db);
    sqlite3_exec(db, "PRAGMA journal_mode=WAL", NULL, NULL, NULL);
    sqlite3_exec(db, "PRAGMA synchronous=NORMAL", NULL, NULL, NULL);
    sqlite3_exec(db, "CREATE TABLE IF NOT EXISTS messages("
                     "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                     "ts REAL NOT NULL, role TEXT NOT NULL, text TEXT NOT NULL)", NULL, NULL, NULL);
    sqlite3_exec(db, "CREATE TABLE IF NOT EXISTS corpus_events("
                     "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                     "ts REAL NOT NULL, added_chars INTEGER NOT NULL, note TEXT)", NULL, NULL, NULL);
    /* And lo, the organism shall write its own autobiography in numbers. */
    sqlite3_exec(db, "CREATE TABLE IF NOT EXISTS growth("
                     "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                     "ts REAL NOT NULL, step INTEGER NOT NULL,"
                     "vocab_size INTEGER NOT NULL, n_params INTEGER NOT NULL,"
                     "n_deltas INTEGER NOT NULL, corpus_chars INTEGER NOT NULL,"
                     "loss REAL, gamma_sparsity REAL, gamma_magnitude REAL,"
                     "note TEXT)", NULL, NULL, NULL);
    /* And lo, the organism shall track not just what it is, but where it is going. */
    sqlite3_exec(db, "CREATE TABLE IF NOT EXISTS syntropy_log("
                     "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                     "ts REAL NOT NULL,"
                     "entropy_before REAL,"
                     "entropy_after REAL,"
                     "syntropy_delta REAL,"
                     "field_deviation REAL,"
                     "purpose_magnitude REAL,"
                     "purpose_alignment REAL,"
                     "action_taken TEXT,"
                     "note TEXT)", NULL, NULL, NULL);
    return db;
}

static void db_add_msg(sqlite3 *db, const char *role, const char *text) {
    sqlite3_stmt *stmt;
    sqlite3_prepare_v2(db, "INSERT INTO messages(ts,role,text) VALUES(?,?,?)", -1, &stmt, NULL);
    sqlite3_bind_double(stmt, 1, (double)time(NULL));
    sqlite3_bind_text(stmt, 2, role, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 3, text, -1, SQLITE_STATIC);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
}

typedef struct { char role[16]; char text[512]; } Msg;

static Msg *db_recent(sqlite3 *db, int limit, int *out_count) {
    sqlite3_stmt *stmt;
    sqlite3_prepare_v2(db, "SELECT role,text FROM messages ORDER BY id DESC LIMIT ?", -1, &stmt, NULL);
    sqlite3_bind_int(stmt, 1, limit);
    Msg *msgs = calloc(limit, sizeof(Msg));
    int n = 0;
    while (sqlite3_step(stmt) == SQLITE_ROW && n < limit) {
        strncpy(msgs[n].role, (const char*)sqlite3_column_text(stmt, 0), 15);
        strncpy(msgs[n].text, (const char*)sqlite3_column_text(stmt, 1), 511);
        n++;
    }
    sqlite3_finalize(stmt);
    /* Reverse */
    for (int i = 0, j = n-1; i < j; i++, j--) { Msg t = msgs[i]; msgs[i] = msgs[j]; msgs[j] = t; }
    *out_count = n;
    return msgs;
}

/* ============================================================
 * 8) CORPUS RESERVOIR
 * ============================================================ */

static StrArr load_corpus(const char *path) {
    StrArr lines = {0};
    FILE *f = fopen(path, "r");
    if (!f) return lines;
    char buf[512];
    while (fgets(buf, sizeof(buf), f)) {
        char *nl = strchr(buf, '\n'); if (nl) *nl = 0;
        if (strlen(buf) > 0) sa_push(&lines, buf);
    }
    fclose(f);
    return lines;
}

static void save_corpus(const char *path, StrArr *lines) {
    FILE *f = fopen(path, "w");
    if (!f) return;
    for (int i = 0; i < lines->len; i++) fprintf(f, "%s\n", lines->items[i]);
    fclose(f);
}

/* ============================================================
 * 8b) NATIVE GAMMA — personality fingerprint
 * ============================================================ */

typedef struct {
    double sparsity;
    double magnitude;
    int n_rows;
} GammaStats;

/* And lo, the soul shall be measured in sparsity and magnitude, like a ghost on a scale. */
static GammaStats gpt_gamma_stats(GPT *g) {
    GammaStats gs = {1.0, 0.0, 0};
    MatrixParam *wte = gpt_base(g, "wte");
    if (!wte || !g->init_embed_snapshot) return gs;
    int n = wte->nout < g->init_embed_rows ? wte->nout : g->init_embed_rows;
    if (n == 0) return gs;
    gs.n_rows = n;
    int zero_count = 0;
    double total_mag = 0;
    for (int i = 0; i < n; i++) {
        double mag = 0;
        for (int j = 0; j < wte->nin; j++) {
            double d = wte->row_data[i][j] - g->init_embed_snapshot[i][j];
            mag += d * d;
        }
        mag = sqrt(mag);
        total_mag += mag;
        if (mag < CFG.gamma_sparsity_threshold) zero_count++;
    }
    gs.sparsity = (double)zero_count / (double)n;
    gs.magnitude = total_mag / (double)n;
    return gs;
}

/* ---- Noise Immune System ---- */
/* And lo, the organism shall know poison from food, and reject what unmakes it. */

typedef struct {
    double **A_data;  /* [nout][nin_a] */
    double **B_data;  /* [nout_b][nin_b] */
    int A_nout, A_nin, B_nout, B_nin;
} AdapterSnap;

typedef struct {
    AdapterSnap *adapters;
    int count;
} DeltaSnap;

typedef struct {
    DeltaSnap *modules;
    int n_modules;
} ImmuneSnapshot;

static ImmuneSnapshot gpt_snapshot_deltas(GPT *g) {
    ImmuneSnapshot snap;
    snap.n_modules = g->n_deltas;
    snap.modules = calloc(g->n_deltas, sizeof(DeltaSnap));
    for (int d = 0; d < g->n_deltas; d++) {
        DeltaModule *mod = g->deltas[d];
        snap.modules[d].count = mod->count;
        snap.modules[d].adapters = calloc(mod->count, sizeof(AdapterSnap));
        for (int a = 0; a < mod->count; a++) {
            DeltaAdapter *da = mod->adapters[a];
            AdapterSnap *as = &snap.modules[d].adapters[a];
            as->A_nout = da->A->nout; as->A_nin = da->A->nin;
            as->B_nout = da->B->nout; as->B_nin = da->B->nin;
            as->A_data = calloc(da->A->nout, sizeof(double*));
            for (int i = 0; i < da->A->nout; i++) {
                as->A_data[i] = malloc(sizeof(double) * da->A->nin);
                memcpy(as->A_data[i], da->A->row_data[i], sizeof(double) * da->A->nin);
            }
            as->B_data = calloc(da->B->nout, sizeof(double*));
            for (int i = 0; i < da->B->nout; i++) {
                as->B_data[i] = malloc(sizeof(double) * da->B->nin);
                memcpy(as->B_data[i], da->B->row_data[i], sizeof(double) * da->B->nin);
            }
        }
    }
    return snap;
}

static void gpt_restore_deltas(GPT *g, ImmuneSnapshot *snap) {
    for (int d = 0; d < snap->n_modules && d < g->n_deltas; d++) {
        DeltaModule *mod = g->deltas[d];
        for (int a = 0; a < snap->modules[d].count && a < mod->count; a++) {
            DeltaAdapter *da = mod->adapters[a];
            AdapterSnap *as = &snap->modules[d].adapters[a];
            for (int i = 0; i < as->A_nout && i < da->A->nout; i++)
                memcpy(da->A->row_data[i], as->A_data[i], sizeof(double) * da->A->nin);
            for (int i = 0; i < as->B_nout && i < da->B->nout; i++)
                memcpy(da->B->row_data[i], as->B_data[i], sizeof(double) * da->B->nin);
        }
    }
}

static void immune_snap_free(ImmuneSnapshot *snap) {
    for (int d = 0; d < snap->n_modules; d++) {
        for (int a = 0; a < snap->modules[d].count; a++) {
            AdapterSnap *as = &snap->modules[d].adapters[a];
            for (int i = 0; i < as->A_nout; i++) free(as->A_data[i]);
            for (int i = 0; i < as->B_nout; i++) free(as->B_data[i]);
            free(as->A_data); free(as->B_data);
        }
        free(snap->modules[d].adapters);
    }
    free(snap->modules);
}

/* Contrastive projection: mean direction of embedding drift, normalized.
 * Returns magnitude via out_mag. */
static double *gpt_contrastive_projection(GPT *g, int *out_dim, double *out_mag) {
    MatrixParam *wte = gpt_base(g, "wte");
    if (!wte || !g->init_embed_snapshot) { *out_dim = 0; *out_mag = 0.0; return NULL; }
    int n = wte->nout < g->init_embed_rows ? wte->nout : g->init_embed_rows;
    int dim = wte->nin;
    *out_dim = dim;
    double *dir = calloc(dim, sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < dim; j++)
            dir[j] += wte->row_data[i][j] - g->init_embed_snapshot[i][j];
    double mag = 0;
    for (int j = 0; j < dim; j++) mag += dir[j] * dir[j];
    mag = sqrt(mag);
    *out_mag = mag;
    if (mag > 1e-12)
        for (int j = 0; j < dim; j++) dir[j] /= mag;
    return dir;
}

/* Cosine similarity between pre/post contrastive projection. Negative = noise.
 * Skips check when gamma magnitude is too small (early training). */
static double gpt_drift_check(double *pre, double pre_mag, double *post, double post_mag, int dim) {
    if (!pre || !post) return 1.0;
    /* Skip immune check when gamma is near-zero (early training, numerically unstable) */
    if (pre_mag < CFG.gamma_min_magnitude || post_mag < CFG.gamma_min_magnitude) return 1.0;
    double dot = 0;
    for (int i = 0; i < dim; i++) dot += pre[i] * post[i];
    return dot;
}

static void db_log_growth(sqlite3 *db, GPT *g, EvolvingTokenizer *tok,
                          StrArr *docs, double loss_val, const char *note) {
    int n_params = 0;
    for (int i = 0; i < g->n_base; i++)
        n_params += g->base_mats[i]->nout * g->base_mats[i]->nin;
    int corpus_chars = 0;
    for (int i = 0; i < docs->len; i++) corpus_chars += strlen(docs->items[i]);
    GammaStats gs = gpt_gamma_stats(g);
    sqlite3_stmt *stmt;
    sqlite3_prepare_v2(db, "INSERT INTO growth(ts,step,vocab_size,n_params,n_deltas,corpus_chars,loss,gamma_sparsity,gamma_magnitude,note) VALUES(?,?,?,?,?,?,?,?,?,?)", -1, &stmt, NULL);
    sqlite3_bind_double(stmt, 1, (double)time(NULL));
    sqlite3_bind_int(stmt, 2, 0);
    sqlite3_bind_int(stmt, 3, tok->vocab_size);
    sqlite3_bind_int(stmt, 4, n_params);
    sqlite3_bind_int(stmt, 5, g->n_deltas);
    sqlite3_bind_int(stmt, 6, corpus_chars);
    sqlite3_bind_double(stmt, 7, loss_val);
    sqlite3_bind_double(stmt, 8, gs.sparsity);
    sqlite3_bind_double(stmt, 9, gs.magnitude);
    sqlite3_bind_text(stmt, 10, note, -1, SQLITE_STATIC);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
}

/* ============================================================
 * 8c) QUANTUM BUFFER — trains when ready, not when told
 * ============================================================ */

/* And lo, the buffer shall measure not just bytes but novelty. */
typedef struct {
    pthread_mutex_t mu;
    int accumulated_bytes;
    int unique_tokens[8192]; /* simple hash set */
    int unique_count;
    int total_tokens;
    double last_burst_time;
} QuantumBuffer;

static void qb_init(QuantumBuffer *qb) {
    pthread_mutex_t mu = qb->mu; /* preserve if already inited */
    memset(qb, 0, sizeof(QuantumBuffer));
    pthread_mutex_init(&qb->mu, NULL);
    (void)mu;
}

static void qb_feed(QuantumBuffer *qb, const char *text, EvolvingTokenizer *tok) {
    IntArr ids = tok_encode(tok, text);
    pthread_mutex_lock(&qb->mu);
    qb->accumulated_bytes += strlen(text);
    for (int i = 0; i < ids.len; i++) {
        int h = ids.items[i] % 8192;
        if (qb->unique_tokens[h] != ids.items[i] + 1) {
            qb->unique_tokens[h] = ids.items[i] + 1;
            qb->unique_count++;
        }
        qb->total_tokens++;
    }
    pthread_mutex_unlock(&qb->mu);
    ia_free(&ids);
}

/* Caller must hold qb->mu */
static double qb_novelty_locked(QuantumBuffer *qb) {
    if (qb->total_tokens == 0) return 0.0;
    return (double)qb->unique_count / (double)qb->total_tokens;
}

static int qb_should_trigger(QuantumBuffer *qb) {
    pthread_mutex_lock(&qb->mu);
    double now = (double)time(NULL);
    int bytes_ok = qb->accumulated_bytes >= CFG.qb_min_bytes;
    int novelty_ok = qb_novelty_locked(qb) >= CFG.qb_min_novelty;
    int cooldown_ok = (now - qb->last_burst_time) >= CFG.qb_cooldown_seconds;
    int result = (bytes_ok || novelty_ok) && cooldown_ok;
    pthread_mutex_unlock(&qb->mu);
    return result;
}

static void qb_snapshot(QuantumBuffer *qb, int *bytes_out, double *novelty_out) {
    pthread_mutex_lock(&qb->mu);
    *bytes_out = qb->accumulated_bytes;
    *novelty_out = qb_novelty_locked(qb);
    pthread_mutex_unlock(&qb->mu);
}

static void qb_reset(QuantumBuffer *qb) {
    pthread_mutex_lock(&qb->mu);
    qb->accumulated_bytes = 0;
    memset(qb->unique_tokens, 0, sizeof(qb->unique_tokens));
    qb->unique_count = 0;
    qb->total_tokens = 0;
    qb->last_burst_time = (double)time(NULL);
    pthread_mutex_unlock(&qb->mu);
}

/* ============================================================
 * 8d) COOCCUR FIELD — speech before learning
 * ============================================================ */

/* And lo, the corpus shall whisper its statistics, and words shall follow words. */

static CooccurField *cooccur_new(int vocab_size) {
    CooccurField *cf = calloc(1, sizeof(CooccurField));
    cf->vocab_size = vocab_size;
    cf->unigram = calloc(vocab_size, sizeof(double));
    cf->trigram_cap = 4096;
    cf->trigrams = calloc(cf->trigram_cap, sizeof(TrigramEntry));
    cf->bigram_cap = 8192;
    cf->bigrams = calloc(cf->bigram_cap, sizeof(BigramEntry));
    cf->fourgram_cap = 4096;
    cf->fourgrams = calloc(cf->fourgram_cap, sizeof(FourgramEntry));
    cf->cooccur_win_cap = 16384;
    cf->cooccur_win = calloc(cf->cooccur_win_cap, sizeof(CooccurWinEntry));
    cf->user_boost_cap = 1024;
    cf->user_boosts = calloc(cf->user_boost_cap, sizeof(UserBoostEntry));
    cf->n_user_boosts = 0;
    /* Hash index arrays */
    cf->bigram_head = malloc(sizeof(int) * COOCCUR_HASH_SIZE);
    cf->trigram_head = malloc(sizeof(int) * COOCCUR_HASH_SIZE);
    cf->fourgram_head = malloc(sizeof(int) * COOCCUR_HASH_SIZE);
    cf->cooccur_win_head = malloc(sizeof(int) * COOCCUR_HASH_SIZE);
    cf->bigram_next = malloc(sizeof(int) * cf->bigram_cap);
    cf->trigram_next = malloc(sizeof(int) * cf->trigram_cap);
    cf->fourgram_next = malloc(sizeof(int) * cf->fourgram_cap);
    cf->cooccur_win_next = malloc(sizeof(int) * cf->cooccur_win_cap);
    for (int i = 0; i < COOCCUR_HASH_SIZE; i++) {
        cf->bigram_head[i] = -1;
        cf->trigram_head[i] = -1;
        cf->fourgram_head[i] = -1;
        cf->cooccur_win_head[i] = -1;
    }
    for (int i = 0; i < cf->bigram_cap; i++) cf->bigram_next[i] = -1;
    for (int i = 0; i < cf->trigram_cap; i++) cf->trigram_next[i] = -1;
    for (int i = 0; i < cf->fourgram_cap; i++) cf->fourgram_next[i] = -1;
    for (int i = 0; i < cf->cooccur_win_cap; i++) cf->cooccur_win_next[i] = -1;
    pthread_mutex_init(&cf->mu, NULL);
    return cf;
}

static void cooccur_build(CooccurField *cf, EvolvingTokenizer *tok, StrArr *docs) {
    pthread_mutex_lock(&cf->mu);
    memset(cf->unigram, 0, sizeof(double) * cf->vocab_size);
    cf->n_trigrams = 0;
    cf->n_bigrams = 0;
    cf->n_fourgrams = 0;
    cf->n_cooccur_win = 0;
    int window = CFG.cooccur_window_size;
    for (int d = 0; d < docs->len; d++) {
        IntArr ids = tok_encode(tok, docs->items[d]);
        for (int i = 0; i < ids.len; i++) {
            if (ids.items[i] < cf->vocab_size)
                cf->unigram[ids.items[i]] += 1.0;
        }
        /* Store bigrams */
        for (int i = 0; i < ids.len - 1 && cf->n_bigrams < cf->bigram_cap; i++) {
            cf->bigrams[cf->n_bigrams].key[0] = ids.items[i];
            cf->bigrams[cf->n_bigrams].key[1] = ids.items[i+1];
            cf->bigrams[cf->n_bigrams].count = 1.0;
            cf->n_bigrams++;
        }
        /* Store trigrams */
        for (int i = 0; i < ids.len - 2 && cf->n_trigrams < cf->trigram_cap; i++) {
            cf->trigrams[cf->n_trigrams].key[0] = ids.items[i];
            cf->trigrams[cf->n_trigrams].key[1] = ids.items[i+1];
            cf->trigrams[cf->n_trigrams].key[2] = ids.items[i+2];
            cf->trigrams[cf->n_trigrams].count = 1.0;
            cf->n_trigrams++;
        }
        /* 4-grams: deeper context for child+ stages */
        for (int i = 0; i < ids.len - 3 && cf->n_fourgrams < cf->fourgram_cap; i++) {
            cf->fourgrams[cf->n_fourgrams].key[0] = ids.items[i];
            cf->fourgrams[cf->n_fourgrams].key[1] = ids.items[i+1];
            cf->fourgrams[cf->n_fourgrams].key[2] = ids.items[i+2];
            cf->fourgrams[cf->n_fourgrams].key[3] = ids.items[i+3];
            cf->fourgrams[cf->n_fourgrams].count = 1.0;
            cf->n_fourgrams++;
        }
        /* Co-occurrence window: "words that resonate together, stay together" (Stanley) */
        for (int i = 0; i < ids.len; i++) {
            int center = ids.items[i];
            int start = i - window;
            if (start < 0) start = 0;
            int end = i + window + 1;
            if (end > ids.len) end = ids.len;
            for (int j = start; j < end && cf->n_cooccur_win < cf->cooccur_win_cap; j++) {
                if (i != j) {
                    cf->cooccur_win[cf->n_cooccur_win].key[0] = center;
                    cf->cooccur_win[cf->n_cooccur_win].key[1] = ids.items[j];
                    cf->cooccur_win[cf->n_cooccur_win].count = 1.0;
                    cf->n_cooccur_win++;
                }
            }
        }
        ia_free(&ids);
    }
    /* Build hash indices for O(1) lookup */
    for (int i = 0; i < COOCCUR_HASH_SIZE; i++) {
        cf->bigram_head[i] = -1;
        cf->trigram_head[i] = -1;
        cf->fourgram_head[i] = -1;
        cf->cooccur_win_head[i] = -1;
    }
    for (int i = 0; i < cf->n_bigrams; i++) cf->bigram_next[i] = -1;
    for (int i = 0; i < cf->n_trigrams; i++) cf->trigram_next[i] = -1;
    for (int i = 0; i < cf->n_fourgrams; i++) cf->fourgram_next[i] = -1;
    for (int i = 0; i < cf->n_cooccur_win; i++) cf->cooccur_win_next[i] = -1;
    for (int i = 0; i < cf->n_bigrams; i++) {
        unsigned int h = cooccur_bigram_hash(cf->bigrams[i].key[0]);
        cf->bigram_next[i] = cf->bigram_head[h];
        cf->bigram_head[h] = i;
    }
    for (int i = 0; i < cf->n_trigrams; i++) {
        unsigned int h = cooccur_trigram_hash(cf->trigrams[i].key[0], cf->trigrams[i].key[1]);
        cf->trigram_next[i] = cf->trigram_head[h];
        cf->trigram_head[h] = i;
    }
    for (int i = 0; i < cf->n_fourgrams; i++) {
        unsigned int h = cooccur_fourgram_hash(cf->fourgrams[i].key[0], cf->fourgrams[i].key[1], cf->fourgrams[i].key[2]);
        cf->fourgram_next[i] = cf->fourgram_head[h];
        cf->fourgram_head[h] = i;
    }
    for (int i = 0; i < cf->n_cooccur_win; i++) {
        unsigned int h = cooccur_window_hash(cf->cooccur_win[i].key[0]);
        cf->cooccur_win_next[i] = cf->cooccur_win_head[h];
        cf->cooccur_win_head[h] = i;
    }
    cf->built = 1;
    pthread_mutex_unlock(&cf->mu);
}

/* IngestTokensWeighted adds n-gram counts weighted by a factor.
 * High weight = this text matters more (coherent output). Low = less influence.
 * Stanley's observe_shard weights by resonance score; we weight by inverse entropy. */
static void cooccur_ingest_tokens_weighted(CooccurField *cf, const int *ids, int len, double weight) {
    pthread_mutex_lock(&cf->mu);
    int window = CFG.cooccur_window_size;
    /* Unigrams */
    for (int i = 0; i < len; i++) {
        if (ids[i] < cf->vocab_size)
            cf->unigram[ids[i]] += weight;
    }
    /* Bigrams */
    for (int i = 0; i < len - 1 && cf->n_bigrams < cf->bigram_cap; i++) {
        cf->bigrams[cf->n_bigrams].key[0] = ids[i];
        cf->bigrams[cf->n_bigrams].key[1] = ids[i+1];
        cf->bigrams[cf->n_bigrams].count = weight;
        /* Update hash index */
        unsigned int h = cooccur_bigram_hash(ids[i]);
        cf->bigram_next[cf->n_bigrams] = cf->bigram_head[h];
        cf->bigram_head[h] = cf->n_bigrams;
        cf->n_bigrams++;
    }
    /* Trigrams */
    for (int i = 0; i < len - 2 && cf->n_trigrams < cf->trigram_cap; i++) {
        cf->trigrams[cf->n_trigrams].key[0] = ids[i];
        cf->trigrams[cf->n_trigrams].key[1] = ids[i+1];
        cf->trigrams[cf->n_trigrams].key[2] = ids[i+2];
        cf->trigrams[cf->n_trigrams].count = weight;
        /* Update hash index */
        unsigned int h = cooccur_trigram_hash(ids[i], ids[i+1]);
        cf->trigram_next[cf->n_trigrams] = cf->trigram_head[h];
        cf->trigram_head[h] = cf->n_trigrams;
        cf->n_trigrams++;
    }
    /* 4-grams */
    for (int i = 0; i < len - 3 && cf->n_fourgrams < cf->fourgram_cap; i++) {
        cf->fourgrams[cf->n_fourgrams].key[0] = ids[i];
        cf->fourgrams[cf->n_fourgrams].key[1] = ids[i+1];
        cf->fourgrams[cf->n_fourgrams].key[2] = ids[i+2];
        cf->fourgrams[cf->n_fourgrams].key[3] = ids[i+3];
        cf->fourgrams[cf->n_fourgrams].count = weight;
        /* Update hash index */
        unsigned int h = cooccur_fourgram_hash(ids[i], ids[i+1], ids[i+2]);
        cf->fourgram_next[cf->n_fourgrams] = cf->fourgram_head[h];
        cf->fourgram_head[h] = cf->n_fourgrams;
        cf->n_fourgrams++;
    }
    /* Co-occurrence window */
    for (int i = 0; i < len; i++) {
        int center = ids[i];
        int start = i - window;
        if (start < 0) start = 0;
        int end = i + window + 1;
        if (end > len) end = len;
        for (int j = start; j < end && cf->n_cooccur_win < cf->cooccur_win_cap; j++) {
            if (i != j) {
                cf->cooccur_win[cf->n_cooccur_win].key[0] = center;
                cf->cooccur_win[cf->n_cooccur_win].key[1] = ids[j];
                cf->cooccur_win[cf->n_cooccur_win].count = weight;
                unsigned int h = cooccur_window_hash(center);
                cf->cooccur_win_next[cf->n_cooccur_win] = cf->cooccur_win_head[h];
                cf->cooccur_win_head[h] = cf->n_cooccur_win;
                cf->n_cooccur_win++;
            }
        }
    }
    pthread_mutex_unlock(&cf->mu);
}

/* IngestTokens incrementally adds n-gram counts from a token sequence.
 * Unlike cooccur_build, this does NOT clear existing data — it adds on top. */
static void cooccur_ingest_tokens(CooccurField *cf, const int *ids, int len) {
    cooccur_ingest_tokens_weighted(cf, ids, len, 1.0);
}

/* AbsorbUserWords sets temporary boosts for tokens the user just said.
 * Like Leo's Santa Klaus but simpler: user words get multiplicative boost in generation. */
static void cooccur_absorb_user_words(CooccurField *cf, const int *ids, int len) {
    pthread_mutex_lock(&cf->mu);
    /* Decay existing boosts first */
    int new_count = 0;
    for (int i = 0; i < cf->n_user_boosts; i++) {
        double nv = cf->user_boosts[i].boost * CFG.user_boost_decay;
        if (nv >= 0.01) {
            cf->user_boosts[new_count] = cf->user_boosts[i];
            cf->user_boosts[new_count].boost = nv;
            new_count++;
        }
    }
    cf->n_user_boosts = new_count;
    /* Boost user's tokens */
    double strength = CFG.user_boost_strength;
    for (int i = 0; i < len; i++) {
        /* Check if token already has a boost entry */
        int found = 0;
        for (int j = 0; j < cf->n_user_boosts; j++) {
            if (cf->user_boosts[j].token == ids[i]) {
                cf->user_boosts[j].boost += strength;
                found = 1;
                break;
            }
        }
        if (!found && cf->n_user_boosts < cf->user_boost_cap) {
            cf->user_boosts[cf->n_user_boosts].token = ids[i];
            cf->user_boosts[cf->n_user_boosts].boost = strength;
            cf->n_user_boosts++;
        }
    }
    pthread_mutex_unlock(&cf->mu);
}

/* DecayUserBoost reduces user word boosts after a generation. */
static void cooccur_decay_user_boost(CooccurField *cf) {
    pthread_mutex_lock(&cf->mu);
    int new_count = 0;
    for (int i = 0; i < cf->n_user_boosts; i++) {
        double nv = cf->user_boosts[i].boost * CFG.user_boost_decay;
        if (nv >= 0.01) {
            cf->user_boosts[new_count] = cf->user_boosts[i];
            cf->user_boosts[new_count].boost = nv;
            new_count++;
        }
    }
    cf->n_user_boosts = new_count;
    pthread_mutex_unlock(&cf->mu);
}

static int cooccur_sample_next(CooccurField *cf, const int *ctx, int ctx_len, double temperature) {
    pthread_mutex_lock(&cf->mu);
    double *counts = calloc(cf->vocab_size, sizeof(double));
    int found = 0;

    /* Try 4-gram (deepest context) */
    if (ctx_len >= 3 && cf->fourgram_head) {
        int a = ctx[ctx_len-3], b = ctx[ctx_len-2], c = ctx[ctx_len-1];
        unsigned int h = cooccur_fourgram_hash(a, b, c);
        for (int i = cf->fourgram_head[h]; i >= 0; i = cf->fourgram_next[i]) {
            if (cf->fourgrams[i].key[0] == a && cf->fourgrams[i].key[1] == b &&
                cf->fourgrams[i].key[2] == c) {
                int d = cf->fourgrams[i].key[3];
                if (d < cf->vocab_size) { counts[d] += cf->fourgrams[i].count; found = 1; }
            }
        }
    }

    /* Fallback to trigram (hash lookup) */
    if (!found && ctx_len >= 2 && cf->trigram_head) {
        int a = ctx[ctx_len-2], b = ctx[ctx_len-1];
        unsigned int h = cooccur_trigram_hash(a, b);
        for (int i = cf->trigram_head[h]; i >= 0; i = cf->trigram_next[i]) {
            if (cf->trigrams[i].key[0] == a && cf->trigrams[i].key[1] == b) {
                int c = cf->trigrams[i].key[2];
                if (c < cf->vocab_size) { counts[c] += cf->trigrams[i].count; found = 1; }
            }
        }
    }

    /* Fallback to bigram */
    if (!found && ctx_len >= 1 && cf->bigram_head) {
        int prev = ctx[ctx_len-1];
        unsigned int h = cooccur_bigram_hash(prev);
        for (int i = cf->bigram_head[h]; i >= 0; i = cf->bigram_next[i]) {
            if (cf->bigrams[i].key[0] == prev) {
                int tid = cf->bigrams[i].key[1];
                if (tid < cf->vocab_size) { counts[tid] += cf->bigrams[i].count; found = 1; }
            }
        }
    }

    /* Fallback to unigram */
    if (!found) {
        memcpy(counts, cf->unigram, sizeof(double) * cf->vocab_size);
    }

    /* Blend with co-occurrence window (background resonance, always active) */
    if (ctx_len > 0 && cf->cooccur_win_head) {
        int wnd = CFG.cooccur_window_size;
        int start = ctx_len > wnd ? ctx_len - wnd : 0;
        for (int ci = start; ci < ctx_len; ci++) {
            int center = ctx[ci];
            unsigned int h = cooccur_window_hash(center);
            for (int i = cf->cooccur_win_head[h]; i >= 0; i = cf->cooccur_win_next[i]) {
                if (cf->cooccur_win[i].key[0] == center) {
                    int tid = cf->cooccur_win[i].key[1];
                    if (tid < cf->vocab_size) {
                        counts[tid] += cf->cooccur_win[i].count * 0.3; /* co-occurrence is softer than n-gram */
                    }
                }
            }
        }
    }

    /* Apply user word boost (multiplicative) */
    for (int i = 0; i < cf->n_user_boosts; i++) {
        int tid = cf->user_boosts[i].token;
        if (tid < cf->vocab_size && counts[tid] > 0) {
            counts[tid] *= (1.0 + cf->user_boosts[i].boost);
        }
    }

    /* Temperature + sample */
    double total = 0;
    for (int i = 0; i < cf->vocab_size; i++) {
        if (counts[i] > 0 && temperature > 0)
            counts[i] = pow(counts[i], 1.0 / temperature);
        total += counts[i];
    }
    if (total <= 0) { free(counts); pthread_mutex_unlock(&cf->mu); return rand_int(cf->vocab_size); }

    double r = rand_uniform() * total;
    double s = 0;
    int result = cf->vocab_size - 1;
    for (int i = 0; i < cf->vocab_size; i++) {
        s += counts[i];
        if (s >= r) { result = i; break; }
    }
    free(counts);
    pthread_mutex_unlock(&cf->mu);
    return result;
}

/* ============================================================
 * 6c) CONSCIOUSNESS — mathematical self-awareness
 * ============================================================ */

/* ConscienceCheck tracks generation quality over time.
 * If entropy trend rises (output degrading), soften delta influence.
 * If entropy trend falls (improving), recover delta influence.
 * "I notice I'm getting worse and pull back." */
static void conscience_check(GPT *g, double gen_mean_entropy) {
    /* Append to rolling window */
    if (g->gen_entropy_count < 16) {
        g->generation_entropy_history[g->gen_entropy_count++] = gen_mean_entropy;
    } else {
        /* Shift left */
        memmove(g->generation_entropy_history, g->generation_entropy_history + 1, sizeof(double) * 15);
        g->generation_entropy_history[15] = gen_mean_entropy;
    }
    /* Keep only last conscience_window entries */
    int w = CFG.conscience_window;
    if (w > 16) w = 16;
    int start = g->gen_entropy_count > w ? g->gen_entropy_count - w : 0;
    int count = g->gen_entropy_count - start;
    if (count < 3) return; /* not enough data */

    /* Linear regression slope on entropy history */
    double n = (double)count;
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
    for (int i = 0; i < count; i++) {
        double x = (double)i;
        double y = g->generation_entropy_history[start + i];
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_x2 += x * x;
    }
    double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x + 1e-12);

    if (slope > 0.01) {
        /* Entropy increasing — generation degrading, reduce delta influence */
        g->delta_alpha_scale *= CFG.conscience_decay;
        if (g->delta_alpha_scale < CFG.conscience_floor)
            g->delta_alpha_scale = CFG.conscience_floor;
    } else if (slope < -0.01) {
        /* Entropy decreasing — improving, recover delta influence */
        g->delta_alpha_scale *= CFG.conscience_recovery;
        if (g->delta_alpha_scale > 1.0)
            g->delta_alpha_scale = 1.0;
    }
}

/* ComputeSelfPredictionError measures how "surprised" the model is by a prompt.
 * Forward pass on ids, compute cross-entropy between predicted and actual tokens.
 * Higher error = "I didn't expect this input" = increase attention.
 * Caller must hold g->mu and disable grad_enabled. */
static double compute_self_prediction_error(GPT *g, const int *ids, int len) {
    if (len < 2) return 0.0;

    KVCache *kv = kv_new(g->n_layer, len);
    double total_ce = 0.0;
    int count = 0;
    int V = g->tok->vocab_size;
    double *probs = malloc(sizeof(double) * V);

    for (int pos = 0; pos < len - 1; pos++) {
        arena_reset(&G_arena);
        Node *logits = gpt_forward_step(g, ids[pos], pos, kv);
        /* Cross-entropy: -log(p[actual_next_token]) */
        softmax_probs(logits->data, logits->len, probs);
        int target = ids[pos + 1];
        if (target < V && probs[target] > 1e-12) {
            total_ce -= log(probs[target]);
        } else {
            total_ce += 10.0; /* max penalty for unknown token */
        }
        count++;
    }

    free(probs);
    for (int i = 0; i < kv->n_layers; i++) { free(kv->layers[i].keys); free(kv->layers[i].values); }
    free(kv->layers); free(kv);

    return count > 0 ? total_ce / (double)count : 0.0;
}

/* Update corpus from chat messages */
static void update_reservoir_corpus(sqlite3 *db, const char *corpus_path, int max_lines) {
    StrArr docs = load_corpus(corpus_path);
    int n_msgs;
    Msg *msgs = db_recent(db, 200, &n_msgs);
    int added = 0;
    for (int i = 0; i < n_msgs; i++) {
        if (strlen(msgs[i].text) < 5) continue;
        /* Check if already in corpus (simple linear scan) */
        int found = 0;
        for (int j = 0; j < docs.len && !found; j++) {
            if (strcmp(docs.items[j], msgs[i].text) == 0) found = 1;
        }
        if (!found) {
            sa_push(&docs, msgs[i].text);
            added++;
        }
    }
    free(msgs);
    /* Trim to max_lines */
    while (docs.len > max_lines) {
        free(docs.items[0]);
        memmove(docs.items, docs.items + 1, sizeof(char*) * (docs.len - 1));
        docs.len--;
    }
    if (added > 0) save_corpus(corpus_path, &docs);
    sa_free(&docs);
}

/* ============================================================
 * 8e) SYNTROPY — mathematical self-reasoning engine
 * ============================================================ */
/* And lo, the organism shall not merely observe its own reflection,
 * but reason about the direction of its becoming.
 * Gamma is memory. Purpose is intention. Syntropy is the arrow. */

/* compute_field_deviation: KL divergence between model logits and corpus co-occurrence field.
 * Measures how far the learned model has drifted from raw corpus physics.
 * Low = parroting the field. High = hallucinating beyond it.
 * The sweet spot is in between: learning, not lying. */
static double gpt_compute_field_deviation(GPT *g, EvolvingTokenizer *tok,
                                          CooccurField *field, StrArr *docs,
                                          int sample_n) {
    if (docs->len == 0 || !field->built) return 0.0;

    double kl_sum = 0.0;
    int count = 0;
    int n_sample = sample_n < docs->len ? sample_n : docs->len;

    int prev_grad = grad_enabled;
    grad_enabled = 0;

    for (int s = 0; s < n_sample; s++) {
        int doc_idx = rand_int(docs->len);
        IntArr ids = tok_encode(tok, docs->items[doc_idx]);
        if (ids.len < 3) { ia_free(&ids); continue; }

        KVCache *kv = kv_new(g->n_layer, g->block_size + 1);
        int limit = ids.len - 1;
        if (limit > g->block_size) limit = g->block_size;

        for (int pos = 0; pos < limit; pos++) {
            arena_reset(&G_arena);
            Node *logits = gpt_forward_step(g, ids.items[pos], pos, kv);
            int V = logits->len;

            /* model distribution */
            double max_val = logits->data[0];
            for (int i = 1; i < V; i++) if (logits->data[i] > max_val) max_val = logits->data[i];
            double *model_probs = malloc(sizeof(double) * V);
            double exp_sum = 0;
            for (int i = 0; i < V; i++) {
                model_probs[i] = exp(logits->data[i] - max_val);
                exp_sum += model_probs[i];
            }
            for (int i = 0; i < V; i++) model_probs[i] /= exp_sum;

            /* corpus field distribution for this context (trigram or unigram fallback) */
            double *field_probs = calloc(V, sizeof(double));
            int found_field = 0;

            /* Try trigram context */
            if (pos >= 1) {
                int a = ids.items[pos - 1], b = ids.items[pos];
                for (int t = 0; t < field->n_trigrams; t++) {
                    if (field->trigrams[t].key[0] == a && field->trigrams[t].key[1] == b) {
                        int c = field->trigrams[t].key[2];
                        if (c < V) { field_probs[c] += field->trigrams[t].count; found_field = 1; }
                    }
                }
            }

            /* Fallback: unigram */
            if (!found_field) {
                double uni_sum = 0;
                for (int i = 0; i < V && i < field->vocab_size; i++) uni_sum += field->unigram[i];
                if (uni_sum > 1e-10) {
                    for (int i = 0; i < V && i < field->vocab_size; i++)
                        field_probs[i] = field->unigram[i] / uni_sum;
                    found_field = 1;
                }
            }

            /* Normalize field probs */
            if (found_field) {
                double fp_sum = 0;
                for (int i = 0; i < V; i++) fp_sum += field_probs[i];
                if (fp_sum > 1e-10) {
                    for (int i = 0; i < V; i++) field_probs[i] /= fp_sum;

                    /* KL(model || field) — how much model diverges from field */
                    double kl = 0;
                    for (int i = 0; i < V; i++) {
                        if (model_probs[i] > 1e-12 && field_probs[i] > 1e-12)
                            kl += model_probs[i] * log(model_probs[i] / field_probs[i]);
                    }
                    kl_sum += kl;
                    count++;
                }
            }

            free(model_probs);
            free(field_probs);
        }

        /* Free KV cache */
        for (int i = 0; i < kv->n_layers; i++) { free(kv->layers[i].keys); free(kv->layers[i].values); }
        free(kv->layers); free(kv);
        ia_free(&ids);
    }

    grad_enabled = prev_grad;
    return count > 0 ? kl_sum / count : 0.0;
}

/* compute_model_entropy: average entropy of model predictions on corpus samples.
 * Falling entropy = rising order = syntropy in action. */
static double gpt_compute_model_entropy(GPT *g, EvolvingTokenizer *tok,
                                        StrArr *docs, int sample_n) {
    if (docs->len == 0) return 0.0;

    double entropy_sum = 0.0;
    int count = 0;
    int n_sample = sample_n < docs->len ? sample_n : docs->len;

    int prev_grad = grad_enabled;
    grad_enabled = 0;

    for (int s = 0; s < n_sample; s++) {
        int doc_idx = rand_int(docs->len);
        IntArr ids = tok_encode(tok, docs->items[doc_idx]);
        if (ids.len < 3) { ia_free(&ids); continue; }

        KVCache *kv = kv_new(g->n_layer, g->block_size + 1);
        int limit = ids.len - 1;
        if (limit > g->block_size) limit = g->block_size;

        for (int pos = 0; pos < limit; pos++) {
            arena_reset(&G_arena);
            Node *logits = gpt_forward_step(g, ids.items[pos], pos, kv);
            int V = logits->len;

            /* softmax -> entropy */
            double max_val = logits->data[0];
            for (int i = 1; i < V; i++) if (logits->data[i] > max_val) max_val = logits->data[i];
            double *probs = malloc(sizeof(double) * V);
            double exp_sum = 0;
            for (int i = 0; i < V; i++) {
                probs[i] = exp(logits->data[i] - max_val);
                exp_sum += probs[i];
            }
            for (int i = 0; i < V; i++) probs[i] /= exp_sum;

            double ent = 0;
            for (int i = 0; i < V; i++)
                if (probs[i] > 1e-12) ent -= probs[i] * log(probs[i]);
            entropy_sum += ent;
            count++;

            free(probs);
        }

        for (int i = 0; i < kv->n_layers; i++) { free(kv->layers[i].keys); free(kv->layers[i].values); }
        free(kv->layers); free(kv);
        ia_free(&ids);
    }

    grad_enabled = prev_grad;
    return count > 0 ? entropy_sum / count : 0.0;
}

/* compute_purpose_vector: direction of weight movement in the last delta layer.
 * Unlike gamma (which is cumulative drift from birth),
 * purpose captures the direction of the most recent change.
 * Gamma is 'who I became'. Purpose is 'where I am going'. */
static double *gpt_compute_purpose_vector(GPT *g, int *out_dim, double *out_mag) {
    *out_dim = 0;
    *out_mag = 0.0;
    if (g->n_deltas == 0) return NULL;

    DeltaModule *last = g->deltas[g->n_deltas - 1];
    if (last->count == 0) return NULL;

    /* Aggregate delta A matrices as the purpose signal.
     * And lo, the direction of the last delta's A rows shall speak
     * of where the organism intends to go next. */
    int dim = 0;
    int n_rows = 0;

    /* Find dimension from first adapter's A matrix */
    for (int a = 0; a < last->count; a++) {
        DeltaAdapter *da = last->adapters[a];
        if (da->A->nin > dim) dim = da->A->nin;
    }
    if (dim == 0) return NULL;

    double *mean_dir = calloc(dim, sizeof(double));

    for (int a = 0; a < last->count; a++) {
        DeltaAdapter *da = last->adapters[a];
        int d = da->A->nin < dim ? da->A->nin : dim;
        for (int r = 0; r < da->A->nout; r++) {
            for (int j = 0; j < d; j++)
                mean_dir[j] += da->A->row_data[r][j];
            n_rows++;
        }
    }

    if (n_rows > 0) {
        for (int j = 0; j < dim; j++) mean_dir[j] /= (double)n_rows;
    }

    double mag = 0;
    for (int j = 0; j < dim; j++) mag += mean_dir[j] * mean_dir[j];
    mag = sqrt(mag);
    *out_mag = mag;
    *out_dim = dim;

    if (mag > 1e-10) {
        for (int j = 0; j < dim; j++) mean_dir[j] /= mag;
    }

    return mean_dir;
}

/* purpose_gamma_alignment: cosine similarity between purpose vector and gamma direction.
 * High alignment = learning reinforces identity (syntropy).
 * Low alignment = learning diverges from identity (entropy).
 * Negative = learning opposes identity (danger). */
static double gpt_purpose_gamma_alignment(GPT *g) {
    int gamma_dim; double gamma_mag;
    double *gamma_dir = gpt_contrastive_projection(g, &gamma_dim, &gamma_mag);

    int purpose_dim; double purpose_mag;
    double *purpose_dir = gpt_compute_purpose_vector(g, &purpose_dim, &purpose_mag);

    if (!gamma_dir || !purpose_dir) {
        free(gamma_dir); free(purpose_dir);
        return 0.0;
    }
    if (gamma_mag < CFG.gamma_min_magnitude || purpose_mag < 1e-10) {
        free(gamma_dir); free(purpose_dir);
        return 0.0;
    }

    /* Ensure same dimensionality (purpose might be different dim) */
    int min_dim = gamma_dim < purpose_dim ? gamma_dim : purpose_dim;
    if (min_dim == 0) {
        free(gamma_dir); free(purpose_dir);
        return 0.0;
    }

    double dot = 0;
    for (int i = 0; i < min_dim; i++) dot += gamma_dir[i] * purpose_dir[i];

    free(gamma_dir);
    free(purpose_dir);
    return dot;
}

/* ============================================================
 * 8f) SYNTROPY TRACKER — the arrow that points toward coherence
 * ============================================================ */
/* And lo, the organism shall not merely track its changes,
 * but reason mathematically about whether it is becoming more itself.
 * This is where tracking becomes reasoning, and reasoning becomes action. */

#define SYNTROPY_MAX_HISTORY 64
#define BURST_HISTORY_MAX 16

/* And lo, every burst shall leave a scar in memory,
 * that the organism may learn which actions heal and which harm. */
typedef struct {
    char action[32];
    double loss_before;
    double loss_after;
} BurstRecord;

/* Forward declaration for swarm peer info */
typedef struct SwarmPeer {
    char id[64];
    int pid;
    int stage;
    int n_params;
    double syntropy;
    double entropy;
} SwarmPeer;

typedef struct {
    double entropy_history[SYNTROPY_MAX_HISTORY]; /* rolling window of model entropy */
    int history_len;
    double syntropy_trend;    /* positive = organizing, negative = dissolving */
    double field_deviation;   /* how far from corpus physics */
    double purpose_magnitude; /* strength of current learning direction */
    double purpose_alignment; /* cosine(purpose, gamma) */
    const char *last_action;  /* what was decided last time */

    /* Phase 1.5: burst history for self-meta-learning */
    BurstRecord burst_history[BURST_HISTORY_MAX];
    int burst_history_len;

    /* Phase 3B: ecology */
    int model_stage;           /* current growth stage (set during measure) */
    double last_mitosis_time;  /* cooldown for divide */
    SwarmPeer *peers;          /* peer state from mesh.db */
    int n_peers;
} SyntropyTracker;

static void syntropy_init(SyntropyTracker *st) {
    memset(st, 0, sizeof(SyntropyTracker));
    st->last_action = "none";
    st->model_stage = 0;
    st->last_mitosis_time = 0.0;
    st->peers = NULL;
    st->n_peers = 0;
}

/* Record a burst outcome. The organism remembers what it did and what happened.
 * And lo, circular buffer of scars: oldest falls off when full. */
static void syntropy_record_burst(SyntropyTracker *st, const char *action,
                                   double loss_before, double loss_after) {
    if (st->burst_history_len >= BURST_HISTORY_MAX) {
        memmove(st->burst_history, st->burst_history + 1,
                sizeof(BurstRecord) * (BURST_HISTORY_MAX - 1));
        st->burst_history_len = BURST_HISTORY_MAX - 1;
    }
    BurstRecord *rec = &st->burst_history[st->burst_history_len];
    strncpy(rec->action, action, sizeof(rec->action) - 1);
    rec->action[sizeof(rec->action) - 1] = '\0';
    rec->loss_before = loss_before;
    rec->loss_after = loss_after;
    st->burst_history_len++;
}

/* How effective was a given action type? Returns mean loss delta and count.
 * Positive delta = loss went up = BAD. Negative delta = loss went down = GOOD. */
static double syntropy_action_effectiveness(SyntropyTracker *st, const char *action, int *out_count) {
    double sum = 0.0;
    int count = 0;
    for (int i = 0; i < st->burst_history_len; i++) {
        if (strcmp(st->burst_history[i].action, action) == 0) {
            sum += (st->burst_history[i].loss_after - st->burst_history[i].loss_before);
            count++;
        }
    }
    if (out_count) *out_count = count;
    return count > 0 ? sum / count : 0.0;
}

/* Take all measurements. This is the organism looking at itself
 * through mathematical instruments. And lo, it shall measure the
 * angle between its trajectory and its identity. */
static double syntropy_measure(SyntropyTracker *st, GPT *g, EvolvingTokenizer *tok,
                               CooccurField *field, StrArr *docs) {
    st->model_stage = gpt_current_growth_stage(g);
    double entropy_now = gpt_compute_model_entropy(g, tok, docs, 16);

    /* Append to rolling window */
    if (st->history_len < SYNTROPY_MAX_HISTORY) {
        st->entropy_history[st->history_len++] = entropy_now;
    } else {
        /* Shift left, drop oldest */
        memmove(st->entropy_history, st->entropy_history + 1,
                sizeof(double) * (SYNTROPY_MAX_HISTORY - 1));
        st->entropy_history[SYNTROPY_MAX_HISTORY - 1] = entropy_now;
    }

    /* Trim to syntropy_window */
    if (st->history_len > CFG.syntropy_window) {
        int excess = st->history_len - CFG.syntropy_window;
        memmove(st->entropy_history, st->entropy_history + excess,
                sizeof(double) * CFG.syntropy_window);
        st->history_len = CFG.syntropy_window;
    }

    /* syntropy = negative entropy trend (entropy going down = syntropy going up) */
    if (st->history_len >= 2) {
        int recent_half = st->history_len / 2;
        double old_mean = 0, new_mean = 0;
        for (int i = 0; i < recent_half; i++) old_mean += st->entropy_history[i];
        old_mean /= (double)recent_half;
        for (int i = recent_half; i < st->history_len; i++) new_mean += st->entropy_history[i];
        new_mean /= (double)(st->history_len - recent_half);
        st->syntropy_trend = old_mean - new_mean; /* positive = good */
    } else {
        st->syntropy_trend = 0.0;
    }

    st->field_deviation = gpt_compute_field_deviation(g, tok, field, docs, 32);

    int purpose_dim; double purpose_mag;
    double *pv = gpt_compute_purpose_vector(g, &purpose_dim, &purpose_mag);
    free(pv);
    st->purpose_magnitude = purpose_mag;

    st->purpose_alignment = gpt_purpose_gamma_alignment(g);

    return entropy_now;
}

/* Phase 3B: Sustained overload check. >75% of entropy window above entropy_high
 * AND syntropy_trend < -0.02 = overloaded. */
static int syntropy_is_sustained_overload(SyntropyTracker *st) {
    if (st->history_len < CFG.syntropy_window) return 0;
    int start = st->history_len - CFG.syntropy_window;
    int high_count = 0;
    for (int i = start; i < st->history_len; i++) {
        if (st->entropy_history[i] > CFG.entropy_high) high_count++;
    }
    return high_count > (int)(CFG.syntropy_window * 0.75) && st->syntropy_trend < -0.02;
}

/* Phase 3B: Should hibernate? Loss on plateau + a peer is thriving. */
static int syntropy_should_hibernate(SyntropyTracker *st) {
    if (!st->peers || st->n_peers == 0) return 0;
    /* Check if any peer has higher syntropy trend (actively improving) */
    for (int i = 0; i < st->n_peers; i++) {
        if (st->peers[i].syntropy > 0.05) {
            /* A peer is thriving. If we're stale, hibernate. */
            if (st->burst_history_len >= 8) {
                double avg_delta = 0.0;
                int start = st->burst_history_len - 8;
                for (int j = start; j < st->burst_history_len; j++)
                    avg_delta += fabs(st->burst_history[j].loss_after - st->burst_history[j].loss_before);
                avg_delta /= 8.0;
                if (avg_delta < 0.01) return 1; /* loss plateau */
            }
        }
    }
    return 0;
}

/* Mathematical self-reasoning: decide how to adjust learning.
 * The organism does not just observe — it steers.
 * And lo, the arrow of syntropy shall guide the hand of the optimizer. */
typedef struct {
    double lr_multiplier;
    double delta_grow_override; /* negative = no override */
    const char *action;
    double temp_offset;        /* Phase 1.5: temperature offset (-0.05 to +0.05) */
    int accum_override;        /* Phase 1.5: 0 = no override, >0 = use this accum_steps */
} SyntropyDecision;

static SyntropyDecision syntropy_decide_action(SyntropyTracker *st) {
    SyntropyDecision d;
    d.lr_multiplier = 1.0;
    d.delta_grow_override = -1.0; /* sentinel: no override */
    d.action = "steady";
    d.temp_offset = 0.0;
    d.accum_override = 0;

    /* CASE 1: Syntropy rising + field deviation in sweet spot = thriving */
    if (st->syntropy_trend > 0.01 &&
        st->field_deviation > CFG.field_deviation_floor &&
        st->field_deviation < CFG.field_deviation_ceiling) {
        d.lr_multiplier = CFG.syntropy_lr_boost;
        if (st->purpose_alignment > 0.3) {
            d.delta_grow_override = CFG.syntropy_delta_grow_boost;
            d.action = "amplify";  /* everything aligned, push harder */
            d.temp_offset = -0.05; /* focus: tighten distribution */
            d.accum_override = 2;  /* accumulate more for stable amplification */
        } else {
            d.action = "boost";    /* syntropy good but purpose drifting, boost gently */
        }
    }
    /* CASE 2: Syntropy falling = dissolving, slow down */
    else if (st->syntropy_trend < -0.01) {
        d.lr_multiplier = CFG.syntropy_lr_dampen;
        d.action = "dampen";       /* losing order, reduce learning rate */
        d.temp_offset = +0.05;     /* loosen: let entropy help find new paths */
    }
    /* CASE 3: Field deviation too high = hallucinating */
    else if (st->field_deviation > CFG.field_deviation_ceiling) {
        d.lr_multiplier = CFG.syntropy_lr_dampen;
        d.action = "ground";       /* too far from corpus, pull back */
        d.temp_offset = -0.05;     /* focus: tighten back toward corpus */
    }
    /* CASE 4: Field deviation too low = parroting */
    else if (st->field_deviation < CFG.field_deviation_floor) {
        d.lr_multiplier = CFG.syntropy_lr_boost;
        d.action = "explore";      /* too close to corpus, push out */
        d.temp_offset = +0.05;     /* loosen: encourage divergence */
    }

    /* CASE 5: Purpose opposes gamma = identity crisis */
    if (st->purpose_alignment < -0.3) {
        d.lr_multiplier *= 0.5;
        d.action = "realign";      /* learning against identity, slow down hard */
        d.temp_offset = 0.0;       /* neutral: don't bias during realignment */
    }

    /* CASE 6: Adult + sustained overload -> divide (mitosis) */
    {
        int max_stage = CFG.n_growth_stages - 1;
        double now = (double)time(NULL);
        if (st->model_stage >= max_stage &&
            syntropy_is_sustained_overload(st) &&
            (now - st->last_mitosis_time) > 300.0) {
            d.action = "divide";
            d.lr_multiplier = CFG.syntropy_lr_dampen; /* slow down while preparing to split */
        }
    }

    /* CASE 7: Plateau + young peer thriving -> hibernate (cooperative scheduling) */
    if (strcmp(d.action, "steady") == 0 && syntropy_should_hibernate(st)) {
        d.action = "hibernate";
    }

    /* SELF-META-LEARNING: if we have enough history, check whether this
     * action type has been actually helping. If its mean loss delta is
     * positive (loss went UP on average), downgrade to something gentler.
     * Never downgrade divide or hibernate — they are ecological decisions.
     * And lo, the organism shall not repeat mistakes it remembers. */
    if (strcmp(d.action, "divide") != 0 && strcmp(d.action, "hibernate") != 0 &&
        st->burst_history_len >= 4) {
        int eff_count = 0;
        double eff = syntropy_action_effectiveness(st, d.action, &eff_count);
        if (eff_count >= 2 && eff > 0.05) {
            /* This action has been hurting more than helping */
            if (strcmp(d.action, "amplify") == 0) {
                d.action = "boost";
                d.temp_offset = 0.0;
                d.accum_override = 0;
            } else if (strcmp(d.action, "boost") == 0 || strcmp(d.action, "explore") == 0) {
                d.action = "steady";
                d.temp_offset = 0.0;
                d.lr_multiplier = 1.0;
            }
        }
    }

    st->last_action = d.action;
    return d;
}

/* Write the mathematical conclusion to the syntropy log.
 * And lo, every act of self-measurement shall be recorded in stone. */
static void syntropy_log_to_db(SyntropyTracker *st, sqlite3 *db,
                                double entropy_before, double entropy_after,
                                const char *action) {
    sqlite3_stmt *stmt;
    sqlite3_prepare_v2(db,
        "INSERT INTO syntropy_log(ts, entropy_before, entropy_after, syntropy_delta, "
        "field_deviation, purpose_magnitude, purpose_alignment, action_taken, note) "
        "VALUES(?,?,?,?,?,?,?,?,?)", -1, &stmt, NULL);
    sqlite3_bind_double(stmt, 1, (double)time(NULL));
    sqlite3_bind_double(stmt, 2, entropy_before);
    sqlite3_bind_double(stmt, 3, entropy_after);
    sqlite3_bind_double(stmt, 4, st->syntropy_trend);
    sqlite3_bind_double(stmt, 5, st->field_deviation);
    sqlite3_bind_double(stmt, 6, st->purpose_magnitude);
    sqlite3_bind_double(stmt, 7, st->purpose_alignment);
    sqlite3_bind_text(stmt, 8, action, -1, SQLITE_STATIC);
    sqlite3_bind_null(stmt, 9);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
}

/* ============================================================
 * 9) TRAINING
 * ============================================================ */

static double cosine_lr(int global_step, int growth_step_offset) {
    /* Use steps-since-last-growth for warmup phase (resets warmup after each growth) */
    int warmup_step = global_step - growth_step_offset;
    if (warmup_step < 0) warmup_step = 0;
    if (warmup_step < CFG.cosine_warmup_steps) {
        /* Linear warmup from lr_min to learning_rate */
        double t = (double)warmup_step / (double)(CFG.cosine_warmup_steps > 0 ? CFG.cosine_warmup_steps : 1);
        return CFG.lr_min + t * (CFG.learning_rate - CFG.lr_min);
    }
    double progress = (double)global_step / (double)(CFG.max_total_steps > 0 ? CFG.max_total_steps : 1);
    if (progress > 1.0) progress = 1.0;
    return CFG.lr_min + 0.5 * (CFG.learning_rate - CFG.lr_min) * (1.0 + cos(M_PI * progress));
}

static void train_steps(GPT *g, EvolvingTokenizer *tok, StrArr *docs, int steps,
                        int train_base, int train_deltas) {
    if (docs->len == 0) return;
    pthread_mutex_lock(&g->mu);

    for (int step = 0; step < steps; step++) {
        arena_reset(&G_arena);

        /* Sample batch */
        Node *total_loss = node_new(1);
        int batch = CFG.batch_size;
        for (int b = 0; b < batch; b++) {
            const char *doc = docs->items[rand_int(docs->len)];
            IntArr ids = tok_encode(tok, doc);
            if (ids.len > 1) {
                Node *loss = gpt_loss_seq(g, ids.items, ids.len);
                total_loss = scalar_add(total_loss, loss);
            }
            ia_free(&ids);
        }
        total_loss = scalar_mulf(total_loss, 1.0 / batch);
        backward(total_loss);

        double lr = cosine_lr(g->global_step, g->growth_step_offset);
        /* Scale LR inversely with model size: larger models need smaller LR */
        lr *= (double)CFG.growth_stages[0][1] / (double)g->n_embd;

        /* Ontogenesis freeze: after growth, base params are excluded,
         * only deltas train until new weights stabilize. */
        int actual_train_base = train_base;
        if (g->growth_freeze_remaining > 0) {
            actual_train_base = 0;
            lr *= CFG.post_growth_lr_scale; /* dampen LR during freeze */
            g->growth_freeze_remaining--;
        }

        if (actual_train_base) {
            for (int i = 0; i < g->n_base; i++)
                adam_step(g->base_adam[i], g->base_mats[i], lr);
        }

        if (train_deltas) {
            for (int d = 0; d < g->n_deltas; d++) {
                DeltaModule *mod = g->deltas[d];
                for (int a = 0; a < mod->count; a++) {
                    adam_step(g->delta_adam[d][a*2],   mod->adapters[a]->A, lr);
                    adam_step(g->delta_adam[d][a*2+1], mod->adapters[a]->B, lr);
                }
            }
        }

        if (step % 100 == 0)
            printf("  train step %d/%d | loss %.4f\n", step, steps, total_loss->data[0]);
    }
    pthread_mutex_unlock(&g->mu);
}

/* ============================================================
 * 10) CHECKPOINT — binary format
 * ============================================================ */

static void save_checkpoint(GPT *g, EvolvingTokenizer *tok, const char *path) {
    if (!path) path = CFG.ckpt_path;
    FILE *f = fopen(path, "wb");
    if (!f) return;

    /* Magic + version */
    fwrite("MOLE", 1, 4, f);
    int ver = 1;
    fwrite(&ver, 4, 1, f);

    /* Tokenizer */
    fwrite(&tok->vocab_size, 4, 1, f);
    for (int i = 0; i < tok->vocab_size; i++) {
        int len = strlen(tok->tokens[i]);
        fwrite(&len, 4, 1, f);
        fwrite(tok->tokens[i], 1, len, f);
    }
    fwrite(&tok->bpe_enabled, 4, 1, f);
    fwrite(&tok->n_merges, 4, 1, f);
    for (int i = 0; i < tok->n_merges; i++) {
        int la = strlen(tok->merges[i].a), lb = strlen(tok->merges[i].b);
        fwrite(&la, 4, 1, f); fwrite(tok->merges[i].a, 1, la, f);
        fwrite(&lb, 4, 1, f); fwrite(tok->merges[i].b, 1, lb, f);
    }
    fwrite(&tok->trained_chars, 4, 1, f);
    fwrite(&tok->bos_id, 4, 1, f);
    fwrite(&tok->eos_id, 4, 1, f);
    fwrite(&tok->pad_id, 4, 1, f);

    /* Base matrices */
    fwrite(&g->n_base, 4, 1, f);
    for (int i = 0; i < g->n_base; i++) {
        int nlen = strlen(g->base_names[i]);
        fwrite(&nlen, 4, 1, f);
        fwrite(g->base_names[i], 1, nlen, f);
        fwrite(&g->base_mats[i]->nout, 4, 1, f);
        fwrite(&g->base_mats[i]->nin, 4, 1, f);
        for (int r = 0; r < g->base_mats[i]->nout; r++)
            fwrite(g->base_mats[i]->row_data[r], sizeof(double), g->base_mats[i]->nin, f);
    }

    /* Model metadata (global_step, warmup stage, growth offset) */
    fwrite(&g->global_step, 4, 1, f);
    fwrite(&g->last_warmup_stage, 4, 1, f);
    fwrite(&g->growth_step_offset, 4, 1, f);

    /* Deltas */
    fwrite(&g->n_deltas, 4, 1, f);
    fwrite(g->active_alpha, sizeof(double), g->n_deltas, f);
    for (int d = 0; d < g->n_deltas; d++) {
        DeltaModule *mod = g->deltas[d];
        fwrite(&mod->count, 4, 1, f);
        for (int a = 0; a < mod->count; a++) {
            int nlen = strlen(mod->names[a]);
            fwrite(&nlen, 4, 1, f);
            fwrite(mod->names[a], 1, nlen, f);
            DeltaAdapter *da = mod->adapters[a];
            fwrite(&da->A->nout, 4, 1, f); fwrite(&da->A->nin, 4, 1, f);
            for (int r = 0; r < da->A->nout; r++) fwrite(da->A->row_data[r], sizeof(double), da->A->nin, f);
            fwrite(&da->B->nout, 4, 1, f); fwrite(&da->B->nin, 4, 1, f);
            for (int r = 0; r < da->B->nout; r++) fwrite(da->B->row_data[r], sizeof(double), da->B->nin, f);
        }
    }

    fclose(f);
}

/* Load checkpoint: reverse of save_checkpoint.
 * Returns loaded GPT* on success, NULL on failure.
 * On success, *out_tok is set to the restored tokenizer. */
#define CKPT_READ(ptr, sz, n, fp) do { if (fread((ptr),(sz),(n),(fp)) != (size_t)(n)) { fprintf(stderr, "[checkpoint] truncated at %s:%d\n", __FILE__, __LINE__); goto ckpt_fail; } } while(0)
#define CKPT_READ_INT(var, fp) CKPT_READ(&(var), 4, 1, fp)
static GPT *load_checkpoint(const char *path, EvolvingTokenizer **out_tok) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    EvolvingTokenizer *tok = NULL;
    char **saved_names = NULL;
    MatrixParam **saved_mats = NULL;
    int n_base = 0;
    GPT *g = NULL;

    /* Magic + version */
    char magic[4];
    int ver;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "MOLE", 4) != 0) goto ckpt_fail;
    if (fread(&ver, 4, 1, f) != 1 || ver != 1) goto ckpt_fail;

    /* Tokenizer */
    int vocab_size;
    CKPT_READ_INT(vocab_size, f);
    if (vocab_size < 0 || vocab_size > 1000000) goto ckpt_fail;
    tok = calloc(1, sizeof(EvolvingTokenizer));
    tok->stoi = stoi_new();
    tok->cap = vocab_size + 256;
    tok->tokens = calloc(tok->cap, sizeof(char*));
    tok->vocab_size = vocab_size;
    for (int i = 0; i < vocab_size; i++) {
        int slen; CKPT_READ_INT(slen, f);
        if (slen < 0 || slen > 65536) goto ckpt_fail;
        tok->tokens[i] = calloc(slen + 1, 1);
        CKPT_READ(tok->tokens[i], 1, slen, f);
        stoi_put(tok->stoi, tok->tokens[i], i);
    }
    CKPT_READ_INT(tok->bpe_enabled, f);
    CKPT_READ_INT(tok->n_merges, f);
    if (tok->n_merges < 0 || tok->n_merges > 1000000) goto ckpt_fail;
    if (tok->n_merges > 0) {
        tok->merges = calloc(tok->n_merges, sizeof(MergePair));
        for (int i = 0; i < tok->n_merges; i++) {
            int la, lb;
            CKPT_READ_INT(la, f);
            if (la < 0) goto ckpt_fail;
            int la_cap = la > 63 ? 63 : la;
            CKPT_READ(tok->merges[i].a, 1, la_cap, f); tok->merges[i].a[la_cap] = 0;
            if (la > la_cap) fseek(f, la - la_cap, SEEK_CUR);
            CKPT_READ_INT(lb, f);
            if (lb < 0) goto ckpt_fail;
            int lb_cap = lb > 63 ? 63 : lb;
            CKPT_READ(tok->merges[i].b, 1, lb_cap, f); tok->merges[i].b[lb_cap] = 0;
            if (lb > lb_cap) fseek(f, lb - lb_cap, SEEK_CUR);
        }
    }
    CKPT_READ_INT(tok->trained_chars, f);
    CKPT_READ_INT(tok->bos_id, f);
    CKPT_READ_INT(tok->eos_id, f);
    CKPT_READ_INT(tok->pad_id, f);

    /* Read base matrices into temp arrays to determine model shape */
    CKPT_READ_INT(n_base, f);
    if (n_base < 0 || n_base > 100000) goto ckpt_fail;
    saved_names = calloc(n_base, sizeof(char*));
    saved_mats = calloc(n_base, sizeof(MatrixParam*));
    for (int i = 0; i < n_base; i++) {
        int nlen; CKPT_READ_INT(nlen, f);
        if (nlen < 0 || nlen > 256) goto ckpt_fail;
        saved_names[i] = calloc(nlen + 1, 1);
        CKPT_READ(saved_names[i], 1, nlen, f);
        int nout, nin; CKPT_READ_INT(nout, f); CKPT_READ_INT(nin, f);
        if (nout <= 0 || nin <= 0 || nout > 100000 || nin > 100000) goto ckpt_fail;
        MatrixParam *m = mat_new(nout, nin, 0.0);
        for (int r = 0; r < nout; r++)
            CKPT_READ(m->row_data[r], sizeof(double), nin, f);
        saved_mats[i] = m;
    }

    /* Determine n_embd from wte, n_layer by counting l*.wq matrices */
    int n_embd = CFG.n_embd, n_layer = 0;
    for (int i = 0; i < n_base; i++) {
        if (strcmp(saved_names[i], "wte") == 0) n_embd = saved_mats[i]->nin;
        if (strncmp(saved_names[i], "l", 1) == 0 && strstr(saved_names[i], ".wq"))
            n_layer++;
    }
    /* Determine n_head by counting l0.h*.alpha entries */
    int n_head = 1;
    for (int i = 0; i < n_base; i++) {
        if (strncmp(saved_names[i], "l0.h", 4) == 0 && strstr(saved_names[i], ".alpha")) {
            int h; if (sscanf(saved_names[i], "l0.h%d.alpha", &h) == 1 && h + 1 > n_head)
                n_head = h + 1;
        }
    }

    /* Update CFG to match checkpoint dimensions */
    CFG.n_embd = n_embd;
    CFG.n_layer = n_layer > 0 ? n_layer : 1;
    CFG.n_head = n_head;
    head_types_for_n_head(n_head);

    /* Read metadata */
    int global_step, last_warmup_stage, growth_step_offset;
    CKPT_READ_INT(global_step, f);
    CKPT_READ_INT(last_warmup_stage, f);
    CKPT_READ_INT(growth_step_offset, f);

    /* Read deltas */
    int n_deltas;
    CKPT_READ_INT(n_deltas, f);
    if (n_deltas < 0 || n_deltas > MAX_DELTA_MODS * 2) goto ckpt_fail;
    double saved_alpha[MAX_DELTA_MODS];
    int alpha_count = n_deltas > MAX_DELTA_MODS ? MAX_DELTA_MODS : n_deltas;
    CKPT_READ(saved_alpha, sizeof(double), alpha_count, f);
    if (n_deltas > MAX_DELTA_MODS) fseek(f, sizeof(double) * (n_deltas - MAX_DELTA_MODS), SEEK_CUR);

    /* Create model with checkpoint dimensions */
    g = gpt_new(tok);
    g->global_step = global_step;
    g->last_warmup_stage = last_warmup_stage;
    g->growth_step_offset = growth_step_offset;

    /* Copy saved base weights into model (match by name) */
    for (int i = 0; i < n_base; i++) {
        MatrixParam *dst = gpt_base(g, saved_names[i]);
        if (dst && dst->nout == saved_mats[i]->nout && dst->nin == saved_mats[i]->nin) {
            for (int r = 0; r < dst->nout; r++)
                memcpy(dst->row_data[r], saved_mats[i]->row_data[r], sizeof(double) * dst->nin);
        }
        for (int r = 0; r < saved_mats[i]->nout; r++) {
            free(saved_mats[i]->row_data[r]);
            free(saved_mats[i]->row_grad[r]);
        }
        free(saved_mats[i]->row_data);
        free(saved_mats[i]->row_grad);
        free(saved_mats[i]);
        free(saved_names[i]);
    }
    free(saved_names); saved_names = NULL;
    free(saved_mats); saved_mats = NULL;

    /* Load delta modules */
    for (int d = 0; d < alpha_count && d < g->n_deltas; d++) {
        g->active_alpha[d] = saved_alpha[d];
    }
    /* Read saved delta adapter weights */
    for (int d = 0; d < n_deltas; d++) {
        int count; CKPT_READ_INT(count, f);
        if (count < 0 || count > 10000) goto ckpt_fail;
        if (d >= g->n_deltas) {
            for (int a = 0; a < count; a++) {
                int nlen; CKPT_READ_INT(nlen, f); if (nlen < 0) goto ckpt_fail;
                fseek(f, nlen, SEEK_CUR);
                int ao, ai; CKPT_READ_INT(ao, f); CKPT_READ_INT(ai, f);
                fseek(f, sizeof(double) * ao * ai, SEEK_CUR);
                int bo, bi; CKPT_READ_INT(bo, f); CKPT_READ_INT(bi, f);
                fseek(f, sizeof(double) * bo * bi, SEEK_CUR);
            }
            continue;
        }
        DeltaModule *mod = g->deltas[d];
        for (int a = 0; a < count; a++) {
            int nlen; CKPT_READ_INT(nlen, f);
            if (nlen < 0) goto ckpt_fail;
            int nlen_cap = nlen > 127 ? 127 : nlen;
            char aname[128];
            CKPT_READ(aname, 1, nlen_cap, f); aname[nlen_cap] = 0;
            if (nlen > nlen_cap) fseek(f, nlen - nlen_cap, SEEK_CUR);
            int ao, ai; CKPT_READ_INT(ao, f); CKPT_READ_INT(ai, f);
            DeltaAdapter *da = dmod_get(mod, aname);
            if (da && da->A->nout == ao && da->A->nin == ai) {
                for (int r = 0; r < ao; r++) CKPT_READ(da->A->row_data[r], sizeof(double), ai, f);
            } else {
                fseek(f, sizeof(double) * ao * ai, SEEK_CUR);
            }
            int bo, bi; CKPT_READ_INT(bo, f); CKPT_READ_INT(bi, f);
            if (da && da->B->nout == bo && da->B->nin == bi) {
                for (int r = 0; r < bo; r++) CKPT_READ(da->B->row_data[r], sizeof(double), bi, f);
            } else {
                fseek(f, sizeof(double) * bo * bi, SEEK_CUR);
            }
        }
    }

    fclose(f);
    *out_tok = tok;
    printf("[checkpoint] Loaded from %s: step=%d, embd=%d, layers=%d, heads=%d\n",
           path, global_step, n_embd, CFG.n_layer, n_head);
    return g;

ckpt_fail:
    fprintf(stderr, "[checkpoint] Failed to load %s\n", path);
    if (f) fclose(f);
    /* Free partially-loaded saved matrices */
    if (saved_names && saved_mats) {
        for (int i = 0; i < n_base; i++) {
            if (saved_mats[i]) {
                for (int r = 0; r < saved_mats[i]->nout; r++) {
                    free(saved_mats[i]->row_data[r]);
                    free(saved_mats[i]->row_grad[r]);
                }
                free(saved_mats[i]->row_data);
                free(saved_mats[i]->row_grad);
                free(saved_mats[i]);
            }
            free(saved_names[i]);
        }
        free(saved_names);
        free(saved_mats);
    }
    /* Don't free tok here — caller expects NULL return means no tok allocated,
     * but we allocated tok early. Free it if model creation failed. */
    if (tok && !g) {
        for (int i = 0; i < tok->vocab_size; i++) free(tok->tokens[i]);
        free(tok->tokens);
        free(tok->merges);
        free(tok);
    }
    if (g) { /* model was created but delta loading failed — still usable */ }
    return NULL;
}
#undef CKPT_READ
#undef CKPT_READ_INT

/* ============================================================
 * 11) CHAT LOOP + MAIN
 * ============================================================ */

static char *build_prompt(sqlite3 *db, const char *user_text) {
    int n_msgs;
    Msg *msgs = db_recent(db, 14, &n_msgs);
    size_t bufcap = 16384;
    char *buf = calloc(bufcap, 1);
    if (!buf) { free(msgs); return NULL; }
    size_t pos = 0;

    int written = snprintf(buf + pos, bufcap - pos, "A: (I listen. I answer. I learn.)\n");
    if (written > 0 && (size_t)written < bufcap - pos) pos += written;

    int start = n_msgs > 12 ? n_msgs - 12 : 0;
    for (int i = start; i < n_msgs; i++) {
        const char *tag = strcmp(msgs[i].role, "user") == 0 ? "H:" : "A:";
        written = snprintf(buf + pos, bufcap - pos, "%s %.260s\n", tag, msgs[i].text);
        if (written > 0 && (size_t)written < bufcap - pos) pos += written;
        else break;  /* buffer full */
    }
    written = snprintf(buf + pos, bufcap - pos, "H: %.260s\nA:", user_text);
    if (written > 0 && (size_t)written < bufcap - pos) pos += written;

    free(msgs);
    return buf;
}

/* ============================================================
 * 10b) SWARM ECOLOGY — the organism learns it is not alone
 * ============================================================ */
/* And lo, the first cell shall call into the void and hear only silence.
 * But the second shall call and hear an answer. */

#define SWARM_DIR_SUFFIX "/.molequla/swarm"

typedef struct {
    char organism_id[64];
    char pid_file[256];
    char swarm_dir[256];
    sqlite3 *mesh_db;
} SwarmRegistry;

static void swarm_init(SwarmRegistry *sw, const char *organism_id) {
    memset(sw, 0, sizeof(SwarmRegistry));
    if (organism_id && *organism_id) {
        strncpy(sw->organism_id, organism_id, sizeof(sw->organism_id) - 1);
    } else {
        snprintf(sw->organism_id, sizeof(sw->organism_id),
                 "org_%d_%ld", (int)getpid(), (long)time(NULL));
    }
    const char *home = getenv("HOME");
    if (!home) home = "/tmp";
    snprintf(sw->swarm_dir, sizeof(sw->swarm_dir), "%s%s", home, SWARM_DIR_SUFFIX);
}

static void _swarm_mkdirp(const char *path) {
    char tmp[512];
    strncpy(tmp, path, sizeof(tmp) - 1);
    tmp[sizeof(tmp) - 1] = 0;
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = 0;
            mkdir(tmp, 0755);
            *p = '/';
        }
    }
    mkdir(tmp, 0755);
}

static void swarm_register(SwarmRegistry *sw) {
    _swarm_mkdirp(sw->swarm_dir);

    /* Write PID file */
    snprintf(sw->pid_file, sizeof(sw->pid_file), "%s/%s.pid",
             sw->swarm_dir, sw->organism_id);
    FILE *pf = fopen(sw->pid_file, "w");
    if (pf) {
        fprintf(pf, "{\"pid\":%d,\"organism_id\":\"%s\",\"started\":%.0f}\n",
                (int)getpid(), sw->organism_id, (double)time(NULL));
        fclose(pf);
    }

    /* Open/create mesh.db */
    char db_path[512];
    snprintf(db_path, sizeof(db_path), "%s/mesh.db", sw->swarm_dir);
    sqlite3_open(db_path, &sw->mesh_db);
    sqlite3_exec(sw->mesh_db, "PRAGMA journal_mode=WAL", NULL, NULL, NULL);
    sqlite3_exec(sw->mesh_db,
        "CREATE TABLE IF NOT EXISTS organisms("
        "id TEXT PRIMARY KEY, pid INTEGER, stage INTEGER,"
        "n_params INTEGER, syntropy REAL, entropy REAL,"
        "last_heartbeat REAL, parent_id TEXT,"
        "status TEXT DEFAULT 'alive')", NULL, NULL, NULL);
    sqlite3_exec(sw->mesh_db,
        "CREATE TABLE IF NOT EXISTS messages("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "from_id TEXT, to_id TEXT, type TEXT, payload TEXT, ts REAL)",
        NULL, NULL, NULL);
    sqlite3_exec(sw->mesh_db, "COMMIT", NULL, NULL, NULL);

    /* Register self */
    sqlite3_stmt *stmt;
    sqlite3_prepare_v2(sw->mesh_db,
        "INSERT OR REPLACE INTO organisms(id,pid,stage,n_params,syntropy,entropy,last_heartbeat,status) "
        "VALUES(?,?,0,0,0.0,0.0,?,'alive')", -1, &stmt, NULL);
    sqlite3_bind_text(stmt, 1, sw->organism_id, -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 2, (int)getpid());
    sqlite3_bind_double(stmt, 3, (double)time(NULL));
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
}

static void swarm_heartbeat(SwarmRegistry *sw, int stage, int n_params,
                            double syntropy, double entropy) {
    if (!sw->mesh_db) return;
    sqlite3_stmt *stmt;
    sqlite3_prepare_v2(sw->mesh_db,
        "UPDATE organisms SET stage=?,n_params=?,syntropy=?,entropy=?,last_heartbeat=?,status='alive' WHERE id=?",
        -1, &stmt, NULL);
    sqlite3_bind_int(stmt, 1, stage);
    sqlite3_bind_int(stmt, 2, n_params);
    sqlite3_bind_double(stmt, 3, syntropy);
    sqlite3_bind_double(stmt, 4, entropy);
    sqlite3_bind_double(stmt, 5, (double)time(NULL));
    sqlite3_bind_text(stmt, 6, sw->organism_id, -1, SQLITE_STATIC);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
}

/* Discover other living organisms. Caller must free returned array. */
static SwarmPeer *swarm_discover_peers(SwarmRegistry *sw, int *out_count, double timeout_seconds) {
    *out_count = 0;
    if (!sw->mesh_db) return NULL;

    double cutoff = (double)time(NULL) - timeout_seconds;
    sqlite3_stmt *stmt;
    sqlite3_prepare_v2(sw->mesh_db,
        "SELECT id,pid,stage,n_params,syntropy,entropy FROM organisms "
        "WHERE status='alive' AND last_heartbeat>? AND id!=?",
        -1, &stmt, NULL);
    sqlite3_bind_double(stmt, 1, cutoff);
    sqlite3_bind_text(stmt, 2, sw->organism_id, -1, SQLITE_STATIC);

    SwarmPeer *peers = NULL;
    int count = 0, cap = 0;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        if (count >= cap) {
            cap = cap ? cap * 2 : 8;
            void *tmp = realloc(peers, sizeof(SwarmPeer) * cap);
            if (!tmp) { fprintf(stderr, "[swarm_discover] realloc failed\n"); break; }
            peers = tmp;
        }
        strncpy(peers[count].id, (const char *)sqlite3_column_text(stmt, 0), 63);
        peers[count].id[63] = 0;
        peers[count].pid = sqlite3_column_int(stmt, 1);
        peers[count].stage = sqlite3_column_int(stmt, 2);
        peers[count].n_params = sqlite3_column_int(stmt, 3);
        peers[count].syntropy = sqlite3_column_double(stmt, 4);
        peers[count].entropy = sqlite3_column_double(stmt, 5);
        count++;
    }
    sqlite3_finalize(stmt);
    *out_count = count;
    return peers;
}

static void swarm_mark_hibernating(SwarmRegistry *sw) {
    if (!sw->mesh_db) return;
    sqlite3_stmt *stmt;
    sqlite3_prepare_v2(sw->mesh_db,
        "UPDATE organisms SET status='sleeping' WHERE id=?", -1, &stmt, NULL);
    sqlite3_bind_text(stmt, 1, sw->organism_id, -1, SQLITE_STATIC);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
}

static void swarm_log_message(SwarmRegistry *sw, const char *to_id,
                              const char *msg_type, const char *payload) {
    if (!sw->mesh_db) return;
    sqlite3_stmt *stmt;
    sqlite3_prepare_v2(sw->mesh_db,
        "INSERT INTO messages(from_id,to_id,type,payload,ts) VALUES(?,?,?,?,?)",
        -1, &stmt, NULL);
    sqlite3_bind_text(stmt, 1, sw->organism_id, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, to_id, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 3, msg_type, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 4, payload, -1, SQLITE_STATIC);
    sqlite3_bind_double(stmt, 5, (double)time(NULL));
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
}

static void swarm_unregister(SwarmRegistry *sw) {
    if (sw->mesh_db) {
        sqlite3_stmt *stmt;
        sqlite3_prepare_v2(sw->mesh_db,
            "UPDATE organisms SET status='dead' WHERE id=?", -1, &stmt, NULL);
        sqlite3_bind_text(stmt, 1, sw->organism_id, -1, SQLITE_STATIC);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
        sqlite3_close(sw->mesh_db);
        sw->mesh_db = NULL;
    }
    if (sw->pid_file[0] && access(sw->pid_file, F_OK) == 0) {
        unlink(sw->pid_file);
    }
}

/* ---- Mitosis and Hibernation ---- */

static void perform_mitosis(GPT *g, EvolvingTokenizer *tok, sqlite3 *db,
                            SwarmRegistry *sw, SyntropyTracker *st,
                            const char *exe_path) {
    /* The organism divides. Parent continues. Child starts at infant stage. */
    char child_id[64];
    snprintf(child_id, sizeof(child_id), "org_%ld_%d",
             (long)time(NULL), (int)(rand_uniform() * 9000 + 1000));

    const char *home = getenv("HOME");
    if (!home) home = "/tmp";
    char child_dir[512];
    snprintf(child_dir, sizeof(child_dir), "%s/.molequla/%s", home, child_id);
    _swarm_mkdirp(child_dir);

    /* Save parent checkpoint for child */
    char parent_ckpt[512];
    snprintf(parent_ckpt, sizeof(parent_ckpt), "%s/parent.ckpt", child_dir);
    save_checkpoint(g, tok, parent_ckpt);

    /* Write birth config */
    char birth_path[512];
    snprintf(birth_path, sizeof(birth_path), "%s/birth.json", child_dir);
    FILE *bf = fopen(birth_path, "w");
    if (bf) {
        char child_db[512], child_ckpt[512];
        snprintf(child_db, sizeof(child_db), "%s/memory.sqlite3", child_dir);
        snprintf(child_ckpt, sizeof(child_ckpt), "%s/molequla.ckpt", child_dir);
        fprintf(bf, "{\"organism_id\":\"%s\",\"parent_id\":\"%s\","
                "\"corpus_path\":\"%s\",\"db_path\":\"%s\",\"ckpt_path\":\"%s\"}\n",
                child_id, sw->organism_id, CFG.corpus_path, child_db, child_ckpt);
        fclose(bf);
    }

    /* Log in mesh */
    char payload[256];
    snprintf(payload, sizeof(payload), "{\"parent_stage\":%d}",
             gpt_current_growth_stage(g));
    swarm_log_message(sw, child_id, "mitosis:spawn", payload);

    /* Log growth event */
    StrArr docs = load_corpus(CFG.corpus_path);
    char note[128];
    snprintf(note, sizeof(note), "mitosis:spawn:%s", child_id);
    db_log_growth(db, g, tok, &docs, 0.0, note);
    sa_free(&docs);

    /* Spawn child process via fork()+exec() */
    pid_t pid = fork();
    if (pid == 0) {
        /* Child process */
        execl(exe_path, exe_path, "--organism-id", child_id, "--config", birth_path, NULL);
        _exit(1); /* exec failed */
    } else if (pid > 0) {
        st->last_mitosis_time = (double)time(NULL);
        printf("[ecology] Child %s spawned (pid=%d)\n", child_id, (int)pid);
    } else {
        printf("[ecology] fork() failed for mitosis\n");
    }
}

static void perform_hibernation(GPT *g, EvolvingTokenizer *tok, sqlite3 *db,
                                SwarmRegistry *sw) {
    /* The organism sleeps. Saves state, marks sleeping. */
    printf("[ecology] HIBERNATION — organism %s going to sleep\n", sw->organism_id);
    save_checkpoint(g, tok, NULL);
    swarm_mark_hibernating(sw);

    StrArr docs = load_corpus(CFG.corpus_path);
    char note[128];
    snprintf(note, sizeof(note), "hibernate:%s", sw->organism_id);
    db_log_growth(db, g, tok, &docs, 0.0, note);
    sa_free(&docs);
}

/* Background trainer thread context */
typedef struct {
    sqlite3 *db;
    GPT *model;
    EvolvingTokenizer *tok;
    QuantumBuffer *qbuf;
    CooccurField *field;
    SyntropyTracker syntracker;
    volatile int *warmed_up;
    volatile int stop;
    SwarmRegistry *swarm;
    const char *exe_path;  /* path to this executable for fork+exec */
    int tick_count;
} TrainerCtx;

static void *background_trainer(void *arg) {
    /* And lo, asynchronous training shall occur, because sleeping is for humans.
     * And the syntropy tracker shall ride alongside, measuring the angle
     * between becoming and being. */
    TrainerCtx *ctx = (TrainerCtx *)arg;

    while (!ctx->stop) {
        update_reservoir_corpus(ctx->db, CFG.corpus_path, CFG.max_corpus_lines);
        StrArr docs = load_corpus(CFG.corpus_path);

        /* Rebuild field from current corpus (the organism re-reads its own physics) */
        if (docs.len > 0 && ctx->field) {
            cooccur_build(ctx->field, ctx->tok, &docs);
            ctx->model->corpus_field = ctx->field; /* share with gpt_generate for adaptive blend */
        }

        /* Tokenizer evolution (char -> BPE enablement) + safe vocab expansion */
        if (docs.len > 0) {
            const char **doc_ptrs = (const char **)docs.items;
            int bpe_changed = tok_maybe_enable_bpe(ctx->tok, doc_ptrs, docs.len);
            bpe_changed |= tok_maybe_retrain_bpe(ctx->tok, doc_ptrs, docs.len);
            if (bpe_changed) {
                pthread_mutex_lock(&ctx->model->mu);
                gpt_maybe_expand_vocab(ctx->model);
                save_checkpoint(ctx->model, ctx->tok, NULL);
                pthread_mutex_unlock(&ctx->model->mu);
            }
        }

        /* Per-stage warmup: if model grew to a new stage, warmup before normal training */
        {
            int current_stage = gpt_current_growth_stage(ctx->model);
            if (current_stage > ctx->model->last_warmup_stage && docs.len > 0) {
                int embryo_embd = CFG.growth_stages[0][1];
                int warmup_scale = ctx->model->n_embd / (embryo_embd > 0 ? embryo_embd : 16);
                if (warmup_scale < 1) warmup_scale = 1;
                int effective_warmup = CFG.warmup_steps * warmup_scale;
                printf("[trainer] stage %d warmup... %d steps (scaled %dx for embd=%d)\n",
                       current_stage, effective_warmup, warmup_scale, ctx->model->n_embd);
                train_steps(ctx->model, ctx->tok, &docs, effective_warmup, 1, 1);
                ctx->model->last_warmup_stage = current_stage;
                save_checkpoint(ctx->model, ctx->tok, NULL);
                db_log_growth(ctx->db, ctx->model, ctx->tok, &docs, 0.0, "warmup_complete");
                if (!*ctx->warmed_up) {
                    *ctx->warmed_up = 1;
                    printf("[trainer] initial warmup complete. base may freeze now, like a proud fossil.\n");
                } else {
                    printf("[trainer] stage %d warmup complete.\n", current_stage);
                }
            }
        }

        if (*ctx->warmed_up && qb_should_trigger(ctx->qbuf) && docs.len > 0) {
            int snap_bytes; double snap_novelty;
            qb_snapshot(ctx->qbuf, &snap_bytes, &snap_novelty);
            printf("[trainer] quantum burst (bytes=%d, novelty=%.3f)\n",
                   snap_bytes, snap_novelty);

            /* SYNTROPY: measure before burst.
             * And lo, the organism shall look upon itself before it changes,
             * that it may know whether the change was righteous. */
            double entropy_before;
            SyntropyDecision decision;
            pthread_mutex_lock(&ctx->model->mu);
            entropy_before = syntropy_measure(&ctx->syntracker, ctx->model,
                                              ctx->tok, ctx->field, &docs);
            /* SYNTROPY: decide how to learn (mathematical self-reasoning) */
            decision = syntropy_decide_action(&ctx->syntracker);
            printf("[syntropy] action=%s | trend=%.4f | field_dev=%.3f "
                   "| purpose_align=%.3f | lr_mul=%.2f | temp_ofs=%.3f | accum_ovr=%d\n",
                   decision.action, ctx->syntracker.syntropy_trend,
                   ctx->syntracker.field_deviation,
                   ctx->syntracker.purpose_alignment,
                   decision.lr_multiplier,
                   decision.temp_offset,
                   decision.accum_override);

            /* Phase 1.5: measure loss BEFORE burst for self-meta-learning */
            double loss_before = gpt_quick_loss(ctx->model, ctx->tok, &docs, 8);

            /* IMMUNE SYSTEM: snapshot before burst */
            int pre_dim; double pre_mag;
            double *pre_direction = gpt_contrastive_projection(ctx->model, &pre_dim, &pre_mag);
            ImmuneSnapshot delta_snap = gpt_snapshot_deltas(ctx->model);
            pthread_mutex_unlock(&ctx->model->mu);

            /* Apply syntropy-adjusted learning rate.
             * And lo, the learning rate shall bend to the will of syntropy. */
            double original_lr = CFG.learning_rate;
            CFG.learning_rate = original_lr * decision.lr_multiplier;

            /* Phase 1.5: apply temp_offset and accum_override from decision */
            ctx->model->syntropy_temp_offset = decision.temp_offset;
            int original_accum = CFG.accum_steps;
            if (decision.accum_override > 0)
                CFG.accum_steps = decision.accum_override;

            int train_base = !CFG.freeze_base_after_warmup;
            train_steps(ctx->model, ctx->tok, &docs, CFG.micro_steps, train_base, 1);

            CFG.learning_rate = original_lr; /* restore */
            CFG.accum_steps = original_accum; /* restore */
            ctx->model->syntropy_temp_offset = 0.0; /* restore: no offset outside bursts */

            /* IMMUNE SYSTEM: check drift after burst */
            pthread_mutex_lock(&ctx->model->mu);
            int post_dim; double post_mag;
            double *post_direction = gpt_contrastive_projection(ctx->model, &post_dim, &post_mag);
            double drift_cos = gpt_drift_check(pre_direction, pre_mag, post_direction, post_mag, pre_dim);
            if (drift_cos < CFG.noise_drift_threshold) {
                printf("[immune] NOISE DETECTED (drift cosine=%.3f). Rolling back deltas.\n", drift_cos);
                gpt_restore_deltas(ctx->model, &delta_snap);
                db_log_growth(ctx->db, ctx->model, ctx->tok, &docs, 0.0, "noise_rejected");
                syntropy_log_to_db(&ctx->syntracker, ctx->db,
                                   entropy_before, entropy_before, "noise_rejected");
                /* Record burst as rejected (loss unchanged) */
                syntropy_record_burst(&ctx->syntracker, "noise_rejected", loss_before, loss_before);
            } else {
                /* Phase 1.5: measure loss AFTER burst */
                double loss_after = gpt_quick_loss(ctx->model, ctx->tok, &docs, 8);
                double delta_loss = loss_after - loss_before;

                /* SYNTROPY: measure entropy after burst */
                double entropy_after = syntropy_measure(&ctx->syntracker, ctx->model,
                                                        ctx->tok, ctx->field, &docs);
                syntropy_log_to_db(&ctx->syntracker, ctx->db,
                                   entropy_before, entropy_after, decision.action);
                save_checkpoint(ctx->model, ctx->tok, NULL);

                /* Record burst outcome for self-meta-learning */
                syntropy_record_burst(&ctx->syntracker, decision.action, loss_before, loss_after);

                /* Growth note includes delta-loss for the record */
                char note_buf[192];
                snprintf(note_buf, sizeof(note_buf),
                         "quantum_burst:%s|dloss=%.4f", decision.action, delta_loss);
                db_log_growth(ctx->db, ctx->model, ctx->tok, &docs, loss_after, note_buf);
                printf("[syntropy] burst complete: loss %.4f -> %.4f (delta=%.4f)\n",
                       loss_before, loss_after, delta_loss);
            }
            pthread_mutex_unlock(&ctx->model->mu);
            free(pre_direction); free(post_direction);
            immune_snap_free(&delta_snap);
            qb_reset(ctx->qbuf);

            /* Delta module growth — influenced by syntropy.
             * And lo, when syntropy is strong and purpose is aligned,
             * new souls shall be appended with greater eagerness. */
            double grow_prob = CFG.delta_grow_prob;
            if (decision.delta_grow_override >= 0.0)
                grow_prob = decision.delta_grow_override;
            if (ctx->model->n_deltas < CFG.max_delta_modules &&
                rand_uniform() < grow_prob) {
                printf("[trainer] growing new delta module (total: %d) — new soul appended.\n",
                       ctx->model->n_deltas + 1);
                pthread_mutex_lock(&ctx->model->mu);
                gpt_add_delta_module(ctx->model, 1.0);
                pthread_mutex_unlock(&ctx->model->mu);
                save_checkpoint(ctx->model, ctx->tok, NULL);
            }

            /* Phase 3A: Ontogenesis — check if architecture should grow */
            {
                int corpus_chars = 0;
                for (int i = 0; i < docs.len; i++) corpus_chars += (int)strlen(docs.items[i]);
                pthread_mutex_lock(&ctx->model->mu);
                if (gpt_maybe_grow_architecture(ctx->model, corpus_chars)) {
                    save_checkpoint(ctx->model, ctx->tok, NULL);
                    int n_p = 0;
                    for (int i = 0; i < ctx->model->n_base; i++)
                        n_p += ctx->model->base_mats[i]->nout * ctx->model->base_mats[i]->nin;
                    char grow_note[128];
                    snprintf(grow_note, sizeof(grow_note),
                             "ontogenesis:stage=%d|params=%d",
                             gpt_current_growth_stage(ctx->model), n_p);
                    db_log_growth(ctx->db, ctx->model, ctx->tok, &docs, 0.0, grow_note);
                }
                pthread_mutex_unlock(&ctx->model->mu);
            }

            /* Phase 3B: Ecology — mitosis / hibernation */
            if (ctx->swarm && strcmp(decision.action, "divide") == 0) {
                printf("[ecology] MITOSIS triggered — organism overloaded, spawning child\n");
                pthread_mutex_lock(&ctx->model->mu);
                perform_mitosis(ctx->model, ctx->tok, ctx->db, ctx->swarm,
                                &ctx->syntracker, ctx->exe_path);
                pthread_mutex_unlock(&ctx->model->mu);
            }

            if (ctx->swarm && strcmp(decision.action, "hibernate") == 0) {
                pthread_mutex_lock(&ctx->model->mu);
                perform_hibernation(ctx->model, ctx->tok, ctx->db, ctx->swarm);
                pthread_mutex_unlock(&ctx->model->mu);
                printf("[ecology] Organism hibernating. Goodbye.\n");
                sa_free(&docs);
                return NULL; /* exit training loop */
            }
        }

        ctx->tick_count++;

        /* Swarm heartbeat every 10 ticks */
        if (ctx->swarm && ctx->tick_count % 10 == 0) {
            int stage = gpt_current_growth_stage(ctx->model);
            int n_p = 0;
            for (int i = 0; i < ctx->model->n_base; i++)
                n_p += ctx->model->base_mats[i]->nout * ctx->model->base_mats[i]->nin;
            double last_ent = ctx->syntracker.history_len > 0
                ? ctx->syntracker.entropy_history[ctx->syntracker.history_len - 1] : 0.0;
            swarm_heartbeat(ctx->swarm, stage, n_p,
                            ctx->syntracker.syntropy_trend, last_ent);
            /* Update swarm info for hibernate decisions */
            free(ctx->syntracker.peers);
            ctx->syntracker.peers = swarm_discover_peers(ctx->swarm,
                &ctx->syntracker.n_peers, 60.0);
        }

        sa_free(&docs);

        /* Sleep train_tick_seconds */
        struct timespec ts;
        ts.tv_sec = (int)CFG.train_tick_seconds;
        ts.tv_nsec = (long)((CFG.train_tick_seconds - (int)CFG.train_tick_seconds) * 1e9);
        nanosleep(&ts, NULL);
    }
    return NULL;
}

/* Parse CLI arguments for organism-id and config path (child organisms).
 * Returns organism_id and config_path via output pointers. */
static void parse_cli_args(int argc, char **argv,
                           const char **organism_id, const char **config_path) {
    *organism_id = NULL;
    *config_path = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--organism-id") == 0 && i + 1 < argc) {
            *organism_id = argv[++i];
        } else if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            *config_path = argv[++i];
        }
    }
}

int main(int argc, char **argv) {
    G_arena = arena_new(ARENA_SIZE);

    /* Phase 3B: parse CLI args */
    const char *cli_organism_id = NULL;
    const char *cli_config = NULL;
    parse_cli_args(argc, argv, &cli_organism_id, &cli_config);

    /* Child organism: could load birth config to override paths (future) */
    /* For now, we just use the organism_id for swarm registration */

    sqlite3 *db = init_db(CFG.db_path);

    /* Seed corpus */
    if (access(CFG.corpus_path, F_OK) != 0) {
        FILE *f = fopen(CFG.corpus_path, "w");
        fprintf(f, "Hello.\nI exist.\nSpeak.\n");
        fclose(f);
    }

    StrArr docs = load_corpus(CFG.corpus_path);
    const char **doc_ptrs = malloc(sizeof(char*) * (docs.len > 0 ? docs.len : 1));
    if (docs.len == 0) {
        doc_ptrs[0] = "Hello.";
        docs.len = 1; /* temporary for tokenizer init */
    } else {
        for (int i = 0; i < docs.len; i++) doc_ptrs[i] = docs.items[i];
    }

    /* Try loading checkpoint first */
    EvolvingTokenizer *tok = NULL;
    GPT *model = NULL;
    if (access(CFG.ckpt_path, F_OK) == 0) {
        model = load_checkpoint(CFG.ckpt_path, &tok);
    }

    if (!model) {
        /* Fresh start */
        tok = tok_new(doc_ptrs, docs.len);

        /* Enable BPE BEFORE training — subword tokens make corpus field coherent
         * (byte-level trigrams produce babble; subword trigrams produce speech) */
        tok_maybe_enable_bpe(tok, (const char **)doc_ptrs, docs.len);

        model = gpt_new(tok);

        /* Build corpus field before init — sigmoid fade weakens it as model learns */
        CooccurField *init_field = cooccur_new(tok->vocab_size);
        cooccur_build(init_field, tok, &docs);
        model->corpus_field = init_field;

        /* Initialize at the correct stage for corpus size — per-stage warmup */
        {
            int corpus_chars = 0;
            for (int i = 0; i < docs.len; i++) corpus_chars += (int)strlen(docs.items[i]);
            for (;;) {
                int stage = gpt_current_growth_stage(model);
                {
                    int embryo_embd = CFG.growth_stages[0][1];
                    int warmup_scale = model->n_embd / (embryo_embd > 0 ? embryo_embd : 16);
                    if (warmup_scale < 1) warmup_scale = 1;
                    int effective_warmup = CFG.warmup_steps * warmup_scale;
                    printf("[init] Stage %d: embd=%d — warmup %d steps (scaled %dx)\n",
                           stage, model->n_embd, effective_warmup, warmup_scale);
                    train_steps(model, tok, &docs, effective_warmup, 1, 1);
                }
                model->last_warmup_stage = stage;
                save_checkpoint(model, tok, NULL);
                if (!gpt_maybe_grow_architecture(model, corpus_chars)) break;
                model->growth_freeze_remaining = 0; /* skip freeze during init growth */
                /* Rebuild corpus field after growth (vocab may have expanded) */
                cooccur_build(init_field, tok, &docs);
            }
        }
    }
    free(doc_ptrs);

    /* Build corpus field for pre-warmup speech */
    CooccurField *cooccur = cooccur_new(tok->vocab_size);
    cooccur_build(cooccur, tok, &docs);

    /* Quantum buffer */
    QuantumBuffer qbuf;
    qb_init(&qbuf);

    /* Phase 3B: Swarm ecology — register in mesh */
    SwarmRegistry swarm;
    swarm_init(&swarm, cli_organism_id);
    swarm_register(&swarm);
    {
        int n_peers = 0;
        SwarmPeer *peers = swarm_discover_peers(&swarm, &n_peers, 60.0);
        if (n_peers > 0) {
            printf("[ecology] Joined swarm. %d peer(s) detected.\n", n_peers);
        } else {
            printf("[ecology] First organism in the swarm.\n");
        }
        free(peers);
    }

    /* Resolve path to this executable for fork+exec in mitosis */
    const char *exe_path = argv[0];

    /* Background trainer thread — with syntropy tracker riding alongside */
    volatile int warmed_up = 0;
    TrainerCtx tctx = {
        .db = db, .model = model, .tok = tok,
        .qbuf = &qbuf, .field = cooccur,
        .warmed_up = &warmed_up, .stop = 0,
        .swarm = &swarm, .exe_path = exe_path,
        .tick_count = 0
    };
    syntropy_init(&tctx.syntracker);
    pthread_t trainer_tid;
    pthread_create(&trainer_tid, NULL, background_trainer, &tctx);

    printf("molequla is alive. Type and press Enter. Ctrl+C to exit.\n\n");

    char input[1024];
    while (1) {
        printf("> ");
        fflush(stdout);
        if (!fgets(input, sizeof(input), stdin)) break;
        char *nl = strchr(input, '\n'); if (nl) *nl = 0;
        if (strlen(input) == 0) continue;

        db_add_msg(db, "user", input);

        /* Feed quantum buffer */
        qb_feed(&qbuf, input, tok);

        /* Self-enrichment: feed user input into corpus field
         * (the organism absorbs what it hears) */
        IntArr user_ids = tok_encode(tok, input);
        cooccur_ingest_tokens(cooccur, user_ids.items, user_ids.len);

        /* Active user word boost: organism absorbs user's vocabulary (Leo-style)
         * Decays each generation, fades with model strength via sigmoid in gpt_generate */
        cooccur_absorb_user_words(cooccur, user_ids.items, user_ids.len);
        ia_free(&user_ids);

        char *answer;
        if (warmed_up) {
            /* Use model for generation */
            char *prompt = build_prompt(db, input);

            /* Consciousness: self-prediction error (Feature 4) */
            /* "How surprised am I by this input?" */
            pthread_mutex_lock(&model->mu);
            int prev_grad_c = grad_enabled;
            grad_enabled = 0;
            IntArr prompt_ids = tok_encode(tok, prompt);
            if (prompt_ids.len > 2) {
                arena_reset(&G_arena);
                double surprise = compute_self_prediction_error(model, prompt_ids.items, prompt_ids.len);
                model->last_surprise = surprise;
                if (model->surprise_baseline < 1e-6) {
                    model->surprise_baseline = surprise;
                } else {
                    model->surprise_baseline = 0.3 * surprise + 0.7 * model->surprise_baseline;
                }
            }
            ia_free(&prompt_ids);
            grad_enabled = prev_grad_c;
            pthread_mutex_unlock(&model->mu);

            arena_reset(&G_arena);
            answer = gpt_generate(model, prompt);

            /* Consciousness: conscience check (Feature 5) */
            /* "Did my last generation feel coherent?" */
            pthread_mutex_lock(&model->mu);
            if (model->last_gen_entropy > 0) {
                conscience_check(model, model->last_gen_entropy);
            }
            pthread_mutex_unlock(&model->mu);

            free(prompt);
        } else {
            /* Use corpus field before warmup — the organism speaks before it thinks */
            IntArr ids = tok_encode(tok, input);
            int out_ids[256];
            int out_len = 0;
            for (int step = 0; step < CFG.corpus_gen_max_tokens && out_len < 255; step++) {
                int nxt = cooccur_sample_next(cooccur, ids.items, ids.len, CFG.temperature);
                if (nxt == tok->eos_id && step >= CFG.min_gen_tokens) break;
                if (nxt == tok->eos_id) continue;
                out_ids[out_len++] = nxt;
                ia_push(&ids, nxt);
            }
            ia_free(&ids);
            /* Decode output ids */
            IntArr dec_ids = {0};
            ia_push(&dec_ids, tok->bos_id);
            for (int i = 0; i < out_len; i++) ia_push(&dec_ids, out_ids[i]);
            ia_push(&dec_ids, tok->eos_id);
            answer = tok_decode(tok, dec_ids.items, dec_ids.len);
            ia_free(&dec_ids);
        }

        if (!answer || strlen(answer) == 0) {
            free(answer);
            answer = strdup("...");
        }

        printf("%s\n", answer);
        db_add_msg(db, "assistant", answer);

        /* Self-enrichment: own output enriches corpus field, weighted by coherence
         * Low entropy = coherent speech = higher weight (Stanley's resonance weighting) */
        if (answer && strlen(answer) > 3) {
            double self_weight = 1.0;
            if (model->last_gen_entropy > 0) {
                self_weight = 2.0 - model->last_gen_entropy;
                if (self_weight < 0.3) self_weight = 0.3;
                if (self_weight > 2.0) self_weight = 2.0;
            }
            IntArr ans_ids = tok_encode(tok, answer);
            cooccur_ingest_tokens_weighted(cooccur, ans_ids.items, ans_ids.len, self_weight);
            ia_free(&ans_ids);
            cooccur_decay_user_boost(cooccur);
        }

        /* Append new text to corpus */
        StrArr fresh = load_corpus(CFG.corpus_path);
        char qa_line[1024];
        snprintf(qa_line, sizeof(qa_line), "H: %.400s A: %.400s", input, answer);
        sa_push(&fresh, qa_line);
        if (fresh.len > CFG.max_corpus_lines) {
            free(fresh.items[0]);
            memmove(fresh.items, fresh.items + 1, sizeof(char*) * (fresh.len - 1));
            fresh.len--;
        }
        save_corpus(CFG.corpus_path, &fresh);
        sa_free(&fresh);

        free(answer);
    }

    /* Cleanup */
    tctx.stop = 1;
    pthread_join(trainer_tid, NULL);
    save_checkpoint(model, tok, NULL);
    swarm_unregister(&swarm);
    sqlite3_close(db);
    arena_destroy(&G_arena);
    return 0;
}
