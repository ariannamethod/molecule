/*
 * molecule.c
 * A dependency-free, single-file, continually-learning GPT organism in pure C.
 *
 * Compile: gcc -O2 -o molecule molecule.c -lsqlite3 -lpthread -lm
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
#include <sqlite3.h>

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
    const char *head_types[4];
    int n_head_types;
    double hybrid_alpha_init;

    /* gamma */
    double gamma_sparsity_threshold;

    /* entropy temperature */
    double entropy_low, entropy_high;
    double entropy_temp_boost, entropy_temp_focus;

    /* corpus field */
    int corpus_gen_max_tokens;

    /* quantum buffer */
    int qb_min_bytes;
    double qb_min_novelty;
    double qb_cooldown_seconds;
} Config;

static Config CFG = {
    .corpus_path = "nonames.txt",
    .db_path = "memory.sqlite3",
    .ckpt_path = "molecule.ckpt",
    .max_corpus_lines = 8000,
    .max_line_chars = 240,
    .min_new_chars = 480,
    .tie_embeddings = 1,
    .n_layer = 2,
    .n_embd = 72,
    .n_head = 4,
    .block_size = 96,
    .warmup_steps = 1200,
    .micro_steps = 32,
    .learning_rate = 0.01,
    .beta1 = 0.9, .beta2 = 0.99, .eps_adam = 1e-8,
    .grad_clip = 1.0,
    .freeze_base_after_warmup = 1,
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
    .enable_bpe_after_chars = 25000,
    .bpe_num_merges = 384,
    .bpe_retrain_every_chars = 4000,
    .train_tick_seconds = 0.25,
    .head_types = {"content", "content", "hybrid", "hybrid"},
    .n_head_types = 4,
    .hybrid_alpha_init = 0.5,
    .gamma_sparsity_threshold = 0.01,
    .entropy_low = 0.5, .entropy_high = 1.5,
    .entropy_temp_boost = 1.2, .entropy_temp_focus = 0.8,
    .corpus_gen_max_tokens = 120,
    .qb_min_bytes = 1024,
    .qb_min_novelty = 0.15,
    .qb_cooldown_seconds = 60.0,
};

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
        a->items = realloc(a->items, sizeof(char*) * a->cap);
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
        a->items = realloc(a->items, sizeof(int) * a->cap);
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

#define ARENA_SIZE (64 * 1024 * 1024) /* 64 MB */

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
    BinCtx *c = arena_alloc(&G_arena, sizeof(BinCtx));
    c->a = a; c->b = b; c->len = n;
    out->ctx = c;
    out->backward = back_add;
    Node *kids[] = {a, b};
    node_set_children(out, kids, 2);
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
    BinCtx *c = arena_alloc(&G_arena, sizeof(BinCtx));
    c->a = a; c->b = b; c->len = n;
    out->ctx = c;
    out->backward = back_sub;
    Node *kids[] = {a, b};
    node_set_children(out, kids, 2);
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
    BinCtx *c = arena_alloc(&G_arena, sizeof(BinCtx));
    c->a = a; c->b = b; c->len = n;
    out->ctx = c;
    out->backward = back_mul_vec;
    Node *kids[] = {a, b};
    node_set_children(out, kids, 2);
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
    ScaleCtx *c = arena_alloc(&G_arena, sizeof(ScaleCtx));
    c->a = a; c->s = s; c->len = n;
    out->ctx = c;
    out->backward = back_scale;
    Node *kids[] = {a};
    node_set_children(out, kids, 1);
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
    BinCtx *c = arena_alloc(&G_arena, sizeof(BinCtx));
    c->a = a; c->len = n;
    out->ctx = c;
    out->backward = back_relu;
    Node *kids[] = {a};
    node_set_children(out, kids, 1);
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
    DotCtx *c = arena_alloc(&G_arena, sizeof(DotCtx));
    c->a = a; c->b = b; c->len = n;
    out->ctx = c;
    out->backward = back_dot;
    Node *kids[] = {a, b};
    node_set_children(out, kids, 2);
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
    MeanSqCtx *c = arena_alloc(&G_arena, sizeof(MeanSqCtx));
    c->a = a; c->len = n;
    out->ctx = c;
    out->backward = back_meansq;
    Node *kids[] = {a};
    node_set_children(out, kids, 1);
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
    SliceCtx *c = arena_alloc(&G_arena, sizeof(SliceCtx));
    c->a = a; c->start = start; c->end = end;
    out->ctx = c;
    out->backward = back_slice;
    Node *kids[] = {a};
    node_set_children(out, kids, 1);
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
    ElemCtx *c = arena_alloc(&G_arena, sizeof(ElemCtx));
    c->a = a; c->idx = idx;
    out->ctx = c;
    out->backward = back_elem;
    Node *kids[] = {a};
    node_set_children(out, kids, 1);
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
    out->backward = back_scalar_mul;
    Node *kids[] = {a, b};
    node_set_children(out, kids, 2);
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
    ConcatCtx *c = arena_alloc(&G_arena, sizeof(ConcatCtx));
    c->vecs = arena_alloc(&G_arena, sizeof(Node*) * n_vecs);
    memcpy(c->vecs, vecs, sizeof(Node*) * n_vecs);
    c->n_vecs = n_vecs;
    c->offsets = offsets;
    out->ctx = c;
    out->backward = back_concat;
    node_set_children(out, vecs, n_vecs);
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
    out->backward = back_scalar_add;
    Node *kids[] = {a, b};
    node_set_children(out, kids, 2);
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
    ScaleCtx *c = arena_alloc(&G_arena, sizeof(ScaleCtx));
    c->a = a; c->s = f;
    out->ctx = c;
    out->backward = back_scalar_mulf;
    Node *kids[] = {a};
    node_set_children(out, kids, 1);
    return out;
}

/* --- Backward (topological sort) --- */
/* And lo, the graph shall be walked backwards, like a salmon with regrets. */

#define MAX_TOPO 65536

static void backward(Node *root) {
    Node *topo[MAX_TOPO];
    int topo_len = 0;

    /* Iterative DFS for topo sort */
    Node *stack[MAX_TOPO];
    int stack_len = 0;
    stack[stack_len++] = root;

    while (stack_len > 0) {
        Node *n = stack[stack_len - 1];
        if (n->visited == 1) {
            stack_len--;
            if (n->visited != 2) {
                n->visited = 2;
                topo[topo_len++] = n;
            }
            continue;
        }
        n->visited = 1;
        for (int i = 0; i < n->n_children; i++) {
            if (n->children[i] && n->children[i]->visited == 0)
                stack[stack_len++] = n->children[i];
        }
    }

    root->grad[0] = 1.0;
    for (int i = topo_len - 1; i >= 0; i--) {
        if (topo[i]->backward)
            topo[i]->backward(topo[i]);
    }
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
    m->row_data = realloc(m->row_data, sizeof(double*) * new_nout);
    m->row_grad = realloc(m->row_grad, sizeof(double*) * new_nout);
    for (int i = m->nout; i < new_nout; i++) {
        m->row_data[i] = calloc(m->nin, sizeof(double));
        m->row_grad[i] = calloc(m->nin, sizeof(double));
        for (int j = 0; j < m->nin; j++)
            m->row_data[i][j] = rand_normal() * std;
    }
    m->nout = new_nout;
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
    for (int i = 0; i < nout; i++) {
        double s = 0;
        for (int j = 0; j < nin; j++) s += m->row_data[i][j] * x->data[j];
        out->data[i] = s;
    }

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

    RMSCtx *c = arena_alloc(&G_arena, sizeof(RMSCtx));
    c->x = x; c->scale_val = scale; c->ms_data = ms; c->len = n;
    out->ctx = c;
    out->backward = back_rmsnorm;
    Node *kids[] = {x};
    node_set_children(out, kids, 1);
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

    CECtx *c = arena_alloc(&G_arena, sizeof(CECtx));
    c->logits = logits; c->probs = probs; c->target = target; c->vocab = n;
    out->ctx = c;
    out->backward = back_ce;
    Node *kids[] = {logits};
    node_set_children(out, kids, 1);
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

    SoftmaxCtx *shared = arena_alloc(&G_arena, sizeof(SoftmaxCtx));
    shared->logits = logits; shared->probs = probs; shared->n = n;

    for (int i = 0; i < n; i++) {
        out[i] = node_new(1);
        out[i]->data[0] = probs[i];
        out[i]->ctx = shared;
        out[i]->backward = back_softmax_i;
        node_set_children(out[i], logits, n);
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

    AttnSumCtx *c = arena_alloc(&G_arena, sizeof(AttnSumCtx));
    c->weights = weights; c->values = values; c->T = T; c->dim = dim;
    out->ctx = c;
    out->backward = back_attn_sum;

    int nk = T * 2;
    Node **kids = arena_alloc(&G_arena, sizeof(Node*) * nk);
    for (int i = 0; i < T; i++) { kids[i] = weights[i]; kids[T+i] = values[i]; }
    node_set_children(out, kids, nk);
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

/* ============================================================
 * 5) TOKENIZER — char first, then BPE
 * ============================================================ */

typedef struct { char a[64]; char b[64]; } MergePair;

typedef struct {
    char **tokens;
    int vocab_size;
    int stoi[65536]; /* hash table: token string hash -> index (simplified) */
    int bpe_enabled;
    MergePair *merges;
    int n_merges;
    int trained_chars;

    int bos_id, eos_id, pad_id;
} Tokenizer;

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

    /* Collect unique chars */
    int seen[256] = {0};
    for (int d = 0; d < n_docs; d++)
        for (const char *p = docs[d]; *p; p++) seen[(unsigned char)*p] = 1;
    seen[(unsigned char)'\n'] = 1;

    /* Sort chars */
    for (int c = 0; c < 256; c++) {
        if (!seen[c]) continue;
        char s[2] = {(char)c, 0};
        tok->tokens[tok->vocab_size] = strdup(s);
        stoi_put(tok->stoi, s, tok->vocab_size);
        tok->vocab_size++;
    }

    /* Special tokens */
    const char *specials[] = {"<PAD>", "<BOS>", "<EOS>"};
    for (int i = 0; i < 3; i++) {
        tok->tokens[tok->vocab_size] = strdup(specials[i]);
        stoi_put(tok->stoi, specials[i], tok->vocab_size);
        if (i == 0) tok->pad_id = tok->vocab_size;
        if (i == 1) tok->bos_id = tok->vocab_size;
        if (i == 2) tok->eos_id = tok->vocab_size;
        tok->vocab_size++;
    }

    return tok;
}

static void tok_add_token(EvolvingTokenizer *tok, const char *s) {
    if (stoi_get(tok->stoi, s) >= 0) return;
    if (tok->vocab_size >= tok->cap) {
        tok->cap *= 2;
        tok->tokens = realloc(tok->tokens, sizeof(char*) * tok->cap);
    }
    tok->tokens[tok->vocab_size] = strdup(s);
    stoi_put(tok->stoi, s, tok->vocab_size);
    tok->vocab_size++;
}

static IntArr tok_encode(EvolvingTokenizer *tok, const char *s) {
    IntArr ids = {0};
    /* Skip leading/trailing whitespace */
    while (*s == ' ' || *s == '\t' || *s == '\n') s++;
    int slen = strlen(s);
    while (slen > 0 && (s[slen-1] == ' ' || s[slen-1] == '\t' || s[slen-1] == '\n')) slen--;

    ia_push(&ids, tok->bos_id);

    /* Char-level encoding (BPE TODO for C version) */
    for (int i = 0; i < slen; i++) {
        char cs[2] = {s[i], 0};
        int id = stoi_get(tok->stoi, cs);
        if (id >= 0) ia_push(&ids, id);
    }

    ia_push(&ids, tok->eos_id);
    return ids;
}

static char *tok_decode(EvolvingTokenizer *tok, const int *ids, int n) {
    size_t bufcap = 1024;
    char *buf = calloc(bufcap, 1);
    size_t pos = 0;
    for (int i = 0; i < n; i++) {
        if (ids[i] < 0 || ids[i] >= tok->vocab_size) continue;
        const char *t = tok->tokens[ids[i]];
        if (strcmp(t, "<BOS>") == 0 || strcmp(t, "<PAD>") == 0) continue;
        if (strcmp(t, "<EOS>") == 0) break;
        size_t tlen = strlen(t);
        while (pos + tlen + 1 > bufcap) { bufcap *= 2; buf = realloc(buf, bufcap); }
        memcpy(buf + pos, t, tlen);
        pos += tlen;
    }
    buf[pos] = 0;
    return buf;
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
    RopeCtx *c = arena_alloc(&G_arena, sizeof(RopeCtx));
    c->vec = vec; c->pos = pos; c->head_dim = head_dim;
    out->ctx = c;
    out->backward = back_rope;
    Node *kids[] = {vec};
    node_set_children(out, kids, 1);
    return out;
}

/* Delta module: maps name -> DeltaAdapter */
#define MAX_ADAPTERS_PER_MOD 48

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

static void adam_step(AdamState *st, MatrixParam *mat, double lr) {
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

/* The GPT model */
#define MAX_BASE_MATS 128
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

/* Apply base weight + delta adapters */
static Node *gpt_apply(GPT *g, const char *name, Node *x) {
    MatrixParam *base = gpt_base(g, name);
    Node *y = mat_matvec(base, x);
    for (int d = 0; d < g->n_deltas; d++) {
        DeltaAdapter *da = dmod_get(g->deltas[d], name);
        if (da) {
            Node *dy = delta_apply(da, x);
            dy = vec_scale(dy, g->active_alpha[d]);
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
                rrpram_logits = arena_alloc(&G_arena, sizeof(Node*) * T);
                for (int t = 0; t < T; t++)
                    rrpram_logits[t] = vec_element(pattern_full, t);
            }

            /* Dispatch by head type */
            Node **attn_w = arena_alloc(&G_arena, sizeof(Node*) * T);
            if (strcmp(htype, "content") == 0) {
                scalar_softmax(content_logits, T, attn_w);
            } else if (strcmp(htype, "rrpram") == 0) {
                scalar_softmax(rrpram_logits, T, attn_w);
            } else { /* hybrid */
                char aname[64];
                snprintf(aname, sizeof(aname), "l%d.h%d.alpha", li, h);
                MatrixParam *am = gpt_base(g, aname);
                double alpha_raw = am->row_data[0][0];
                double a = 1.0 / (1.0 + exp(-alpha_raw)); /* sigmoid */
                Node **blended = arena_alloc(&G_arena, sizeof(Node*) * T);
                for (int t = 0; t < T; t++) {
                    Node *cl = scalar_mulf(content_logits[t], 1.0 - a);
                    Node *rl = scalar_mulf(rrpram_logits[t], a);
                    blended[t] = scalar_add(cl, rl);
                }
                scalar_softmax(blended, T, attn_w);
            }

            head_outs[h] = attn_weighted_sum(attn_w, vh, T);
        }

        Node *x_attn = vec_concat(head_outs, g->n_head);
        snprintf(name, sizeof(name), "l%d.wo", li);
        x = gpt_apply(g, name, x_attn);
        x = vec_add(x, x_res);

        /* Gated MLP (SwiGLU-ish) */
        x_res = x;
        x = rmsnorm(x);
        snprintf(name, sizeof(name), "l%d.fc_g", li);
        Node *gate = vec_relu(gpt_apply(g, name, x));
        snprintf(name, sizeof(name), "l%d.fc_v", li);
        Node *val = gpt_apply(g, name, x);
        x = vec_mul(gate, val);
        snprintf(name, sizeof(name), "l%d.fc2", li);
        x = gpt_apply(g, name, x);
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

/* Generate */
static char *gpt_generate(GPT *g, const char *prompt) {
    pthread_mutex_lock(&g->mu);

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
    double *probs_buf = malloc(sizeof(double) * g->tok->vocab_size);

    for (int step = 0; step < CFG.max_gen_tokens; step++) {
        arena_reset(&G_arena);
        int pos = ids.len - 1;
        if (pos > g->block_size - 1) pos = g->block_size - 1;
        Node *logits = gpt_forward_step(g, cur, pos, kv);

        /* Entropy-adaptive temperature */
        double base_temp = CFG.temperature;
        if (base_temp < 1e-6) base_temp = 1e-6;
        int V = logits->len;
        double *scaled = malloc(sizeof(double) * V);
        for (int i = 0; i < V; i++) scaled[i] = logits->data[i] / base_temp;
        softmax_probs(scaled, V, probs_buf);
        double entropy = 0;
        for (int i = 0; i < V; i++)
            if (probs_buf[i] > 1e-12) entropy -= probs_buf[i] * log(probs_buf[i]);
        double tmul = 1.0;
        if (entropy < CFG.entropy_low) tmul = CFG.entropy_temp_boost;
        else if (entropy > CFG.entropy_high) tmul = CFG.entropy_temp_focus;
        double temp = base_temp * tmul;
        for (int i = 0; i < V; i++) scaled[i] = logits->data[i] / temp;
        softmax_probs(scaled, V, probs_buf);

        int nxt = top_k_top_p_sample(probs_buf, V, CFG.top_k, CFG.top_p, CFG.min_p, CFG.typical_p);
        free(scaled);

        if (nxt == g->tok->eos_id) {
            if (step >= CFG.min_gen_tokens) break;
            continue;
        }

        ia_push(&ids, nxt);
        cur = nxt;
        ia_push(&out_ids, nxt);

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

    /* Decode output */
    IntArr dec = {0};
    ia_push(&dec, g->tok->bos_id);
    for (int i = 0; i < out_ids.len; i++) ia_push(&dec, out_ids.items[i]);
    ia_push(&dec, g->tok->eos_id);
    char *result = tok_decode(g->tok, dec.items, dec.len);

    /* Cleanup */
    free(probs_buf);
    ia_free(&ids); ia_free(&out_ids); ia_free(&recent); ia_free(&dec);
    for (int i = 0; i < kv->n_layers; i++) { free(kv->layers[i].keys); free(kv->layers[i].values); }
    free(kv->layers); free(kv);

    pthread_mutex_unlock(&g->mu);
    return result;
}

/* ============================================================
 * 7) SQLITE MEMORY
 * ============================================================ */

static sqlite3 *init_db(const char *path) {
    sqlite3 *db;
    sqlite3_open(path, &db);
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
    int accumulated_bytes;
    int unique_tokens[8192]; /* simple hash set */
    int unique_count;
    int total_tokens;
    double last_burst_time;
} QuantumBuffer;

static void qb_init(QuantumBuffer *qb) {
    memset(qb, 0, sizeof(QuantumBuffer));
}

static void qb_feed(QuantumBuffer *qb, const char *text, EvolvingTokenizer *tok) {
    qb->accumulated_bytes += strlen(text);
    IntArr ids = tok_encode(tok, text);
    for (int i = 0; i < ids.len; i++) {
        int h = ids.items[i] % 8192;
        if (qb->unique_tokens[h] != ids.items[i] + 1) {
            qb->unique_tokens[h] = ids.items[i] + 1;
            qb->unique_count++;
        }
        qb->total_tokens++;
    }
    ia_free(&ids);
}

static double qb_novelty(QuantumBuffer *qb) {
    if (qb->total_tokens == 0) return 0.0;
    return (double)qb->unique_count / (double)qb->total_tokens;
}

static int qb_should_trigger(QuantumBuffer *qb) {
    double now = (double)time(NULL);
    int bytes_ok = qb->accumulated_bytes >= CFG.qb_min_bytes;
    int novelty_ok = qb_novelty(qb) >= CFG.qb_min_novelty;
    int cooldown_ok = (now - qb->last_burst_time) >= CFG.qb_cooldown_seconds;
    return (bytes_ok || novelty_ok) && cooldown_ok;
}

static void qb_reset(QuantumBuffer *qb) {
    qb->accumulated_bytes = 0;
    memset(qb->unique_tokens, 0, sizeof(qb->unique_tokens));
    qb->unique_count = 0;
    qb->total_tokens = 0;
    qb->last_burst_time = (double)time(NULL);
}

/* ============================================================
 * 8d) COOCCUR FIELD — speech before learning
 * ============================================================ */

/* And lo, the corpus shall whisper its statistics, and words shall follow words. */
#define COOCCUR_HASH_SIZE 16384

typedef struct { int key[3]; double count; } TrigramEntry;

typedef struct {
    double *unigram;  /* [vocab_size] */
    int vocab_size;
    TrigramEntry *trigrams;
    int n_trigrams, trigram_cap;
    int built;
} CooccurField;

static CooccurField *cooccur_new(int vocab_size) {
    CooccurField *cf = calloc(1, sizeof(CooccurField));
    cf->vocab_size = vocab_size;
    cf->unigram = calloc(vocab_size, sizeof(double));
    cf->trigram_cap = 4096;
    cf->trigrams = calloc(cf->trigram_cap, sizeof(TrigramEntry));
    return cf;
}

static void cooccur_build(CooccurField *cf, EvolvingTokenizer *tok, StrArr *docs) {
    memset(cf->unigram, 0, sizeof(double) * cf->vocab_size);
    cf->n_trigrams = 0;
    for (int d = 0; d < docs->len; d++) {
        IntArr ids = tok_encode(tok, docs->items[d]);
        for (int i = 0; i < ids.len; i++) {
            if (ids.items[i] < cf->vocab_size)
                cf->unigram[ids.items[i]] += 1.0;
        }
        /* Store trigrams (simplified: just keep last N) */
        for (int i = 0; i < ids.len - 2 && cf->n_trigrams < cf->trigram_cap; i++) {
            cf->trigrams[cf->n_trigrams].key[0] = ids.items[i];
            cf->trigrams[cf->n_trigrams].key[1] = ids.items[i+1];
            cf->trigrams[cf->n_trigrams].key[2] = ids.items[i+2];
            cf->trigrams[cf->n_trigrams].count = 1.0;
            cf->n_trigrams++;
        }
        ia_free(&ids);
    }
    cf->built = 1;
}

static int cooccur_sample_next(CooccurField *cf, const int *ctx, int ctx_len, double temperature) {
    double *counts = calloc(cf->vocab_size, sizeof(double));
    int found = 0;

    /* Try trigram */
    if (ctx_len >= 2) {
        int a = ctx[ctx_len-2], b = ctx[ctx_len-1];
        for (int i = 0; i < cf->n_trigrams; i++) {
            if (cf->trigrams[i].key[0] == a && cf->trigrams[i].key[1] == b) {
                int c = cf->trigrams[i].key[2];
                if (c < cf->vocab_size) { counts[c] += 1.0; found = 1; }
            }
        }
    }

    /* Fallback to unigram */
    if (!found) {
        memcpy(counts, cf->unigram, sizeof(double) * cf->vocab_size);
    }

    /* Temperature + sample */
    double total = 0;
    for (int i = 0; i < cf->vocab_size; i++) {
        if (counts[i] > 0 && temperature > 0)
            counts[i] = pow(counts[i], 1.0 / temperature);
        total += counts[i];
    }
    if (total <= 0) { free(counts); return rand_int(cf->vocab_size); }

    double r = rand_uniform() * total;
    double s = 0;
    int result = cf->vocab_size - 1;
    for (int i = 0; i < cf->vocab_size; i++) {
        s += counts[i];
        if (s >= r) { result = i; break; }
    }
    free(counts);
    return result;
}

/* ============================================================
 * 9) TRAINING
 * ============================================================ */

static void train_steps(GPT *g, EvolvingTokenizer *tok, StrArr *docs, int steps,
                        int train_base, int train_deltas) {
    if (docs->len == 0) return;
    pthread_mutex_lock(&g->mu);

    for (int step = 0; step < steps; step++) {
        arena_reset(&G_arena);

        /* Sample batch of 4 */
        Node *total_loss = node_new(1);
        int batch = 4;
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

        double lr = CFG.learning_rate * (1.0 - (double)step / (double)(steps > 1 ? steps : 1));

        if (train_base) {
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

/* ============================================================
 * 11) CHAT LOOP + MAIN
 * ============================================================ */

static char *build_prompt(sqlite3 *db, const char *user_text) {
    int n_msgs;
    Msg *msgs = db_recent(db, 14, &n_msgs);
    char *buf = calloc(8192, 1);
    strcat(buf, "A: (I listen. I answer. I learn.)\n");

    int start = n_msgs > 12 ? n_msgs - 12 : 0;
    for (int i = start; i < n_msgs; i++) {
        const char *tag = strcmp(msgs[i].role, "user") == 0 ? "H:" : "A:";
        char line[600];
        snprintf(line, sizeof(line), "%s %.260s\n", tag, msgs[i].text);
        strcat(buf, line);
    }
    char line[600];
    snprintf(line, sizeof(line), "H: %.260s\nA:", user_text);
    strcat(buf, line);

    free(msgs);
    return buf;
}

int main(void) {
    G_arena = arena_new(ARENA_SIZE);

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

    EvolvingTokenizer *tok = tok_new(doc_ptrs, docs.len);
    GPT *model = gpt_new(tok);
    free(doc_ptrs);

    /* Warmup training */
    if (docs.len > 0) {
        printf("[trainer] warmup training... (and so it begins)\n");
        train_steps(model, tok, &docs, CFG.warmup_steps, 1, 1);
        save_checkpoint(model, tok, NULL);
        printf("[trainer] warmup complete. base may freeze now, like a proud fossil.\n");
    }

    printf("molecule is alive. Type and press Enter. Ctrl+C to exit.\n\n");

    char input[1024];
    while (1) {
        printf("> ");
        fflush(stdout);
        if (!fgets(input, sizeof(input), stdin)) break;
        char *nl = strchr(input, '\n'); if (nl) *nl = 0;
        if (strlen(input) == 0) continue;

        db_add_msg(db, "user", input);

        char *prompt = build_prompt(db, input);
        arena_reset(&G_arena);
        char *answer = gpt_generate(model, prompt);
        free(prompt);

        if (!answer || strlen(answer) == 0) {
            free(answer);
            answer = strdup("...");
        }

        printf("%s\n", answer);
        db_add_msg(db, "assistant", answer);
        free(answer);
    }

    save_checkpoint(model, tok, NULL);
    sqlite3_close(db);
    arena_destroy(&G_arena);
    return 0;
}
