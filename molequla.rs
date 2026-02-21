// molequla.rs
// The Fifth Element: a GPT organism + distributed cognition metabolism.
// Rust port — full feature parity with Go/C/Python/JS + mesh coordinator.
// In the beginning there was nonames.txt. And Rust said, "hold my borrow checker."

use std::cell::Cell;
use std::collections::HashMap;
use std::fs;
use std::io::{self, BufRead, Write as IoWrite};
use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};
use rand::Rng;
use rand::seq::SliceRandom;
use rusqlite::{Connection, params};
use serde::{Serialize, Deserialize};

// BLAS FFI — activated with --features blas
#[cfg(feature = "blas")]
mod blas_ffi {
    // CblasRowMajor = 101, CblasNoTrans = 111
    extern "C" {
        pub fn cblas_dgemv(order: i32, trans: i32, m: i32, n: i32,
                           alpha: f64, a: *const f64, lda: i32,
                           x: *const f64, incx: i32,
                           beta: f64, y: *mut f64, incy: i32);
    }
    pub unsafe fn dgemv(nout: usize, nin: usize, a: &[f64], x: &[f64], y: &mut [f64]) {
        unsafe {
            cblas_dgemv(101, 111, nout as i32, nin as i32,
                        1.0, a.as_ptr(), nin as i32,
                        x.as_ptr(), 1, 0.0, y.as_mut_ptr(), 1);
        }
    }
}

#[cfg(feature = "blas")]
const HAS_BLAS: bool = true;
#[cfg(not(feature = "blas"))]
const HAS_BLAS: bool = false;

thread_local! { static GRAD_ENABLED: Cell<bool> = Cell::new(true); }
fn grad_on() -> bool { GRAD_ENABLED.with(|g| g.get()) }
fn set_grad(v: bool) { GRAD_ENABLED.with(|g| g.set(v)); }
fn now_secs() -> f64 { SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64() }

// ============================================================
// 0) CONFIG
// ============================================================

#[derive(Clone, Serialize, Deserialize)]
struct Config {
    corpus_path: String, db_path: String, ckpt_path: String,
    max_corpus_lines: usize, max_line_chars: usize, min_new_chars_to_train: usize,
    tie_embeddings: bool,
    n_layer: usize, n_embd: usize, n_head: usize, block_size: usize,
    growth_stages: Vec<[usize; 4]>, freeze_after_growth_steps: usize, post_growth_lr_scale: f64,
    warmup_steps: usize, micro_steps: usize,
    learning_rate: f64, beta1: f64, beta2: f64, eps_adam: f64, grad_clip: f64,
    freeze_base_after_warmup: bool, batch_size: usize,
    lr_min: f64, max_total_steps: usize, cosine_warmup_steps: usize, accum_steps: usize,
    delta_rank: usize, max_delta_modules: usize, delta_grow_prob: f64,
    temperature: f64, top_k: usize, top_p: f64, min_p: f64, typical_p: f64,
    max_gen_tokens: usize, min_gen_tokens: usize, repetition_guard: usize,
    enable_bpe_after_chars: usize, bpe_num_merges: usize, bpe_retrain_every_chars: usize,
    train_tick_seconds: f64,
    head_types: Vec<String>, hybrid_alpha_init: f64,
    gamma_sparsity_threshold: f64, noise_drift_threshold: f64, gamma_min_magnitude: f64,
    entropy_low: f64, entropy_high: f64, entropy_temp_boost: f64, entropy_temp_focus: f64,
    corpus_gen_max_tokens: usize, corpus_fade_k: f64, corpus_fade_threshold: f64,
    qb_min_bytes: usize, qb_min_novelty: f64, qb_cooldown_seconds: f64,
    syntropy_window: usize, field_deviation_ceiling: f64, field_deviation_floor: f64,
    syntropy_lr_boost: f64, syntropy_lr_dampen: f64, syntropy_delta_grow_boost: f64,

    // consciousness: per-token dissonance feedback
    #[serde(default = "default_dissonance_ema_alpha")]
    dissonance_ema_alpha: f64,
    #[serde(default = "default_dissonance_spike_k")]
    dissonance_spike_k: f64,
    #[serde(default = "default_dissonance_drop_k")]
    dissonance_drop_k: f64,
    #[serde(default = "default_dissonance_spike_threshold")]
    dissonance_spike_threshold: f64,
    #[serde(default = "default_dissonance_drop_threshold")]
    dissonance_drop_threshold: f64,

    // consciousness: pattern breaking (anti-field generation)
    #[serde(default = "default_anti_field_prob")]
    anti_field_prob: f64,
    #[serde(default = "default_anti_field_min_step")]
    anti_field_min_step: usize,

    // consciousness: conscience (self-editing)
    #[serde(default = "default_conscience_window")]
    conscience_window: usize,
    #[serde(default = "default_conscience_decay")]
    conscience_decay: f64,
    #[serde(default = "default_conscience_recovery")]
    conscience_recovery: f64,
    #[serde(default = "default_conscience_floor")]
    conscience_floor: f64,

    // frequency/presence penalty
    #[serde(default = "default_freq_penalty")]
    freq_penalty: f64,
    #[serde(default = "default_presence_penalty")]
    presence_penalty: f64,

    // co-occurrence enhancements (Stanley/Leo-style)
    #[serde(default = "default_cooccur_window_size")]
    cooccur_window_size: usize,
    #[serde(default = "default_user_boost_strength")]
    user_boost_strength: f64,
    #[serde(default = "default_user_boost_decay")]
    user_boost_decay: f64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            corpus_path: "nonames.txt".into(), db_path: "memory.sqlite3".into(),
            ckpt_path: "molequla_ckpt.json".into(),
            max_corpus_lines: 8000, max_line_chars: 240, min_new_chars_to_train: 480,
            tie_embeddings: true, n_layer: 1, n_embd: 16, n_head: 1, block_size: 96,
            // embryo: ~10K params, infant: ~28K params, child: ~154K params, adolescent: ~1.1M params, teen: ~4.1M params, adult: ~10M params
            growth_stages: vec![[0,16,1,1],[20000,32,1,2],[50000,64,2,4],[200000,128,4,4],[350000,224,5,8],[500000,320,6,8]],
            freeze_after_growth_steps: 500, post_growth_lr_scale: 0.3,
            warmup_steps: 1200, micro_steps: 32,
            learning_rate: 0.01, beta1: 0.9, beta2: 0.99, eps_adam: 1e-8, grad_clip: 1.0,
            freeze_base_after_warmup: true, batch_size: 4,
            lr_min: 0.001, max_total_steps: 50000, cosine_warmup_steps: 200, accum_steps: 1,
            delta_rank: 8, max_delta_modules: 12, delta_grow_prob: 0.08,
            temperature: 0.85, top_k: 40, top_p: 0.92, min_p: 0.06, typical_p: 0.95,
            max_gen_tokens: 180, min_gen_tokens: 16, repetition_guard: 4,
            enable_bpe_after_chars: 20000, bpe_num_merges: 384, bpe_retrain_every_chars: 4000,
            train_tick_seconds: 0.25,
            head_types: vec!["content".into()], hybrid_alpha_init: 0.5,
            gamma_sparsity_threshold: 0.01, noise_drift_threshold: -0.1, gamma_min_magnitude: 1e-6,
            entropy_low: 0.5, entropy_high: 1.5, entropy_temp_boost: 1.2, entropy_temp_focus: 0.8,
            corpus_gen_max_tokens: 120, corpus_fade_k: 3.0, corpus_fade_threshold: 1.5,
            qb_min_bytes: 1024, qb_min_novelty: 0.15, qb_cooldown_seconds: 60.0,
            syntropy_window: 8, field_deviation_ceiling: 12.0, field_deviation_floor: 0.1,
            syntropy_lr_boost: 1.3, syntropy_lr_dampen: 0.6, syntropy_delta_grow_boost: 0.15,

            // consciousness defaults
            dissonance_ema_alpha: 0.3, dissonance_spike_k: 0.8, dissonance_drop_k: 1.2,
            dissonance_spike_threshold: 1.5, dissonance_drop_threshold: 0.5,
            anti_field_prob: 0.05, anti_field_min_step: 8,
            conscience_window: 8, conscience_decay: 0.95, conscience_recovery: 1.005, conscience_floor: 0.3,
            freq_penalty: 0.1, presence_penalty: 0.1,
            cooccur_window_size: 5, user_boost_strength: 0.3, user_boost_decay: 0.7,
        }
    }
}

fn head_types_for(n: usize) -> Vec<String> {
    if n <= 1 { return vec!["content".into()]; }
    if n == 2 { return vec!["content".into(), "hybrid".into()]; }
    let half = (n + 1) / 2; // majority content
    let mut t: Vec<String> = (0..half).map(|_| "content".into()).collect();
    t.extend((half..n).map(|_| "hybrid".into()));
    t
}

// ============================================================
// 1) AUTOGRAD TAPE — index-based computation graph
// ============================================================

type NodeId = usize;

#[derive(Clone)]
enum MatRef {
    Base(String),
    DeltaA(usize, String),
    DeltaB(usize, String),
}

#[derive(Clone)]
enum Op {
    Leaf,
    Add(NodeId, NodeId),
    Sub(NodeId, NodeId),
    Mul(NodeId, NodeId),
    Scale(NodeId, f64),
    AddScalar(NodeId, f64),
    Neg(NodeId),
    ReLU(NodeId),
    SiLU(NodeId),
    Dot(NodeId, NodeId),
    MeanSq(NodeId),
    Element(NodeId, usize),
    Slice(NodeId, usize, usize),
    Concat(Vec<(NodeId, usize)>),
    MatVec { mat: MatRef, x: NodeId },
    EmbedLookup { mat: MatRef, row: usize },
    RMSNorm(NodeId),
    CrossEntropy { logits: NodeId, target: usize },
    SoftmaxAttn { logits: Vec<NodeId>, values: Vec<NodeId> },
    RoPE { input: NodeId, pos: usize, head_dim: usize },
    ScalarAdd(NodeId, NodeId),
    ScalarMul(NodeId, NodeId),
    ScalarMulF(NodeId, f64),
    ScalarAddF(NodeId, f64),
    Sigmoid(NodeId),
}

struct TapeNode {
    data: Vec<f64>,
    grad: Vec<f64>,
    op: Op,
    aux: Vec<f64>,
}

struct Tape {
    nodes: Vec<TapeNode>,
}

impl Tape {
    fn new() -> Self { Tape { nodes: Vec::with_capacity(4096) } }

    fn push(&mut self, data: Vec<f64>, op: Op, aux: Vec<f64>) -> NodeId {
        let n = data.len();
        let grad = vec![0.0; n];
        let id = self.nodes.len();
        self.nodes.push(TapeNode { data, grad, op, aux });
        id
    }

    fn leaf(&mut self, data: &[f64]) -> NodeId {
        self.push(data.to_vec(), Op::Leaf, vec![])
    }

    fn scalar_leaf(&mut self, v: f64) -> NodeId {
        self.push(vec![v], Op::Leaf, vec![])
    }

    fn data(&self, id: NodeId) -> &[f64] { &self.nodes[id].data }

    fn embed_lookup(&mut self, mat: MatRef, row: usize, row_data: &[f64]) -> NodeId {
        let op = if grad_on() { Op::EmbedLookup { mat, row } } else { Op::Leaf };
        self.push(row_data.to_vec(), op, vec![])
    }

    fn add(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let n = self.nodes[a].data.len();
        let mut d = vec![0.0; n];
        for i in 0..n { d[i] = self.nodes[a].data[i] + self.nodes[b].data[i]; }
        let op = if grad_on() { Op::Add(a, b) } else { Op::Leaf };
        self.push(d, op, vec![])
    }

    fn sub(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let n = self.nodes[a].data.len();
        let mut d = vec![0.0; n];
        for i in 0..n { d[i] = self.nodes[a].data[i] - self.nodes[b].data[i]; }
        let op = if grad_on() { Op::Sub(a, b) } else { Op::Leaf };
        self.push(d, op, vec![])
    }

    fn mul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let n = self.nodes[a].data.len();
        let mut d = vec![0.0; n];
        for i in 0..n { d[i] = self.nodes[a].data[i] * self.nodes[b].data[i]; }
        let op = if grad_on() { Op::Mul(a, b) } else { Op::Leaf };
        self.push(d, op, vec![])
    }

    fn scale(&mut self, a: NodeId, s: f64) -> NodeId {
        let d: Vec<f64> = self.nodes[a].data.iter().map(|&x| x * s).collect();
        let op = if grad_on() { Op::Scale(a, s) } else { Op::Leaf };
        self.push(d, op, vec![])
    }

    fn neg(&mut self, a: NodeId) -> NodeId {
        let d: Vec<f64> = self.nodes[a].data.iter().map(|&x| -x).collect();
        let op = if grad_on() { Op::Neg(a) } else { Op::Leaf };
        self.push(d, op, vec![])
    }

    fn relu(&mut self, a: NodeId) -> NodeId {
        let d: Vec<f64> = self.nodes[a].data.iter().map(|&x| x.max(0.0)).collect();
        let op = if grad_on() { Op::ReLU(a) } else { Op::Leaf };
        self.push(d, op, vec![])
    }

    fn silu(&mut self, a: NodeId) -> NodeId {
        let n = self.nodes[a].data.len();
        let mut sig = vec![0.0; n];
        let mut d = vec![0.0; n];
        for i in 0..n {
            sig[i] = 1.0 / (1.0 + (-self.nodes[a].data[i]).exp());
            d[i] = self.nodes[a].data[i] * sig[i];
        }
        let op = if grad_on() { Op::SiLU(a) } else { Op::Leaf };
        self.push(d, op, sig)
    }

    fn dot(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let n = self.nodes[a].data.len();
        let mut val = 0.0;
        for i in 0..n { val += self.nodes[a].data[i] * self.nodes[b].data[i]; }
        let op = if grad_on() { Op::Dot(a, b) } else { Op::Leaf };
        self.push(vec![val], op, vec![])
    }

    fn element(&mut self, a: NodeId, idx: usize) -> NodeId {
        let val = self.nodes[a].data[idx];
        let op = if grad_on() { Op::Element(a, idx) } else { Op::Leaf };
        self.push(vec![val], op, vec![])
    }

    fn slice(&mut self, a: NodeId, start: usize, end: usize) -> NodeId {
        let d = self.nodes[a].data[start..end].to_vec();
        let op = if grad_on() { Op::Slice(a, start, end) } else { Op::Leaf };
        self.push(d, op, vec![])
    }

    fn concat(&mut self, parts: &[NodeId]) -> NodeId {
        let mut d = Vec::new();
        let mut info = Vec::new();
        for &p in parts {
            let len = self.nodes[p].data.len();
            d.extend_from_slice(&self.nodes[p].data);
            info.push((p, len));
        }
        let op = if grad_on() { Op::Concat(info) } else { Op::Leaf };
        self.push(d, op, vec![])
    }

    fn matvec(&mut self, mat: MatRef, mat_data: &[Vec<f64>], x: NodeId) -> NodeId {
        let nout = mat_data.len();
        let nin = if nout > 0 { mat_data[0].len() } else { 0 };
        let x_data = &self.nodes[x].data;
        let mut d = vec![0.0; nout];
        #[cfg(feature = "blas")]
        {
            if nout * nin >= 256 {
                let mut buf = vec![0.0f64; nout * nin];
                for i in 0..nout {
                    buf[i * nin..(i + 1) * nin].copy_from_slice(&mat_data[i][..nin]);
                }
                unsafe { blas_ffi::dgemv(nout, nin, &buf, x_data, &mut d); }
            } else {
                for i in 0..nout {
                    let mut s = 0.0;
                    for j in 0..nin { s += mat_data[i][j] * x_data[j]; }
                    d[i] = s;
                }
            }
        }
        #[cfg(not(feature = "blas"))]
        {
            for i in 0..nout {
                let mut s = 0.0;
                for j in 0..nin { s += mat_data[i][j] * x_data[j]; }
                d[i] = s;
            }
        }
        let op = if grad_on() { Op::MatVec { mat, x } } else { Op::Leaf };
        self.push(d, op, vec![])
    }

    fn rmsnorm(&mut self, a: NodeId) -> NodeId {
        let x = &self.nodes[a].data;
        let n = x.len();
        let nf = n as f64;
        let mut ms = 0.0;
        for &v in x.iter() { ms += v * v; }
        ms /= nf;
        let rms = (ms + 1e-5).sqrt();
        let inv = 1.0 / rms;
        let d: Vec<f64> = x.iter().map(|&v| v * inv).collect();
        let mut aux = vec![inv, ms];
        aux.extend_from_slice(x);
        let op = if grad_on() { Op::RMSNorm(a) } else { Op::Leaf };
        self.push(d, op, aux)
    }

    fn cross_entropy(&mut self, logits: NodeId, target: usize) -> NodeId {
        let x = &self.nodes[logits].data;
        let max_v = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut exps: Vec<f64> = x.iter().map(|&v| (v - max_v).exp()).collect();
        let sum_exp: f64 = exps.iter().sum();
        for v in exps.iter_mut() { *v /= sum_exp; }
        let loss = -(exps[target].max(1e-12)).ln();
        let op = if grad_on() { Op::CrossEntropy { logits, target } } else { Op::Leaf };
        self.push(vec![loss], op, exps)
    }

    fn softmax_attn(&mut self, logits: &[NodeId], values: &[NodeId]) -> NodeId {
        let t = logits.len();
        let dim = if t > 0 { self.nodes[values[0]].data.len() } else { 0 };
        // softmax over logit scalars
        let raw: Vec<f64> = logits.iter().map(|&l| self.nodes[l].data[0]).collect();
        let max_v = raw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut probs: Vec<f64> = raw.iter().map(|&v| (v - max_v).exp()).collect();
        let sum_p: f64 = probs.iter().sum();
        for p in probs.iter_mut() { *p /= sum_p; }
        // weighted sum of values
        let mut d = vec![0.0; dim];
        for i in 0..t {
            for j in 0..dim { d[j] += probs[i] * self.nodes[values[i]].data[j]; }
        }
        let op = if grad_on() {
            Op::SoftmaxAttn { logits: logits.to_vec(), values: values.to_vec() }
        } else { Op::Leaf };
        self.push(d, op, probs)
    }

    fn rope(&mut self, input: NodeId, pos: usize, head_dim: usize) -> NodeId {
        let x = &self.nodes[input].data;
        let half = head_dim / 2;
        let mut d = vec![0.0; head_dim];
        for j in 0..half {
            let theta = (pos as f64) / (10000.0_f64).powf(2.0 * j as f64 / head_dim as f64);
            let (sin_t, cos_t) = theta.sin_cos();
            d[2*j]   = x[2*j] * cos_t - x[2*j+1] * sin_t;
            d[2*j+1] = x[2*j] * sin_t + x[2*j+1] * cos_t;
        }
        let op = if grad_on() { Op::RoPE { input, pos, head_dim } } else { Op::Leaf };
        self.push(d, op, vec![])
    }

    fn scalar_add(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let v = self.nodes[a].data[0] + self.nodes[b].data[0];
        let op = if grad_on() { Op::ScalarAdd(a, b) } else { Op::Leaf };
        self.push(vec![v], op, vec![])
    }

    fn scalar_mul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let v = self.nodes[a].data[0] * self.nodes[b].data[0];
        let op = if grad_on() { Op::ScalarMul(a, b) } else { Op::Leaf };
        self.push(vec![v], op, vec![])
    }

    fn scalar_mulf(&mut self, a: NodeId, f: f64) -> NodeId {
        let v = self.nodes[a].data[0] * f;
        let op = if grad_on() { Op::ScalarMulF(a, f) } else { Op::Leaf };
        self.push(vec![v], op, vec![])
    }

    fn scalar_addf(&mut self, a: NodeId, f: f64) -> NodeId {
        let v = self.nodes[a].data[0] + f;
        let op = if grad_on() { Op::ScalarAddF(a, f) } else { Op::Leaf };
        self.push(vec![v], op, vec![])
    }

    fn sigmoid(&mut self, a: NodeId) -> NodeId {
        let v = 1.0 / (1.0 + (-self.nodes[a].data[0]).exp());
        let op = if grad_on() { Op::Sigmoid(a) } else { Op::Leaf };
        self.push(vec![v], op, vec![])
    }

    // ---- BACKWARD ----
    fn backward(&mut self, root: NodeId, model: &mut GPT) {
        self.nodes[root].grad[0] = 1.0;
        for i in (0..=root).rev() {
            let grad = self.nodes[i].grad.clone();
            if grad.iter().all(|&g| g == 0.0) { continue; }
            let op = self.nodes[i].op.clone();
            let aux = self.nodes[i].aux.clone();
            match op {
                Op::Leaf => {},
                Op::Add(a, b) => {
                    for j in 0..grad.len() {
                        self.nodes[a].grad[j] += grad[j];
                        self.nodes[b].grad[j] += grad[j];
                    }
                },
                Op::Sub(a, b) => {
                    for j in 0..grad.len() {
                        self.nodes[a].grad[j] += grad[j];
                        self.nodes[b].grad[j] -= grad[j];
                    }
                },
                Op::Mul(a, b) => {
                    let ad = self.nodes[a].data.clone();
                    let bd = self.nodes[b].data.clone();
                    for j in 0..grad.len() {
                        self.nodes[a].grad[j] += bd[j] * grad[j];
                        self.nodes[b].grad[j] += ad[j] * grad[j];
                    }
                },
                Op::Scale(a, s) => {
                    for j in 0..grad.len() { self.nodes[a].grad[j] += s * grad[j]; }
                },
                Op::AddScalar(a, _) => {
                    for j in 0..grad.len() { self.nodes[a].grad[j] += grad[j]; }
                },
                Op::Neg(a) => {
                    for j in 0..grad.len() { self.nodes[a].grad[j] -= grad[j]; }
                },
                Op::ReLU(a) => {
                    let ad = self.nodes[a].data.clone();
                    for j in 0..grad.len() {
                        if ad[j] > 0.0 { self.nodes[a].grad[j] += grad[j]; }
                    }
                },
                Op::SiLU(a) => {
                    let ad = self.nodes[a].data.clone();
                    for j in 0..grad.len() {
                        let sig = aux[j];
                        self.nodes[a].grad[j] += (sig * (1.0 + ad[j] * (1.0 - sig))) * grad[j];
                    }
                },
                Op::Dot(a, b) => {
                    let ad = self.nodes[a].data.clone();
                    let bd = self.nodes[b].data.clone();
                    for j in 0..ad.len() {
                        self.nodes[a].grad[j] += bd[j] * grad[0];
                        self.nodes[b].grad[j] += ad[j] * grad[0];
                    }
                },
                Op::MeanSq(a) => {
                    let ad = self.nodes[a].data.clone();
                    let nf = ad.len() as f64;
                    for j in 0..ad.len() {
                        self.nodes[a].grad[j] += (2.0 * ad[j] / nf) * grad[0];
                    }
                },
                Op::Element(a, idx) => {
                    self.nodes[a].grad[idx] += grad[0];
                },
                Op::Slice(a, start, _end) => {
                    for j in 0..grad.len() {
                        self.nodes[a].grad[start + j] += grad[j];
                    }
                },
                Op::Concat(parts) => {
                    let mut off = 0;
                    for (nid, len) in parts {
                        for j in 0..len { self.nodes[nid].grad[j] += grad[off + j]; }
                        off += len;
                    }
                },
                Op::MatVec { mat, x } => {
                    let x_data = self.nodes[x].data.clone();
                    let nout = grad.len();
                    let nin = x_data.len();
                    let mat_data = model.get_mat_data(&mat);
                    // grad for x
                    for r in 0..nout {
                        for c in 0..nin {
                            self.nodes[x].grad[c] += mat_data[r][c] * grad[r];
                        }
                    }
                    // grad for matrix
                    let mat_grad = model.get_mat_grad_mut(&mat);
                    for r in 0..nout {
                        for c in 0..nin {
                            mat_grad[r][c] += x_data[c] * grad[r];
                        }
                    }
                },
                Op::EmbedLookup { mat, row } => {
                    let mat_grad = model.get_mat_grad_mut(&mat);
                    for j in 0..grad.len() { mat_grad[row][j] += grad[j]; }
                },
                Op::RMSNorm(a) => {
                    let inv_rms = aux[0];
                    let ms = aux[1];
                    let orig_x = &aux[2..];
                    let n = orig_x.len();
                    let nf = n as f64;
                    let mut dot_grad = 0.0;
                    for j in 0..n { dot_grad += grad[j] * orig_x[j]; }
                    let scale = dot_grad * inv_rms / (nf * (ms + 1e-5));
                    for j in 0..n {
                        self.nodes[a].grad[j] += inv_rms * grad[j] - scale * orig_x[j];
                    }
                },
                Op::CrossEntropy { logits, target } => {
                    let probs = &aux;
                    for j in 0..probs.len() {
                        let indicator = if j == target { 1.0 } else { 0.0 };
                        self.nodes[logits].grad[j] += (probs[j] - indicator) * grad[0];
                    }
                },
                Op::SoftmaxAttn { logits, values } => {
                    let probs = &aux;
                    let t = logits.len();
                    let dim = grad.len();
                    // grad for values
                    for ti in 0..t {
                        for j in 0..dim {
                            self.nodes[values[ti]].grad[j] += probs[ti] * grad[j];
                        }
                    }
                    // grad for logits via softmax jacobian
                    let mut d_probs = vec![0.0; t];
                    for ti in 0..t {
                        let vd = &self.nodes[values[ti]].data;
                        for j in 0..dim { d_probs[ti] += vd[j] * grad[j]; }
                    }
                    let sum_dp: f64 = (0..t).map(|ti| probs[ti] * d_probs[ti]).sum();
                    for ti in 0..t {
                        self.nodes[logits[ti]].grad[0] += probs[ti] * (d_probs[ti] - sum_dp);
                    }
                },
                Op::RoPE { input, pos, head_dim } => {
                    let half = head_dim / 2;
                    for j in 0..half {
                        let theta = (pos as f64) / (10000.0_f64).powf(2.0 * j as f64 / head_dim as f64);
                        let (sin_t, cos_t) = theta.sin_cos();
                        self.nodes[input].grad[2*j]   += grad[2*j] * cos_t + grad[2*j+1] * sin_t;
                        self.nodes[input].grad[2*j+1] += -grad[2*j] * sin_t + grad[2*j+1] * cos_t;
                    }
                },
                Op::ScalarAdd(a, b) => {
                    self.nodes[a].grad[0] += grad[0];
                    self.nodes[b].grad[0] += grad[0];
                },
                Op::ScalarMul(a, b) => {
                    let av = self.nodes[a].data[0];
                    let bv = self.nodes[b].data[0];
                    self.nodes[a].grad[0] += bv * grad[0];
                    self.nodes[b].grad[0] += av * grad[0];
                },
                Op::ScalarMulF(a, f) => {
                    self.nodes[a].grad[0] += f * grad[0];
                },
                Op::ScalarAddF(a, _) => {
                    self.nodes[a].grad[0] += grad[0];
                },
                Op::Sigmoid(a) => {
                    let out = self.nodes[i].data[0];
                    self.nodes[a].grad[0] += out * (1.0 - out) * grad[0];
                },
            }
        }
    }
}

// ============================================================
// 2) MATRIX PARAM
// ============================================================

#[derive(Clone, Serialize, Deserialize)]
struct MatrixParam {
    data: Vec<Vec<f64>>,
    #[serde(skip)]
    grad: Vec<Vec<f64>>,
    nout: usize,
    nin: usize,
}

impl MatrixParam {
    fn new(nout: usize, nin: usize, std: f64) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<Vec<f64>> = (0..nout)
            .map(|_| (0..nin).map(|_| rng.gen::<f64>() * std * 2.0 - std).collect())
            .collect();
        let grad = vec![vec![0.0; nin]; nout];
        MatrixParam { data, grad, nout, nin }
    }

    fn zeros(nout: usize, nin: usize) -> Self {
        MatrixParam {
            data: vec![vec![0.0; nin]; nout],
            grad: vec![vec![0.0; nin]; nout],
            nout, nin,
        }
    }

    fn row(&self, i: usize) -> &[f64] { &self.data[i] }

    fn matvec_raw(&self, x: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.nout];
        #[cfg(feature = "blas")]
        {
            if self.nout * self.nin >= 256 {
                let mut buf = vec![0.0f64; self.nout * self.nin];
                for i in 0..self.nout {
                    buf[i * self.nin..(i + 1) * self.nin].copy_from_slice(&self.data[i][..self.nin]);
                }
                unsafe { blas_ffi::dgemv(self.nout, self.nin, &buf, x, &mut out); }
                return out;
            }
        }
        for i in 0..self.nout {
            let mut s = 0.0;
            for j in 0..self.nin { s += self.data[i][j] * x[j]; }
            out[i] = s;
        }
        out
    }

    fn grow_rows(&mut self, new_nout: usize, std: f64) {
        let mut rng = rand::thread_rng();
        while self.data.len() < new_nout {
            self.data.push((0..self.nin).map(|_| rng.gen::<f64>() * std * 2.0 - std).collect());
            self.grad.push(vec![0.0; self.nin]);
        }
        self.nout = new_nout;
    }

    fn grow_cols(&mut self, new_nin: usize, std: f64) {
        let mut rng = rand::thread_rng();
        for i in 0..self.nout {
            while self.data[i].len() < new_nin {
                self.data[i].push(rng.gen::<f64>() * std * 2.0 - std);
                self.grad[i].push(0.0);
            }
        }
        self.nin = new_nin;
    }

    fn grow(&mut self, new_nout: usize, new_nin: usize, std: f64) {
        self.grow_rows(new_nout, std);
        self.grow_cols(new_nin, std);
    }

    fn zero_grad(&mut self) {
        for row in self.grad.iter_mut() {
            for v in row.iter_mut() { *v = 0.0; }
        }
    }

    fn clip_grad(&mut self, clip: f64) {
        let mut sq_sum = 0.0;
        for row in &self.grad {
            for &v in row { sq_sum += v * v; }
        }
        let norm = sq_sum.sqrt();
        if norm > clip {
            let scale = clip / norm;
            for row in self.grad.iter_mut() {
                for v in row.iter_mut() { *v *= scale; }
            }
        }
    }

    fn ensure_grad(&mut self) {
        if self.grad.len() != self.nout {
            self.grad = vec![vec![0.0; self.nin]; self.nout];
        }
    }
}

// ============================================================
// 3) BPE TOKENIZER
// ============================================================

#[derive(Clone, Serialize, Deserialize)]
struct MergePair { a: String, b: String }

#[derive(Clone, Serialize, Deserialize)]
struct EvolvingTokenizer {
    tokens: Vec<String>,
    #[serde(default)]
    stoi: HashMap<String, usize>,
    #[serde(default)]
    vocab_size: usize,
    #[serde(default = "default_bos")]
    bos: String,
    #[serde(default = "default_eos")]
    eos: String,
    #[serde(default = "default_pad")]
    pad: String,
    #[serde(default)]
    bos_id: usize,
    #[serde(default)]
    eos_id: usize,
    #[serde(default)]
    pad_id: usize,
    bpe_enabled: bool,
    merges: Vec<MergePair>,
    trained_chars: usize,
}

fn default_bos() -> String { "<BOS>".to_string() }
fn default_eos() -> String { "<EOS>".to_string() }
fn default_pad() -> String { "<PAD>".to_string() }

// Consciousness config serde defaults (for backward-compatible checkpoint loading)
fn default_dissonance_ema_alpha() -> f64 { 0.3 }
fn default_dissonance_spike_k() -> f64 { 0.8 }
fn default_dissonance_drop_k() -> f64 { 1.2 }
fn default_dissonance_spike_threshold() -> f64 { 1.5 }
fn default_dissonance_drop_threshold() -> f64 { 0.5 }
fn default_anti_field_prob() -> f64 { 0.05 }
fn default_anti_field_min_step() -> usize { 8 }
fn default_conscience_window() -> usize { 8 }
fn default_conscience_decay() -> f64 { 0.95 }
fn default_conscience_recovery() -> f64 { 1.005 }
fn default_conscience_floor() -> f64 { 0.3 }
fn default_freq_penalty() -> f64 { 0.1 }
fn default_presence_penalty() -> f64 { 0.1 }
fn default_cooccur_window_size() -> usize { 5 }
fn default_user_boost_strength() -> f64 { 0.3 }
fn default_user_boost_decay() -> f64 { 0.7 }

impl EvolvingTokenizer {
    /// Rebuild stoi and special token IDs from tokens vec (for cross-impl compat)
    fn rebuild_indices(&mut self) {
        self.stoi.clear();
        for (i, t) in self.tokens.iter().enumerate() {
            self.stoi.insert(t.clone(), i);
        }
        self.vocab_size = self.tokens.len();
        self.bos_id = *self.stoi.get("<BOS>").unwrap_or(&(self.vocab_size.saturating_sub(3)));
        self.eos_id = *self.stoi.get("<EOS>").unwrap_or(&(self.vocab_size.saturating_sub(2)));
        self.pad_id = *self.stoi.get("<PAD>").unwrap_or(&(self.vocab_size.saturating_sub(1)));
        self.bos = "<BOS>".to_string();
        self.eos = "<EOS>".to_string();
        self.pad = "<PAD>".to_string();
    }
}

fn unicode_category(c: char) -> u8 {
    if c.is_alphabetic() || c == '\'' { 0 } // L
    else if c.is_numeric() { 1 } // N
    else if c.is_whitespace() { 2 } // Z
    else { 3 } // P
}

fn unicode_segment(text: &str) -> Vec<Vec<u8>> {
    let mut segments = Vec::new();
    let mut current = Vec::new();
    let mut last_cat: Option<u8> = None;
    for c in text.chars() {
        let cat = unicode_category(c);
        if last_cat.is_some() && last_cat != Some(cat) {
            if !current.is_empty() { segments.push(current.clone()); current.clear(); }
        }
        let mut buf = [0u8; 4];
        let s = c.encode_utf8(&mut buf);
        current.extend_from_slice(s.as_bytes());
        last_cat = Some(cat);
    }
    if !current.is_empty() { segments.push(current); }
    segments
}

fn byte_to_token(b: u8) -> String { format!("0x{:02x}", b) }

fn token_to_bytes(tok: &str) -> Vec<u8> {
    if tok.starts_with("0x") {
        tok.split('+').filter_map(|part| {
            u8::from_str_radix(part.trim_start_matches("0x"), 16).ok()
        }).collect()
    } else {
        vec![]
    }
}

impl EvolvingTokenizer {
    fn new(_docs: &[String]) -> Self {
        let mut tokens: Vec<String> = (0..=255u8).map(|b| byte_to_token(b)).collect();
        let bos = "<BOS>".to_string(); let eos = "<EOS>".to_string(); let pad = "<PAD>".to_string();
        tokens.push(bos.clone()); tokens.push(eos.clone()); tokens.push(pad.clone());
        let mut stoi = HashMap::new();
        for (i, t) in tokens.iter().enumerate() { stoi.insert(t.clone(), i); }
        let vocab_size = tokens.len();
        EvolvingTokenizer {
            bos_id: vocab_size - 3, eos_id: vocab_size - 2, pad_id: vocab_size - 1,
            tokens, stoi, vocab_size,
            bos, eos, pad,
            bpe_enabled: false, merges: Vec::new(), trained_chars: 0,
        }
    }

    fn add_token(&mut self, tok: String) -> usize {
        if let Some(&id) = self.stoi.get(&tok) { return id; }
        let id = self.tokens.len();
        self.stoi.insert(tok.clone(), id);
        self.tokens.push(tok);
        self.vocab_size = self.tokens.len();
        id
    }

    fn train_bpe(&mut self, docs: &[String], num_merges: usize) {
        let all_text: String = docs.join(" ");
        let segments = unicode_segment(&all_text);
        // Build word list: each word = list of byte tokens
        let mut words: Vec<Vec<String>> = segments.iter().map(|seg| {
            seg.iter().map(|&b| byte_to_token(b)).collect()
        }).collect();
        // Merge loop
        for _ in 0..num_merges {
            let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();
            for word in &words {
                for w in word.windows(2) {
                    *pair_counts.entry((w[0].clone(), w[1].clone())).or_insert(0) += 1;
                }
            }
            if pair_counts.is_empty() { break; }
            let best = pair_counts.into_iter().max_by_key(|&(_, c)| c).unwrap().0;
            let new_tok = format!("{}+{}", best.0, best.1);
            self.add_token(new_tok.clone());
            self.merges.push(MergePair { a: best.0.clone(), b: best.1.clone() });
            // Apply merge to all words
            for word in words.iter_mut() {
                let mut i = 0;
                while i + 1 < word.len() {
                    if word[i] == best.0 && word[i+1] == best.1 {
                        word[i] = new_tok.clone();
                        word.remove(i + 1);
                    } else {
                        i += 1;
                    }
                }
            }
        }
        self.bpe_enabled = true;
    }

    fn apply_bpe(&self, tokens: &[String]) -> Vec<String> {
        if self.merges.is_empty() { return tokens.to_vec(); }
        let rank: HashMap<(String,String), usize> = self.merges.iter().enumerate()
            .map(|(i, m)| ((m.a.clone(), m.b.clone()), i)).collect();
        let mut syms = tokens.to_vec();
        loop {
            if syms.len() < 2 { break; }
            let mut best_rank = usize::MAX;
            let mut best_i = 0;
            for i in 0..syms.len()-1 {
                if let Some(&r) = rank.get(&(syms[i].clone(), syms[i+1].clone())) {
                    if r < best_rank { best_rank = r; best_i = i; }
                }
            }
            if best_rank == usize::MAX { break; }
            let merged = format!("{}+{}", syms[best_i], syms[best_i+1]);
            syms[best_i] = merged;
            syms.remove(best_i + 1);
        }
        syms
    }

    fn maybe_enable_bpe(&mut self, docs: &[String], cfg: &Config) -> bool {
        let total_chars: usize = docs.iter().map(|d| d.len()).sum();
        if !self.bpe_enabled && total_chars >= cfg.enable_bpe_after_chars {
            self.train_bpe(docs, cfg.bpe_num_merges);
            self.trained_chars = total_chars;
            return true;
        }
        if self.bpe_enabled && total_chars >= self.trained_chars + cfg.bpe_retrain_every_chars {
            self.train_bpe(docs, cfg.bpe_num_merges);
            self.trained_chars = total_chars;
            return true;
        }
        false
    }

    fn encode(&self, s: &str) -> Vec<usize> {
        let trimmed = s.trim();
        if trimmed.is_empty() { return vec![self.bos_id, self.eos_id]; }
        let segments = unicode_segment(trimmed);
        let mut ids = vec![self.bos_id];
        for seg in &segments {
            let byte_toks: Vec<String> = seg.iter().map(|&b| byte_to_token(b)).collect();
            let toks = if self.bpe_enabled { self.apply_bpe(&byte_toks) } else { byte_toks };
            for t in &toks {
                if let Some(&id) = self.stoi.get(t) { ids.push(id); }
            }
        }
        ids.push(self.eos_id);
        ids
    }

    fn decode(&self, ids: &[usize]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if id == self.bos_id || id == self.pad_id { continue; }
            if id == self.eos_id { break; }
            if id < self.tokens.len() {
                bytes.extend(token_to_bytes(&self.tokens[id]));
            }
        }
        String::from_utf8_lossy(&bytes).trim().to_string()
    }
}

// ============================================================
// 4) DELTA ADAPTERS (LoRA)
// ============================================================

#[derive(Clone, Serialize, Deserialize)]
struct DeltaAdapter {
    a: MatrixParam,  // nout x rank
    b: MatrixParam,  // rank x nin
}

impl DeltaAdapter {
    fn new(nout: usize, nin: usize, rank: usize, std: f64) -> Self {
        DeltaAdapter { a: MatrixParam::new(nout, rank, std), b: MatrixParam::new(rank, nin, std) }
    }

    fn apply_raw(&self, x: &[f64]) -> Vec<f64> {
        let bx = self.b.matvec_raw(x);
        self.a.matvec_raw(&bx)
    }

    fn grow_dims(&mut self, new_nout: usize, new_nin: usize) {
        self.a.grow_rows(new_nout, 0.01);
        self.b.grow_cols(new_nin, 0.01);
    }
}

type DeltaModule = HashMap<String, DeltaAdapter>;

// ============================================================
// 5) GPT MODEL
// ============================================================

struct AdamState {
    m: Vec<Vec<f64>>,
    v: Vec<Vec<f64>>,
    t: usize,
}

impl AdamState {
    fn new(nout: usize, nin: usize) -> Self {
        AdamState { m: vec![vec![0.0; nin]; nout], v: vec![vec![0.0; nin]; nout], t: 0 }
    }
}

struct GPT {
    cfg: Config,
    tok: EvolvingTokenizer,
    n_layer: usize, n_embd: usize, n_head: usize, head_dim: usize, block_size: usize,
    base: HashMap<String, MatrixParam>,
    deltas: Vec<DeltaModule>,
    active_alpha: Vec<f64>,
    adam_base: HashMap<String, AdamState>,
    adam_delta: Vec<HashMap<String, (AdamState, AdamState)>>, // per delta: name -> (adam_A, adam_B)
    init_embed_snapshot: Vec<Vec<f64>>,
    residual_alpha: f64,
    global_step: usize,
    syntropy_temp_offset: f64,
    growth_freeze_remaining: i32,
    last_warmup_stage: i32,       // per-stage warmup: last stage that completed warmup (-1 = none)
    growth_step_offset: usize,    // global_step at last growth event (for LR warmup reset)
    head_types: Vec<String>,

    // consciousness state
    delta_alpha_scale: f64,               // conscience: multiplier on all delta contributions (1.0 = normal)
    generation_entropy_history: Vec<f64>, // conscience: rolling window of per-generation mean entropy
    last_surprise: f64,                   // self-prediction error on last prompt
    surprise_baseline: f64,               // EMA of surprise over time
    last_gen_entropy: f64,                // mean entropy of last generation (for conscience)
}

impl GPT {
    fn new(tok: EvolvingTokenizer, cfg: &Config) -> Self {
        let ne = cfg.n_embd;
        let nl = cfg.n_layer;
        let nh = cfg.n_head;
        let hd = ne / nh;
        let bs = cfg.block_size;
        let vs = tok.vocab_size;
        let std_e = 0.02;
        let std_l = 0.02 / (nl as f64).sqrt().max(1.0);

        let mut base = HashMap::new();
        base.insert("wte".into(), MatrixParam::new(vs, ne, std_e));
        base.insert("wpe".into(), MatrixParam::new(bs, ne, std_e));
        if !cfg.tie_embeddings {
            base.insert("lm_head".into(), MatrixParam::new(vs, ne, std_l));
        }
        let htypes = head_types_for(nh);
        for li in 0..nl {
            let p = format!("l{}", li);
            base.insert(format!("{}.wq", p), MatrixParam::new(ne, ne, std_l));
            base.insert(format!("{}.wk", p), MatrixParam::new(ne, ne, std_l));
            base.insert(format!("{}.wv", p), MatrixParam::new(ne, ne, std_l));
            base.insert(format!("{}.wo", p), MatrixParam::new(ne, ne, std_l));
            base.insert(format!("{}.fc_g", p), MatrixParam::new(4*ne, ne, std_l));
            base.insert(format!("{}.fc_v", p), MatrixParam::new(4*ne, ne, std_l));
            base.insert(format!("{}.fc2", p), MatrixParam::new(ne, 4*ne, std_l));
            for hi in 0..nh {
                if htypes[hi] == "hybrid" || htypes[hi] == "rrpram" {
                    base.insert(format!("{}.h{}.w_pattern", p, hi), MatrixParam::new(bs, hd, std_l));
                    let mut alpha_mat = MatrixParam::zeros(1, 1);
                    alpha_mat.data[0][0] = cfg.hybrid_alpha_init;
                    base.insert(format!("{}.h{}.alpha", p, hi), alpha_mat);
                }
            }
        }
        let snapshot: Vec<Vec<f64>> = base["wte"].data.clone();

        let mut gpt = GPT {
            cfg: cfg.clone(), tok, n_layer: nl, n_embd: ne, n_head: nh,
            head_dim: hd, block_size: bs, base,
            deltas: Vec::new(), active_alpha: Vec::new(),
            adam_base: HashMap::new(),
            adam_delta: Vec::new(),
            init_embed_snapshot: snapshot,
            residual_alpha: 1.0 / (nl as f64).sqrt().max(1.0),
            global_step: 0, syntropy_temp_offset: 0.0,
            growth_freeze_remaining: 0, last_warmup_stage: -1, growth_step_offset: 0,
            head_types: htypes,
            delta_alpha_scale: 1.0,
            generation_entropy_history: Vec::new(),
            last_surprise: 0.0, surprise_baseline: 0.0, last_gen_entropy: 0.0,
        };
        gpt.add_delta_module(1.0);
        gpt
    }

    fn lm_head_name(&self) -> String {
        if self.cfg.tie_embeddings { "wte".into() } else { "lm_head".into() }
    }

    fn get_mat_data(&self, r: &MatRef) -> &Vec<Vec<f64>> {
        match r {
            MatRef::Base(name) => &self.base[name].data,
            MatRef::DeltaA(di, name) => &self.deltas[*di][name].a.data,
            MatRef::DeltaB(di, name) => &self.deltas[*di][name].b.data,
        }
    }

    fn get_mat_grad_mut(&mut self, r: &MatRef) -> &mut Vec<Vec<f64>> {
        match r {
            MatRef::Base(name) => &mut self.base.get_mut(name).unwrap().grad,
            MatRef::DeltaA(di, name) => &mut self.deltas[*di].get_mut(name).unwrap().a.grad,
            MatRef::DeltaB(di, name) => &mut self.deltas[*di].get_mut(name).unwrap().b.grad,
        }
    }

    fn add_delta_module(&mut self, alpha: f64) {
        if self.deltas.len() >= self.cfg.max_delta_modules { return; }
        let ne = self.n_embd;
        let r = self.cfg.delta_rank;
        let std = 0.01;
        let mut dm = DeltaModule::new();
        for li in 0..self.n_layer {
            let p = format!("l{}", li);
            for suffix in &["wq", "wk", "wv", "wo"] {
                dm.insert(format!("{}.{}", p, suffix), DeltaAdapter::new(ne, ne, r, std));
            }
            dm.insert(format!("{}.fc_g", p), DeltaAdapter::new(4*ne, ne, r, std));
            dm.insert(format!("{}.fc_v", p), DeltaAdapter::new(4*ne, ne, r, std));
            dm.insert(format!("{}.fc2", p), DeltaAdapter::new(ne, 4*ne, r, std));
            for (hi, htype) in self.head_types.iter().enumerate() {
                if htype == "rrpram" || htype == "hybrid" {
                    dm.insert(format!("{}.h{}.w_pattern", p, hi),
                        DeltaAdapter::new(self.block_size, self.head_dim, r, std));
                }
            }
        }
        dm.insert("lm_head".into(), DeltaAdapter::new(self.tok.vocab_size, ne, r, std));
        self.deltas.push(dm);
        self.active_alpha.push(alpha);
        self.adam_delta.push(HashMap::new());
    }

    fn apply_with_deltas_raw(&self, name: &str, x: &[f64]) -> Vec<f64> {
        let base_name = if name == "lm_head" { self.lm_head_name() } else { name.to_string() };
        let mut out = self.base[&base_name].matvec_raw(x);
        for (di, dm) in self.deltas.iter().enumerate() {
            if let Some(adapter) = dm.get(name) {
                let delta_out = adapter.apply_raw(x);
                // Consciousness: conscience scales delta influence (Feature 5)
                let effective_alpha = self.active_alpha[di] * self.delta_alpha_scale;
                for j in 0..out.len() { out[j] += effective_alpha * delta_out[j]; }
            }
        }
        out
    }

    fn apply_with_deltas_tape(&self, tape: &mut Tape, name: &str, x: NodeId) -> NodeId {
        let base_name = if name == "lm_head" { self.lm_head_name() } else { name.to_string() };
        let mut out = tape.matvec(MatRef::Base(base_name), &self.base[&if name == "lm_head" { self.lm_head_name() } else { name.to_string() }].data, x);
        for (di, dm) in self.deltas.iter().enumerate() {
            if let Some(adapter) = dm.get(name) {
                let bx = tape.matvec(MatRef::DeltaB(di, name.to_string()), &adapter.b.data, x);
                let abx = tape.matvec(MatRef::DeltaA(di, name.to_string()), &adapter.a.data, bx);
                let scaled = tape.scale(abx, self.active_alpha[di] * self.delta_alpha_scale);
                out = tape.add(out, scaled);
            }
        }
        out
    }

    // Forward step for inference (no KV cache version — full sequence)
    fn forward_infer(&self, token_ids: &[usize]) -> Vec<Vec<f64>> {
        let seq_len = token_ids.len();
        let ne = self.n_embd;
        let nh = self.n_head;
        let hd = self.head_dim;
        let ra = self.residual_alpha;

        // Embed all tokens
        let mut xs: Vec<Vec<f64>> = token_ids.iter().enumerate().map(|(pos, &tid)| {
            let wte = self.base["wte"].row(tid);
            let wpe = self.base["wpe"].row(pos % self.block_size);
            let mut x = vec![0.0; ne];
            for j in 0..ne { x[j] = wte[j] + wpe[j]; }
            x
        }).collect();

        for li in 0..self.n_layer {
            let p = format!("l{}", li);
            // Pre-attention RMSNorm + QKV
            let mut qs = Vec::with_capacity(seq_len);
            let mut ks = Vec::with_capacity(seq_len);
            let mut vs = Vec::with_capacity(seq_len);
            let x_residuals: Vec<Vec<f64>> = xs.clone();

            for t in 0..seq_len {
                let xn = rmsnorm_raw(&xs[t]);
                qs.push(self.apply_with_deltas_raw(&format!("{}.wq", p), &xn));
                ks.push(self.apply_with_deltas_raw(&format!("{}.wk", p), &xn));
                vs.push(self.apply_with_deltas_raw(&format!("{}.wv", p), &xn));
            }

            // Attention per head (content, rrpram, or hybrid)
            let mut attn_outs: Vec<Vec<f64>> = vec![vec![0.0; ne]; seq_len];
            for h in 0..nh {
                let hs = h * hd;
                let he = hs + hd;
                let htype = if h < self.head_types.len() {
                    self.head_types[h].as_str()
                } else { "content" };

                for t in 0..seq_len {
                    let causal_len = t + 1;

                    // Content attention logits (QK^T / sqrt(d) + RoPE)
                    let content_logits: Option<Vec<f64>> = if htype == "content" || htype == "hybrid" {
                        let qh = rope_raw(&qs[t][hs..he], t, hd);
                        let mut logits = Vec::with_capacity(causal_len);
                        let inv_sqrt = 1.0 / (hd as f64).sqrt();
                        for s in 0..causal_len {
                            let kh = rope_raw(&ks[s][hs..he], s, hd);
                            let mut dot = 0.0;
                            for j in 0..hd { dot += qh[j] * kh[j]; }
                            logits.push(dot * inv_sqrt);
                        }
                        Some(logits)
                    } else { None };

                    // RRPRAM attention logits (W_pattern @ xh -> positional scores)
                    let rrpram_logits: Option<Vec<f64>> = if htype == "rrpram" || htype == "hybrid" {
                        let xh: Vec<f64> = xs[t][hs..he].to_vec();
                        let pkey = format!("{}.h{}.w_pattern", p, h);
                        let pattern_full = self.apply_with_deltas_raw(&pkey, &xh);
                        let p_len = pattern_full.len();
                        let mut logits = Vec::with_capacity(causal_len);
                        for s in 0..causal_len {
                            let idx = if s < p_len { s } else { p_len - 1 };
                            logits.push(pattern_full[idx]);
                        }
                        Some(logits)
                    } else { None };

                    // Dispatch by head type
                    let final_logits = match htype {
                        "content" => content_logits.unwrap(),
                        "rrpram" => rrpram_logits.unwrap(),
                        _ => { // hybrid: sigmoid(alpha) blend
                            let akey = format!("{}.h{}.alpha", p, h);
                            let alpha_raw = if let Some(m) = self.base.get(&akey) {
                                m.data[0][0]
                            } else { 0.5 };
                            let a = 1.0 / (1.0 + (-alpha_raw).exp()); // sigmoid
                            let cl = content_logits.unwrap();
                            let rl = rrpram_logits.unwrap();
                            cl.iter().zip(rl.iter())
                                .map(|(&c, &r)| (1.0 - a) * c + a * r)
                                .collect()
                        }
                    };

                    let probs = softmax_raw(&final_logits);
                    let mut head_out = vec![0.0; hd];
                    for s in 0..causal_len {
                        for j in 0..hd { head_out[j] += probs[s] * vs[s][hs + j]; }
                    }
                    for j in 0..hd { attn_outs[t][hs + j] = head_out[j]; }
                }
            }

            // Apply wo + residual
            for t in 0..seq_len {
                let wo_out = self.apply_with_deltas_raw(&format!("{}.wo", p), &attn_outs[t]);
                xs[t] = x_residuals[t].clone();
                for j in 0..ne { xs[t][j] += ra * wo_out[j]; }
            }

            // MLP: SwiGLU
            let x_residuals2: Vec<Vec<f64>> = xs.clone();
            for t in 0..seq_len {
                let xn = rmsnorm_raw(&xs[t]);
                let gate = self.apply_with_deltas_raw(&format!("{}.fc_g", p), &xn);
                let val = self.apply_with_deltas_raw(&format!("{}.fc_v", p), &xn);
                let mut gated = vec![0.0; gate.len()];
                for j in 0..gate.len() {
                    let sig = 1.0 / (1.0 + (-gate[j]).exp());
                    gated[j] = (gate[j] * sig) * val[j]; // SiLU(gate) * val
                }
                let mlp_out = self.apply_with_deltas_raw(&format!("{}.fc2", p), &gated);
                xs[t] = x_residuals2[t].clone();
                for j in 0..ne { xs[t][j] += ra * mlp_out[j]; }
            }
        }

        // Final RMSNorm + LM head
        let mut all_logits = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let xn = rmsnorm_raw(&xs[t]);
            all_logits.push(self.apply_with_deltas_raw("lm_head", &xn));
        }
        all_logits
    }

    // Forward step for training (tape-based, single sequence)
    fn loss_on_sequence(&self, tape: &mut Tape, ids: &[usize]) -> NodeId {
        let len = ids.len() - 1;
        let ne = self.n_embd;
        let nh = self.n_head;
        let hd = self.head_dim;
        let ra = self.residual_alpha;

        // Embed
        let mut x_nodes: Vec<NodeId> = ids[..len].iter().enumerate().map(|(pos, &tid)| {
            let te = tape.embed_lookup(MatRef::Base("wte".into()), tid, self.base["wte"].row(tid));
            let pe = tape.embed_lookup(MatRef::Base("wpe".into()), pos % self.block_size,
                                        self.base["wpe"].row(pos % self.block_size));
            tape.add(te, pe)
        }).collect();

        for li in 0..self.n_layer {
            let p = format!("l{}", li);
            let x_res: Vec<NodeId> = x_nodes.clone();

            // RMSNorm + QKV for all positions
            let mut qs = Vec::with_capacity(len);
            let mut ks = Vec::with_capacity(len);
            let mut vs_n = Vec::with_capacity(len);
            for t in 0..len {
                let xn = tape.rmsnorm(x_nodes[t]);
                qs.push(self.apply_with_deltas_tape(tape, &format!("{}.wq", p), xn));
                ks.push(self.apply_with_deltas_tape(tape, &format!("{}.wk", p), xn));
                vs_n.push(self.apply_with_deltas_tape(tape, &format!("{}.wv", p), xn));
            }

            // Attention per head (content, rrpram, or hybrid)
            let mut head_concat = vec![Vec::new(); len];
            for h in 0..nh {
                let hs = h * hd;
                let he = hs + hd;
                let htype = if h < self.head_types.len() {
                    self.head_types[h].as_str()
                } else { "content" };

                for t in 0..len {
                    let causal_len = t + 1;
                    let mut val_nodes = Vec::with_capacity(causal_len);
                    for s in 0..causal_len {
                        val_nodes.push(tape.slice(vs_n[s], hs, he));
                    }

                    // Content attention logits
                    let content_logits: Option<Vec<NodeId>> = if htype == "content" || htype == "hybrid" {
                        let qh = tape.slice(qs[t], hs, he);
                        let qh_r = tape.rope(qh, t, hd);
                        let inv_sqrt = 1.0 / (hd as f64).sqrt();
                        let mut logits = Vec::with_capacity(causal_len);
                        for s in 0..causal_len {
                            let kh = tape.slice(ks[s], hs, he);
                            let kh_r = tape.rope(kh, s, hd);
                            let dot_val = tape.dot(qh_r, kh_r);
                            logits.push(tape.scalar_mulf(dot_val, inv_sqrt));
                        }
                        Some(logits)
                    } else { None };

                    // RRPRAM attention logits
                    let rrpram_logits: Option<Vec<NodeId>> = if htype == "rrpram" || htype == "hybrid" {
                        let xh = tape.slice(x_nodes[t], hs, he);
                        let pkey = format!("{}.h{}.w_pattern", p, h);
                        let pattern_full = self.apply_with_deltas_tape(tape, &pkey, xh);
                        let p_len = self.base.get(&pkey).map_or(self.block_size, |m| m.data.len());
                        let mut logits = Vec::with_capacity(causal_len);
                        for s in 0..causal_len {
                            let idx = if s < p_len { s } else { p_len - 1 };
                            logits.push(tape.element(pattern_full, idx));
                        }
                        Some(logits)
                    } else { None };

                    // Dispatch by head type
                    let final_logits = match htype {
                        "content" => content_logits.unwrap(),
                        "rrpram" => rrpram_logits.unwrap(),
                        _ => { // hybrid: sigmoid(alpha) blend
                            let akey = format!("{}.h{}.alpha", p, h);
                            let alpha_row = tape.embed_lookup(
                                MatRef::Base(akey.clone()), 0, &self.base[&akey].data[0]);
                            let alpha_scalar = tape.element(alpha_row, 0);
                            let a = tape.sigmoid(alpha_scalar);
                            let neg_a = tape.scalar_mulf(a, -1.0);
                            let one_minus_a = tape.scalar_addf(neg_a, 1.0);
                            let cl = content_logits.unwrap();
                            let rl = rrpram_logits.unwrap();
                            let mut blended = Vec::with_capacity(cl.len());
                            for i in 0..cl.len() {
                                let c_scaled = tape.scalar_mul(cl[i], one_minus_a);
                                let r_scaled = tape.scalar_mul(rl[i], a);
                                blended.push(tape.scalar_add(c_scaled, r_scaled));
                            }
                            blended
                        }
                    };

                    let attn_out = tape.softmax_attn(&final_logits, &val_nodes);
                    head_concat[t].push(attn_out);
                }
            }

            // Concat heads + wo + residual
            for t in 0..len {
                let cat = tape.concat(&head_concat[t]);
                let wo = self.apply_with_deltas_tape(tape, &format!("{}.wo", p), cat);
                let wo_scaled = tape.scale(wo, ra);
                x_nodes[t] = tape.add(x_res[t], wo_scaled);
            }

            // MLP
            let x_res2: Vec<NodeId> = x_nodes.clone();
            for t in 0..len {
                let xn = tape.rmsnorm(x_nodes[t]);
                let gate = self.apply_with_deltas_tape(tape, &format!("{}.fc_g", p), xn);
                let gate_act = tape.silu(gate);
                let val = self.apply_with_deltas_tape(tape, &format!("{}.fc_v", p), xn);
                let gated = tape.mul(gate_act, val);
                let mlp = self.apply_with_deltas_tape(tape, &format!("{}.fc2", p), gated);
                let mlp_scaled = tape.scale(mlp, ra);
                x_nodes[t] = tape.add(x_res2[t], mlp_scaled);
            }
        }

        // Loss
        let mut total_loss = tape.scalar_leaf(0.0);
        for t in 0..len {
            let xn = tape.rmsnorm(x_nodes[t]);
            let logits = self.apply_with_deltas_tape(tape, "lm_head", xn);
            let loss = tape.cross_entropy(logits, ids[t + 1]);
            total_loss = tape.scalar_add(total_loss, loss);
        }
        tape.scalar_mulf(total_loss, 1.0 / len as f64)
    }

    fn maybe_expand_vocab(&mut self, new_vocab: usize) {
        let old = self.base["wte"].nout;
        if new_vocab <= old { return; }
        let std = 0.02;
        self.base.get_mut("wte").unwrap().grow_rows(new_vocab, std);
        if !self.cfg.tie_embeddings {
            self.base.get_mut("lm_head").unwrap().grow_rows(new_vocab, std);
        }
        for dm in &mut self.deltas {
            if let Some(a) = dm.get_mut("lm_head") {
                a.a.grow_rows(new_vocab, 0.01);
            }
        }
        // Extend snapshot
        while self.init_embed_snapshot.len() < new_vocab {
            self.init_embed_snapshot.push(vec![0.0; self.n_embd]);
        }
    }

    /// Return index of current stage based on model dimensions, or -1 if unknown.
    fn current_growth_stage(&self) -> i32 {
        for (i, stage) in self.cfg.growth_stages.iter().enumerate() {
            if self.n_embd == stage[1] && self.n_layer == stage[2] && self.n_head == stage[3] {
                return i as i32;
            }
        }
        -1 // legacy checkpoint or unknown dims
    }

    /// Return the target stage index based on corpus size.
    fn target_growth_stage(&self, corpus_chars: usize) -> usize {
        let mut target = 0;
        for (i, stage) in self.cfg.growth_stages.iter().enumerate() {
            if corpus_chars >= stage[0] {
                target = i;
            }
        }
        target
    }

    /// Check if growth is needed and execute it. Returns true if grew.
    fn maybe_grow_architecture(&mut self, corpus_chars: usize) -> bool {
        let current = self.current_growth_stage();
        if current < 0 { return false; } // legacy checkpoint, skip growth
        if self.growth_freeze_remaining > 0 { return false; } // still stabilizing
        let target_full = self.target_growth_stage(corpus_chars);
        if target_full <= current as usize { return false; }
        // Grow only one stage at a time — prevent catastrophic multi-stage jumps
        let target = (current + 1) as usize;

        let new_embd = self.cfg.growth_stages[target][1];
        let new_layer = self.cfg.growth_stages[target][2];
        let new_head = self.cfg.growth_stages[target][3];
        let old_embd = self.n_embd;
        let old_layer = self.n_layer;
        let old_head = self.n_head;
        let new_head_dim = new_embd / new_head;
        let std_g = 0.001;

        eprintln!("[growth] ONTOGENESIS: stage {} -> {}", current, target);
        eprintln!("  embd: {} -> {}, layer: {} -> {}, head: {} -> {}",
                  old_embd, new_embd, old_layer, new_layer, old_head, new_head);

        // 1. Grow embedding matrices (columns = embd dimension)
        self.base.get_mut("wte").unwrap().grow_cols(new_embd, std_g);
        self.base.get_mut("wpe").unwrap().grow_cols(new_embd, std_g);
        if !self.cfg.tie_embeddings {
            if let Some(lm) = self.base.get_mut("lm_head") {
                lm.grow_cols(new_embd, std_g);
            }
        }

        // Update head types for new head count
        let new_htypes = head_types_for(new_head);

        // 2. Grow existing layer matrices
        for li in 0..old_layer {
            let p = format!("l{}", li);
            for suffix in &["wq", "wk", "wv", "wo"] {
                let key = format!("{}.{}", p, suffix);
                self.base.get_mut(&key).unwrap().grow(new_embd, new_embd, std_g);
            }
            self.base.get_mut(&format!("{}.fc_g", p)).unwrap().grow(4 * new_embd, new_embd, std_g);
            self.base.get_mut(&format!("{}.fc_v", p)).unwrap().grow(4 * new_embd, new_embd, std_g);
            self.base.get_mut(&format!("{}.fc2", p)).unwrap().grow(new_embd, 4 * new_embd, std_g);
            // Grow existing head pattern matrices
            for h in 0..old_head {
                let pkey = format!("{}.h{}.w_pattern", p, h);
                if self.base.contains_key(&pkey) {
                    self.base.get_mut(&pkey).unwrap().grow_cols(new_head_dim, std_g);
                }
            }
            // Add new heads for existing layers
            for h in old_head..new_head {
                let htype = if h < new_htypes.len() { new_htypes[h].as_str() } else { "content" };
                if htype == "rrpram" || htype == "hybrid" {
                    self.base.insert(format!("{}.h{}.w_pattern", p, h),
                        MatrixParam::new(self.block_size, new_head_dim, std_g));
                    let mut alpha_mat = MatrixParam::zeros(1, 1);
                    alpha_mat.data[0][0] = self.cfg.hybrid_alpha_init;
                    self.base.insert(format!("{}.h{}.alpha", p, h), alpha_mat);
                }
            }
        }

        // 3. Add new layers
        let std_l = 0.02 / (new_layer as f64).sqrt().max(1.0);
        for li in old_layer..new_layer {
            let p = format!("l{}", li);
            self.base.insert(format!("{}.wq", p), MatrixParam::new(new_embd, new_embd, std_l));
            self.base.insert(format!("{}.wk", p), MatrixParam::new(new_embd, new_embd, std_l));
            self.base.insert(format!("{}.wv", p), MatrixParam::new(new_embd, new_embd, std_l));
            self.base.insert(format!("{}.wo", p), MatrixParam::new(new_embd, new_embd, std_l));
            self.base.insert(format!("{}.fc_g", p), MatrixParam::new(4 * new_embd, new_embd, std_l));
            self.base.insert(format!("{}.fc_v", p), MatrixParam::new(4 * new_embd, new_embd, std_l));
            self.base.insert(format!("{}.fc2", p), MatrixParam::new(new_embd, 4 * new_embd, std_l));
            for (hi, htype) in new_htypes.iter().enumerate() {
                if htype == "rrpram" || htype == "hybrid" {
                    self.base.insert(format!("{}.h{}.w_pattern", p, hi),
                        MatrixParam::new(self.block_size, new_head_dim, std_l));
                    let mut alpha_mat = MatrixParam::zeros(1, 1);
                    alpha_mat.data[0][0] = self.cfg.hybrid_alpha_init;
                    self.base.insert(format!("{}.h{}.alpha", p, hi), alpha_mat);
                }
            }
        }

        // 4. Grow deltas
        for dm in &mut self.deltas {
            for li in 0..old_layer {
                let p = format!("l{}", li);
                for suffix in &["wq", "wk", "wv", "wo"] {
                    let key = format!("{}.{}", p, suffix);
                    if let Some(adapter) = dm.get_mut(&key) {
                        adapter.grow_dims(new_embd, new_embd);
                    }
                }
                let key_fg = format!("{}.fc_g", p);
                if let Some(a) = dm.get_mut(&key_fg) { a.grow_dims(4 * new_embd, new_embd); }
                let key_fv = format!("{}.fc_v", p);
                if let Some(a) = dm.get_mut(&key_fv) { a.grow_dims(4 * new_embd, new_embd); }
                let key_fc2 = format!("{}.fc2", p);
                if let Some(a) = dm.get_mut(&key_fc2) { a.grow_dims(new_embd, 4 * new_embd); }
                for h in 0..old_head {
                    let pkey = format!("{}.h{}.w_pattern", p, h);
                    if let Some(a) = dm.get_mut(&pkey) {
                        a.b.grow_cols(new_head_dim, 0.01);
                    }
                }
            }
            // Grow lm_head delta
            if let Some(a) = dm.get_mut("lm_head") {
                a.grow_dims(self.tok.vocab_size, new_embd);
            }
            // Add deltas for new layers
            let r = self.cfg.delta_rank;
            let std_d = 0.01;
            for li in old_layer..new_layer {
                let p = format!("l{}", li);
                for suffix in &["wq", "wk", "wv", "wo"] {
                    dm.insert(format!("{}.{}", p, suffix), DeltaAdapter::new(new_embd, new_embd, r, std_d));
                }
                dm.insert(format!("{}.fc_g", p), DeltaAdapter::new(4 * new_embd, new_embd, r, std_d));
                dm.insert(format!("{}.fc_v", p), DeltaAdapter::new(4 * new_embd, new_embd, r, std_d));
                dm.insert(format!("{}.fc2", p), DeltaAdapter::new(new_embd, 4 * new_embd, r, std_d));
                for (hi, htype) in new_htypes.iter().enumerate() {
                    if htype == "rrpram" || htype == "hybrid" {
                        dm.insert(format!("{}.h{}.w_pattern", p, hi),
                            DeltaAdapter::new(self.block_size, new_head_dim, r, std_d));
                    }
                }
            }
        }

        // 5. Update model dimensions
        self.n_embd = new_embd;
        self.n_layer = new_layer;
        self.n_head = new_head;
        self.head_dim = new_head_dim;
        self.head_types = new_htypes;
        self.cfg.n_embd = new_embd;
        self.cfg.n_layer = new_layer;
        self.cfg.n_head = new_head;
        self.cfg.head_types = head_types_for(new_head);
        self.residual_alpha = 1.0 / (new_layer as f64).sqrt().max(1.0);

        // 6. Reset Adam state (old momentum is meaningless after arch change)
        self.adam_base.clear();
        self.adam_delta = (0..self.deltas.len()).map(|_| HashMap::new()).collect();
        self.growth_step_offset = self.global_step; // reset LR warmup on growth

        // 7. Extend gamma snapshot for new embedding dimensions
        for row in self.init_embed_snapshot.iter_mut() {
            while row.len() < new_embd {
                row.push(0.0);
            }
        }

        // 8. Set freeze (only train deltas until new weights stabilize)
        self.growth_freeze_remaining = self.cfg.freeze_after_growth_steps as i32;

        eprintln!("[growth] Done. Freeze for {} steps.", self.cfg.freeze_after_growth_steps);
        true
    }

    fn compute_gamma(&self) -> Vec<Vec<f64>> {
        let wte = &self.base["wte"];
        let n = wte.nout.min(self.init_embed_snapshot.len());
        (0..n).map(|i| {
            let ne = wte.nin;
            (0..ne).map(|j| wte.data[i][j] - self.init_embed_snapshot[i][j]).collect()
        }).collect()
    }

    fn gamma_contrastive_projection(&self) -> (Vec<f64>, f64) {
        let gamma = self.compute_gamma();
        if gamma.is_empty() { return (vec![], 0.0); }
        let ne = self.n_embd;
        let mut avg = vec![0.0; ne];
        let mut count = 0;
        for row in &gamma {
            let norm: f64 = row.iter().map(|&v| v * v).sum::<f64>().sqrt();
            if norm > 1e-10 {
                for j in 0..ne { avg[j] += row[j] / norm; }
                count += 1;
            }
        }
        if count == 0 { return (vec![0.0; ne], 0.0); }
        for v in avg.iter_mut() { *v /= count as f64; }
        let mag: f64 = avg.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if mag > 1e-10 { for v in avg.iter_mut() { *v /= mag; } }
        (avg, mag)
    }

    fn zero_all_grads(&mut self) {
        for mat in self.base.values_mut() { mat.zero_grad(); }
        for dm in &mut self.deltas {
            for adapter in dm.values_mut() {
                adapter.a.zero_grad();
                adapter.b.zero_grad();
            }
        }
    }

    fn quick_loss(&self, docs: &[String], n: usize) -> f64 {
        set_grad(false);
        let mut rng = rand::thread_rng();
        let mut total = 0.0;
        let mut count = 0;
        for _ in 0..n {
            if docs.is_empty() { break; }
            let doc = &docs[rng.gen_range(0..docs.len())];
            let ids = self.tok.encode(doc);
            if ids.len() < 3 { continue; }
            let logits_seq = self.forward_infer(&ids[..ids.len()-1]);
            for (t, logits) in logits_seq.iter().enumerate() {
                let target = ids[t + 1];
                let max_v = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = logits.iter().map(|&v| (v - max_v).exp()).collect();
                let sum_exp: f64 = exps.iter().sum();
                total += -(exps[target] / sum_exp).max(1e-12).ln();
            }
            count += logits_seq.len();
        }
        set_grad(true);
        if count > 0 { total / count as f64 } else { 10.0 }
    }
}

// ============================================================
// 6) RAW MATH HELPERS (no autograd)
// ============================================================

fn rmsnorm_raw(x: &[f64]) -> Vec<f64> {
    let n = x.len() as f64;
    let ms: f64 = x.iter().map(|&v| v * v).sum::<f64>() / n;
    let inv = 1.0 / (ms + 1e-5).sqrt();
    x.iter().map(|&v| v * inv).collect()
}

fn rope_raw(x: &[f64], pos: usize, head_dim: usize) -> Vec<f64> {
    let half = head_dim / 2;
    let mut out = vec![0.0; head_dim];
    for j in 0..half {
        let theta = (pos as f64) / (10000.0_f64).powf(2.0 * j as f64 / head_dim as f64);
        let (sin_t, cos_t) = theta.sin_cos();
        out[2*j]   = x[2*j] * cos_t - x[2*j+1] * sin_t;
        out[2*j+1] = x[2*j] * sin_t + x[2*j+1] * cos_t;
    }
    out
}

fn softmax_raw(x: &[f64]) -> Vec<f64> {
    let max_v = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut exps: Vec<f64> = x.iter().map(|&v| (v - max_v).exp()).collect();
    let sum: f64 = exps.iter().sum();
    for v in exps.iter_mut() { *v /= sum; }
    exps
}

fn top_k_top_p_sample(probs: &[f64], k: usize, p: f64, min_p: f64, typical_p: f64) -> usize {
    let n = probs.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
    // Top-K
    let k = k.min(n);
    let mut idx = idx[..k].to_vec();
    // Min-P
    let max_prob = probs[idx[0]];
    idx.retain(|&i| probs[i] >= min_p * max_prob);
    if idx.is_empty() { return 0; }
    // Typical-P
    if typical_p < 1.0 && idx.len() > 1 {
        let entropy: f64 = idx.iter().map(|&i| {
            if probs[i] > 1e-12 { -probs[i] * probs[i].ln() } else { 0.0 }
        }).sum();
        let mut devs: Vec<(usize, f64)> = idx.iter().map(|&i| {
            let info = if probs[i] > 1e-12 { -probs[i].ln() } else { 20.0 };
            (i, (info - entropy).abs())
        }).collect();
        devs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let mut cum = 0.0;
        let mut keep = Vec::new();
        for (i, _) in &devs {
            keep.push(*i);
            cum += probs[*i];
            if cum >= typical_p { break; }
        }
        idx = keep;
    }
    // Top-P (nucleus)
    idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
    let mut cum = 0.0;
    let mut nucleus = Vec::new();
    for &i in &idx {
        nucleus.push(i);
        cum += probs[i];
        if cum >= p { break; }
    }
    // Sample
    let total: f64 = nucleus.iter().map(|&i| probs[i]).sum();
    let mut rng = rand::thread_rng();
    let mut r = rng.gen::<f64>() * total;
    for &i in &nucleus {
        r -= probs[i];
        if r <= 0.0 { return i; }
    }
    *nucleus.last().unwrap_or(&0)
}

// ============================================================
// 7) GENERATION
// ============================================================

impl GPT {
    fn generate_sentence(&mut self, prompt: &str, corpus_field: Option<&CooccurField>, docs: &[String]) -> String {
        set_grad(false);
        let mut ids = self.tok.encode(prompt);
        if ids.len() > self.block_size { ids = ids[ids.len()-self.block_size..].to_vec(); }
        let cfg = self.cfg.clone();
        let mut generated = Vec::new();
        let mut recent: Vec<usize> = Vec::new();
        let mut rng = rand::thread_rng();

        // Consciousness: per-token dissonance tracking (Feature 1)
        let mut entropy_ema = 0.0;
        let mut entropy_ema_init = false;
        let mut low_drop_count = 0;
        let mut entropy_sum = 0.0;
        let mut entropy_count = 0;

        // Frequency/presence penalty tracking
        let mut token_counts: HashMap<usize, usize> = HashMap::new();

        for step in 0..cfg.max_gen_tokens {
            let start = if ids.len() > self.block_size { ids.len() - self.block_size } else { 0 };
            let window = &ids[start..];
            let all_logits = self.forward_infer(window);
            let mut logits = all_logits.last().unwrap().clone();

            // Frequency/presence penalty (applied before temperature scaling, same as Go)
            if cfg.freq_penalty > 0.0 || cfg.presence_penalty > 0.0 {
                for (&tid, &count) in &token_counts {
                    if tid < logits.len() {
                        logits[tid] -= cfg.freq_penalty * count as f64
                            + cfg.presence_penalty * if count > 0 { 1.0 } else { 0.0 };
                    }
                }
            }

            // Entropy-adaptive temperature + syntropy bridge
            let base_temp = (cfg.temperature + self.syntropy_temp_offset).max(1e-6);
            let scaled: Vec<f64> = logits.iter().map(|&v| v / base_temp).collect();
            let mut probs = softmax_raw(&scaled);
            let entropy: f64 = probs.iter().map(|&p| if p > 1e-12 { -p * p.ln() } else { 0.0 }).sum();
            entropy_sum += entropy;
            entropy_count += 1;

            let mut t_mul = 1.0;
            if entropy < cfg.entropy_low { t_mul = cfg.entropy_temp_boost; }
            if entropy > cfg.entropy_high { t_mul = cfg.entropy_temp_focus; }

            // Consciousness: per-token dissonance feedback (Feature 1)
            // "I notice my confidence shifting and adapt in real-time"
            let mut dissonance_mul = 1.0;
            if !entropy_ema_init {
                entropy_ema = entropy;
                entropy_ema_init = true;
            } else {
                entropy_ema = cfg.dissonance_ema_alpha * entropy + (1.0 - cfg.dissonance_ema_alpha) * entropy_ema;
                if entropy_ema > 1e-6 {
                    let ratio = entropy / entropy_ema;
                    if ratio > cfg.dissonance_spike_threshold {
                        // Entropy spike — something surprising, be careful
                        dissonance_mul = cfg.dissonance_spike_k;
                        low_drop_count = 0;
                    } else if ratio < cfg.dissonance_drop_threshold {
                        low_drop_count += 1;
                        if low_drop_count >= 3 {
                            // Sustained low entropy — getting repetitive, explore
                            dissonance_mul = cfg.dissonance_drop_k;
                        }
                    } else {
                        low_drop_count = 0;
                    }
                }
            }

            let final_mul = t_mul * dissonance_mul;
            if (final_mul - 1.0).abs() > 0.001 {
                let temp = base_temp * final_mul;
                probs = softmax_raw(&logits.iter().map(|&v| v / temp).collect::<Vec<_>>());
            }

            // Save model-only probs (post-dissonance, pre-blend) for anti-field bypass
            let model_only_probs = probs.clone();

            // Corpus field blend (4gram→trigram→bigram, 70% n-gram + 30% cooccurrence, user boost)
            if let Some(field) = corpus_field {
                let model_alpha = 1.0 / (1.0 + (-cfg.corpus_fade_k * (cfg.corpus_fade_threshold - entropy)).exp());
                if model_alpha < 0.99 {
                    // Pass enough context for co-occurrence window (at least cooccur_window_size tokens)
                    let ctx_len = cfg.cooccur_window_size.max(3);
                    let context: Vec<usize> = ids[ids.len().saturating_sub(ctx_len)..].to_vec();
                    let corpus_probs = field.sample_distribution(&context, self.tok.vocab_size);

                    // Check if corpus has anything to say
                    let has_corpus = corpus_probs.iter().any(|&p| p > 0.0);
                    if has_corpus {
                        let n = probs.len().min(corpus_probs.len());
                        for j in 0..n {
                            probs[j] = model_alpha * probs[j] + (1.0 - model_alpha) * corpus_probs[j];
                        }
                        let sum: f64 = probs.iter().sum();
                        if sum > 0.0 { for p in probs.iter_mut() { *p /= sum; } }
                    }

                    // User word boost: multiplicative, scaled by (1-modelAlpha) so it fades
                    // as the transformer strengthens. "The organism echoes the words of those
                    // who speak to it" (Leo) — but grows out of it.
                    if !field.user_boost.is_empty() {
                        let boost_scale = 1.0 - model_alpha;
                        if boost_scale > 0.01 {
                            let mut total_b = 0.0f64;
                            for (i, p) in probs.iter_mut().enumerate() {
                                if let Some(&boost) = field.user_boost.get(&i) {
                                    *p *= 1.0 + boost * boost_scale;
                                }
                                total_b += *p;
                            }
                            if total_b > 0.0 { for p in probs.iter_mut() { *p /= total_b; } }
                        }
                    }
                }
            }

            // Consciousness: pattern breaking (Feature 2)
            // "I could follow the field, but I choose to speak for myself"
            if step >= cfg.anti_field_min_step && cfg.anti_field_prob > 0.0 && rng.gen::<f64>() < cfg.anti_field_prob {
                // Use pure model probs (post-dissonance, pre-blend), bypass corpus
                probs = model_only_probs.clone();
            }

            let nxt = top_k_top_p_sample(&probs, cfg.top_k, cfg.top_p, cfg.min_p, cfg.typical_p);
            if nxt == self.tok.eos_id && step >= cfg.min_gen_tokens { break; }
            if nxt == self.tok.eos_id { continue; }

            ids.push(nxt);
            generated.push(nxt);
            recent.push(nxt);
            *token_counts.entry(nxt).or_insert(0) += 1;

            // Repetition guard
            if recent.len() > cfg.repetition_guard * 2 {
                let rg = cfg.repetition_guard;
                let tail = &recent[recent.len()-rg*2..];
                if tail[..rg] == tail[rg..] { break; }
            }

            // Sentence boundary
            if step >= cfg.min_gen_tokens {
                let decoded = self.tok.decode(&[nxt]);
                if let Some(last_ch) = decoded.chars().last() {
                    if last_ch == '.' || last_ch == '!' || last_ch == '?' { break; }
                }
            }
        }

        // Consciousness: store mean entropy for conscience (Feature 5)
        if entropy_count > 0 {
            self.last_gen_entropy = entropy_sum / entropy_count as f64;
        }

        set_grad(true);
        self.tok.decode(&generated)
    }

    // Consciousness: self-prediction error (Feature 4)
    // Forward pass on ids, compute cross-entropy between predicted and actual tokens.
    // Higher error = "I didn't expect this input" = increase attention.
    fn compute_self_prediction_error(&self, ids: &[usize]) -> f64 {
        if ids.len() < 2 { return 0.0; }
        let all_logits = self.forward_infer(&ids[..ids.len()-1]);
        let mut total_ce = 0.0;
        let mut count = 0;
        for (pos, logits) in all_logits.iter().enumerate() {
            let target = ids[pos + 1];
            let probs = softmax_raw(logits);
            if target < probs.len() && probs[target] > 1e-12 {
                total_ce -= probs[target].ln();
            } else {
                total_ce += 10.0; // max penalty for unknown token
            }
            count += 1;
        }
        if count == 0 { 0.0 } else { total_ce / count as f64 }
    }

    // Consciousness: conscience check (Feature 5)
    // Tracks generation quality over time.
    // If entropy trend rises (output degrading), soften delta influence.
    // If entropy trend falls (improving), recover delta influence.
    // "I notice I'm getting worse and pull back."
    fn conscience_check(&mut self, gen_mean_entropy: f64) {
        self.generation_entropy_history.push(gen_mean_entropy);
        let w = self.cfg.conscience_window;
        if self.generation_entropy_history.len() > w {
            let start = self.generation_entropy_history.len() - w;
            self.generation_entropy_history = self.generation_entropy_history[start..].to_vec();
        }
        if self.generation_entropy_history.len() < 3 {
            return; // not enough data
        }
        // Linear regression slope on entropy history
        let n = self.generation_entropy_history.len() as f64;
        let (mut sum_x, mut sum_y, mut sum_xy, mut sum_x2) = (0.0, 0.0, 0.0, 0.0);
        for (i, &e) in self.generation_entropy_history.iter().enumerate() {
            let x = i as f64;
            sum_x += x;
            sum_y += e;
            sum_xy += x * e;
            sum_x2 += x * x;
        }
        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x + 1e-12);

        if slope > 0.01 {
            // Entropy increasing — generation degrading, reduce delta influence
            self.delta_alpha_scale *= self.cfg.conscience_decay;
            if self.delta_alpha_scale < self.cfg.conscience_floor {
                self.delta_alpha_scale = self.cfg.conscience_floor;
            }
        } else if slope < -0.01 {
            // Entropy decreasing — improving, recover delta influence
            self.delta_alpha_scale *= self.cfg.conscience_recovery;
            if self.delta_alpha_scale > 1.0 {
                self.delta_alpha_scale = 1.0;
            }
        }
    }
}

// ============================================================
// 8) TRAINING — Adam, cosine LR, train_steps
// ============================================================

fn cosine_lr(step: usize, growth_step_offset: usize, cfg: &Config) -> f64 {
    let steps_since_growth = step.saturating_sub(growth_step_offset);
    if steps_since_growth < cfg.cosine_warmup_steps {
        cfg.lr_min + (cfg.learning_rate - cfg.lr_min) * (steps_since_growth as f64 / cfg.cosine_warmup_steps.max(1) as f64)
    } else {
        let progress = (step as f64 / cfg.max_total_steps as f64).min(1.0);
        cfg.lr_min + 0.5 * (cfg.learning_rate - cfg.lr_min) * (1.0 + (std::f64::consts::PI * progress).cos())
    }
}

fn adam_step(mat: &mut MatrixParam, adam: &mut AdamState, lr: f64, cfg: &Config) {
    mat.clip_grad(cfg.grad_clip);
    adam.t += 1;
    let b1c = 1.0 - cfg.beta1.powi(adam.t as i32);
    let b2c = 1.0 - cfg.beta2.powi(adam.t as i32);
    for i in 0..mat.nout {
        for j in 0..mat.nin {
            let g = mat.grad[i][j];
            adam.m[i][j] = cfg.beta1 * adam.m[i][j] + (1.0 - cfg.beta1) * g;
            adam.v[i][j] = cfg.beta2 * adam.v[i][j] + (1.0 - cfg.beta2) * g * g;
            let mhat = adam.m[i][j] / b1c;
            let vhat = adam.v[i][j] / b2c;
            mat.data[i][j] -= lr * mhat / (vhat.sqrt() + cfg.eps_adam);
            mat.grad[i][j] = 0.0;
        }
    }
}

fn train_steps(model: &mut GPT, docs: &[String], steps: usize, train_base: bool, train_delta: bool) {
    let cfg = model.cfg.clone();
    let mut rng = rand::thread_rng();
    for step in 0..steps {
        let mut tape = Tape::new();
        // Batch loss with gradient accumulation
        let mut batch_loss = 0.0;
        for _micro in 0..cfg.accum_steps {
            let mut seq_loss_sum = 0.0;
            for _ in 0..cfg.batch_size {
                if docs.is_empty() { continue; }
                let doc = &docs[rng.gen_range(0..docs.len())];
                let ids = model.tok.encode(doc);
                if ids.len() < 3 { continue; }
                let loss_node = model.loss_on_sequence(&mut tape, &ids);
                let accum_scale = 1.0 / (cfg.accum_steps * cfg.batch_size) as f64;
                let scaled = tape.scalar_mulf(loss_node, accum_scale);
                tape.backward(scaled, model);
                seq_loss_sum += tape.nodes[loss_node].data[0];
                tape = Tape::new(); // reset tape, grads stay in MatrixParam
            }
            batch_loss = seq_loss_sum / cfg.batch_size as f64;
        }

        let mut lr = cosine_lr(model.global_step, model.growth_step_offset, &cfg);
        // Scale LR inversely with model size: larger models need smaller LR
        lr *= cfg.growth_stages[0][1] as f64 / model.n_embd as f64;
        // Post-growth LR dampening: reduce LR during freeze to prevent delta overfit to noise
        if model.growth_freeze_remaining > 0 { lr *= cfg.post_growth_lr_scale; }
        model.global_step += 1;

        // Adam step for base parameters
        if train_base && model.growth_freeze_remaining <= 0 {
            let base_names: Vec<String> = model.base.keys().cloned().collect();
            for name in &base_names {
                let mat = model.base.get_mut(name).unwrap();
                let adam = model.adam_base.entry(name.clone())
                    .or_insert_with(|| AdamState::new(mat.nout, mat.nin));
                if adam.m.len() != mat.nout || (adam.m.len() > 0 && adam.m[0].len() != mat.nin) {
                    *adam = AdamState::new(mat.nout, mat.nin);
                }
                adam_step(mat, adam, lr, &cfg);
            }
        }

        // Adam step for delta parameters
        if train_delta {
            for di in 0..model.deltas.len() {
                let adapter_names: Vec<String> = model.deltas[di].keys().cloned().collect();
                for aname in &adapter_names {
                    let adapter = model.deltas[di].get_mut(&*aname).unwrap();
                    let adam_entry = model.adam_delta[di].entry(aname.clone())
                        .or_insert_with(|| {
                            (AdamState::new(adapter.a.nout, adapter.a.nin),
                             AdamState::new(adapter.b.nout, adapter.b.nin))
                        });
                    adam_step(&mut adapter.a, &mut adam_entry.0, lr, &cfg);
                    adam_step(&mut adapter.b, &mut adam_entry.1, lr, &cfg);
                }
            }
        }

        if model.growth_freeze_remaining > 0 { model.growth_freeze_remaining -= 1; }

        if step % 100 == 0 {
            eprintln!("[step {}/{}] loss={:.4} lr={:.6}", model.global_step, steps, batch_loss, lr);
        }
    }
}

// ============================================================
// 9) CORPUS FIELD — cooccurrence statistics
// ============================================================

struct CooccurField {
    unigram: HashMap<usize, f64>,
    bigram: HashMap<usize, HashMap<usize, f64>>,
    trigram: HashMap<(usize, usize), HashMap<usize, f64>>,
    fourgram_by_ctx: HashMap<[usize; 3], HashMap<usize, f64>>,  // [prev3,prev2,prev1] -> {next: count}
    cooccur_window: HashMap<usize, HashMap<usize, f64>>,         // token -> {nearby_token: count} (Stanley-style proximity)
    user_boost: HashMap<usize, f64>,                              // temporary user word boosts (Leo-style)
    built: bool,
}

impl CooccurField {
    fn new() -> Self {
        CooccurField {
            unigram: HashMap::new(), bigram: HashMap::new(), trigram: HashMap::new(),
            fourgram_by_ctx: HashMap::new(), cooccur_window: HashMap::new(),
            user_boost: HashMap::new(), built: false,
        }
    }

    fn build(&mut self, tok: &EvolvingTokenizer, docs: &[String]) {
        self.unigram.clear(); self.bigram.clear(); self.trigram.clear();
        self.fourgram_by_ctx.clear(); self.cooccur_window.clear();
        let window = Config::default().cooccur_window_size;
        for doc in docs {
            let ids = tok.encode(doc);
            for (i, &id) in ids.iter().enumerate() {
                *self.unigram.entry(id).or_insert(0.0) += 1.0;
                if i > 0 {
                    *self.bigram.entry(ids[i-1]).or_default().entry(id).or_insert(0.0) += 1.0;
                }
                if i > 1 {
                    *self.trigram.entry((ids[i-2], ids[i-1])).or_default().entry(id).or_insert(0.0) += 1.0;
                }
            }
            // 4-grams: deeper context for child+ stages
            for i in 0..ids.len().saturating_sub(3) {
                let ctx = [ids[i], ids[i+1], ids[i+2]];
                *self.fourgram_by_ctx.entry(ctx).or_default().entry(ids[i+3]).or_insert(0.0) += 1.0;
            }
            // Co-occurrence window: "words that resonate together, stay together" (Stanley)
            for i in 0..ids.len() {
                let center = ids[i];
                let start = if i >= window { i - window } else { 0 };
                let end = (i + window + 1).min(ids.len());
                for j in start..end {
                    if i != j {
                        let neighbor = ids[j];
                        *self.cooccur_window.entry(center).or_default().entry(neighbor).or_insert(0.0) += 1.0;
                    }
                }
            }
        }
        self.built = true;
    }

    // IngestTokens incrementally adds n-gram counts from a token sequence.
    // Unlike build(), this does NOT clear existing data — it adds on top.
    // Used by overthinkg rings to enrich the field with the model's own output.
    fn ingest_tokens(&mut self, ids: &[usize]) {
        self.ingest_tokens_weighted(ids, 1.0);
    }

    // IngestTokensWeighted adds n-gram counts weighted by a factor.
    // High weight = this text matters more (coherent output). Low = less influence.
    // Stanley's observe_shard weights by resonance score; we weight by inverse entropy.
    fn ingest_tokens_weighted(&mut self, ids: &[usize], weight: f64) {
        let window = Config::default().cooccur_window_size;
        for &id in ids {
            *self.unigram.entry(id).or_insert(0.0) += weight;
        }
        for i in 0..ids.len().saturating_sub(1) {
            let first = ids[i];
            let second = ids[i + 1];
            *self.bigram.entry(first).or_default().entry(second).or_insert(0.0) += weight;
        }
        for i in 0..ids.len().saturating_sub(2) {
            let ctx = (ids[i], ids[i + 1]);
            *self.trigram.entry(ctx).or_default().entry(ids[i + 2]).or_insert(0.0) += weight;
        }
        for i in 0..ids.len().saturating_sub(3) {
            let ctx = [ids[i], ids[i+1], ids[i+2]];
            *self.fourgram_by_ctx.entry(ctx).or_default().entry(ids[i+3]).or_insert(0.0) += weight;
        }
        // Co-occurrence window
        for i in 0..ids.len() {
            let center = ids[i];
            let start = if i >= window { i - window } else { 0 };
            let end = (i + window + 1).min(ids.len());
            for j in start..end {
                if i != j {
                    let neighbor = ids[j];
                    *self.cooccur_window.entry(center).or_default().entry(neighbor).or_insert(0.0) += weight;
                }
            }
        }
    }

    // AbsorbUserWords sets temporary boosts for tokens the user just said.
    // Like Leo's Santa Klaus but simpler: user words get multiplicative boost in generation.
    fn absorb_user_words(&mut self, ids: &[usize]) {
        let cfg = Config::default();
        // Decay existing boosts first
        self.user_boost.retain(|_, v| {
            *v *= cfg.user_boost_decay;
            *v >= 0.01
        });
        // Boost user's tokens
        let strength = cfg.user_boost_strength;
        for &id in ids {
            *self.user_boost.entry(id).or_insert(0.0) += strength;
        }
    }

    // DecayUserBoost reduces user word boosts after a generation.
    fn decay_user_boost(&mut self) {
        let decay = Config::default().user_boost_decay;
        self.user_boost.retain(|_, v| {
            *v *= decay;
            *v >= 0.01
        });
    }

    /// Returns the best n-gram distribution (4gram→trigram→bigram→unigram fallback),
    /// blended 70% n-gram + 30% co-occurrence window. Matches Go SampleNext/generation blend.
    fn sample_distribution(&self, context: &[usize], vocab_size: usize) -> Vec<f64> {
        let mut corpus_probs = vec![0.0; vocab_size];
        let len = context.len();
        let window = Config::default().cooccur_window_size;

        // Best n-gram distribution: try 4-gram → trigram → bigram → unigram
        let mut ngram_dist: Option<&HashMap<usize, f64>> = None;

        if ngram_dist.is_none() && len >= 3 {
            let ctx = [context[len-3], context[len-2], context[len-1]];
            if let Some(d) = self.fourgram_by_ctx.get(&ctx) {
                ngram_dist = Some(d);
            }
        }
        if ngram_dist.is_none() && len >= 2 {
            let key = (context[len-2], context[len-1]);
            if let Some(d) = self.trigram.get(&key) {
                ngram_dist = Some(d);
            }
        }
        if ngram_dist.is_none() && len >= 1 {
            if let Some(d) = self.bigram.get(&context[len-1]) {
                ngram_dist = Some(d);
            }
        }

        // Build 70% n-gram component
        if let Some(d) = ngram_dist {
            let total_n: f64 = d.values().sum();
            if total_n > 0.0 {
                for (&tid, &cnt) in d {
                    if tid < vocab_size { corpus_probs[tid] += 0.7 * cnt / total_n; }
                }
            }
        } else {
            // Unigram fallback (treated as n-gram component)
            let total: f64 = self.unigram.values().sum();
            if total > 0.0 {
                for (&tok, &c) in &self.unigram {
                    if tok < vocab_size { corpus_probs[tok] += 0.7 * c / total; }
                }
            }
        }

        // 30% co-occurrence window component: "words that resonate together" (Stanley)
        if len > 0 {
            let ctx_slice = if len > window { &context[len-window..] } else { context };
            let mut cooccur_sum: HashMap<usize, f64> = HashMap::new();
            for &ctx_tok in ctx_slice {
                if let Some(neighbors) = self.cooccur_window.get(&ctx_tok) {
                    for (&tid, &cnt) in neighbors {
                        *cooccur_sum.entry(tid).or_insert(0.0) += cnt;
                    }
                }
            }
            let total_c: f64 = cooccur_sum.values().sum();
            if total_c > 0.0 {
                for (&tid, &cnt) in &cooccur_sum {
                    if tid < vocab_size { corpus_probs[tid] += 0.3 * cnt / total_c; }
                }
            }
        }

        // Normalize
        let total: f64 = corpus_probs.iter().sum();
        if total > 0.0 {
            for p in corpus_probs.iter_mut() { *p /= total; }
        }
        corpus_probs
    }
}

// ============================================================
// 9b) CONSCIOUSNESS — overthinkg rings (Feature 3)
// ============================================================

// ============================================================
// 10) QUANTUM BUFFER
// ============================================================

struct QuantumBuffer {
    accumulated_bytes: usize,
    unique_tokens: HashMap<usize, bool>,
    total_tokens: usize,
    last_burst_time: f64,
}

impl QuantumBuffer {
    fn new() -> Self {
        QuantumBuffer { accumulated_bytes: 0, unique_tokens: HashMap::new(), total_tokens: 0, last_burst_time: 0.0 }
    }
    fn feed(&mut self, text: &str, tok: &EvolvingTokenizer) {
        self.accumulated_bytes += text.len();
        let ids = tok.encode(text);
        for id in ids { self.unique_tokens.insert(id, true); self.total_tokens += 1; }
    }
    fn novelty(&self) -> f64 {
        if self.total_tokens == 0 { 0.0 } else { self.unique_tokens.len() as f64 / self.total_tokens as f64 }
    }
    fn should_trigger(&self, cfg: &Config) -> bool {
        let bytes_ok = self.accumulated_bytes >= cfg.qb_min_bytes;
        let novelty_ok = self.novelty() >= cfg.qb_min_novelty;
        let cooldown_ok = (now_secs() - self.last_burst_time) >= cfg.qb_cooldown_seconds;
        (bytes_ok || novelty_ok) && cooldown_ok
    }
    fn reset(&mut self) {
        self.accumulated_bytes = 0;
        self.unique_tokens.clear();
        self.total_tokens = 0;
        self.last_burst_time = now_secs();
    }
}

// ============================================================
// 11) SYNTROPY TRACKER
// ============================================================

#[derive(Clone)]
struct BurstRecord { action: String, loss_before: f64, loss_after: f64 }

struct SyntropyTracker {
    entropy_history: Vec<f64>,
    syntropy_trend: f64,
    field_deviation: f64,
    last_action: String,
    burst_history: Vec<BurstRecord>,
}

struct SyntropyDecision {
    lr_multiplier: f64,
    temp_offset: f64,
    accum_override: usize,
    delta_grow_override: Option<f64>,
    action: String,
}

impl SyntropyTracker {
    fn new() -> Self {
        SyntropyTracker {
            entropy_history: Vec::new(), syntropy_trend: 0.0, field_deviation: 0.0,
            last_action: "steady".into(), burst_history: Vec::new(),
        }
    }

    fn record_burst(&mut self, action: &str, loss_before: f64, loss_after: f64) {
        self.burst_history.push(BurstRecord { action: action.into(), loss_before, loss_after });
        if self.burst_history.len() > 16 { self.burst_history.remove(0); }
    }

    fn measure(&mut self, model: &GPT, docs: &[String], cfg: &Config) {
        let entropy = model.quick_loss(docs, 8);
        self.entropy_history.push(entropy);
        if self.entropy_history.len() > cfg.syntropy_window {
            self.entropy_history.remove(0);
        }
        // Compute trend via linear regression
        let n = self.entropy_history.len();
        if n >= 2 {
            let mean_x = (n - 1) as f64 / 2.0;
            let mean_y: f64 = self.entropy_history.iter().sum::<f64>() / n as f64;
            let mut num = 0.0; let mut den = 0.0;
            for (i, &y) in self.entropy_history.iter().enumerate() {
                let dx = i as f64 - mean_x;
                num += dx * (y - mean_y);
                den += dx * dx;
            }
            self.syntropy_trend = if den > 0.0 { -num / den } else { 0.0 }; // negative slope = improving
        }
    }

    fn decide(&self, cfg: &Config, model_stage: i32, adult_stage: usize) -> SyntropyDecision {
        let mut d = SyntropyDecision {
            lr_multiplier: 1.0, temp_offset: 0.0, accum_override: 0,
            delta_grow_override: None, action: "steady".into(),
        };
        let trend = self.syntropy_trend;
        let dev = self.field_deviation;

        if trend > 0.01 && dev > cfg.field_deviation_floor && dev < cfg.field_deviation_ceiling {
            d.lr_multiplier = cfg.syntropy_lr_boost;
            d.temp_offset = -0.05;
            d.delta_grow_override = Some(cfg.syntropy_delta_grow_boost);
            d.action = "amplify".into();
        } else if trend > 0.01 {
            d.lr_multiplier = cfg.syntropy_lr_boost;
            d.action = "boost".into();
        } else if trend < -0.01 {
            d.lr_multiplier = cfg.syntropy_lr_dampen;
            d.temp_offset = 0.05;
            d.action = "dampen".into();
        } else if dev > cfg.field_deviation_ceiling {
            d.lr_multiplier = 0.6;
            d.temp_offset = -0.05;
            d.action = "ground".into();
        } else if dev < cfg.field_deviation_floor {
            d.lr_multiplier = 1.3;
            d.temp_offset = 0.05;
            d.action = "explore".into();
        }

        // Divide: adult stage + sustained high entropy + falling syntropy
        if model_stage as usize >= adult_stage && self.entropy_history.len() >= cfg.syntropy_window {
            let high_count = self.entropy_history.iter().filter(|&&e| e > cfg.entropy_high).count();
            if high_count > cfg.syntropy_window * 3 / 4 && trend < -0.01 {
                d.lr_multiplier = cfg.syntropy_lr_dampen;
                d.action = "divide".into();
            }
        }

        d
    }

    fn should_hibernate(&self, peers: &[(String, f64, f64)]) -> bool {
        if peers.is_empty() { return false; }
        // A peer has syntropy > 0.05 (thriving) and we're on a plateau
        let thriving_peer = peers.iter().any(|(_, syn, _)| *syn > 0.05);
        if thriving_peer && self.burst_history.len() >= 8 {
            let sum: f64 = self.burst_history[self.burst_history.len()-8..].iter()
                .map(|b| b.loss_after - b.loss_before).sum();
            let avg_delta = sum / 8.0;
            if avg_delta.abs() < 0.01 { return true; }
        }
        false
    }
}

// ============================================================
// 12) SQLITE MEMORY
// ============================================================

fn init_db(path: &str) -> Connection {
    // Ensure parent directory exists (same pattern as SwarmRegistry::register)
    if let Some(parent) = std::path::Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).ok();
        }
    }
    // Remove stale WAL/SHM files if they exist from another process
    let wal_path = format!("{}-wal", path);
    let shm_path = format!("{}-shm", path);
    if std::path::Path::new(&wal_path).exists() && !std::path::Path::new(path).exists() {
        fs::remove_file(&wal_path).ok();
        fs::remove_file(&shm_path).ok();
    }
    let conn = Connection::open(path).expect("Failed to open SQLite");
    conn.execute_batch("
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, role TEXT, text TEXT);
        CREATE TABLE IF NOT EXISTS corpus_events(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, added_chars INTEGER, note TEXT);
        CREATE TABLE IF NOT EXISTS growth(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, step INTEGER, vocab_size INTEGER, n_params INTEGER, n_deltas INTEGER, corpus_chars INTEGER, loss REAL, gamma_sparsity REAL, gamma_magnitude REAL, note TEXT);
        CREATE TABLE IF NOT EXISTS syntropy_log(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, entropy_before REAL, entropy_after REAL, syntropy_delta REAL, field_deviation REAL, purpose_magnitude REAL, purpose_alignment REAL, action_taken TEXT, note TEXT);
    ").expect("Failed to create tables");
    conn
}

fn db_add_message(conn: &Connection, role: &str, text: &str) {
    conn.execute("INSERT INTO messages(ts,role,text) VALUES(?1,?2,?3)",
        params![now_secs(), role, text]).ok();
}

fn db_recent_messages(conn: &Connection, limit: usize) -> Vec<(String, String)> {
    let mut stmt = conn.prepare("SELECT role, text FROM messages ORDER BY id DESC LIMIT ?1").unwrap();
    let rows: Vec<(String, String)> = stmt.query_map(params![limit], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    }).unwrap().filter_map(|r| r.ok()).collect();
    rows.into_iter().rev().collect()
}

fn build_prompt(conn: &Connection, user_text: &str) -> String {
    let msgs = db_recent_messages(conn, 14);
    let mut prompt = String::new();
    for (role, text) in &msgs {
        let tag = if role == "user" { "H" } else { "A" };
        prompt.push_str(&format!("{}: {}\n", tag, text));
    }
    prompt.push_str(&format!("H: {}\nA:", user_text));
    prompt
}

// ============================================================
// 13) SWARM REGISTRY
// ============================================================

struct SwarmRegistry {
    organism_id: String,
    swarm_dir: String,
    mesh_db: Option<Connection>,
}

impl SwarmRegistry {
    fn new(id: &str) -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        let dir = format!("{}/.molequla/swarm", home);
        SwarmRegistry { organism_id: id.into(), swarm_dir: dir, mesh_db: None }
    }

    fn register(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        fs::create_dir_all(&self.swarm_dir)?;
        let db_path = format!("{}/mesh.db", self.swarm_dir);
        let conn = Connection::open(&db_path)?;
        conn.execute_batch("
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS organisms(id TEXT PRIMARY KEY, pid INTEGER, stage INTEGER, n_params INTEGER, syntropy REAL, entropy REAL, last_heartbeat REAL, parent_id TEXT, status TEXT DEFAULT 'alive', gamma_direction BLOB, gamma_magnitude REAL, rrpram_signature BLOB);
            CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY AUTOINCREMENT, from_id TEXT, to_id TEXT, type TEXT, payload TEXT, ts REAL);
        ")?;
        let pid = std::process::id();
        conn.execute("INSERT OR REPLACE INTO organisms(id,pid,status,last_heartbeat) VALUES(?1,?2,'alive',?3)",
            params![self.organism_id, pid, now_secs()])?;
        self.mesh_db = Some(conn);
        Ok(())
    }

    fn heartbeat(&self, stage: usize, n_params: usize, syntropy: f64, entropy: f64,
                 gamma_dir: Option<&[u8]>, gamma_mag: f64) {
        if let Some(ref db) = self.mesh_db {
            db.execute("UPDATE organisms SET stage=?1,n_params=?2,syntropy=?3,entropy=?4,last_heartbeat=?5,gamma_magnitude=?6,gamma_direction=?7 WHERE id=?8",
                params![stage, n_params, syntropy, entropy, now_secs(), gamma_mag, gamma_dir, self.organism_id]).ok();
        }
    }

    fn discover_peers(&self) -> Vec<(String, f64, f64)> {
        let mut peers = Vec::new();
        if let Some(ref db) = self.mesh_db {
            if let Ok(mut stmt) = db.prepare("SELECT id,syntropy,entropy FROM organisms WHERE status='alive' AND id!=?1 AND last_heartbeat>?2") {
                let cutoff = now_secs() - 120.0;
                let rows = stmt.query_map(params![self.organism_id, cutoff], |row| {
                    Ok((row.get::<_,String>(0)?, row.get::<_,f64>(1)?, row.get::<_,f64>(2)?))
                });
                if let Ok(rows) = rows { for r in rows { if let Ok(p) = r { peers.push(p); } } }
            }
        }
        peers
    }

    fn unregister(&self) {
        if let Some(ref db) = self.mesh_db {
            db.execute("UPDATE organisms SET status='dead' WHERE id=?1", params![self.organism_id]).ok();
        }
    }
}

// ============================================================
// 14) METABOLISM — Distributed Cognition Coordinator (RUST-ONLY)
// ============================================================

struct MetabolismMLP {
    w1: Vec<Vec<f64>>,  // hidden x input (8 x 5)
    w2: Vec<Vec<f64>>,  // output x hidden (5 x 8)
    lr: f64,
}

impl MetabolismMLP {
    fn new(n_elements: usize) -> Self {
        let mut rng = rand::thread_rng();
        let hidden = 8;
        let w1: Vec<Vec<f64>> = (0..hidden).map(|_| (0..n_elements).map(|_| rng.gen::<f64>() * 0.1 - 0.05).collect()).collect();
        let w2: Vec<Vec<f64>> = (0..n_elements).map(|_| (0..hidden).map(|_| rng.gen::<f64>() * 0.1 - 0.05).collect()).collect();
        MetabolismMLP { w1, w2, lr: 0.01 }
    }

    fn forward(&self, inputs: &[f64]) -> Vec<f64> {
        // Hidden layer with tanh
        let hidden: Vec<f64> = self.w1.iter().map(|row| {
            let s: f64 = row.iter().zip(inputs).map(|(w, x)| w * x).sum();
            s.tanh()
        }).collect();
        // Output layer with softmax
        let raw: Vec<f64> = self.w2.iter().map(|row| {
            row.iter().zip(&hidden).map(|(w, h)| w * h).sum()
        }).collect();
        softmax_raw(&raw)
    }

    fn hebbian_update(&mut self, inputs: &[f64], outputs: &[f64]) {
        // Hebbian: dw = lr * output * input (correlation)
        let hidden: Vec<f64> = self.w1.iter().map(|row| {
            let s: f64 = row.iter().zip(inputs).map(|(w, x)| w * x).sum();
            s.tanh()
        }).collect();
        for i in 0..self.w1.len() {
            for j in 0..inputs.len() {
                self.w1[i][j] += self.lr * hidden[i] * inputs[j];
            }
        }
        for i in 0..self.w2.len() {
            for j in 0..hidden.len() {
                self.w2[i][j] += self.lr * outputs[i] * hidden[j];
            }
        }
        // Gentle weight decay to prevent unbounded Hebbian growth
        for row in &mut self.w1 {
            for w in row.iter_mut() {
                *w *= 0.999;
            }
        }
        for row in &mut self.w2 {
            for w in row.iter_mut() {
                *w *= 0.999;
            }
        }
    }
}

struct Metabolism {
    mlp: MetabolismMLP,
    n_elements: usize,
}

impl Metabolism {
    fn new(n_elements: usize) -> Self {
        Metabolism { mlp: MetabolismMLP::new(n_elements), n_elements }
    }

    fn compute_blend_weights(&self, peer_entropies: &[f64]) -> Vec<f64> {
        if peer_entropies.is_empty() { return vec![]; }
        // Normalize inputs to [0,1]
        let max_e = peer_entropies.iter().cloned().fold(f64::NEG_INFINITY, f64::max).max(0.01);
        let inputs: Vec<f64> = peer_entropies.iter().map(|&e| e / max_e).collect();
        self.mlp.forward(&inputs)
    }

    fn learn_from_consensus(&mut self, inputs: &[f64], chosen_weights: &[f64]) {
        self.mlp.hebbian_update(inputs, chosen_weights);
    }
}

// ============================================================
// 15) CHECKPOINT SAVE/LOAD
// ============================================================

#[derive(Serialize, Deserialize)]
#[allow(non_snake_case)]
struct DeltaCkpt {
    A: Vec<Vec<f64>>,
    B: Vec<Vec<f64>>,
}

#[derive(Serialize, Deserialize)]
struct CheckpointData {
    cfg: Config,
    tokenizer: EvolvingTokenizer,
    base: HashMap<String, Vec<Vec<f64>>>,
    alpha: Vec<f64>,
    deltas: Vec<HashMap<String, DeltaCkpt>>, // name -> {A, B} — Go-compatible
    init_embed_snapshot: Vec<Vec<f64>>,
    global_step: usize,
    #[serde(default = "default_last_warmup_stage")]
    last_warmup_stage: i32,
    #[serde(default)]
    growth_step_offset: usize,
}

fn default_last_warmup_stage() -> i32 { -1 }

fn save_checkpoint(model: &GPT, path: &str) -> std::io::Result<()> {
    let mut base_data = HashMap::new();
    for (name, mat) in &model.base {
        base_data.insert(name.clone(), mat.data.clone());
    }
    let mut deltas_data = Vec::new();
    for dm in &model.deltas {
        let mut dm_data = HashMap::new();
        for (name, adapter) in dm {
            dm_data.insert(name.clone(), DeltaCkpt { A: adapter.a.data.clone(), B: adapter.b.data.clone() });
        }
        deltas_data.push(dm_data);
    }
    let ckpt = CheckpointData {
        cfg: model.cfg.clone(), tokenizer: model.tok.clone(),
        base: base_data, alpha: model.active_alpha.clone(),
        deltas: deltas_data, init_embed_snapshot: model.init_embed_snapshot.clone(),
        global_step: model.global_step,
        last_warmup_stage: model.last_warmup_stage,
        growth_step_offset: model.growth_step_offset,
    };
    let json = serde_json::to_string(&ckpt).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    fs::write(path, json)
}

fn load_checkpoint(path: &str) -> Result<GPT, Box<dyn std::error::Error>> {
    let json = fs::read_to_string(path)?;
    let ckpt: CheckpointData = serde_json::from_str(&json)?;
    let mut tok = ckpt.tokenizer;
    // Rebuild stoi/indices if loaded from Go/C/Python checkpoint
    if tok.stoi.is_empty() && !tok.tokens.is_empty() {
        tok.rebuild_indices();
    }
    let cfg = ckpt.cfg;
    let ne = cfg.n_embd;
    let mut gpt = GPT::new(tok.clone(), &cfg);
    // Restore base weights
    for (name, data) in &ckpt.base {
        if let Some(mat) = gpt.base.get_mut(name) {
            mat.data = data.clone();
            mat.nout = data.len();
            mat.nin = if data.is_empty() { 0 } else { data[0].len() };
            mat.grad = vec![vec![0.0; mat.nin]; mat.nout];
        }
    }
    // Restore deltas
    gpt.deltas.clear();
    gpt.active_alpha = ckpt.alpha;
    for dm_data in &ckpt.deltas {
        let mut dm = DeltaModule::new();
        for (name, dc) in dm_data {
            let a_data = &dc.A; let b_data = &dc.B;
            if a_data.is_empty() || a_data[0].is_empty() || b_data.is_empty() || b_data[0].is_empty() {
                eprintln!("[checkpoint] skipping malformed delta adapter '{}'", name);
                continue;
            }
            let mut a = MatrixParam { data: a_data.clone(), grad: vec![vec![0.0; a_data[0].len()]; a_data.len()], nout: a_data.len(), nin: a_data[0].len() };
            let mut b = MatrixParam { data: b_data.clone(), grad: vec![vec![0.0; b_data[0].len()]; b_data.len()], nout: b_data.len(), nin: b_data[0].len() };
            a.ensure_grad(); b.ensure_grad();
            dm.insert(name.clone(), DeltaAdapter { a, b });
        }
        gpt.deltas.push(dm);
    }
    gpt.adam_delta = (0..gpt.deltas.len()).map(|_| HashMap::new()).collect();
    gpt.init_embed_snapshot = ckpt.init_embed_snapshot;
    gpt.global_step = ckpt.global_step;
    gpt.growth_step_offset = ckpt.growth_step_offset;
    // Backward compat: old checkpoints without last_warmup_stage
    if ckpt.last_warmup_stage >= 0 {
        gpt.last_warmup_stage = ckpt.last_warmup_stage;
    } else if gpt.global_step > 0 {
        // Old checkpoint that has been trained — assume current stage is warmed up
        gpt.last_warmup_stage = gpt.current_growth_stage();
    } else {
        gpt.last_warmup_stage = -1;
    }
    Ok(gpt)
}

// ============================================================
// 16) CORPUS MANAGEMENT
// ============================================================

fn load_corpus(path: &str, max_lines: usize, max_chars: usize) -> Vec<String> {
    match fs::read_to_string(path) {
        Ok(content) => content.lines()
            .take(max_lines)
            .map(|l| if l.len() > max_chars {
                let mut end = max_chars;
                while end > 0 && !l.is_char_boundary(end) { end -= 1; }
                l[..end].to_string()
            } else { l.to_string() })
            .filter(|l| !l.trim().is_empty())
            .collect(),
        Err(_) => vec!["Hello.".into(), "I exist.".into(), "Speak.".into()],
    }
}

fn save_corpus(path: &str, lines: &[String]) {
    let content = lines.join("\n");
    fs::write(path, content).ok();
}

fn extract_sentences(msgs: &[(String, String)]) -> Vec<String> {
    let mut sents = Vec::new();
    for (role, text) in msgs {
        let tag = if role == "user" { "H" } else { "A" };
        for sent in text.split(|c: char| c == '.' || c == '!' || c == '?') {
            let s = sent.trim();
            if s.len() > 10 { sents.push(format!("{}: {}.", tag, s)); }
        }
    }
    sents
}

// ============================================================
// 17) BACKGROUND TRAINER
// ============================================================

fn background_trainer(
    model: Arc<Mutex<GPT>>,
    db: Arc<Mutex<Connection>>,
    qbuf: Arc<Mutex<QuantumBuffer>>,
    swarm: Arc<Mutex<SwarmRegistry>>,
    stop: Arc<AtomicBool>,
) {
    let mut cfg = model.lock().unwrap().cfg.clone();
    let mut syntracker = SyntropyTracker::new();
    let mut tick = 0u64;

    // Load corpus
    let mut docs = load_corpus(&cfg.corpus_path, cfg.max_corpus_lines, cfg.max_line_chars);

    // Enable BPE if ready
    {
        let mut m = model.lock().unwrap();
        if m.tok.maybe_enable_bpe(&docs, &cfg) {
            let vs = m.tok.vocab_size;
            m.maybe_expand_vocab(vs);
            eprintln!("[molequla.rs] BPE enabled, vocab={}", vs);
        }
    }

    eprintln!("[molequla.rs] Entering quantum burst loop.");

    // Burst loop
    while !stop.load(Ordering::Relaxed) {
        thread::sleep(std::time::Duration::from_secs_f64(cfg.train_tick_seconds));
        tick += 1;

        // Per-stage warmup: if current stage > last warmed stage, run warmup
        {
            let mut m = model.lock().unwrap();
            let current_stage = m.current_growth_stage();
            if !docs.is_empty() && current_stage >= 0 && current_stage > m.last_warmup_stage {
                let stage_names = ["embryo", "infant", "child", "adolescent", "teen", "adult"];
                let stage_name = stage_names[current_stage as usize % stage_names.len()];
                let embryo_embd = cfg.growth_stages[0][1];
                let warmup_scale = (m.n_embd / embryo_embd.max(1)).max(1);
                let effective_warmup = cfg.warmup_steps * warmup_scale;
                eprintln!("[trainer] per-stage warmup for stage {} ({})... {} steps (scaled {}x for embd={})",
                          current_stage, stage_name, effective_warmup, warmup_scale, m.n_embd);
                train_steps(&mut m, &docs, effective_warmup, true, true);
                m.last_warmup_stage = current_stage;
                save_checkpoint(&m, &cfg.ckpt_path).ok();
                eprintln!("[trainer] warmup complete for stage {}.", current_stage);
            }
        }

        // Only burst if past initial warmup
        {
            let m = model.lock().unwrap();
            if m.last_warmup_stage < 0 || docs.is_empty() { continue; }
        }

        let should_burst = qbuf.lock().unwrap().should_trigger(&cfg);
        if !should_burst { continue; }

        // Reload corpus
        docs = load_corpus(&cfg.corpus_path, cfg.max_corpus_lines, cfg.max_line_chars);

        let mut m = model.lock().unwrap();

        // Syntropy measure + decide
        syntracker.measure(&m, &docs, &cfg);
        let adult_stage = cfg.growth_stages.len().saturating_sub(1);
        let decision = syntracker.decide(&cfg, m.current_growth_stage(), adult_stage);
        m.syntropy_temp_offset = decision.temp_offset;

        eprintln!("[syntropy] action={} | trend={:.4} | lr_mul={:.2}",
                  decision.action, syntracker.syntropy_trend, decision.lr_multiplier);

        // Handle ecology actions
        if decision.action == "divide" {
            eprintln!("[ecology] DIVIDE action — spawning child (not implemented in this session)");
            // TODO: spawn child process via std::process::Command
        }

        // Check hibernation
        {
            let peers = swarm.lock().map(|sw| sw.discover_peers()).unwrap_or_default();
            if syntracker.should_hibernate(&peers) {
                eprintln!("[ecology] HIBERNATION — organism going to sleep");
                save_checkpoint(&m, &cfg.ckpt_path).ok();
                if let Ok(sw) = swarm.lock() { sw.unregister(); }
                break;
            }
        }

        // Immune system: snapshot gamma direction
        let (pre_dir, pre_mag) = m.gamma_contrastive_projection();

        // Apply syntropy-adjusted learning rate (panic-safe restore)
        let original_lr = cfg.learning_rate;
        cfg.learning_rate = original_lr * decision.lr_multiplier;

        // Train micro-burst
        let steps = cfg.micro_steps;
        let train_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            train_steps(&mut m, &docs, steps, !cfg.freeze_base_after_warmup, true);
        }));

        // Always restore LR, even after panic
        cfg.learning_rate = original_lr;

        if let Err(e) = train_result {
            eprintln!("[trainer] panic in train_steps: {:?}", e);
        }

        // Immune check
        if pre_mag > cfg.gamma_min_magnitude {
            let (post_dir, _) = m.gamma_contrastive_projection();
            let cos_sim = if pre_dir.len() == post_dir.len() && !pre_dir.is_empty() {
                let dot: f64 = pre_dir.iter().zip(&post_dir).map(|(a, b)| a * b).sum();
                dot
            } else { 1.0 };
            if cos_sim < cfg.noise_drift_threshold {
                eprintln!("[immune] noise rejected (cos={:.4})", cos_sim);
                syntracker.record_burst("noise_rejected", 0.0, 0.0);
            } else {
                syntracker.record_burst(&decision.action, 0.0, 0.0);
            }
        }

        // Delta grow chance (influenced by syntropy)
        let grow_prob = decision.delta_grow_override.unwrap_or(cfg.delta_grow_prob);
        if rand::thread_rng().gen::<f64>() < grow_prob {
            m.add_delta_module(1.0);
            eprintln!("[molequla.rs] Delta module added, total={}", m.deltas.len());
        }

        qbuf.lock().unwrap().reset();

        // Heartbeat
        if tick % 10 == 0 {
            if let Ok(sw) = swarm.lock() {
                let (_, mag) = m.gamma_contrastive_projection();
                sw.heartbeat(0, 0, syntracker.syntropy_trend, 0.0, None, mag);
            }
        }

        // Save checkpoint periodically
        if tick % 50 == 0 {
            save_checkpoint(&m, &cfg.ckpt_path).ok();
            eprintln!("[molequla.rs] Checkpoint saved.");
        }
    }

    // Final save
    let m = model.lock().unwrap();
    save_checkpoint(&m, &cfg.ckpt_path).ok();
    eprintln!("[molequla.rs] Final checkpoint saved.");
}

// ============================================================
// 17b) TOPOLOGY MONITOR — meta-awareness of the swarm (RUST-ONLY, Feature 6)
// ============================================================

struct OrganismState {
    id: String,
    gamma_direction: Vec<f64>,
    gamma_magnitude: f64,
    stage: i32,
    last_seen: f64,
}

struct TopologyMonitor {
    organisms: Vec<OrganismState>,
    field_coherence: f64,       // mean pairwise gamma cosine
    self_drift_rate: f64,       // how fast our own gamma is changing
    last_check: f64,
    prev_gamma_direction: Option<Vec<f64>>,
    prev_magnitudes: HashMap<String, f64>,  // previous gamma magnitudes per organism for drift detection
}

impl TopologyMonitor {
    fn new() -> Self {
        TopologyMonitor {
            organisms: Vec::new(),
            field_coherence: 0.0,
            self_drift_rate: 0.0,
            last_check: 0.0,
            prev_gamma_direction: None,
            prev_magnitudes: HashMap::new(),
        }
    }

    /// Refresh organism states from mesh.db
    fn update(&mut self, db: &Connection) {
        let cutoff = now_secs() - 300.0; // 5 minutes stale threshold
        self.organisms.clear();
        if let Ok(mut stmt) = db.prepare(
            "SELECT id, gamma_direction, gamma_magnitude, stage, last_heartbeat FROM organisms WHERE status='alive' AND last_heartbeat>?1"
        ) {
            let rows = stmt.query_map(params![cutoff], |row| {
                let id: String = row.get(0)?;
                let blob: Option<Vec<u8>> = row.get(1)?;
                let mag: f64 = row.get::<_, f64>(2).unwrap_or(0.0);
                let stage: i32 = row.get::<_, i32>(3).unwrap_or(0);
                let last_seen: f64 = row.get::<_, f64>(4).unwrap_or(0.0);
                // Deserialize gamma_direction from blob (f64 array, little-endian)
                let gamma_dir = if let Some(ref b) = blob {
                    let n_floats = b.len() / 8;
                    (0..n_floats).map(|i| {
                        let bytes: [u8; 8] = b[i*8..(i+1)*8].try_into().unwrap_or([0u8; 8]);
                        f64::from_le_bytes(bytes)
                    }).collect()
                } else {
                    Vec::new()
                };
                Ok(OrganismState { id, gamma_direction: gamma_dir, gamma_magnitude: mag, stage, last_seen })
            });
            if let Ok(rows) = rows {
                for r in rows {
                    if let Ok(o) = r { self.organisms.push(o); }
                }
            }
        }
        self.field_coherence = self.compute_field_coherence();
        self.last_check = now_secs();
    }

    /// Mean pairwise gamma cosine across all organisms
    fn compute_field_coherence(&self) -> f64 {
        let n = self.organisms.len();
        if n < 2 { return 1.0; }
        let mut sum = 0.0;
        let mut count = 0;
        for i in 0..n {
            if self.organisms[i].gamma_direction.is_empty() { continue; }
            for j in (i+1)..n {
                if self.organisms[j].gamma_direction.is_empty() { continue; }
                let cos = cosine_similarity(&self.organisms[i].gamma_direction, &self.organisms[j].gamma_direction);
                sum += cos;
                count += 1;
            }
        }
        if count == 0 { 1.0 } else { sum / count as f64 }
    }

    /// Pairs with high cosine similarity (> threshold) — resonance detected
    fn detect_resonance(&self) -> Vec<(String, String, f64)> {
        let threshold = 0.8;
        let n = self.organisms.len();
        let mut pairs = Vec::new();
        for i in 0..n {
            if self.organisms[i].gamma_direction.is_empty() { continue; }
            for j in (i+1)..n {
                if self.organisms[j].gamma_direction.is_empty() { continue; }
                let cos = cosine_similarity(&self.organisms[i].gamma_direction, &self.organisms[j].gamma_direction);
                if cos > threshold {
                    pairs.push((self.organisms[i].id.clone(), self.organisms[j].id.clone(), cos));
                }
            }
        }
        pairs
    }

    /// Organisms drifting too fast (gamma_magnitude changing rapidly)
    fn detect_drift(&mut self, threshold: f64) -> Vec<String> {
        let mut drifting = Vec::new();
        for o in &self.organisms {
            if let Some(&prev_mag) = self.prev_magnitudes.get(&o.id) {
                if prev_mag.abs() > 1e-12 {
                    let rate = (o.gamma_magnitude - prev_mag).abs() / prev_mag;
                    if rate > threshold {
                        drifting.push(o.id.clone());
                    }
                }
            }
        }
        // Update prev_magnitudes for next check
        for o in &self.organisms {
            self.prev_magnitudes.insert(o.id.clone(), o.gamma_magnitude);
        }
        drifting
    }

    /// Self-reflection: (our_drift_rate, are_we_the_outlier)
    fn self_reflection(&self, our_id: &str) -> (f64, bool) {
        let us = self.organisms.iter().find(|o| o.id == our_id);
        if us.is_none() || self.organisms.len() < 2 {
            return (self.self_drift_rate, false);
        }
        let us = us.unwrap();
        if us.gamma_direction.is_empty() { return (self.self_drift_rate, false); }

        // Compute our average cosine to all others
        let mut our_cos_sum = 0.0;
        let mut our_count = 0;
        for o in &self.organisms {
            if o.id == our_id || o.gamma_direction.is_empty() { continue; }
            our_cos_sum += cosine_similarity(&us.gamma_direction, &o.gamma_direction);
            our_count += 1;
        }
        let our_avg_cos = if our_count > 0 { our_cos_sum / our_count as f64 } else { 1.0 };

        // We're an outlier if our avg cosine is much lower than field coherence
        let is_outlier = our_avg_cos < self.field_coherence - 0.3;

        (self.self_drift_rate, is_outlier)
    }

    /// Update self drift rate based on our current gamma direction
    fn update_self_drift(&mut self, current_gamma_dir: &[f64]) {
        if let Some(ref prev) = self.prev_gamma_direction {
            if prev.len() == current_gamma_dir.len() && !prev.is_empty() {
                let cos = cosine_similarity(prev, current_gamma_dir);
                // drift = 1 - cos (0 = no drift, 2 = full reversal)
                self.self_drift_rate = 1.0 - cos;
            }
        }
        self.prev_gamma_direction = Some(current_gamma_dir.to_vec());
    }
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() { return 0.0; }
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 { 0.0 } else { dot / denom }
}

// Background thread: topology monitoring loop
fn topology_monitor_thread(
    mesh_db_path: String,
    organism_id: String,
    model: Arc<Mutex<GPT>>,
    stop: Arc<AtomicBool>,
) {
    let check_interval = std::time::Duration::from_secs(30);
    let mut topo = TopologyMonitor::new();

    // Open our own read-only connection to mesh.db
    let db = match Connection::open(&mesh_db_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[topology] Failed to open mesh.db: {}", e);
            return;
        }
    };

    while !stop.load(Ordering::Relaxed) {
        thread::sleep(check_interval);

        topo.update(&db);

        // Update our self-drift from current model gamma
        if let Ok(m) = model.lock() {
            let (gamma_dir, _gamma_mag) = m.gamma_contrastive_projection();
            if !gamma_dir.is_empty() {
                topo.update_self_drift(&gamma_dir);
            }
        }

        // Log findings
        let resonance_pairs = topo.detect_resonance();
        if !resonance_pairs.is_empty() {
            for (a, b, cos) in &resonance_pairs {
                eprintln!("[topology] Resonance detected: {} <-> {} (cos={:.4})", a, b, cos);
            }
        }

        let drifters = topo.detect_drift(2.0);
        if !drifters.is_empty() {
            eprintln!("[topology] High-drift organisms: {:?}", drifters);
        }

        let (drift_rate, is_outlier) = topo.self_reflection(&organism_id);
        if is_outlier {
            eprintln!("[topology] WARNING: We are an outlier! drift_rate={:.4}, field_coherence={:.4}",
                drift_rate, topo.field_coherence);
        }

        if topo.organisms.len() > 1 {
            eprintln!("[topology] {} organisms, coherence={:.4}, self_drift={:.4}",
                topo.organisms.len(), topo.field_coherence, drift_rate);
        }
    }
}

// ============================================================
// 18) MAIN
// ============================================================

fn main() {
    eprintln!("╔══════════════════════════════════════════════════╗");
    eprintln!("║  MOLEQULA.RS — The Fifth Element                ║");
    eprintln!("║  GPT organism + distributed cognition metabolism ║");
    eprintln!("╚══════════════════════════════════════════════════╝");

    let cfg = Config::default();

    // Init SQLite
    let db = Arc::new(Mutex::new(init_db(&cfg.db_path)));

    // Load corpus
    let docs = load_corpus(&cfg.corpus_path, cfg.max_corpus_lines, cfg.max_line_chars);
    eprintln!("[init] Corpus: {} lines, {} chars", docs.len(),
        docs.iter().map(|d| d.len()).sum::<usize>());

    // Try loading checkpoint, else create new
    let mut model = if std::path::Path::new(&cfg.ckpt_path).exists() {
        match load_checkpoint(&cfg.ckpt_path) {
            Ok(m) => { eprintln!("[init] Loaded checkpoint, step={}", m.global_step); m },
            Err(e) => { eprintln!("[init] Checkpoint load failed: {}, starting fresh", e);
                        let mut tok = EvolvingTokenizer::new(&docs);
                        tok.maybe_enable_bpe(&docs, &cfg);
                        GPT::new(tok, &cfg) },
        }
    } else {
        let mut tok = EvolvingTokenizer::new(&docs);
        // Enable BPE before warmup — subword tokens for coherent corpus field
        tok.maybe_enable_bpe(&docs, &cfg);
        GPT::new(tok, &cfg)
    };

    // Build corpus field before init — sigmoid fade weakens it as model learns
    let mut init_corpus_field = CooccurField::new();
    init_corpus_field.build(&model.tok, &docs);

    // Per-stage warmup: grow model to match corpus size, warming up at each stage
    let corpus_chars: usize = docs.iter().map(|d| d.len()).sum();
    let target_stage = model.target_growth_stage(corpus_chars);
    let current_stage = model.current_growth_stage();
    if target_stage > current_stage as usize && !docs.is_empty() {
        let stage_names = ["embryo", "infant", "child", "adolescent", "teen", "adult"];
        eprintln!("[init] Per-stage warmup: current={}, target={}", current_stage, target_stage);
        loop {
            let stage = model.current_growth_stage();
            let stage_name = stage_names[stage.max(0) as usize % stage_names.len()];
            let embryo_embd = cfg.growth_stages[0][1].max(1);
            let warmup_scale = (model.n_embd / embryo_embd).max(1);
            let effective_warmup = cfg.warmup_steps * warmup_scale;
            eprintln!("[init] Stage {} ({}): embd={}, warmup {} steps (scaled {}x)",
                      stage, stage_name, model.n_embd, effective_warmup, warmup_scale);
            train_steps(&mut model, &docs, effective_warmup, true, true);
            model.last_warmup_stage = stage;
            save_checkpoint(&model, &cfg.ckpt_path).ok();
            if !model.maybe_grow_architecture(corpus_chars) { break; }
            model.growth_freeze_remaining = 0; // skip freeze during init growth
            // Rebuild corpus field after growth (vocab may have expanded)
            init_corpus_field.build(&model.tok, &docs);
        }
        eprintln!("[init] Per-stage warmup complete. Stage={}", model.current_growth_stage());
    }

    let model = Arc::new(Mutex::new(model));

    // Use the init corpus field (already built)
    let corpus_field = init_corpus_field;

    // Swarm
    let organism_id = format!("rust-{}", std::process::id());
    let swarm = Arc::new(Mutex::new(SwarmRegistry::new(&organism_id)));
    swarm.lock().unwrap().register().ok();
    eprintln!("[swarm] Registered as {}", organism_id);

    // Quantum buffer
    let qbuf = Arc::new(Mutex::new(QuantumBuffer::new()));

    // Background trainer
    let stop = Arc::new(AtomicBool::new(false));
    let trainer_handle = {
        let m = Arc::clone(&model);
        let d = Arc::clone(&db);
        let q = Arc::clone(&qbuf);
        let s = Arc::clone(&swarm);
        let st = Arc::clone(&stop);
        thread::spawn(move || background_trainer(m, d, q, s, st))
    };

    // Topology Monitor (Feature 6 — Rust-only)
    let topo_handle = {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        let mesh_path = format!("{}/.molequla/swarm/mesh.db", home);
        let m = Arc::clone(&model);
        let st = Arc::clone(&stop);
        let oid = organism_id.clone();
        thread::spawn(move || topology_monitor_thread(mesh_path, oid, m, st))
    };
    eprintln!("[topology] Monitor thread started (30s interval)");

    // Metabolism
    let mut metabolism = Metabolism::new(5);
    eprintln!("[metabolism] 4.C MLP initialized (5 elements, Hebbian)");

    // Corpus field needs to be shared for overthinkg
    let corpus_field = Arc::new(Mutex::new(corpus_field));

    // Chat loop
    eprintln!("[chat] Ready. Type something.");
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = match line { Ok(l) => l, Err(_) => break };
        let text = line.trim().to_string();
        if text.is_empty() { continue; }
        if text == "/quit" || text == "/exit" { break; }

        // Store user message
        db_add_message(&db.lock().unwrap(), "user", &text);

        // Feed quantum buffer
        { let m = model.lock().unwrap(); qbuf.lock().unwrap().feed(&text, &m.tok); }

        // Rebuild cooccur field with updated corpus
        let fresh_docs = load_corpus(&cfg.corpus_path, cfg.max_corpus_lines, cfg.max_line_chars);
        if !fresh_docs.is_empty() {
            let m = model.lock().unwrap();
            corpus_field.lock().unwrap().build(&m.tok, &fresh_docs);
        }

        // Enrich corpus field with user input (after rebuild, so it's not lost)
        // Active user word boost: organism absorbs user's vocabulary (Leo-style)
        // Decays each generation, fades with model strength via sigmoid in generate_sentence
        {
            let m = model.lock().unwrap();
            let user_ids = m.tok.encode(&text);
            let mut cf = corpus_field.lock().unwrap();
            cf.ingest_tokens(&user_ids);
            cf.absorb_user_words(&user_ids);
        }

        let prompt = build_prompt(&db.lock().unwrap(), &text);

        // Consciousness: self-prediction error (Feature 4)
        // "How surprised am I by this input?"
        {
            let mut m = model.lock().unwrap();
            set_grad(false);
            let prompt_ids = m.tok.encode(&prompt);
            if prompt_ids.len() > 2 {
                let surprise = m.compute_self_prediction_error(&prompt_ids);
                m.last_surprise = surprise;
                if m.surprise_baseline < 1e-6 {
                    m.surprise_baseline = surprise;
                } else {
                    m.surprise_baseline = 0.3 * surprise + 0.7 * m.surprise_baseline;
                }
            }
            set_grad(true);
        }

        // Generate response
        let answer = {
            let mut m = model.lock().unwrap();
            let cf = corpus_field.lock().unwrap();
            m.generate_sentence(&prompt, Some(&cf), &fresh_docs)
        };
        let answer = if answer.is_empty() { "...".to_string() } else { answer };

        // Consciousness: conscience check (Feature 5)
        // "Did my last generation feel coherent?"
        {
            let mut m = model.lock().unwrap();
            let last_ent = m.last_gen_entropy;
            if last_ent > 0.0 {
                m.conscience_check(last_ent);
            }
        }

        println!("{}", answer);
        db_add_message(&db.lock().unwrap(), "assistant", &answer);

        // Self-enrichment: own output enriches corpus field, weighted by coherence
        // Low entropy = coherent speech = higher weight (Stanley's resonance weighting)
        if answer.len() > 3 {
            let m = model.lock().unwrap();
            let last_ent = m.last_gen_entropy;
            let answer_ids = m.tok.encode(&answer);
            let mut self_weight = 1.0;
            if last_ent > 0.0 {
                self_weight = (2.0 - last_ent).clamp(0.3, 2.0);
            }
            let mut cf = corpus_field.lock().unwrap();
            cf.ingest_tokens_weighted(&answer_ids, self_weight);
            cf.decay_user_boost();
        }

        // Check mesh peers for metabolism
        if let Ok(sw) = swarm.lock() {
            let peers = sw.discover_peers();
            if !peers.is_empty() {
                let entropies: Vec<f64> = peers.iter().map(|p| p.2).collect();
                let weights = metabolism.compute_blend_weights(&entropies);
                eprintln!("[metabolism] {} peers, blend: {:?}", peers.len(),
                    weights.iter().map(|w| format!("{:.2}", w)).collect::<Vec<_>>());
            }
        }
    }

    // Shutdown
    eprintln!("[shutdown] Saving...");
    stop.store(true, Ordering::Relaxed);
    trainer_handle.join().ok();
    topo_handle.join().ok();
    swarm.lock().unwrap().unregister();
    eprintln!("[shutdown] Done. The fifth element rests.");
}
