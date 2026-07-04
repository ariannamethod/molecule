package main

/*
#cgo CFLAGS: -I/usr/local/include/ariannamethod -O2
#cgo darwin CFLAGS: -I/opt/homebrew/include
#cgo linux CFLAGS: -DUSE_BLAS -I/usr/include/x86_64-linux-gnu/openblas-pthread/
#include <notorch.h>
#include <string.h>

// Copy a Go float32 slice into a tensor's data buffer.
static void ntx_set(nt_tensor* t, float* src, int n) {
    if (t && src && n > 0 && n <= t->len) memcpy(t->data, src, (size_t)n * sizeof(float));
}
// Copy a tensor's data buffer out into a caller buffer.
static void ntx_get(nt_tensor* t, float* dst, int n) {
    if (t && dst && n > 0 && n <= t->len) memcpy(dst, t->data, (size_t)n * sizeof(float));
}
// Read the scalar at a tape entry's output (loss lives at output->data[0]).
static float ntx_entry_scalar(int idx) {
    nt_tape* tp = nt_tape_get();
    if (!tp || idx < 0 || idx >= tp->count) return 0.0f;
    nt_tensor* o = tp->entries[idx].output;
    return (o && o->len > 0) ? o->data[0] : 0.0f;
}
// Copy a tape entry's full output tensor out (for op-output parity checks).
static void ntx_entry_data(int idx, float* dst, int n) {
    nt_tape* tp = nt_tape_get();
    if (!tp || idx < 0 || idx >= tp->count || !dst) return;
    nt_tensor* o = tp->entries[idx].output;
    if (!o) return;
    int m = (n < o->len) ? n : o->len;
    for (int i = 0; i < m; i++) dst[i] = o->data[i];
}
*/
import "C"
import "unsafe"

// ═══════════════════════════════════════════════════════════════════════════════
// CGO bridge to notorch — molequla's training path.
//
// AML stays the inference / field-physics language (cgo_aml.go); notorch is how
// the organism *learns* — fast C tape autograd, BLAS, optional CUDA. This bridge
// is training-only. Op semantics mirror notorch/examples/train_llama3_bpe.c.
//
// GPU mode (nt_set_gpu_mode / gpu_init) is compiled out of libnotorch without
// USE_CUDA, so it is NOT bound here — it lives in a `cuda`-tagged file.
// ═══════════════════════════════════════════════════════════════════════════════

// ntTensor is an opaque handle to a notorch nt_tensor.
type ntTensor = *C.nt_tensor

func ntTensorNew(length int) ntTensor       { return C.nt_tensor_new(C.int(length)) }
func ntTensorNew2D(rows, cols int) ntTensor { return C.nt_tensor_new2d(C.int(rows), C.int(cols)) }
func ntTensorFree(t ntTensor)               { C.nt_tensor_free(t) }

// ntTensorSet copies a Go float32 slice into the tensor (len ≤ tensor len).
func ntTensorSet(t ntTensor, data []float32) {
	if t == nil || len(data) == 0 {
		return
	}
	C.ntx_set(t, (*C.float)(unsafe.Pointer(&data[0])), C.int(len(data)))
}

// ntTensorGet copies the tensor's first n floats out into a fresh Go slice.
func ntTensorGet(t ntTensor, n int) []float32 {
	if t == nil || n <= 0 {
		return nil
	}
	out := make([]float32, n)
	C.ntx_get(t, (*C.float)(unsafe.Pointer(&out[0])), C.int(n))
	return out
}

// ntTensorSyncCPU refreshes a tensor's host mirror (t->data) from device before
// a host read. ntx_get does a raw memcpy from t->data, so on the cuda build a
// GPU-trained tensor must be synced first or the readback into model.Base is
// stale. No-op on the non-cuda build (nt_tensor_sync_cpu is (void)t there).
func ntTensorSyncCPU(t ntTensor) {
	if t == nil {
		return
	}
	C.nt_tensor_sync_cpu(t)
}

// ── Tape lifecycle ──
// nt_tape_clear keeps Chuck m/v state (positional, keyed by registration order);
// nt_tape_destroy wipes it — call destroy+start only after a growth event.

func ntTapeStart()   { C.nt_tape_start() }
func ntTapeClear()   { C.nt_tape_clear() }
func ntTapeDestroy() { C.nt_tape_destroy() }

// ntTapeParam registers a trainable tensor, returns its tape index. The caller
// MUST register params in a byte-identical order every burst (an explicitly
// ordered slice — never a Go map range) so Chuck's positional m/v slots stay
// bound to the same tensor.
func ntTapeParam(t ntTensor) int { return int(C.nt_tape_param(t)) }
func ntTapeNoDecay(idx int)      { C.nt_tape_no_decay(C.int(idx)) }

// ntTapeInput records a non-trainable input tensor (tokens / targets) on the tape.
func ntTapeInput(t ntTensor) int {
	return int(C.nt_tape_record(t, C.NT_OP_NONE, -1, -1, 0))
}

func ntTapeBackward(lossIdx int)              { C.nt_tape_backward(C.int(lossIdx)) }
func ntTapeClipGrads(maxNorm float64) float64 { return float64(C.nt_tape_clip_grads(C.float(maxNorm))) }
func ntTapeChuckStep(lr, lossVal float64)     { C.nt_tape_chuck_step(C.float(lr), C.float(lossVal)) }

// ntEntryScalar reads output->data[0] of a tape entry (e.g. the loss).
func ntEntryScalar(idx int) float64 { return float64(C.ntx_entry_scalar(C.int(idx))) }

// ntEntryData copies a tape entry's full output tensor into a Go slice (first n
// floats). Used to compare an op's output against a Go reimplementation.
func ntEntryData(idx, n int) []float32 {
	if n <= 0 {
		return nil
	}
	out := make([]float32, n)
	C.ntx_entry_data(C.int(idx), (*C.float)(unsafe.Pointer(&out[0])), C.int(n))
	return out
}

// ── NaN/Inf guard ──

type ntNanGuard struct{ g C.nt_nan_guard }

func newNTNanGuard() ntNanGuard { return ntNanGuard{g: C.nt_nan_guard_new()} }

// check returns true if grads are clean, false if NaN/Inf was detected (grads zeroed).
func (n *ntNanGuard) check() bool { return C.nt_nan_guard_check(&n.g) != 0 }

// ── Forward ops — each records on the tape and returns a tape entry index ──

func ntSeqEmbedding(wteIdx, wpeIdx, tokensIdx, T, D int) int {
	return int(C.nt_seq_embedding(C.int(wteIdx), C.int(wpeIdx), C.int(tokensIdx), C.int(T), C.int(D)))
}
func ntRope(xIdx, T, headDim int) int {
	return int(C.nt_rope(C.int(xIdx), C.int(T), C.int(headDim)))
}
func ntSeqRMSNorm(xIdx, gammaIdx, T, D int) int {
	return int(C.nt_seq_rmsnorm(C.int(xIdx), C.int(gammaIdx), C.int(T), C.int(D)))
}
func ntSeqLinear(wIdx, xIdx, T int) int {
	return int(C.nt_seq_linear(C.int(wIdx), C.int(xIdx), C.int(T)))
}
func ntMHCausalAttention(qIdx, kIdx, vIdx, T, headDim int) int {
	return int(C.nt_mh_causal_attention(C.int(qIdx), C.int(kIdx), C.int(vIdx), C.int(T), C.int(headDim)))
}
func ntAdd(aIdx, bIdx int) int { return int(C.nt_add(C.int(aIdx), C.int(bIdx))) }
func ntMul(aIdx, bIdx int) int { return int(C.nt_mul(C.int(aIdx), C.int(bIdx))) }
func ntSilu(xIdx int) int      { return int(C.nt_silu(C.int(xIdx))) }
func ntSeqCrossEntropy(logitsIdx, targetsIdx, T, V int) int {
	return int(C.nt_seq_cross_entropy(C.int(logitsIdx), C.int(targetsIdx), C.int(T), C.int(V)))
}

// ── Mode ──

func ntTrainMode(on bool) {
	if on {
		C.nt_train_mode(1)
	} else {
		C.nt_train_mode(0)
	}
}

func ntSeed(s uint64) { C.nt_seed(C.uint64_t(s)) }

// ── Increment 2: low-rank RRPRAM (Resonance form, op 33) ──
// molequla's never-trained position-bias w_pattern is replaced by notorch's
// proven low-rank attention. Reference: notorch examples/train_resonance_lora.c.

// ntRrpramLowrankAttention — op 33, packed low-rank RRPRAM attention.
// wrCombined packs Wr_a[H,E,R] then Wr_b[H,R,T_r] in one tensor (T_r == T);
// rank is derived as len/(H·(E+T_r)). xIdx is the post-RMSNorm hidden (full E),
// vIdx the same value tensor the content head uses. Returns the post-softmax
// attention-weighted V, shape [T × (nrHeads·headDim)] — same layout as content.
func ntRrpramLowrankAttention(wrCombinedIdx, xIdx, vIdx, T, nEmbd, nrHeads, headDim int) int {
	return int(C.nt_rrpram_lowrank_attention(C.int(wrCombinedIdx), C.int(xIdx), C.int(vIdx),
		C.int(T), C.int(nEmbd), C.int(nrHeads), C.int(headDim)))
}

// ntTapeParamFrozen registers a tensor as a FROZEN tape param: it takes part in
// the forward and gradient flows through it, but the optimizer step skips it.
// Used for the precomputed per-head gate vectors (g_sig / g_one) — the gate is
// frozen this increment, which keeps sigmoid/scale_by_t off the tape and so
// sidesteps the notorch GPU-sync bug class on those ops.
func ntTapeParamFrozen(t ntTensor) int { return int(C.nt_tape_param_frozen(t)) }
