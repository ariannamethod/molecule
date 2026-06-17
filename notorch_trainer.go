package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// ═══════════════════════════════════════════════════════════════════════════════
// notorch trainer — molequla's content transformer trained on notorch's C tape.
//
// Replaces the AML-interpreter path (aml_trainer.go): instead of running the
// transformer as a re-parsed AML script per step on the AML core's CPU
// autograd, the model is built once in notorch ops and trained on notorch's
// compiled tape (BLAS, optional CUDA), Chuck optimizer. See
// 06_PLAN_gpu_training.md, Increment 1.
//
// Content model only — RoPE + MHA + SwiGLU, non-parametric RMSNorm. molequla's
// dormant RRPRAM / hybrid heads are Increment 2. model.Base stays the canonical
// float64 weight store; per burst it is mirrored into notorch tensors and back.
//
// Naming: every symbol here is nt-prefixed — molequla already has a (disabled)
// `notorchTrainSteps` Hebbian stub; these must not collide with it.
// ═══════════════════════════════════════════════════════════════════════════════

// ntTapeNeedsReset is set after a growth event (Net2Net changed dims) so the
// next burst wipes the positional Chuck moment slots before training — old
// slots are meaningless once the param set changes (06_PLAN §6, audit S1).
var ntTapeNeedsReset bool

// ntOnGrowth signals the notorch trainer to reset its tape state before the
// next burst. Call whenever MaybeGrowArchitecture has grown the model.
func ntOnGrowth() { ntTapeNeedsReset = true }

// ntOrderedParam pairs a model.Base weight with its key. The slice order is
// fixed and deterministic — Chuck moment slots are positional (keyed by
// registration order), so registration MUST be byte-identical every burst.
// Never derive this from a Go map range (map iteration is randomized).
type ntOrderedParam struct {
	name string
	mp   *MatrixParam
}

// ntContentParams returns the content-transformer weights in a fixed order:
// wte, then per layer {wq,wk,wv,wo,fc_g,fc_v,fc2}, then lm_head. wpe is omitted
// — the trainer uses RoPE for position (06_PLAN §6, audit #3).
func ntContentParams(model *GPT) []ntOrderedParam {
	out := make([]ntOrderedParam, 0, 2+7*model.NLayer)
	out = append(out, ntOrderedParam{"wte", model.Base["wte"]})
	for l := 0; l < model.NLayer; l++ {
		pfx := fmt.Sprintf("l%d.", l)
		for _, suf := range []string{"wq", "wk", "wv", "wo", "fc_g", "fc_v", "fc2"} {
			out = append(out, ntOrderedParam{pfx + suf, model.Base[pfx+suf]})
		}
	}
	out = append(out, ntOrderedParam{"lm_head", model.Base["lm_head"]})
	return out
}

// ntFlattenMatrix copies a MatrixParam (Nout×Nin float64) into a row-major
// float32 slice — the layout notorch tensors expect.
func ntFlattenMatrix(mp *MatrixParam) []float32 {
	flat := make([]float32, mp.Nout*mp.Nin)
	for i := 0; i < mp.Nout && i < len(mp.Rows); i++ {
		if mp.Rows[i] == nil {
			continue
		}
		row := mp.Rows[i].Data
		for j := 0; j < mp.Nin && j < len(row); j++ {
			flat[i*mp.Nin+j] = float32(row[j])
		}
	}
	return flat
}

// ntUnflattenMatrix writes a row-major float32 slice back into a MatrixParam.
func ntUnflattenMatrix(mp *MatrixParam, flat []float32) {
	for i := 0; i < mp.Nout && i < len(mp.Rows); i++ {
		if mp.Rows[i] == nil {
			continue
		}
		row := mp.Rows[i].Data
		for j := 0; j < mp.Nin && j < len(row); j++ {
			if i*mp.Nin+j < len(flat) {
				row[j] = float64(flat[i*mp.Nin+j])
			}
		}
	}
}

// ntBuildForward builds molequla's transformer on the active notorch tape and
// returns the cross-entropy loss tape index. pIdx holds the CONTENT param tape
// indices in ntContentParams order (unchanged from Inc1 — wte, 7·NLayer content,
// lm_head). wrIdx/gateCIdx/gateRIdx are the Inc2 per-layer RRPRAM tape indices
// (registered separately, AFTER content, so content Chuck slots are untouched —
// B1); a layer with no hybrid head has wrIdx[l] < 0. tokIdx/tgtIdx are inputs.
func ntBuildForward(model *GPT, pIdx, wrIdx, gateCIdx, gateRIdx []int, tokIdx, tgtIdx, T, vocab int) int {
	D := model.NEmbd
	headDim := D / model.NHead
	wte := pIdx[0]
	lmHead := pIdx[len(pIdx)-1]

	h := ntSeqEmbedding(wte, -1, tokIdx, T, D) // WTE only — RoPE handles position
	for l := 0; l < model.NLayer; l++ {
		b := 1 + l*7
		wq, wk, wv, wo := pIdx[b], pIdx[b+1], pIdx[b+2], pIdx[b+3]
		fcG, fcV, fc2 := pIdx[b+4], pIdx[b+5], pIdx[b+6]

		hn := ntSeqRMSNorm(h, -1, T, D) // gamma -1 → non-parametric (matches molequla)
		q := ntRope(ntSeqLinear(wq, hn, T), T, headDim)
		k := ntRope(ntSeqLinear(wk, hn, T), T, headDim)
		v := ntSeqLinear(wv, hn, T)
		attn := ntMHCausalAttention(q, k, v, T, headDim)

		// Inc2: low-rank RRPRAM head (Resonance form, op 33), output-level blend.
		// rrpram_out = (xn @ Wr_a) @ Wr_b → causal softmax → @ v, packed over all
		// heads (full D input, same v as content). Per-head frozen gate masks
		// content-only heads (gateR=0) and weights hybrid heads by sigmoid(alpha):
		//   out = gateC ⊙ content_out + gateR ⊙ rrpram_out
		if l < len(wrIdx) && wrIdx[l] >= 0 {
			rAttn := ntRrpramLowrankAttention(wrIdx[l], hn, v, T, D, model.NHead, headDim)
			attn = ntAdd(ntMul(attn, gateCIdx[l]), ntMul(rAttn, gateRIdx[l]))
		}
		h = ntAdd(h, ntSeqLinear(wo, attn, T))

		hn = ntSeqRMSNorm(h, -1, T, D)
		gate := ntSilu(ntSeqLinear(fcG, hn, T))
		up := ntSeqLinear(fcV, hn, T)
		h = ntAdd(h, ntSeqLinear(fc2, ntMul(gate, up), T))
	}
	hf := ntSeqRMSNorm(h, -1, T, D)
	logits := ntSeqLinear(lmHead, hf, T)
	return ntSeqCrossEntropy(logits, tgtIdx, T, vocab)
}

// ntPackWr flattens the per-layer factors wr_a [NHead·NEmbd × R] then
// wr_b [NHead·R × BlockSize] into the single combined buffer notorch op-33 reads
// (all Wr_a then all Wr_b, row-major). Returns nil if the layer has no factors.
func ntPackWr(model *GPT, l int) []float32 {
	a := model.Base[fmt.Sprintf("l%d.wr_a", l)]
	b := model.Base[fmt.Sprintf("l%d.wr_b", l)]
	if a == nil || b == nil {
		return nil
	}
	out := make([]float32, 0, a.Nout*a.Nin+b.Nout*b.Nin)
	out = append(out, ntFlattenMatrix(a)...)
	out = append(out, ntFlattenMatrix(b)...)
	return out
}

// ntUnpackWr splits a trained combined buffer back into the wr_a / wr_b Base
// matrices (the canonical Go-side store), mirroring ntPackWr.
func ntUnpackWr(model *GPT, l int, combined []float32) {
	a := model.Base[fmt.Sprintf("l%d.wr_a", l)]
	b := model.Base[fmt.Sprintf("l%d.wr_b", l)]
	if a == nil || b == nil {
		return
	}
	aLen := a.Nout * a.Nin
	if aLen > len(combined) {
		return
	}
	ntUnflattenMatrix(a, combined[:aLen])
	ntUnflattenMatrix(b, combined[aLen:])
}

// ntBuildGateVectors returns the per-head output-level blend masks for layer l,
// each of length T·NEmbd. content-out weight gateC = 1 - g_h, rrpram-out weight
// gateR = g_h, where g_h = sigmoid(alpha_{l,h}) for a hybrid head and 0 for a
// content head (so content heads stay pure content and their factors get zero
// gradient). The gate is FROZEN this increment — sigmoid is precomputed Go-side,
// keeping nt_sigmoid/nt_scale_by_t off the tape (notorch GPU-sync bug class).
func ntBuildGateVectors(model *GPT, l, T int) (gateC, gateR []float32) {
	D := model.NEmbd
	hd := model.HeadDim
	htypes := headTypesForNHead(model.NHead)
	hg := make([]float32, model.NHead)
	for h := 0; h < model.NHead; h++ {
		g := float32(0.0) // content head → pure content (gateR = 0)
		if h < len(htypes) && (htypes[h] == "hybrid" || htypes[h] == "rrpram") {
			if mp := model.Base[fmt.Sprintf("l%d.h%d.alpha", l, h)]; mp != nil &&
				len(mp.Rows) > 0 && len(mp.Rows[0].Data) > 0 {
				g = float32(1.0 / (1.0 + math.Exp(-mp.Rows[0].Data[0]))) // sigmoid(alpha)
			} else {
				g = float32(1.0 / (1.0 + math.Exp(-CFG.HybridAlphaInit)))
			}
		}
		hg[h] = g
	}
	gateC = make([]float32, T*D)
	gateR = make([]float32, T*D)
	for t := 0; t < T; t++ {
		for h := 0; h < model.NHead; h++ {
			for d := 0; d < hd; d++ {
				idx := t*D + h*hd + d
				gateR[idx] = hg[h]
				gateC[idx] = 1.0 - hg[h]
			}
		}
	}
	return gateC, gateR
}

// ntTrainCore runs `steps` training steps of molequla's content model on
// notorch. lrFor(step) supplies the per-step learning rate. Caller holds
// model.mu. Returns (avg loss, counted steps).
// Returns (avg loss, counted steps, step-loop wall ms) — the wall time is the
// pure training cost, criterion-2 metric (06_PLAN §11.2), measured over the
// step loop only, excluding the per-burst weight mirror in/out.
func ntTrainCore(model *GPT, tok *EvolvingTokenizer, docs []string, steps, seqLen int, lrFor func(int) float64) (float64, int, float64) {
	if len(docs) == 0 || steps <= 0 {
		return 0, 0, 0
	}
	vocab := tok.VocabSize
	// Inc2 (B3): op-33 assumes T_r == T and the combined Wr is packed at width
	// BlockSize, so RRPRAM-bearing bursts MUST run at T = BlockSize (this also
	// satisfies the documented gpu_rrpram_lr T-vs-T_max stride workaround). Pin it.
	hasRRPRAM := layerHasHybrid()
	if hasRRPRAM {
		seqLen = model.BlockSize
	}
	params := ntContentParams(model)

	// Mirror model.Base weights into notorch tensors (created once per burst).
	tensors := make([]ntTensor, len(params))
	for i, p := range params {
		t := ntTensorNew2D(p.mp.Nout, p.mp.Nin)
		ntTensorSet(t, ntFlattenMatrix(p.mp))
		tensors[i] = t
	}
	defer func() {
		for _, t := range tensors {
			ntTensorFree(t)
		}
	}()

	// Inc2: per-layer combined Wr tensors (packed wr_a++wr_b) + frozen per-head
	// gate vectors, created once per burst (the gate is frozen — alpha does not
	// train this increment, so the vectors are constant across the burst).
	wrTensors := make([]ntTensor, model.NLayer) // nil when a layer has no factors
	gateCT := make([]ntTensor, model.NLayer)
	gateRT := make([]ntTensor, model.NLayer)
	if hasRRPRAM {
		for l := 0; l < model.NLayer; l++ {
			combined := ntPackWr(model, l)
			if combined == nil {
				continue
			}
			wt := ntTensorNew(len(combined))
			ntTensorSet(wt, combined)
			wrTensors[l] = wt
			gc, gr := ntBuildGateVectors(model, l, seqLen)
			gct := ntTensorNew(len(gc))
			ntTensorSet(gct, gc)
			grt := ntTensorNew(len(gr))
			ntTensorSet(grt, gr)
			gateCT[l] = gct
			gateRT[l] = grt
		}
		defer func() {
			for l := 0; l < model.NLayer; l++ {
				if wrTensors[l] != nil {
					ntTensorFree(wrTensors[l])
				}
				if gateCT[l] != nil {
					ntTensorFree(gateCT[l])
				}
				if gateRT[l] != nil {
					ntTensorFree(gateRT[l])
				}
			}
		}()
	}

	// Post-growth: wipe positional Chuck slots before the first step (S1).
	if ntTapeNeedsReset {
		ntTapeDestroy()
		ntTapeNeedsReset = false
	}

	guard := newNTNanGuard()
	tokBuf := make([]float32, seqLen)
	tgtBuf := make([]float32, seqLen)
	var lossSum float64
	var lossN int

	t0 := time.Now()
	for step := 0; step < steps; step++ {
		ids := tok.Encode(docs[rand.Intn(len(docs))])
		if len(ids) < 2 {
			continue
		}
		start := 0
		if len(ids) > seqLen+1 {
			start = rand.Intn(len(ids) - seqLen - 1)
		}
		for i := 0; i < seqLen; i++ {
			idx := start + i
			if idx < len(ids) {
				tokBuf[i] = float32(ids[idx])
			} else {
				tokBuf[i] = 0
			}
			if idx+1 < len(ids) {
				tgtBuf[i] = float32(ids[idx+1])
			} else {
				tgtBuf[i] = 0
			}
		}

		ntTapeStart()
		// Register params in the fixed ntContentParams order (B1).
		pIdx := make([]int, len(tensors))
		for i, t := range tensors {
			pIdx[i] = ntTapeParam(t)
		}
		ntTapeNoDecay(pIdx[0]) // wte — no weight decay on embeddings

		// Inc2: register the combined Wr (trainable, AFTER the content params so
		// content Chuck slots keep their identity — B1) and the frozen gate
		// vectors (frozen → consume no param slot). wrIdx[l] < 0 → layer has none.
		wrIdx := make([]int, model.NLayer)
		gateCIdx := make([]int, model.NLayer)
		gateRIdx := make([]int, model.NLayer)
		for l := 0; l < model.NLayer; l++ {
			wrIdx[l] = -1
			if wrTensors[l] == nil {
				continue
			}
			wrIdx[l] = ntTapeParam(wrTensors[l])
			ntTapeNoDecay(wrIdx[l]) // low-rank factors: no weight decay (double-shrink)
			gateCIdx[l] = ntTapeParamFrozen(gateCT[l])
			gateRIdx[l] = ntTapeParamFrozen(gateRT[l])
		}

		tokT := ntTensorNew(seqLen)
		ntTensorSet(tokT, tokBuf)
		tgtT := ntTensorNew(seqLen)
		ntTensorSet(tgtT, tgtBuf)
		tokIdx := ntTapeInput(tokT)
		tgtIdx := ntTapeInput(tgtT)
		ntTensorFree(tokT)
		ntTensorFree(tgtT)

		lossIdx := ntBuildForward(model, pIdx, wrIdx, gateCIdx, gateRIdx, tokIdx, tgtIdx, seqLen, vocab)
		loss := ntEntryScalar(lossIdx)
		ntTapeBackward(lossIdx)
		if guard.check() {
			ntTapeClipGrads(1.0)
			ntTapeChuckStep(lrFor(step), loss)
		}
		ntTapeClear()

		if !math.IsNaN(loss) && !math.IsInf(loss, 0) {
			lossSum += loss
			lossN++
		}
		model.globalStep++
	}
	elapsedMs := float64(time.Since(t0).Microseconds()) / 1000.0

	// Mirror trained weights back into the canonical model.Base store.
	// Sync each tensor's host mirror from device first — ntx_get is a raw memcpy
	// from t->data, which is stale after GPU training on the cuda build (no-op on CPU).
	for i, p := range params {
		ntTensorSyncCPU(tensors[i])
		ntUnflattenMatrix(p.mp, ntTensorGet(tensors[i], p.mp.Nout*p.mp.Nin))
	}
	// Inc2: split the trained combined Wr back into the wr_a / wr_b Base matrices.
	if hasRRPRAM {
		for l := 0; l < model.NLayer; l++ {
			if wrTensors[l] == nil {
				continue
			}
			a := model.Base[fmt.Sprintf("l%d.wr_a", l)]
			b := model.Base[fmt.Sprintf("l%d.wr_b", l)]
			if a == nil || b == nil {
				continue
			}
			ntTensorSyncCPU(wrTensors[l])
			ntUnpackWr(model, l, ntTensorGet(wrTensors[l], a.Nout*a.Nin+b.Nout*b.Nin))
		}
	}
	if lossN > 0 {
		return lossSum / float64(lossN), lossN, elapsedMs
	}
	return 0, 0, elapsedMs
}

// ntBurstTrain — ecology micro-burst on the notorch path. Mirrors amlBurstTrain
// (aml_trainer.go:252): fixed burst LR scaled by embryo/current embd.
func ntBurstTrain(model *GPT, tok *EvolvingTokenizer, docs []string, steps int, burstLR float64) {
	if CFG.Trainer == "aml" {
		amlBurstTrain(model, tok, docs, steps, burstLR)
		return
	}
	if len(docs) == 0 || steps <= 0 {
		return
	}
	model.mu.Lock()
	defer model.mu.Unlock()
	embryoEmbd := CFG.GrowthStages[0][1]
	lr := burstLR * float64(embryoEmbd) / float64(model.NEmbd)
	avg, n, ms := ntTrainCore(model, tok, docs, steps, model.BlockSize, func(int) float64 { return lr })
	if model.growthFreezeRemaining > 0 {
		model.growthFreezeRemaining -= steps
		if model.growthFreezeRemaining < 0 {
			model.growthFreezeRemaining = 0
		}
	}
	if n > 0 {
		fmt.Printf("[notorch] burst complete: %d steps, avg loss %.4f | %.0fms %.1f steps/s | gpu-dispatch=%d\n",
			steps, avg, ms, ntStepsPerSec(n, ms), ntGPUDispatchCount())
	}
}

// ntStepsPerSec — steps/sec from a counted-step total and wall ms (criterion 2).
func ntStepsPerSec(n int, ms float64) float64 {
	if ms <= 0 {
		return 0
	}
	return float64(n) / (ms / 1000.0)
}

// ntWarmupTrain — per-stage warmup on the notorch path. Mirrors amlTrainSteps
// (aml_trainer.go:139): cosine LR driven by molequla's cosineLR (so the
// post-growth Chuck-state reset, S1, costs no LR-schedule continuity — the
// schedule lives in cosineLR, not in Chuck's internal macro counter).
func ntWarmupTrain(model *GPT, tok *EvolvingTokenizer, docs []string, steps int, overrides ...int) {
	if CFG.Trainer == "aml" {
		amlTrainSteps(model, tok, docs, steps, overrides...)
		return
	}
	if len(docs) == 0 || steps <= 0 {
		return
	}
	model.mu.Lock()
	defer model.mu.Unlock()
	seqLen := model.BlockSize
	if len(overrides) > 0 && overrides[0] > 0 && overrides[0] < seqLen {
		seqLen = overrides[0]
	}
	embryoEmbd := CFG.GrowthStages[0][1]
	g0 := model.globalStep
	lrFor := func(step int) float64 {
		gs := g0 + step
		lr := cosineLR(gs, gs-model.growthStepOffset)
		lr *= float64(embryoEmbd) / float64(model.NEmbd)
		if model.growthFreezeRemaining > 0 {
			lr *= CFG.PostGrowthLRScale
		}
		return lr
	}
	avg, n, ms := ntTrainCore(model, tok, docs, steps, seqLen, lrFor)
	if model.growthFreezeRemaining > 0 {
		model.growthFreezeRemaining -= steps
		if model.growthFreezeRemaining < 0 {
			model.growthFreezeRemaining = 0
		}
	}
	if n > 0 {
		fmt.Printf("[notorch] warmup complete: %d steps, avg loss %.4f | %.0fms %.1f steps/s | gpu-dispatch=%d\n",
			steps, avg, ms, ntStepsPerSec(n, ms), ntGPUDispatchCount())
	}
}
