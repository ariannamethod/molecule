// macrogpt.go
// A dependency-free*, single-file, goroutine-powered, continually-learning GPT organism.
// Go port of macrogpt.py — same architecture, same checkpoint format, 100x faster.
//
// * "dependency-free" = no PyTorch, no numpy, no C. One Go dep: modernc.org/sqlite (pure Go).
//
// In the beginning there was nonames.txt.
// And it was good. Mostly. Sometimes cursed.

package main

import (
	"bufio"
	"database/sql"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
	"sync"
	"time"

	_ "modernc.org/sqlite"
)

// ============================================================
// 0) CONFIG — bend reality here (carefully, mortals)
// ============================================================

type Config struct {
	// data
	CorpusPath     string  `json:"corpus_path"`
	DBPath         string  `json:"db_path"`
	CkptPath       string  `json:"ckpt_path"`
	MaxCorpusLines int     `json:"max_corpus_lines"`
	MaxLineChars   int     `json:"max_line_chars"`
	MinNewChars    int     `json:"min_new_chars_to_train"`

	// model
	TieEmbeddings bool `json:"tie_embeddings"`
	NLayer        int  `json:"n_layer"`
	NEmbd         int  `json:"n_embd"`
	NHead         int  `json:"n_head"`
	BlockSize     int  `json:"block_size"`

	// training
	WarmupSteps         int     `json:"warmup_steps"`
	MicroSteps          int     `json:"micro_steps"`
	LearningRate        float64 `json:"learning_rate"`
	Beta1               float64 `json:"beta1"`
	Beta2               float64 `json:"beta2"`
	EpsAdam             float64 `json:"eps_adam"`
	GradClip            float64 `json:"grad_clip"`
	FreezeBaseAfterWarm bool    `json:"freeze_base_after_warmup"`

	// deltas
	DeltaRank      int     `json:"delta_rank"`
	MaxDeltaModules int    `json:"max_delta_modules"`
	DeltaGrowProb  float64 `json:"delta_grow_prob"`

	// generation
	Temperature    float64 `json:"temperature"`
	TopK           int     `json:"top_k"`
	TopP           float64 `json:"top_p"`
	MaxGenTokens   int     `json:"max_gen_tokens"`
	MinGenTokens   int     `json:"min_gen_tokens"`
	RepetitionGuard int    `json:"repetition_guard"`

	// tokenizer evolution
	EnableBPEAfterChars  int `json:"enable_bpe_after_chars"`
	BPENumMerges         int `json:"bpe_num_merges"`
	BPERetrainEveryChars int `json:"bpe_retrain_every_chars"`

	// async
	TrainTickSeconds float64 `json:"train_tick_seconds"`
}

var CFG = Config{
	CorpusPath:           "nonames.txt",
	DBPath:               "memory.sqlite3",
	CkptPath:             "macrogpt_ckpt.json",
	MaxCorpusLines:       8000,
	MaxLineChars:         240,
	MinNewChars:          480,
	TieEmbeddings:        true,
	NLayer:               2,
	NEmbd:                72,
	NHead:                4,
	BlockSize:            96,
	WarmupSteps:          1200,
	MicroSteps:           32,
	LearningRate:         0.01,
	Beta1:                0.9,
	Beta2:                0.99,
	EpsAdam:              1e-8,
	GradClip:             1.0,
	FreezeBaseAfterWarm:  true,
	DeltaRank:            8,
	MaxDeltaModules:      12,
	DeltaGrowProb:        0.08,
	Temperature:          0.85,
	TopK:                 40,
	TopP:                 0.92,
	MaxGenTokens:         180,
	MinGenTokens:         16,
	RepetitionGuard:      4,
	EnableBPEAfterChars:  25000,
	BPENumMerges:         384,
	BPERetrainEveryChars: 4000,
	TrainTickSeconds:     0.25,
}

// ============================================================
// 1) AUTOGRAD — vectors, not scalar confetti
// ============================================================

// Node is anything in the autograd compute graph.
type Node interface {
	getChildren() []Node
	doBackward()
}

// Vec is a differentiable vector. One object = one embedding / hidden state.
type Vec struct {
	Data     []float64
	Grad     []float64
	children []Node
	backFn   func()
}

func NewVec(data []float64) *Vec {
	g := make([]float64, len(data))
	return &Vec{Data: data, Grad: g}
}

func NewVecZero(n int) *Vec {
	return NewVec(make([]float64, n))
}

func (v *Vec) getChildren() []Node { return v.children }
func (v *Vec) doBackward() {
	if v.backFn != nil {
		v.backFn()
	}
}

// Add returns a new Vec = self + other (element-wise).
func (v *Vec) Add(other *Vec) *Vec {
	n := len(v.Data)
	d := make([]float64, n)
	for i := 0; i < n; i++ {
		d[i] = v.Data[i] + other.Data[i]
	}
	out := NewVec(d)
	out.children = []Node{v, other}
	out.backFn = func() {
		for i := 0; i < n; i++ {
			v.Grad[i] += out.Grad[i]
			other.Grad[i] += out.Grad[i]
		}
	}
	return out
}

// Sub returns a new Vec = self - other.
func (v *Vec) Sub(other *Vec) *Vec {
	n := len(v.Data)
	d := make([]float64, n)
	for i := 0; i < n; i++ {
		d[i] = v.Data[i] - other.Data[i]
	}
	out := NewVec(d)
	out.children = []Node{v, other}
	out.backFn = func() {
		for i := 0; i < n; i++ {
			v.Grad[i] += out.Grad[i]
			other.Grad[i] -= out.Grad[i]
		}
	}
	return out
}

// Neg returns -self.
func (v *Vec) Neg() *Vec {
	n := len(v.Data)
	d := make([]float64, n)
	for i := 0; i < n; i++ {
		d[i] = -v.Data[i]
	}
	out := NewVec(d)
	out.children = []Node{v}
	out.backFn = func() {
		for i := 0; i < n; i++ {
			v.Grad[i] -= out.Grad[i]
		}
	}
	return out
}

// MulVec returns element-wise product self * other.
func (v *Vec) MulVec(other *Vec) *Vec {
	n := len(v.Data)
	d := make([]float64, n)
	for i := 0; i < n; i++ {
		d[i] = v.Data[i] * other.Data[i]
	}
	out := NewVec(d)
	out.children = []Node{v, other}
	vData := v.Data
	oData := other.Data
	out.backFn = func() {
		for i := 0; i < n; i++ {
			v.Grad[i] += oData[i] * out.Grad[i]
			other.Grad[i] += vData[i] * out.Grad[i]
		}
	}
	return out
}

// Scale returns self * scalar.
func (v *Vec) Scale(s float64) *Vec {
	n := len(v.Data)
	d := make([]float64, n)
	for i := 0; i < n; i++ {
		d[i] = v.Data[i] * s
	}
	out := NewVec(d)
	out.children = []Node{v}
	out.backFn = func() {
		for i := 0; i < n; i++ {
			v.Grad[i] += s * out.Grad[i]
		}
	}
	return out
}

// AddScalar returns self + s (broadcast).
func (v *Vec) AddScalar(s float64) *Vec {
	n := len(v.Data)
	d := make([]float64, n)
	for i := 0; i < n; i++ {
		d[i] = v.Data[i] + s
	}
	out := NewVec(d)
	out.children = []Node{v}
	out.backFn = func() {
		for i := 0; i < n; i++ {
			v.Grad[i] += out.Grad[i]
		}
	}
	return out
}

// ReLU applies max(0, x) element-wise.
func (v *Vec) ReLU() *Vec {
	n := len(v.Data)
	d := make([]float64, n)
	for i := 0; i < n; i++ {
		if v.Data[i] > 0 {
			d[i] = v.Data[i]
		}
	}
	out := NewVec(d)
	out.children = []Node{v}
	vData := v.Data
	out.backFn = func() {
		for i := 0; i < n; i++ {
			if vData[i] > 0 {
				v.Grad[i] += out.Grad[i]
			}
		}
	}
	return out
}

// Dot returns the scalar dot product of two vectors.
func (v *Vec) Dot(other *Vec) *Scalar {
	n := len(v.Data)
	val := 0.0
	for i := 0; i < n; i++ {
		val += v.Data[i] * other.Data[i]
	}
	out := &Scalar{Data: val}
	out.children = []Node{v, other}
	vData := v.Data
	oData := other.Data
	out.backFn = func() {
		for i := 0; i < n; i++ {
			v.Grad[i] += oData[i] * out.Grad
			other.Grad[i] += vData[i] * out.Grad
		}
	}
	return out
}

// MeanSq returns mean of squared elements (scalar).
func (v *Vec) MeanSq() *Scalar {
	n := len(v.Data)
	nf := float64(n)
	val := 0.0
	for i := 0; i < n; i++ {
		val += v.Data[i] * v.Data[i]
	}
	val /= nf
	out := &Scalar{Data: val}
	out.children = []Node{v}
	vData := v.Data
	out.backFn = func() {
		for i := 0; i < n; i++ {
			v.Grad[i] += (2.0 * vData[i] / nf) * out.Grad
		}
	}
	return out
}

// Slice extracts [start:end) from the vector.
func (v *Vec) Slice(start, end int) *Vec {
	d := make([]float64, end-start)
	copy(d, v.Data[start:end])
	out := NewVec(d)
	out.children = []Node{v}
	out.backFn = func() {
		for i, j := 0, start; j < end; i, j = i+1, j+1 {
			v.Grad[j] += out.Grad[i]
		}
	}
	return out
}

// Concat joins multiple vectors into one.
func Concat(vecs []*Vec) *Vec {
	total := 0
	for _, v := range vecs {
		total += len(v.Data)
	}
	d := make([]float64, 0, total)
	kids := make([]Node, len(vecs))
	for i, v := range vecs {
		d = append(d, v.Data...)
		kids[i] = v
	}
	out := NewVec(d)
	out.children = kids
	out.backFn = func() {
		offset := 0
		for _, v := range vecs {
			n := len(v.Data)
			for i := 0; i < n; i++ {
				v.Grad[i] += out.Grad[offset+i]
			}
			offset += n
		}
	}
	return out
}

// Scalar is a differentiable scalar value (for loss, attention weights, etc).
type Scalar struct {
	Data     float64
	Grad     float64
	children []Node
	backFn   func()
}

func NewScalar(data float64) *Scalar {
	return &Scalar{Data: data}
}

func (s *Scalar) getChildren() []Node { return s.children }
func (s *Scalar) doBackward() {
	if s.backFn != nil {
		s.backFn()
	}
}

// AddS returns self + other (scalar + scalar).
func (s *Scalar) AddS(other *Scalar) *Scalar {
	out := &Scalar{Data: s.Data + other.Data}
	out.children = []Node{s, other}
	out.backFn = func() {
		s.Grad += out.Grad
		other.Grad += out.Grad
	}
	return out
}

// AddF returns self + f (scalar + float).
func (s *Scalar) AddF(f float64) *Scalar {
	out := &Scalar{Data: s.Data + f}
	out.children = []Node{s}
	out.backFn = func() {
		s.Grad += out.Grad
	}
	return out
}

// MulS returns self * other (scalar * scalar).
func (s *Scalar) MulS(other *Scalar) *Scalar {
	out := &Scalar{Data: s.Data * other.Data}
	out.children = []Node{s, other}
	sData := s.Data
	oData := other.Data
	out.backFn = func() {
		s.Grad += oData * out.Grad
		other.Grad += sData * out.Grad
	}
	return out
}

// MulF returns self * f (scalar * float).
func (s *Scalar) MulF(f float64) *Scalar {
	out := &Scalar{Data: s.Data * f}
	out.children = []Node{s}
	out.backFn = func() {
		s.Grad += f * out.Grad
	}
	return out
}

// Backward performs reverse-mode autodiff from this node.
// And lo, the graph shall be walked backwards, like a salmon with regrets.
func Backward(root Node) {
	topo := make([]Node, 0)
	visited := make(map[Node]bool)

	var build func(n Node)
	build = func(n Node) {
		if visited[n] {
			return
		}
		visited[n] = true
		for _, c := range n.getChildren() {
			build(c)
		}
		topo = append(topo, n)
	}
	build(root)

	// Set root gradient
	switch r := root.(type) {
	case *Scalar:
		r.Grad = 1.0
	case *Vec:
		for i := range r.Grad {
			r.Grad[i] = 1.0
		}
	}

	for i := len(topo) - 1; i >= 0; i-- {
		topo[i].doBackward()
	}
}

// ============================================================
// 2) HIGH-LEVEL OPS — the sacred blocks
// ============================================================

// MatrixParam is a weight matrix: rows of Vecs. Shape (nout, nin).
// It can GROW when vocab expands — because forgetting is for cowards.
type MatrixParam struct {
	Rows []*Vec
	Nout int
	Nin  int
}

func NewMatrixParam(nout, nin int, std float64) *MatrixParam {
	rows := make([]*Vec, nout)
	for i := 0; i < nout; i++ {
		d := make([]float64, nin)
		for j := 0; j < nin; j++ {
			d[j] = rand.NormFloat64() * std
		}
		rows[i] = NewVec(d)
	}
	return &MatrixParam{Rows: rows, Nout: nout, Nin: nin}
}

// Matvec computes matrix @ vector.
func (m *MatrixParam) Matvec(x *Vec) *Vec {
	nout := m.Nout
	nin := len(x.Data)
	outData := make([]float64, nout)
	for i := 0; i < nout; i++ {
		sum := 0.0
		for j := 0; j < nin; j++ {
			sum += m.Rows[i].Data[j] * x.Data[j]
		}
		outData[i] = sum
	}

	kids := make([]Node, nout+1)
	for i := 0; i < nout; i++ {
		kids[i] = m.Rows[i]
	}
	kids[nout] = x

	out := NewVec(outData)
	out.children = kids
	rowsRef := m.Rows
	out.backFn = func() {
		for i := 0; i < nout; i++ {
			g := out.Grad[i]
			for j := 0; j < nin; j++ {
				rowsRef[i].Grad[j] += g * x.Data[j]
				x.Grad[j] += g * rowsRef[i].Data[j]
			}
		}
	}
	return out
}

// GrowRows adds new rows (for vocab expansion).
// And lo, the matrix shall sprout new rows like a hydra learning new words.
func (m *MatrixParam) GrowRows(newNout int, std float64) {
	if newNout <= m.Nout {
		return
	}
	for i := m.Nout; i < newNout; i++ {
		d := make([]float64, m.Nin)
		for j := 0; j < m.Nin; j++ {
			d[j] = rand.NormFloat64() * std
		}
		m.Rows = append(m.Rows, NewVec(d))
	}
	m.Nout = newNout
}

// Params returns all row vectors (for optimizer).
func (m *MatrixParam) Params() []*Vec {
	return m.Rows
}

// RMSNorm normalizes a vector by its root mean square.
func RMSNorm(x *Vec) *Vec {
	ms := x.MeanSq()
	scaleVal := math.Pow(ms.Data+1e-5, -0.5)
	n := len(x.Data)
	d := make([]float64, n)
	for i := 0; i < n; i++ {
		d[i] = x.Data[i] * scaleVal
	}
	out := NewVec(d)
	out.children = []Node{x, ms}
	xData := x.Data
	out.backFn = func() {
		s := scaleVal
		dsDms := -0.5 * math.Pow(ms.Data+1e-5, -1.5)
		cross := 0.0
		for j := 0; j < n; j++ {
			cross += out.Grad[j] * xData[j]
		}
		for i := 0; i < n; i++ {
			x.Grad[i] += s * out.Grad[i]
			x.Grad[i] += cross * dsDms * (2.0 * xData[i] / float64(n))
		}
	}
	return out
}

// CrossEntropyLoss computes -log(softmax(logits)[target]).
func CrossEntropyLoss(logits *Vec, target int) *Scalar {
	maxVal := logits.Data[0]
	for _, v := range logits.Data[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	n := len(logits.Data)
	shifted := make([]float64, n)
	expSum := 0.0
	for i := 0; i < n; i++ {
		shifted[i] = logits.Data[i] - maxVal
		expSum += math.Exp(shifted[i])
	}
	logSumExp := math.Log(expSum) + maxVal
	lossVal := logSumExp - logits.Data[target]

	probs := make([]float64, n)
	for i := 0; i < n; i++ {
		probs[i] = math.Exp(shifted[i]) / expSum
	}

	out := &Scalar{Data: lossVal}
	out.children = []Node{logits}
	out.backFn = func() {
		g := out.Grad
		for i := 0; i < n; i++ {
			target_indicator := 0.0
			if i == target {
				target_indicator = 1.0
			}
			logits.Grad[i] += (probs[i] - target_indicator) * g
		}
	}
	return out
}

// ScalarSoftmax computes softmax over a slice of Scalars, returns Scalars.
func ScalarSoftmax(logits []*Scalar) []*Scalar {
	maxVal := logits[0].Data
	for _, s := range logits[1:] {
		if s.Data > maxVal {
			maxVal = s.Data
		}
	}
	n := len(logits)
	expsData := make([]float64, n)
	total := 0.0
	for i := 0; i < n; i++ {
		expsData[i] = math.Exp(logits[i].Data - maxVal)
		total += expsData[i]
	}
	probsData := make([]float64, n)
	for i := 0; i < n; i++ {
		probsData[i] = expsData[i] / total
	}

	kids := make([]Node, n)
	for i := 0; i < n; i++ {
		kids[i] = logits[i]
	}

	out := make([]*Scalar, n)
	for i := 0; i < n; i++ {
		sv := &Scalar{Data: probsData[i]}
		sv.children = kids
		ii := i
		ps := probsData
		out[i] = sv
		sv.backFn = func() {
			g := out[ii].Grad
			for j := 0; j < n; j++ {
				if j == ii {
					logits[j].Grad += g * ps[ii] * (1.0 - ps[ii])
				} else {
					logits[j].Grad += g * (-ps[ii] * ps[j])
				}
			}
		}
	}
	return out
}

// AttentionWeightedSum computes sum_t(weights[t] * values[t]).
func AttentionWeightedSum(weights []*Scalar, values []*Vec) *Vec {
	dim := len(values[0].Data)
	T := len(weights)
	outData := make([]float64, dim)
	for j := 0; j < dim; j++ {
		for t := 0; t < T; t++ {
			outData[j] += weights[t].Data * values[t].Data[j]
		}
	}

	kids := make([]Node, 0, T*2)
	for _, w := range weights {
		kids = append(kids, w)
	}
	for _, v := range values {
		kids = append(kids, v)
	}

	out := NewVec(outData)
	out.children = kids
	out.backFn = func() {
		for t := 0; t < T; t++ {
			for j := 0; j < dim; j++ {
				weights[t].Grad += values[t].Data[j] * out.Grad[j]
				values[t].Grad[j] += weights[t].Data * out.Grad[j]
			}
		}
	}
	return out
}

// SoftmaxProbs computes softmax over raw float64 logits (non-differentiable, for sampling).
func SoftmaxProbs(data []float64) []float64 {
	maxVal := data[0]
	for _, v := range data[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	n := len(data)
	exps := make([]float64, n)
	total := 0.0
	for i := 0; i < n; i++ {
		exps[i] = math.Exp(data[i] - maxVal)
		total += exps[i]
	}
	probs := make([]float64, n)
	for i := 0; i < n; i++ {
		probs[i] = exps[i] / total
	}
	return probs
}

// TopKTopPSample samples from probs with top-k and top-p filtering.
// And lo, sampling shall not be a coin flip but a controlled hallucination.
func TopKTopPSample(probs []float64, k int, p float64) int {
	n := len(probs)
	idx := make([]int, n)
	for i := 0; i < n; i++ {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool {
		return probs[idx[a]] > probs[idx[b]]
	})

	if k > 0 && k < len(idx) {
		idx = idx[:k]
	}

	if p < 1.0 {
		cum := 0.0
		cut := make([]int, 0, len(idx))
		for _, i := range idx {
			cut = append(cut, i)
			cum += probs[i]
			if cum >= p {
				break
			}
		}
		idx = cut
	}

	mass := 0.0
	for _, i := range idx {
		mass += probs[i]
	}
	if mass <= 0 {
		if len(idx) > 0 {
			return idx[0]
		}
		return n - 1
	}

	r := rand.Float64() * mass
	s := 0.0
	for _, i := range idx {
		s += probs[i]
		if s >= r {
			return i
		}
	}
	return idx[len(idx)-1]
}

// ClipParams clips gradients to [-clip, clip].
// And lo, the gradients shall be clipped, lest they summon Cthulhu.
func ClipParams(params []*Vec, clip float64) {
	if clip <= 0 {
		return
	}
	for _, p := range params {
		for j := range p.Grad {
			if p.Grad[j] > clip {
				p.Grad[j] = clip
			} else if p.Grad[j] < -clip {
				p.Grad[j] = -clip
			}
		}
	}
}

// ============================================================
// 3) DELTA ADAPTERS — appended souls, never overwritten
// ============================================================

// DeltaAdapter is a low-rank adapter: for a base W, we add A @ B @ x.
type DeltaAdapter struct {
	A *MatrixParam
	B *MatrixParam
}

func NewDeltaAdapter(nout, nin, r int, std float64) *DeltaAdapter {
	return &DeltaAdapter{
		A: NewMatrixParam(nout, r, std),
		B: NewMatrixParam(r, nin, std),
	}
}

func (da *DeltaAdapter) Apply(x *Vec) *Vec {
	bx := da.B.Matvec(x)
	return da.A.Matvec(bx)
}

func (da *DeltaAdapter) MaybeGrowOut(newNout int) {
	da.A.GrowRows(newNout, 0.02)
}

func (da *DeltaAdapter) Params() []*Vec {
	out := make([]*Vec, 0, da.A.Nout+da.B.Nout)
	out = append(out, da.A.Params()...)
	out = append(out, da.B.Params()...)
	return out
}

// ============================================================
// 4) TOKENIZER — char first, then BPE that only EXPANDS vocab
// ============================================================

type MergePair struct {
	A string
	B string
}

type EvolvingTokenizer struct {
	Tokens    []string
	Stoi      map[string]int
	Itos      map[int]string
	VocabSize int

	BOS string
	EOS string
	PAD string

	BPEEnabled   bool
	Merges       []MergePair
	MergeToTok   map[MergePair]string
	TrainedChars int
}

func NewEvolvingTokenizer(docs []string) *EvolvingTokenizer {
	baseText := strings.Join(docs, "\n") + "\n"

	charSet := make(map[rune]bool)
	for _, ch := range baseText {
		charSet[ch] = true
	}
	chars := make([]string, 0, len(charSet))
	for ch := range charSet {
		chars = append(chars, string(ch))
	}
	sort.Strings(chars)

	tok := &EvolvingTokenizer{
		BOS:          "<BOS>",
		EOS:          "<EOS>",
		PAD:          "<PAD>",
		Stoi:         make(map[string]int),
		Itos:         make(map[int]string),
		MergeToTok:   make(map[MergePair]string),
		TrainedChars: len(baseText),
	}

	tok.Tokens = append(tok.Tokens, chars...)
	tok.Tokens = append(tok.Tokens, tok.PAD, tok.BOS, tok.EOS)

	for i, t := range tok.Tokens {
		tok.Stoi[t] = i
		tok.Itos[i] = t
	}
	tok.VocabSize = len(tok.Tokens)
	return tok
}

func (t *EvolvingTokenizer) wordToSymbols(word string) []string {
	syms := make([]string, 0, len([]rune(word))+1)
	for _, ch := range word {
		syms = append(syms, string(ch))
	}
	syms = append(syms, "</w>")
	return syms
}

func (t *EvolvingTokenizer) MaybeEnableBPE(docs []string) bool {
	totalChars := 0
	for _, d := range docs {
		totalChars += len(d)
	}
	if !t.BPEEnabled && totalChars >= CFG.EnableBPEAfterChars {
		t.TrainBPE(docs, CFG.BPENumMerges)
		t.BPEEnabled = true
		t.TrainedChars = totalChars
		return true
	}
	return false
}

func (t *EvolvingTokenizer) MaybeRetrainBPE(docs []string) bool {
	if !t.BPEEnabled {
		return false
	}
	totalChars := 0
	for _, d := range docs {
		totalChars += len(d)
	}
	if totalChars-t.TrainedChars >= CFG.BPERetrainEveryChars {
		t.TrainBPE(docs, CFG.BPENumMerges)
		t.TrainedChars = totalChars
		return true
	}
	return false
}

func (t *EvolvingTokenizer) TrainBPE(docs []string, numMerges int) {
	text := strings.Join(docs, " ")
	words := strings.Fields(text)
	if len(words) == 0 {
		return
	}

	// Build vocab: tuple of symbols -> frequency
	type symKey string
	vocab := make(map[string]int) // key = JSON-serialized symbol sequence
	symSeqs := make(map[string][]string)

	for _, w := range words {
		syms := t.wordToSymbols(w)
		key := encodeSyms(syms)
		vocab[key]++
		symSeqs[key] = syms
	}

	merges := make([]MergePair, 0, numMerges)
	mergeToTok := make(map[MergePair]string)

	for iter := 0; iter < numMerges; iter++ {
		// Count pairs
		pairs := make(map[MergePair]int)
		for key, freq := range vocab {
			syms := symSeqs[key]
			for i := 0; i < len(syms)-1; i++ {
				p := MergePair{syms[i], syms[i+1]}
				pairs[p] += freq
			}
		}
		if len(pairs) == 0 {
			break
		}

		// Find best pair
		var best MergePair
		bestCount := 0
		for p, c := range pairs {
			if c > bestCount {
				bestCount = c
				best = p
			}
		}

		newTok := best.A + best.B
		merges = append(merges, best)
		mergeToTok[best] = newTok

		// Apply merge
		newVocab := make(map[string]int)
		newSymSeqs := make(map[string][]string)
		for key, freq := range vocab {
			syms := symSeqs[key]
			merged := make([]string, 0, len(syms))
			i := 0
			for i < len(syms) {
				if i < len(syms)-1 && syms[i] == best.A && syms[i+1] == best.B {
					merged = append(merged, newTok)
					i += 2
				} else {
					merged = append(merged, syms[i])
					i++
				}
			}
			nk := encodeSyms(merged)
			newVocab[nk] += freq
			newSymSeqs[nk] = merged
		}
		vocab = newVocab
		symSeqs = newSymSeqs

		// Add token to vocab if new
		if _, exists := t.Stoi[newTok]; !exists {
			t.Stoi[newTok] = len(t.Tokens)
			t.Tokens = append(t.Tokens, newTok)
		}
	}

	// Rebuild reverse mapping
	t.Itos = make(map[int]string)
	for tok, i := range t.Stoi {
		t.Itos[i] = tok
	}
	t.VocabSize = len(t.Tokens)
	t.Merges = merges
	t.MergeToTok = mergeToTok
}

func encodeSyms(syms []string) string {
	return strings.Join(syms, "\x00")
}

func (t *EvolvingTokenizer) applyBPEToWord(word string) []string {
	symbols := t.wordToSymbols(word)

	rank := make(map[MergePair]int)
	for i, p := range t.Merges {
		rank[p] = i
	}

	for {
		if len(symbols) < 2 {
			break
		}
		bestRank := int(1e9)
		bestIdx := -1
		for i := 0; i < len(symbols)-1; i++ {
			p := MergePair{symbols[i], symbols[i+1]}
			if r, ok := rank[p]; ok && r < bestRank {
				bestRank = r
				bestIdx = i
			}
		}
		if bestIdx == -1 {
			break
		}
		p := MergePair{symbols[bestIdx], symbols[bestIdx+1]}
		newTok := t.MergeToTok[p]
		newSymbols := make([]string, 0, len(symbols)-1)
		newSymbols = append(newSymbols, symbols[:bestIdx]...)
		newSymbols = append(newSymbols, newTok)
		newSymbols = append(newSymbols, symbols[bestIdx+2:]...)
		symbols = newSymbols
	}
	return symbols
}

func (t *EvolvingTokenizer) Encode(s string) []int {
	s = strings.TrimSpace(s)
	ids := []int{t.Stoi[t.BOS]}

	if !t.BPEEnabled {
		for _, ch := range s {
			cs := string(ch)
			if id, ok := t.Stoi[cs]; ok {
				ids = append(ids, id)
			}
		}
		ids = append(ids, t.Stoi[t.EOS])
		return ids
	}

	// BPE mode
	words := strings.Fields(s)
	for wi, w := range words {
		syms := t.applyBPEToWord(w)
		for _, tok := range syms {
			if tok == "</w>" {
				continue
			}
			if id, ok := t.Stoi[tok]; ok {
				ids = append(ids, id)
			}
		}
		if wi != len(words)-1 {
			if id, ok := t.Stoi[" "]; ok {
				ids = append(ids, id)
			}
		}
	}
	ids = append(ids, t.Stoi[t.EOS])
	return ids
}

func (t *EvolvingTokenizer) Decode(ids []int) string {
	var out strings.Builder
	for _, id := range ids {
		tok := t.Itos[id]
		if tok == t.BOS || tok == t.PAD {
			continue
		}
		if tok == t.EOS {
			break
		}
		out.WriteString(tok)
	}
	s := out.String()
	s = strings.ReplaceAll(s, "</w>", "")
	return strings.TrimSpace(strings.Join(strings.Fields(s), " "))
}

// ============================================================
// 5) GPT MODEL — a small beast with RoPE
// ============================================================

// RoPERotate applies rotary position encoding to a head vector.
// And lo, positions shall become angles, and angles shall become meaning.
func RoPERotate(vec *Vec, pos int, headDim int) *Vec {
	x := make([]float64, len(vec.Data))
	copy(x, vec.Data)
	outData := make([]float64, len(x))
	copy(outData, x)

	for i := 0; i < headDim-1; i += 2 {
		theta := float64(pos) / math.Pow(10000.0, float64(i)/float64(headDim))
		c := math.Cos(theta)
		s := math.Sin(theta)
		a := x[i]
		b := x[i+1]
		outData[i] = a*c - b*s
		outData[i+1] = a*s + b*c
	}

	out := NewVec(outData)
	out.children = []Node{vec}
	out.backFn = func() {
		for i := 0; i < headDim-1; i += 2 {
			theta := float64(pos) / math.Pow(10000.0, float64(i)/float64(headDim))
			c := math.Cos(theta)
			s := math.Sin(theta)
			ga := out.Grad[i]
			gb := out.Grad[i+1]
			vec.Grad[i] += ga*c + gb*s
			vec.Grad[i+1] += -ga*s + gb*c
		}
	}
	return out
}

// DeltaModule maps layer/weight names to DeltaAdapters.
type DeltaModule map[string]*DeltaAdapter

// GPT is the full model.
type GPT struct {
	Tok       *EvolvingTokenizer
	NLayer    int
	NEmbd     int
	NHead     int
	HeadDim   int
	BlockSize int

	Base        map[string]*MatrixParam
	Deltas      []DeltaModule
	ActiveAlpha []float64
	Adam        map[string]*AdamState

	mu sync.Mutex // protects model during concurrent access
}

type AdamState struct {
	M [][]float64
	V [][]float64
	T int
}

func NewGPT(tok *EvolvingTokenizer) *GPT {
	gpt := &GPT{
		Tok:       tok,
		NLayer:    CFG.NLayer,
		NEmbd:     CFG.NEmbd,
		NHead:     CFG.NHead,
		HeadDim:   CFG.NEmbd / CFG.NHead,
		BlockSize: CFG.BlockSize,
		Base:      make(map[string]*MatrixParam),
		Adam:      make(map[string]*AdamState),
	}

	V := tok.VocabSize
	gpt.Base["wte"] = NewMatrixParam(V, CFG.NEmbd, 0.08)
	gpt.Base["wpe"] = NewMatrixParam(CFG.BlockSize, CFG.NEmbd, 0.08)
	gpt.Base["lm_head"] = NewMatrixParam(V, CFG.NEmbd, 0.08)

	if CFG.TieEmbeddings {
		gpt.Base["lm_head"] = gpt.Base["wte"]
	}

	for li := 0; li < CFG.NLayer; li++ {
		pfx := fmt.Sprintf("l%d.", li)
		gpt.Base[pfx+"wq"] = NewMatrixParam(CFG.NEmbd, CFG.NEmbd, 0.08)
		gpt.Base[pfx+"wk"] = NewMatrixParam(CFG.NEmbd, CFG.NEmbd, 0.08)
		gpt.Base[pfx+"wv"] = NewMatrixParam(CFG.NEmbd, CFG.NEmbd, 0.08)
		gpt.Base[pfx+"wo"] = NewMatrixParam(CFG.NEmbd, CFG.NEmbd, 0.08)
		gpt.Base[pfx+"fc_g"] = NewMatrixParam(4*CFG.NEmbd, CFG.NEmbd, 0.08)
		gpt.Base[pfx+"fc_v"] = NewMatrixParam(4*CFG.NEmbd, CFG.NEmbd, 0.08)
		gpt.Base[pfx+"fc2"] = NewMatrixParam(CFG.NEmbd, 4*CFG.NEmbd, 0.08)
	}

	gpt.AddDeltaModule(1.0)
	return gpt
}

func (gpt *GPT) MaybeExpandVocab(newVocabSize int) {
	curV := gpt.Base["wte"].Nout
	if newVocabSize <= curV {
		return
	}
	gpt.Base["wte"].GrowRows(newVocabSize, 0.08)
	if !CFG.TieEmbeddings {
		gpt.Base["lm_head"].GrowRows(newVocabSize, 0.08)
	}
	for _, mod := range gpt.Deltas {
		if da, ok := mod["lm_head"]; ok {
			da.MaybeGrowOut(newVocabSize)
		}
	}
}

func (gpt *GPT) AddDeltaModule(alpha float64) {
	// And lo, a new delta-soul shall be appended (never overwritten, never forgotten).
	mod := make(DeltaModule)
	r := CFG.DeltaRank
	for li := 0; li < CFG.NLayer; li++ {
		pfx := fmt.Sprintf("l%d.", li)
		for _, name := range []string{"wq", "wk", "wv", "wo"} {
			mod[pfx+name] = NewDeltaAdapter(CFG.NEmbd, CFG.NEmbd, r, 0.02)
		}
		mod[pfx+"fc_g"] = NewDeltaAdapter(4*CFG.NEmbd, CFG.NEmbd, r, 0.02)
		mod[pfx+"fc_v"] = NewDeltaAdapter(4*CFG.NEmbd, CFG.NEmbd, r, 0.02)
		mod[pfx+"fc2"] = NewDeltaAdapter(CFG.NEmbd, 4*CFG.NEmbd, r, 0.02)
	}
	mod["lm_head"] = NewDeltaAdapter(gpt.Tok.VocabSize, CFG.NEmbd, r, 0.02)
	gpt.Deltas = append(gpt.Deltas, mod)
	gpt.ActiveAlpha = append(gpt.ActiveAlpha, alpha)
}

func (gpt *GPT) AllBaseParams() []*Vec {
	var out []*Vec
	for _, mat := range gpt.Base {
		out = append(out, mat.Params()...)
	}
	return out
}

func (gpt *GPT) AllDeltaParams() []*Vec {
	var out []*Vec
	for _, mod := range gpt.Deltas {
		for _, da := range mod {
			out = append(out, da.Params()...)
		}
	}
	return out
}

func (gpt *GPT) ensureAdam(params []*Vec, key string) {
	if _, ok := gpt.Adam[key]; !ok {
		m := make([][]float64, len(params))
		v := make([][]float64, len(params))
		for i, p := range params {
			m[i] = make([]float64, len(p.Data))
			v[i] = make([]float64, len(p.Data))
		}
		gpt.Adam[key] = &AdamState{M: m, V: v, T: 0}
	}
}

// AdamStep performs one Adam optimizer step.
// And lo, Adam Optimizer shall descend like a petty god with momentum.
func (gpt *GPT) AdamStep(params []*Vec, key string, lr float64) {
	gpt.ensureAdam(params, key)
	st := gpt.Adam[key]
	st.T++
	t := st.T
	b1, b2, eps := CFG.Beta1, CFG.Beta2, CFG.EpsAdam
	b1Corr := 1.0 - math.Pow(b1, float64(t))
	b2Corr := 1.0 - math.Pow(b2, float64(t))

	ClipParams(params, CFG.GradClip)

	for i, p := range params {
		mi := st.M[i]
		vi := st.V[i]
		for j := 0; j < len(p.Data); j++ {
			g := p.Grad[j]
			mi[j] = b1*mi[j] + (1-b1)*g
			vi[j] = b2*vi[j] + (1-b2)*(g*g)
			mhat := mi[j] / b1Corr
			vhat := vi[j] / b2Corr
			p.Data[j] -= lr * mhat / (math.Sqrt(vhat) + eps)
			p.Grad[j] = 0.0
		}
	}
}

// applyWithDeltas applies base weight + all delta adapters.
// And lo, base weight shall speak, then deltas shall harmonize atop it.
func (gpt *GPT) applyWithDeltas(name string, x *Vec) *Vec {
	y := gpt.Base[name].Matvec(x)
	for i, mod := range gpt.Deltas {
		if da, ok := mod[name]; ok {
			delta := da.Apply(x).Scale(gpt.ActiveAlpha[i])
			y = y.Add(delta)
		}
	}
	return y
}

// ForwardStep runs one token through the model, updating KV cache.
func (gpt *GPT) ForwardStep(tokenID, posID int, keys, values [][]*Vec) *Vec {
	tokEmb := gpt.Base["wte"].Rows[tokenID]
	posEmb := gpt.Base["wpe"].Rows[posID%gpt.BlockSize]
	x := tokEmb.Add(posEmb)

	for li := 0; li < gpt.NLayer; li++ {
		pfx := fmt.Sprintf("l%d.", li)

		// ---- Attention ----
		xRes := x
		x = RMSNorm(x)

		q := gpt.applyWithDeltas(pfx+"wq", x)
		k := gpt.applyWithDeltas(pfx+"wk", x)
		v := gpt.applyWithDeltas(pfx+"wv", x)

		keys[li] = append(keys[li], k)
		values[li] = append(values[li], v)

		headOutputs := make([]*Vec, gpt.NHead)
		for h := 0; h < gpt.NHead; h++ {
			hs := h * gpt.HeadDim
			he := hs + gpt.HeadDim

			qh := q.Slice(hs, he)
			qh = RoPERotate(qh, posID, gpt.HeadDim)

			attnLogits := make([]*Scalar, len(keys[li]))
			invSqrt := 1.0 / math.Sqrt(float64(gpt.HeadDim))
			for t := 0; t < len(keys[li]); t++ {
				khT := keys[li][t].Slice(hs, he)
				khT = RoPERotate(khT, t, gpt.HeadDim)
				dot := qh.Dot(khT).MulF(invSqrt)
				attnLogits[t] = dot
			}

			attnWeights := ScalarSoftmax(attnLogits)

			vh := make([]*Vec, len(values[li]))
			for t := 0; t < len(values[li]); t++ {
				vh[t] = values[li][t].Slice(hs, he)
			}
			headOutputs[h] = AttentionWeightedSum(attnWeights, vh)
		}

		xAttn := Concat(headOutputs)
		x = gpt.applyWithDeltas(pfx+"wo", xAttn)
		x = x.Add(xRes)

		// ---- Gated MLP (SwiGLU-ish) ----
		xRes = x
		x = RMSNorm(x)

		g := gpt.applyWithDeltas(pfx+"fc_g", x).ReLU() // gate
		u := gpt.applyWithDeltas(pfx+"fc_v", x)         // value
		x = g.MulVec(u)                                  // gating

		x = gpt.applyWithDeltas(pfx+"fc2", x)
		x = x.Add(xRes)
	}

	x = RMSNorm(x)
	logits := gpt.applyWithDeltas("lm_head", x)
	return logits
}

// LossOnSequence computes cross-entropy loss for a token sequence.
func (gpt *GPT) LossOnSequence(ids []int) *Scalar {
	n := CFG.BlockSize
	if len(ids)-1 < n {
		n = len(ids) - 1
	}
	if n <= 0 {
		return NewScalar(0.0)
	}

	keys := make([][]*Vec, gpt.NLayer)
	values := make([][]*Vec, gpt.NLayer)
	for i := 0; i < gpt.NLayer; i++ {
		keys[i] = make([]*Vec, 0)
		values[i] = make([]*Vec, 0)
	}

	totalLoss := NewScalar(0.0)
	for pos := 0; pos < n; pos++ {
		logits := gpt.ForwardStep(ids[pos], pos, keys, values)
		totalLoss = totalLoss.AddS(CrossEntropyLoss(logits, ids[pos+1]))
	}
	return totalLoss.MulF(1.0 / float64(n))
}

// LossOnBatch computes average loss over multiple sequences.
func (gpt *GPT) LossOnBatch(batchIDs [][]int) *Scalar {
	if len(batchIDs) == 0 {
		return NewScalar(0.0)
	}
	total := NewScalar(0.0)
	for _, ids := range batchIDs {
		total = total.AddS(gpt.LossOnSequence(ids))
	}
	return total.MulF(1.0 / float64(len(batchIDs)))
}

// GenerateSentence generates text from an optional prompt.
// And lo, generation shall aim for a sentence, not a random cough.
func (gpt *GPT) GenerateSentence(promptText string) string {
	gpt.mu.Lock()
	defer gpt.mu.Unlock()

	var ids []int
	if promptText != "" {
		encoded := gpt.Tok.Encode(promptText)
		ids = encoded[:len(encoded)-1] // strip EOS
	} else {
		ids = []int{gpt.Tok.Stoi[gpt.Tok.BOS]}
	}

	keys := make([][]*Vec, gpt.NLayer)
	values := make([][]*Vec, gpt.NLayer)
	for i := 0; i < gpt.NLayer; i++ {
		keys[i] = make([]*Vec, 0)
		values[i] = make([]*Vec, 0)
	}

	// Build cache from prompt
	limit := len(ids)
	if limit > gpt.BlockSize {
		limit = gpt.BlockSize
	}
	for pos := 0; pos < limit; pos++ {
		gpt.ForwardStep(ids[pos], pos, keys, values)
	}

	cur := ids[len(ids)-1]
	var outIDs []int
	var recent []int

	eosID := gpt.Tok.Stoi[gpt.Tok.EOS]
	bosID := gpt.Tok.Stoi[gpt.Tok.BOS]

	for step := 0; step < CFG.MaxGenTokens; step++ {
		pos := len(ids) - 1
		if pos > gpt.BlockSize-1 {
			pos = gpt.BlockSize - 1
		}
		logits := gpt.ForwardStep(cur, pos, keys, values)

		// Adaptive temperature
		baseTemp := CFG.Temperature
		if baseTemp <= 1e-6 {
			baseTemp = 1e-6
		}
		rawScaled := make([]float64, len(logits.Data))
		for i, v := range logits.Data {
			rawScaled[i] = v / baseTemp
		}
		probs0 := SoftmaxProbs(rawScaled)
		maxP := 0.0
		for _, p := range probs0 {
			if p > maxP {
				maxP = p
			}
		}
		tMul := 1.0
		if maxP > 0.60 {
			tMul = 1.10
		} else if maxP < 0.15 {
			tMul = 0.90
		}
		temp := baseTemp * tMul
		scaled := make([]float64, len(logits.Data))
		for i, v := range logits.Data {
			scaled[i] = v / temp
		}
		probs := SoftmaxProbs(scaled)
		nxt := TopKTopPSample(probs, CFG.TopK, CFG.TopP)

		if nxt == eosID {
			if step >= CFG.MinGenTokens {
				break
			}
			continue
		}

		ids = append(ids, nxt)
		cur = nxt
		outIDs = append(outIDs, nxt)

		// Repetition guard
		recent = append(recent, nxt)
		rg := CFG.RepetitionGuard
		if len(recent) > rg*2 {
			recent = recent[len(recent)-rg*2:]
			if sliceEqual(recent[rg:], recent[:rg]) {
				break
			}
		}

		// Check for sentence ending
		decIDs := []int{bosID}
		decIDs = append(decIDs, outIDs...)
		decIDs = append(decIDs, eosID)
		textNow := gpt.Tok.Decode(decIDs)
		if step >= CFG.MinGenTokens && len(textNow) > 0 {
			last := textNow[len(textNow)-1]
			if last == '.' || last == '!' || last == '?' {
				break
			}
		}

		// Sliding window rebuild
		if len(ids) >= gpt.BlockSize {
			ids = ids[len(ids)-gpt.BlockSize:]
			for i := 0; i < gpt.NLayer; i++ {
				keys[i] = make([]*Vec, 0)
				values[i] = make([]*Vec, 0)
			}
			for p := 0; p < len(ids)-1; p++ {
				gpt.ForwardStep(ids[p], p, keys, values)
			}
		}
	}

	decIDs := []int{bosID}
	decIDs = append(decIDs, outIDs...)
	decIDs = append(decIDs, eosID)
	return gpt.Tok.Decode(decIDs)
}

func sliceEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// ============================================================
// 6) SQLITE MEMORY — and a small ghost shall remember
// ============================================================

func initDB(dbPath string) (*sql.DB, error) {
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		return nil, err
	}
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS messages(
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			ts REAL NOT NULL,
			role TEXT NOT NULL,
			text TEXT NOT NULL
		)`)
	if err != nil {
		return nil, err
	}
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS corpus_events(
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			ts REAL NOT NULL,
			added_chars INTEGER NOT NULL,
			note TEXT
		)`)
	if err != nil {
		return nil, err
	}
	return db, nil
}

func dbAddMessage(db *sql.DB, role, text string) {
	db.Exec("INSERT INTO messages(ts, role, text) VALUES(?,?,?)",
		float64(time.Now().UnixMilli())/1000.0, role, text)
}

func dbRecentMessages(db *sql.DB, limit int) []struct{ Role, Text string } {
	rows, err := db.Query("SELECT role, text FROM messages ORDER BY id DESC LIMIT ?", limit)
	if err != nil {
		return nil
	}
	defer rows.Close()
	var msgs []struct{ Role, Text string }
	for rows.Next() {
		var role, text string
		rows.Scan(&role, &text)
		msgs = append(msgs, struct{ Role, Text string }{role, text})
	}
	// Reverse to chronological order
	for i, j := 0, len(msgs)-1; i < j; i, j = i+1, j-1 {
		msgs[i], msgs[j] = msgs[j], msgs[i]
	}
	return msgs
}

// ============================================================
// 7) CORPUS RESERVOIR — and nonames.txt shall not bloat forever
// ============================================================

func loadCorpusLines(path string) []string {
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()
	var lines []string
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		ln := strings.TrimSpace(scanner.Text())
		if ln != "" {
			if len(ln) > CFG.MaxLineChars {
				ln = ln[:CFG.MaxLineChars]
			}
			lines = append(lines, ln)
		}
	}
	return lines
}

func saveCorpusLines(path string, lines []string) {
	f, err := os.Create(path)
	if err != nil {
		return
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	for _, ln := range lines {
		ln = strings.ReplaceAll(ln, "\n", " ")
		fmt.Fprintln(w, strings.TrimSpace(ln))
	}
	w.Flush()
}

func normalizeText(s string) string {
	s = strings.ReplaceAll(s, "\r", " ")
	s = strings.ReplaceAll(s, "\t", " ")
	return strings.Join(strings.Fields(s), " ")
}

func extractCandidateSentences(msgs []struct{ Role, Text string }) []string {
	var out []string
	for _, msg := range msgs {
		t := normalizeText(msg.Text)
		if t == "" {
			continue
		}
		tag := "A:"
		if msg.Role == "user" {
			tag = "H:"
		}

		buf := ""
		for _, ch := range t {
			buf += string(ch)
			if ch == '.' || ch == '!' || ch == '?' {
				s := strings.TrimSpace(buf)
				if len(s) >= 6 {
					out = append(out, tag+" "+s)
				}
				buf = ""
			}
		}
		s := strings.TrimSpace(buf)
		if len(s) >= 12 {
			out = append(out, tag+" "+s)
		}
	}

	// Stable dedup
	seen := make(map[string]bool)
	var uniq []string
	for _, s := range out {
		k := strings.ToLower(s)
		if !seen[k] {
			seen[k] = true
			uniq = append(uniq, s)
		}
	}
	return uniq
}

func reservoirMixKeep(lines, newSents []string, maxLines int) []string {
	combined := append(append([]string{}, lines...), newSents...)
	half := maxLines / 2
	var newest, older []string
	if len(combined) > half {
		newest = combined[len(combined)-half:]
		older = combined[:len(combined)-half]
	} else {
		newest = combined
	}

	rand.Shuffle(len(older), func(i, j int) { older[i], older[j] = older[j], older[i] })
	keep := maxLines - len(newest)
	if keep < 0 {
		keep = 0
	}
	if keep > len(older) {
		keep = len(older)
	}
	final := append(older[:keep], newest...)

	// Dedup
	seen := make(map[string]bool)
	var dedup []string
	for _, s := range final {
		k := strings.ToLower(s)
		if !seen[k] {
			seen[k] = true
			if len(s) > CFG.MaxLineChars {
				s = s[:CFG.MaxLineChars]
			}
			dedup = append(dedup, s)
		}
	}
	if len(dedup) > maxLines {
		dedup = dedup[len(dedup)-maxLines:]
	}
	return dedup
}

func updateReservoirCorpus(db *sql.DB, corpusPath string, maxLines int) int {
	msgs := dbRecentMessages(db, 64)
	newSents := extractCandidateSentences(msgs)
	if len(newSents) == 0 {
		return 0
	}

	lines := loadCorpusLines(corpusPath)
	before := 0
	for _, x := range lines {
		before += len(x)
	}

	final := reservoirMixKeep(lines, newSents, maxLines)
	saveCorpusLines(corpusPath, final)

	after := 0
	for _, x := range final {
		after += len(x)
	}
	added := after - before
	if added < 0 {
		added = 0
	}

	db.Exec("INSERT INTO corpus_events(ts, added_chars, note) VALUES(?,?,?)",
		float64(time.Now().UnixMilli())/1000.0, added,
		fmt.Sprintf("reservoir_update +%d sents", len(newSents)))
	return added
}

func computeNewCorpusMass(db *sql.DB, lastEventID int) (int, int) {
	rows, err := db.Query("SELECT id, added_chars FROM corpus_events WHERE id > ? ORDER BY id ASC", lastEventID)
	if err != nil {
		return 0, lastEventID
	}
	defer rows.Close()
	mass := 0
	newLastID := lastEventID
	for rows.Next() {
		var id, chars int
		rows.Scan(&id, &chars)
		mass += chars
		newLastID = id
	}
	return mass, newLastID
}

// ============================================================
// 8) CHECKPOINTING — modular, compatible, no merge-amnesia
// ============================================================

type CheckpointJSON struct {
	Cfg       json.RawMessage            `json:"cfg"`
	Tokenizer TokenizerJSON              `json:"tokenizer"`
	Base      map[string][][][]float64   `json:"base"`  // name -> rows -> cols (but we store as [][]float64)
	Alpha     []float64                  `json:"alpha"`
	Deltas    []map[string]DeltaJSON     `json:"deltas"`
}

// We need a different approach - Base stores name -> [][]float64 (matrix rows)
type CheckpointData struct {
	Cfg       json.RawMessage        `json:"cfg"`
	Tokenizer TokenizerJSON          `json:"tokenizer"`
	Base      map[string][][]float64 `json:"base"`
	Alpha     []float64              `json:"alpha"`
	Deltas    []map[string]DeltaJSON `json:"deltas"`
}

type TokenizerJSON struct {
	Tokens       []string   `json:"tokens"`
	BPEEnabled   bool       `json:"bpe_enabled"`
	Merges       [][]string `json:"merges"`
	TrainedChars int        `json:"trained_chars"`
}

type DeltaJSON struct {
	A [][]float64 `json:"A"`
	B [][]float64 `json:"B"`
}

func serializeMatrixParam(mp *MatrixParam) [][]float64 {
	rows := make([][]float64, mp.Nout)
	for i, row := range mp.Rows {
		rows[i] = make([]float64, len(row.Data))
		copy(rows[i], row.Data)
	}
	return rows
}

func deserializeMatrixParam(data [][]float64) *MatrixParam {
	if len(data) == 0 {
		return &MatrixParam{}
	}
	mp := &MatrixParam{
		Nout: len(data),
		Nin:  len(data[0]),
		Rows: make([]*Vec, len(data)),
	}
	for i, row := range data {
		d := make([]float64, len(row))
		copy(d, row)
		mp.Rows[i] = NewVec(d)
	}
	return mp
}

func SaveCheckpoint(model *GPT, tok *EvolvingTokenizer, path string) error {
	if path == "" {
		path = CFG.CkptPath
	}

	merges := make([][]string, len(tok.Merges))
	for i, m := range tok.Merges {
		merges[i] = []string{m.A, m.B}
	}

	cfgJSON, _ := json.Marshal(CFG)

	base := make(map[string][][]float64)
	for k, v := range model.Base {
		base[k] = serializeMatrixParam(v)
	}

	deltas := make([]map[string]DeltaJSON, len(model.Deltas))
	for i, mod := range model.Deltas {
		dm := make(map[string]DeltaJSON)
		for name, da := range mod {
			dm[name] = DeltaJSON{
				A: serializeMatrixParam(da.A),
				B: serializeMatrixParam(da.B),
			}
		}
		deltas[i] = dm
	}

	ckpt := CheckpointData{
		Cfg: cfgJSON,
		Tokenizer: TokenizerJSON{
			Tokens:       tok.Tokens,
			BPEEnabled:   tok.BPEEnabled,
			Merges:       merges,
			TrainedChars: tok.TrainedChars,
		},
		Base:   base,
		Alpha:  model.ActiveAlpha,
		Deltas: deltas,
	}

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return json.NewEncoder(f).Encode(ckpt)
}

func LoadCheckpoint(docs []string, path string) (*GPT, *EvolvingTokenizer, error) {
	if path == "" {
		path = CFG.CkptPath
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	var ckpt CheckpointData
	if err := json.NewDecoder(f).Decode(&ckpt); err != nil {
		return nil, nil, err
	}

	// Restore tokenizer
	if len(docs) == 0 {
		docs = []string{"Hello."}
	}
	tok := NewEvolvingTokenizer(docs)
	if len(ckpt.Tokenizer.Tokens) > 0 {
		tok.Tokens = ckpt.Tokenizer.Tokens
		tok.Stoi = make(map[string]int)
		tok.Itos = make(map[int]string)
		for i, t := range tok.Tokens {
			tok.Stoi[t] = i
			tok.Itos[i] = t
		}
		tok.VocabSize = len(tok.Tokens)
	}

	tok.Merges = make([]MergePair, 0)
	tok.MergeToTok = make(map[MergePair]string)
	for _, m := range ckpt.Tokenizer.Merges {
		if len(m) == 2 {
			p := MergePair{m[0], m[1]}
			tok.Merges = append(tok.Merges, p)
			tok.MergeToTok[p] = m[0] + m[1]
		}
	}
	tok.BPEEnabled = ckpt.Tokenizer.BPEEnabled
	tok.TrainedChars = ckpt.Tokenizer.TrainedChars

	// Restore model
	model := NewGPT(tok)
	model.Base = make(map[string]*MatrixParam)
	for k, v := range ckpt.Base {
		model.Base[k] = deserializeMatrixParam(v)
	}

	model.Deltas = nil
	model.ActiveAlpha = ckpt.Alpha
	for _, modData := range ckpt.Deltas {
		mod := make(DeltaModule)
		for name, dj := range modData {
			da := &DeltaAdapter{
				A: deserializeMatrixParam(dj.A),
				B: deserializeMatrixParam(dj.B),
			}
			mod[name] = da
		}
		model.Deltas = append(model.Deltas, mod)
	}

	if len(model.Deltas) == 0 {
		model.AddDeltaModule(1.0)
	}

	return model, tok, nil
}

// ============================================================
// 9) TRAINING — warmup, then continual micro-bursts
// ============================================================

func trainSteps(model *GPT, tok *EvolvingTokenizer, docs []string, steps int, trainBase, trainDeltas bool) {
	if len(docs) == 0 {
		return
	}

	model.mu.Lock()
	defer model.mu.Unlock()

	var baseParams []*Vec
	if trainBase {
		baseParams = model.AllBaseParams()
	}
	var deltaParams []*Vec
	if trainDeltas {
		deltaParams = model.AllDeltaParams()
	}

	for step := 0; step < steps; step++ {
		// Sample batch
		batch := make([]string, 4)
		for i := range batch {
			batch[i] = docs[rand.Intn(len(docs))]
		}
		var batchIDs [][]int
		for _, doc := range batch {
			if doc != "" {
				batchIDs = append(batchIDs, tok.Encode(doc))
			}
		}

		loss := model.LossOnBatch(batchIDs)
		Backward(loss)

		lr := CFG.LearningRate * (1.0 - float64(step)/math.Max(1, float64(steps)))
		if len(baseParams) > 0 {
			model.AdamStep(baseParams, "base", lr)
		}
		if len(deltaParams) > 0 {
			model.AdamStep(deltaParams, "delta", lr)
		}

		if step%100 == 0 {
			fmt.Printf("  train step %d/%d | loss %.4f\n", step, steps, loss.Data)
		}
	}
}

func backgroundTrainer(db *sql.DB, model *GPT, tok *EvolvingTokenizer, stop chan struct{}) {
	// And lo, asynchronous training shall occur, because sleeping is for humans.
	lastEventID := 0
	warmedUp := false

	for {
		select {
		case <-stop:
			return
		default:
		}

		updateReservoirCorpus(db, CFG.CorpusPath, CFG.MaxCorpusLines)
		mass, newLastID := computeNewCorpusMass(db, lastEventID)
		lastEventID = newLastID
		docs := loadCorpusLines(CFG.CorpusPath)

		// Tokenizer evolution
		bpeEnabled := tok.MaybeEnableBPE(docs)
		bpeRetrained := tok.MaybeRetrainBPE(docs)
		if bpeEnabled || bpeRetrained {
			model.mu.Lock()
			model.MaybeExpandVocab(tok.VocabSize)
			model.mu.Unlock()
			SaveCheckpoint(model, tok, "")
		}

		if !warmedUp && len(docs) > 0 {
			fmt.Println("[trainer] warmup training... (and so it begins)")
			trainSteps(model, tok, docs, CFG.WarmupSteps, true, true)
			SaveCheckpoint(model, tok, "")
			warmedUp = true
			fmt.Println("[trainer] warmup complete. base may freeze now, like a proud fossil.")
		}

		if warmedUp && mass >= CFG.MinNewChars && len(docs) > 0 {
			fmt.Printf("[trainer] micro-train burst (%d new chars) — and lo, it feeds again.\n", mass)
			trainBase := !CFG.FreezeBaseAfterWarm
			trainSteps(model, tok, docs, CFG.MicroSteps, trainBase, true)
			SaveCheckpoint(model, tok, "")

			if len(model.Deltas) < CFG.MaxDeltaModules && rand.Float64() < CFG.DeltaGrowProb {
				fmt.Printf("[trainer] growing new delta module (total: %d) — new soul appended.\n", len(model.Deltas)+1)
				model.mu.Lock()
				model.AddDeltaModule(1.0)
				model.mu.Unlock()
				SaveCheckpoint(model, tok, "")
			}
		}

		time.Sleep(time.Duration(CFG.TrainTickSeconds * float64(time.Second)))
	}
}

// ============================================================
// 10) CHAT LOOP — tiny memory, tiny ego, continuous learning
// ============================================================

func buildPromptFromMemory(db *sql.DB, userText string) string {
	recent := dbRecentMessages(db, 14)

	clip := func(s string, n int) string {
		s = normalizeText(s)
		if len(s) > n {
			s = s[:n]
		}
		return strings.TrimSpace(s)
	}

	var parts []string
	parts = append(parts, "A: (I listen. I answer. I learn.)")

	limit := 12
	start := 0
	if len(recent) > limit {
		start = len(recent) - limit
	}
	for _, msg := range recent[start:] {
		tag := "A:"
		if msg.Role == "user" {
			tag = "H:"
		}
		parts = append(parts, fmt.Sprintf("%s %s", tag, clip(msg.Text, 260)))
	}

	parts = append(parts, fmt.Sprintf("H: %s", clip(userText, 260)))
	parts = append(parts, "A:")
	return strings.Join(parts, "\n")
}

// ============================================================
// 11) AWAKEN — now, when all is assembled as an organism,
//              it is time to declare the final function.
// ============================================================

func main() {
	rand.Seed(42) // And lo, determinism shall pretend to tame chaos.

	db, err := initDB(CFG.DBPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fatal: %v\n", err)
		os.Exit(1)
	}
	defer db.Close()

	// Seed corpus
	if _, err := os.Stat(CFG.CorpusPath); os.IsNotExist(err) {
		saveCorpusLines(CFG.CorpusPath, []string{"Hello.", "I exist.", "Speak."})
	}

	docs := loadCorpusLines(CFG.CorpusPath)

	model, tok, err := LoadCheckpoint(docs, "")
	if err != nil || model == nil || tok == nil {
		if len(docs) == 0 {
			docs = []string{"Hello."}
		}
		tok = NewEvolvingTokenizer(docs)
		model = NewGPT(tok)
	}

	model.MaybeExpandVocab(tok.VocabSize)

	// Start background trainer
	stop := make(chan struct{})
	go backgroundTrainer(db, model, tok, stop)

	fmt.Println("macrogpt is alive. Type and press Enter. Ctrl+C to exit.\n")

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("> ")
		if !scanner.Scan() {
			break
		}
		userText := strings.TrimSpace(scanner.Text())
		if userText == "" {
			continue
		}

		dbAddMessage(db, "user", userText)
		updateReservoirCorpus(db, CFG.CorpusPath, CFG.MaxCorpusLines)

		prompt := buildPromptFromMemory(db, userText)
		answer := model.GenerateSentence(prompt)
		if answer == "" {
			answer = "..."
		}

		fmt.Println(answer)
		dbAddMessage(db, "assistant", answer)
	}

	close(stop)
	SaveCheckpoint(model, tok, "")
}
