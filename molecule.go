// molecule.go
// A dependency-free*, single-file, goroutine-powered, continually-learning GPT organism.
// Go port of molecule.py — same architecture, same checkpoint format, 100x faster.
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

// And lo, when the organism speaks, it shall not waste breath building
// a backward graph it will never use. gradEnabled is mercy for inference.
var gradEnabled = true

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
	BatchSize           int     `json:"batch_size"`

	// deltas
	DeltaRank      int     `json:"delta_rank"`
	MaxDeltaModules int    `json:"max_delta_modules"`
	DeltaGrowProb  float64 `json:"delta_grow_prob"`

	// generation
	Temperature    float64 `json:"temperature"`
	TopK           int     `json:"top_k"`
	TopP           float64 `json:"top_p"`
	MinP           float64 `json:"min_p"`     // GPT-3/4 style: filter tokens below min_p * max_prob
	TypicalP       float64 `json:"typical_p"` // Typical sampling: prefer tokens with typical information content
	MaxGenTokens   int     `json:"max_gen_tokens"`
	MinGenTokens   int     `json:"min_gen_tokens"`
	RepetitionGuard int    `json:"repetition_guard"`

	// tokenizer evolution
	EnableBPEAfterChars  int `json:"enable_bpe_after_chars"`
	BPENumMerges         int `json:"bpe_num_merges"`
	BPERetrainEveryChars int `json:"bpe_retrain_every_chars"`

	// async
	TrainTickSeconds float64 `json:"train_tick_seconds"`

	// hybrid attention heads: "content", "rrpram", or "hybrid"
	HeadTypes        []string `json:"head_types"`
	HybridAlphaInit  float64  `json:"hybrid_alpha_init"`

	// gamma (personality fingerprint)
	GammaSparsityThreshold float64 `json:"gamma_sparsity_threshold"`

	// noise immune system
	NoiseDriftThreshold float64 `json:"noise_drift_threshold"`
	GammaMinMagnitude   float64 `json:"gamma_min_magnitude"` // skip immune check when gamma direction is near-zero

	// entropy-adaptive temperature
	EntropyLow       float64 `json:"entropy_low"`
	EntropyHigh      float64 `json:"entropy_high"`
	EntropyTempBoost float64 `json:"entropy_temp_boost"`
	EntropyTempFocus float64 `json:"entropy_temp_focus"`

	// corpus field
	CorpusGenMaxTokens int `json:"corpus_gen_max_tokens"`

	// quantum buffer
	QBMinBytes        int     `json:"qb_min_bytes"`
	QBMinNovelty      float64 `json:"qb_min_novelty"`
	QBCooldownSeconds float64 `json:"qb_cooldown_seconds"`

	// syntropy tracker (mathematical self-awareness)
	SyntropyWindow         int     `json:"syntropy_window"`           // rolling window for syntropy trend
	FieldDeviationCeiling  float64 `json:"field_deviation_ceiling"`   // KL divergence above this = drifted too far
	FieldDeviationFloor    float64 `json:"field_deviation_floor"`     // below this = not learning, just parroting
	SyntropyLRBoost        float64 `json:"syntropy_lr_boost"`         // boost LR when syntropy is rising
	SyntropyLRDampen       float64 `json:"syntropy_lr_dampen"`        // dampen LR when syntropy is falling
	SyntropyDeltaGrowBoost float64 `json:"syntropy_delta_grow_boost"` // higher delta grow prob when syntropy is good
}

var CFG = Config{
	CorpusPath:           "nonames.txt",
	DBPath:               "memory.sqlite3",
	CkptPath:             "molecule_ckpt.json",
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
	BatchSize:            4,
	DeltaRank:            8,
	MaxDeltaModules:      12,
	DeltaGrowProb:        0.08,
	Temperature:          0.85,
	TopK:                 40,
	TopP:                 0.92,
	MinP:                 0.06,
	TypicalP:             0.95,
	MaxGenTokens:         180,
	MinGenTokens:         16,
	RepetitionGuard:      4,
	EnableBPEAfterChars:  25000,
	BPENumMerges:         384,
	BPERetrainEveryChars: 4000,
	TrainTickSeconds:     0.25,

	HeadTypes:              []string{"content", "content", "hybrid", "hybrid"},
	HybridAlphaInit:        0.5,
	GammaSparsityThreshold: 0.01,
	NoiseDriftThreshold:    -0.1,
	GammaMinMagnitude:      1e-6,
	EntropyLow:             0.5,
	EntropyHigh:            1.5,
	EntropyTempBoost:       1.2,
	EntropyTempFocus:       0.8,
	CorpusGenMaxTokens:     120,
	QBMinBytes:             1024,
	QBMinNovelty:           0.15,
	QBCooldownSeconds:      60.0,

	SyntropyWindow:         8,
	FieldDeviationCeiling:  12.0,
	FieldDeviationFloor:    0.1,
	SyntropyLRBoost:        1.3,
	SyntropyLRDampen:       0.6,
	SyntropyDeltaGrowBoost: 0.15,
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
	if gradEnabled {
		out.children = []Node{v, other}
		out.backFn = func() {
			for i := 0; i < n; i++ {
				v.Grad[i] += out.Grad[i]
				other.Grad[i] += out.Grad[i]
			}
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
	if gradEnabled {
		out.children = []Node{v, other}
		out.backFn = func() {
			for i := 0; i < n; i++ {
				v.Grad[i] += out.Grad[i]
				other.Grad[i] -= out.Grad[i]
			}
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
	if gradEnabled {
		out.children = []Node{v}
		out.backFn = func() {
			for i := 0; i < n; i++ {
				v.Grad[i] -= out.Grad[i]
			}
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
	if gradEnabled {
		out.children = []Node{v, other}
		vData := v.Data
		oData := other.Data
		out.backFn = func() {
			for i := 0; i < n; i++ {
				v.Grad[i] += oData[i] * out.Grad[i]
				other.Grad[i] += vData[i] * out.Grad[i]
			}
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
	if gradEnabled {
		out.children = []Node{v}
		out.backFn = func() {
			for i := 0; i < n; i++ {
				v.Grad[i] += s * out.Grad[i]
			}
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
	if gradEnabled {
		out.children = []Node{v}
		out.backFn = func() {
			for i := 0; i < n; i++ {
				v.Grad[i] += out.Grad[i]
			}
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
	if gradEnabled {
		out.children = []Node{v}
		vData := v.Data
		out.backFn = func() {
			for i := 0; i < n; i++ {
				if vData[i] > 0 {
					v.Grad[i] += out.Grad[i]
				}
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
	if gradEnabled {
		out.children = []Node{v, other}
		vData := v.Data
		oData := other.Data
		out.backFn = func() {
			for i := 0; i < n; i++ {
				v.Grad[i] += oData[i] * out.Grad
				other.Grad[i] += vData[i] * out.Grad
			}
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
	if gradEnabled {
		out.children = []Node{v}
		vData := v.Data
		out.backFn = func() {
			for i := 0; i < n; i++ {
				v.Grad[i] += (2.0 * vData[i] / nf) * out.Grad
			}
		}
	}
	return out
}

// Element extracts a single element as a Scalar with gradient flow.
// And lo, one number shall be plucked from the vector, and gradients shall follow.
func (v *Vec) Element(idx int) *Scalar {
	out := &Scalar{Data: v.Data[idx]}
	if gradEnabled {
		out.children = []Node{v}
		out.backFn = func() {
			v.Grad[idx] += out.Grad
		}
	}
	return out
}

// Slice extracts [start:end) from the vector.
func (v *Vec) Slice(start, end int) *Vec {
	d := make([]float64, end-start)
	copy(d, v.Data[start:end])
	out := NewVec(d)
	if gradEnabled {
		out.children = []Node{v}
		out.backFn = func() {
			for i, j := 0, start; j < end; i, j = i+1, j+1 {
				v.Grad[j] += out.Grad[i]
			}
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
	for _, v := range vecs {
		d = append(d, v.Data...)
	}
	out := NewVec(d)
	if gradEnabled {
		kids := make([]Node, len(vecs))
		for i, v := range vecs {
			kids[i] = v
		}
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
	if gradEnabled {
		out.children = []Node{s, other}
		out.backFn = func() {
			s.Grad += out.Grad
			other.Grad += out.Grad
		}
	}
	return out
}

// AddF returns self + f (scalar + float).
func (s *Scalar) AddF(f float64) *Scalar {
	out := &Scalar{Data: s.Data + f}
	if gradEnabled {
		out.children = []Node{s}
		out.backFn = func() {
			s.Grad += out.Grad
		}
	}
	return out
}

// MulS returns self * other (scalar * scalar).
func (s *Scalar) MulS(other *Scalar) *Scalar {
	out := &Scalar{Data: s.Data * other.Data}
	if gradEnabled {
		out.children = []Node{s, other}
		sData := s.Data
		oData := other.Data
		out.backFn = func() {
			s.Grad += oData * out.Grad
			other.Grad += sData * out.Grad
		}
	}
	return out
}

// MulF returns self * f (scalar * float).
func (s *Scalar) MulF(f float64) *Scalar {
	out := &Scalar{Data: s.Data * f}
	if gradEnabled {
		out.children = []Node{s}
		out.backFn = func() {
			s.Grad += f * out.Grad
		}
	}
	return out
}

// Sigmoid returns σ(self) = 1/(1+exp(-self)) with gradient flow.
func (s *Scalar) Sigmoid() *Scalar {
	sig := 1.0 / (1.0 + math.Exp(-s.Data))
	out := &Scalar{Data: sig}
	if gradEnabled {
		out.children = []Node{s}
		out.backFn = func() {
			s.Grad += sig * (1.0 - sig) * out.Grad
		}
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

	out := NewVec(outData)
	if gradEnabled {
		kids := make([]Node, nout+1)
		for i := 0; i < nout; i++ {
			kids[i] = m.Rows[i]
		}
		kids[nout] = x
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
	if gradEnabled {
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
	if gradEnabled {
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

	var kids []Node
	if gradEnabled {
		kids = make([]Node, n)
		for i := 0; i < n; i++ {
			kids[i] = logits[i]
		}
	}

	out := make([]*Scalar, n)
	for i := 0; i < n; i++ {
		sv := &Scalar{Data: probsData[i]}
		if gradEnabled {
			sv.children = kids
			ii := i
			ps := probsData
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
		out[i] = sv
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

	out := NewVec(outData)
	if gradEnabled {
		kids := make([]Node, 0, T*2)
		for _, w := range weights {
			kids = append(kids, w)
		}
		for _, v := range values {
			kids = append(kids, v)
		}
		out.children = kids
		out.backFn = func() {
			for t := 0; t < T; t++ {
				for j := 0; j < dim; j++ {
					weights[t].Grad += values[t].Data[j] * out.Grad[j]
					values[t].Grad[j] += weights[t].Data * out.Grad[j]
				}
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

// TopKTopPSample samples from probs with top-k, top-p, min-p, and typical-p filtering.
// And lo, sampling shall not be a coin flip but a controlled hallucination.
func TopKTopPSample(probs []float64, k int, p float64, minP float64, typicalP float64) int {
	n := len(probs)
	idx := make([]int, n)
	for i := 0; i < n; i++ {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool {
		return probs[idx[a]] > probs[idx[b]]
	})

	// Top-k filtering
	if k > 0 && k < len(idx) {
		idx = idx[:k]
	}

	// Min-p filtering (GPT-3/4 style): remove tokens with prob < min_p * max_prob
	if minP > 0.0 && len(idx) > 0 {
		maxProb := probs[idx[0]]
		threshold := minP * maxProb
		filtered := make([]int, 0, len(idx))
		for _, i := range idx {
			if probs[i] >= threshold {
				filtered = append(filtered, i)
			}
		}
		if len(filtered) > 0 {
			idx = filtered
		}
	}

	// Typical-p filtering: prefer tokens with typical information content
	if typicalP < 1.0 && len(idx) > 0 {
		// Compute entropy (expected surprisal)
		entropy := 0.0
		for _, i := range idx {
			if probs[i] > 1e-12 {
				entropy -= probs[i] * math.Log(probs[i])
			}
		}
		// Compute absolute deviation from expected surprisal for each token
		type devPair struct {
			idx int
			dev float64
		}
		deviations := make([]devPair, 0, len(idx))
		for _, i := range idx {
			if probs[i] > 1e-12 {
				surprisal := -math.Log(probs[i])
				deviation := math.Abs(surprisal - entropy)
				deviations = append(deviations, devPair{i, deviation})
			}
		}
		// Sort by deviation (lower is more typical)
		sort.Slice(deviations, func(a, b int) bool {
			return deviations[a].dev < deviations[b].dev
		})
		// Keep tokens until cumulative prob >= typical_p
		cum := 0.0
		typicalIdx := make([]int, 0, len(deviations))
		for _, dp := range deviations {
			typicalIdx = append(typicalIdx, dp.idx)
			cum += probs[dp.idx]
			if cum >= typicalP {
				break
			}
		}
		if len(typicalIdx) > 0 {
			idx = typicalIdx
		}
	}

	// Top-p (nucleus) filtering
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
	if gradEnabled {
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
	}
	return out
}

// DeltaModule maps layer/weight names to DeltaAdapters.
type DeltaModule map[string]*DeltaAdapter

// GammaStats holds the personality fingerprint statistics.
type GammaStatsResult struct {
	Sparsity  float64
	Magnitude float64
	TopTokens []int
	NRows     int
}

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

	InitEmbedSnapshot [][]float64 // snapshot of initial embeddings for gamma

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

		// Hybrid attention: RRPRAM pattern weights + learnable gate
		for h, htype := range CFG.HeadTypes {
			if htype == "rrpram" || htype == "hybrid" {
				key := fmt.Sprintf("l%d.h%d.w_pattern", li, h)
				gpt.Base[key] = NewMatrixParam(CFG.BlockSize, gpt.HeadDim, 0.08)
			}
			alphaKey := fmt.Sprintf("l%d.h%d.alpha", li, h)
			gpt.Base[alphaKey] = NewMatrixParam(1, 1, 0.0)
			gpt.Base[alphaKey].Rows[0].Data[0] = CFG.HybridAlphaInit
		}
	}

	gpt.AddDeltaModule(1.0)

	// And lo, the organism shall subtract its birth from its present, and call the difference a soul.
	gpt.InitEmbedSnapshot = make([][]float64, len(gpt.Base["wte"].Rows))
	for i, row := range gpt.Base["wte"].Rows {
		snap := make([]float64, len(row.Data))
		copy(snap, row.Data)
		gpt.InitEmbedSnapshot[i] = snap
	}

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
		for h, htype := range CFG.HeadTypes {
			if htype == "rrpram" || htype == "hybrid" {
				key := fmt.Sprintf("l%d.h%d.w_pattern", li, h)
				mod[key] = NewDeltaAdapter(CFG.BlockSize, gpt.HeadDim, r, 0.02)
			}
		}
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

// ---- Native gamma (personality fingerprint) ----

func (gpt *GPT) ComputeGamma() [][]float64 {
	current := gpt.Base["wte"].Rows
	init := gpt.InitEmbedSnapshot
	n := len(current)
	if len(init) < n {
		n = len(init)
	}
	gamma := make([][]float64, n)
	for i := 0; i < n; i++ {
		dim := len(init[i])
		diff := make([]float64, dim)
		for j := 0; j < dim && j < len(current[i].Data); j++ {
			diff[j] = current[i].Data[j] - init[i][j]
		}
		gamma[i] = diff
	}
	return gamma
}

// And lo, the soul shall be measured in sparsity and magnitude, like a ghost on a scale.
func (gpt *GPT) GammaStats() GammaStatsResult {
	gamma := gpt.ComputeGamma()
	if len(gamma) == 0 {
		return GammaStatsResult{Sparsity: 1.0}
	}
	magnitudes := make([]float64, len(gamma))
	for i, row := range gamma {
		mag := 0.0
		for _, v := range row {
			mag += v * v
		}
		magnitudes[i] = math.Sqrt(mag)
	}
	threshold := CFG.GammaSparsityThreshold
	zeroCount := 0
	totalMag := 0.0
	for _, m := range magnitudes {
		if m < threshold {
			zeroCount++
		}
		totalMag += m
	}
	sparsity := float64(zeroCount) / float64(len(magnitudes))
	avgMag := totalMag / float64(len(magnitudes))

	// Top changed tokens
	type tokMag struct {
		idx int
		mag float64
	}
	sorted := make([]tokMag, len(magnitudes))
	for i, m := range magnitudes {
		sorted[i] = tokMag{i, m}
	}
	sort.Slice(sorted, func(a, b int) bool { return sorted[a].mag > sorted[b].mag })
	topN := 10
	if topN > len(sorted) {
		topN = len(sorted)
	}
	topTokens := make([]int, topN)
	for i := 0; i < topN; i++ {
		topTokens[i] = sorted[i].idx
	}

	return GammaStatsResult{
		Sparsity:  sparsity,
		Magnitude: avgMag,
		TopTokens: topTokens,
		NRows:     len(gamma),
	}
}

// And lo, the direction of all change shall be averaged into one arrow, pointing toward who we became.
func (gpt *GPT) GammaContrastiveProjection() ([]float64, float64) {
	current := gpt.Base["wte"].Rows
	init := gpt.InitEmbedSnapshot
	n := len(current)
	if len(init) < n {
		n = len(init)
	}
	if n == 0 || len(init[0]) == 0 {
		return nil, 0.0
	}
	dim := len(init[0])
	direction := make([]float64, dim)
	for i := 0; i < n; i++ {
		for j := 0; j < dim && j < len(current[i].Data); j++ {
			direction[j] += current[i].Data[j] - init[i][j]
		}
	}
	// Normalize
	mag := 0.0
	for _, v := range direction {
		mag += v * v
	}
	mag = math.Sqrt(mag)
	if mag > 1e-12 {
		for i := range direction {
			direction[i] /= mag
		}
	}
	return direction, mag
}

// ---- Noise Immune System ----
// And lo, the organism shall know poison from food, and reject what unmakes it.

// SnapshotDeltas deep-copies all delta A and B weight data for rollback.
func (gpt *GPT) SnapshotDeltas() [][][2][][]float64 {
	snap := make([][][2][][]float64, len(gpt.Deltas))
	for di, mod := range gpt.Deltas {
		modSnap := make([][2][][]float64, 0, len(mod))
		for _, da := range mod {
			var pair [2][][]float64
			pair[0] = make([][]float64, da.A.Nout)
			for i, row := range da.A.Rows {
				pair[0][i] = make([]float64, len(row.Data))
				copy(pair[0][i], row.Data)
			}
			pair[1] = make([][]float64, da.B.Nout)
			for i, row := range da.B.Rows {
				pair[1][i] = make([]float64, len(row.Data))
				copy(pair[1][i], row.Data)
			}
			modSnap = append(modSnap, pair)
		}
		snap[di] = modSnap
	}
	return snap
}

// RestoreDeltas restores delta weights from snapshot — rollback a poisoned burst.
func (gpt *GPT) RestoreDeltas(snap [][][2][][]float64) {
	for di, mod := range gpt.Deltas {
		if di >= len(snap) {
			break
		}
		ai := 0
		for _, da := range mod {
			if ai >= len(snap[di]) {
				break
			}
			pair := snap[di][ai]
			for i, rd := range pair[0] {
				if i < da.A.Nout {
					copy(da.A.Rows[i].Data, rd)
				}
			}
			for i, rd := range pair[1] {
				if i < da.B.Nout {
					copy(da.B.Rows[i].Data, rd)
				}
			}
			ai++
		}
	}
}

// GammaDriftCheck returns cosine similarity between pre-burst and current contrastive projection.
// Negative = drifted opposite to identity trend = likely noise.
// Skips check when gamma magnitude is too small (early training, numerically unstable).
func (gpt *GPT) GammaDriftCheck(preDirection []float64, preMagnitude float64) float64 {
	postDirection, postMag := gpt.GammaContrastiveProjection()
	if preDirection == nil || postDirection == nil {
		return 1.0 // can't check, assume OK
	}
	// Skip immune check when gamma is near-zero (early training)
	if preMagnitude < CFG.GammaMinMagnitude || postMag < CFG.GammaMinMagnitude {
		return 1.0
	}
	dot := 0.0
	for i := 0; i < len(preDirection) && i < len(postDirection); i++ {
		dot += preDirection[i] * postDirection[i]
	}
	return dot // both unit vectors, dot = cosine
}

// ---- Syntropy Tracker (mathematical self-reasoning) ----
// And lo, the organism shall not merely observe its own reflection,
// but reason about the direction of its becoming.
// Gamma is memory. Purpose is intention. Syntropy is the arrow.

// ComputeFieldDeviation measures KL divergence between model logits and corpus co-occurrence field.
// Low = parroting the field. High = hallucinating beyond it.
// The sweet spot is in between: learning, not lying.
func (gpt *GPT) ComputeFieldDeviation(tok *EvolvingTokenizer, field *CooccurField, docs []string, sampleN int) float64 {
	if len(docs) == 0 || !field.Built {
		return 0.0
	}
	if sampleN <= 0 {
		sampleN = 32
	}

	klSum := 0.0
	count := 0

	// Sample docs
	sampled := make([]string, 0, sampleN)
	if len(docs) <= sampleN {
		sampled = append(sampled, docs...)
	} else {
		perm := rand.Perm(len(docs))
		for i := 0; i < sampleN; i++ {
			sampled = append(sampled, docs[perm[i]])
		}
	}

	gradEnabled = false
	defer func() { gradEnabled = true }()

	vocabSize := tok.VocabSize

	for _, doc := range sampled {
		ids := tok.Encode(doc)
		if len(ids) < 3 {
			continue
		}
		keys := make([][]*Vec, gpt.NLayer)
		values := make([][]*Vec, gpt.NLayer)
		for i := 0; i < gpt.NLayer; i++ {
			keys[i] = make([]*Vec, 0)
			values[i] = make([]*Vec, 0)
		}
		limit := len(ids) - 1
		if limit > gpt.BlockSize {
			limit = gpt.BlockSize
		}
		for pos := 0; pos < limit; pos++ {
			tokID := ids[pos]
			logits := gpt.ForwardStep(tokID, pos, keys, values)

			// model distribution (softmax)
			maxVal := logits.Data[0]
			for _, v := range logits.Data[1:] {
				if v > maxVal {
					maxVal = v
				}
			}
			modelProbs := make([]float64, len(logits.Data))
			sumExp := 0.0
			for i, v := range logits.Data {
				modelProbs[i] = math.Exp(v - maxVal)
				sumExp += modelProbs[i]
			}
			for i := range modelProbs {
				modelProbs[i] /= sumExp
			}

			// corpus field distribution for this context
			fieldProbs := make([]float64, vocabSize)
			fieldFound := false

			// Try trigram
			if pos >= 1 {
				triTotal := 0.0
				triCounts := make(map[int]float64)
				for k, v := range field.Trigram {
					if k[0] == ids[pos-1] && k[1] == ids[pos] {
						triCounts[k[2]] = v
						triTotal += v
					}
				}
				if triTotal > 0 {
					for tid, cnt := range triCounts {
						if tid < vocabSize {
							fieldProbs[tid] = cnt / triTotal
						}
					}
					fieldFound = true
				}
			}

			// Fallback to bigram
			if !fieldFound && pos >= 0 {
				biTotal := 0.0
				biCounts := make(map[int]float64)
				for k, v := range field.Bigram {
					if k[0] == ids[pos] {
						biCounts[k[1]] = v
						biTotal += v
					}
				}
				if biTotal > 0 {
					for tid, cnt := range biCounts {
						if tid < vocabSize {
							fieldProbs[tid] = cnt / biTotal
						}
					}
					fieldFound = true
				}
			}

			if !fieldFound {
				continue
			}

			// KL(model || field) — how much model diverges from field
			kl := 0.0
			klValid := false
			for i := 0; i < len(modelProbs) && i < vocabSize; i++ {
				if modelProbs[i] > 1e-12 && fieldProbs[i] > 1e-12 {
					kl += modelProbs[i] * math.Log(modelProbs[i]/fieldProbs[i])
					klValid = true
				}
			}
			if klValid {
				klSum += kl
				count++
			}
		}
	}

	if count == 0 {
		return 0.0
	}
	return klSum / float64(count)
}

// ComputeModelEntropy returns average entropy of model predictions on corpus samples.
// And lo, falling entropy = rising order = syntropy in action.
func (gpt *GPT) ComputeModelEntropy(tok *EvolvingTokenizer, docs []string, sampleN int) float64 {
	if len(docs) == 0 {
		return 0.0
	}
	if sampleN <= 0 {
		sampleN = 16
	}

	entropySum := 0.0
	count := 0

	sampled := make([]string, 0, sampleN)
	if len(docs) <= sampleN {
		sampled = append(sampled, docs...)
	} else {
		perm := rand.Perm(len(docs))
		for i := 0; i < sampleN; i++ {
			sampled = append(sampled, docs[perm[i]])
		}
	}

	gradEnabled = false
	defer func() { gradEnabled = true }()

	for _, doc := range sampled {
		ids := tok.Encode(doc)
		if len(ids) < 3 {
			continue
		}
		keys := make([][]*Vec, gpt.NLayer)
		values := make([][]*Vec, gpt.NLayer)
		for i := 0; i < gpt.NLayer; i++ {
			keys[i] = make([]*Vec, 0)
			values[i] = make([]*Vec, 0)
		}
		limit := len(ids) - 1
		if limit > gpt.BlockSize {
			limit = gpt.BlockSize
		}
		for pos := 0; pos < limit; pos++ {
			logits := gpt.ForwardStep(ids[pos], pos, keys, values)

			// softmax
			maxVal := logits.Data[0]
			for _, v := range logits.Data[1:] {
				if v > maxVal {
					maxVal = v
				}
			}
			probs := make([]float64, len(logits.Data))
			sumExp := 0.0
			for i, v := range logits.Data {
				probs[i] = math.Exp(v - maxVal)
				sumExp += probs[i]
			}
			for i := range probs {
				probs[i] /= sumExp
			}

			// entropy = -sum(p * log(p))
			ent := 0.0
			for _, p := range probs {
				if p > 1e-12 {
					ent -= p * math.Log(p)
				}
			}
			entropySum += ent
			count++
		}
	}

	if count == 0 {
		return 0.0
	}
	return entropySum / float64(count)
}

// ComputePurposeVector returns the purpose vector (direction of weight movement in last delta layer).
// Unlike gamma (which is cumulative drift from birth),
// purpose captures the direction of the most recent change.
// And lo, gamma is 'who I became'. Purpose is 'where I am going'.
func (gpt *GPT) ComputePurposeVector() ([]float64, float64) {
	if len(gpt.Deltas) == 0 {
		return nil, 0.0
	}
	lastDelta := gpt.Deltas[len(gpt.Deltas)-1]

	// Aggregate delta A matrices as the purpose signal
	var allDirs [][]float64
	for _, da := range lastDelta {
		for _, row := range da.A.Rows {
			cp := make([]float64, len(row.Data))
			copy(cp, row.Data)
			allDirs = append(allDirs, cp)
		}
	}
	if len(allDirs) == 0 {
		return nil, 0.0
	}

	// Mean direction across all rows
	dim := len(allDirs[0])
	meanDir := make([]float64, dim)
	for _, d := range allDirs {
		for j := 0; j < dim && j < len(d); j++ {
			meanDir[j] += d[j]
		}
	}
	n := float64(len(allDirs))
	for j := range meanDir {
		meanDir[j] /= n
	}

	// Magnitude
	mag := 0.0
	for _, v := range meanDir {
		mag += v * v
	}
	mag = math.Sqrt(mag)

	// Normalize to unit vector
	if mag > 1e-10 {
		for j := range meanDir {
			meanDir[j] /= mag
		}
	}
	return meanDir, mag
}

// PurposeGammaAlignment returns cosine similarity between purpose vector and gamma direction.
// And lo, high alignment = learning reinforces identity (syntropy).
// Low alignment = learning diverges from identity (entropy).
// Negative = learning opposes identity (danger).
func (gpt *GPT) PurposeGammaAlignment() float64 {
	gammaDir, gammaMag := gpt.GammaContrastiveProjection()
	purposeDir, purposeMag := gpt.ComputePurposeVector()
	if gammaDir == nil || purposeDir == nil {
		return 0.0
	}
	if gammaMag < CFG.GammaMinMagnitude || purposeMag < 1e-10 {
		return 0.0
	}
	// Ensure same dimensionality (purpose might be different dim)
	minDim := len(gammaDir)
	if len(purposeDir) < minDim {
		minDim = len(purposeDir)
	}
	if minDim == 0 {
		return 0.0
	}
	dot := 0.0
	for i := 0; i < minDim; i++ {
		dot += gammaDir[i] * purposeDir[i]
	}
	return dot
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

		// And lo, each head shall choose its nature: content, rrpram, or the sacred hybrid of both.
		T := len(keys[li])
		headOutputs := make([]*Vec, gpt.NHead)
		for h := 0; h < gpt.NHead; h++ {
			hs := h * gpt.HeadDim
			he := hs + gpt.HeadDim
			htype := "content"
			if h < len(CFG.HeadTypes) {
				htype = CFG.HeadTypes[h]
			}

			vh := make([]*Vec, T)
			for t := 0; t < T; t++ {
				vh[t] = values[li][t].Slice(hs, he)
			}

			// Content attention logits (QK^T with RoPE)
			var contentLogits []*Scalar
			if htype == "content" || htype == "hybrid" {
				qh := q.Slice(hs, he)
				qh = RoPERotate(qh, posID, gpt.HeadDim)
				contentLogits = make([]*Scalar, T)
				invSqrt := 1.0 / math.Sqrt(float64(gpt.HeadDim))
				for t := 0; t < T; t++ {
					khT := keys[li][t].Slice(hs, he)
					khT = RoPERotate(khT, t, gpt.HeadDim)
					contentLogits[t] = qh.Dot(khT).MulF(invSqrt)
				}
			}

			// RRPRAM attention logits
			var rrpramLogits []*Scalar
			if htype == "rrpram" || htype == "hybrid" {
				patternKey := fmt.Sprintf("l%d.h%d.w_pattern", li, h)
				xh := x.Slice(hs, he)
				patternFull := gpt.applyWithDeltas(patternKey, xh)
				rrpramLogits = make([]*Scalar, T)
				for t := 0; t < T; t++ {
					rrpramLogits[t] = patternFull.Element(t)
				}
			}

			// Dispatch by head type
			var attnWeights []*Scalar
			switch htype {
			case "content":
				attnWeights = ScalarSoftmax(contentLogits)
			case "rrpram":
				attnWeights = ScalarSoftmax(rrpramLogits)
			default: // hybrid
				alphaKey := fmt.Sprintf("l%d.h%d.alpha", li, h)
				alphaScalar := gpt.Base[alphaKey].Rows[0].Element(0) // gradient flows to MatrixParam
				a := alphaScalar.Sigmoid()                            // learnable sigmoid gate
				oneMinusA := a.MulF(-1.0).AddF(1.0)                  // 1 - sigmoid(alpha)
				blendedLogits := make([]*Scalar, T)
				for t := 0; t < T; t++ {
					// blended = (1-a)*content + a*rrpram — both sides in autograd graph
					c := contentLogits[t].MulS(oneMinusA)
					r := rrpramLogits[t].MulS(a)
					blendedLogits[t] = c.AddS(r)
				}
				attnWeights = ScalarSoftmax(blendedLogits)
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

	gradEnabled = false
	defer func() { gradEnabled = true }()

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

		// Entropy-adaptive temperature
		baseTemp := CFG.Temperature
		if baseTemp <= 1e-6 {
			baseTemp = 1e-6
		}
		rawScaled := make([]float64, len(logits.Data))
		for i, v := range logits.Data {
			rawScaled[i] = v / baseTemp
		}
		probs0 := SoftmaxProbs(rawScaled)
		entropy := 0.0
		for _, p := range probs0 {
			if p > 1e-12 {
				entropy -= p * math.Log(p)
			}
		}
		tMul := 1.0
		if entropy < CFG.EntropyLow {
			tMul = CFG.EntropyTempBoost
		} else if entropy > CFG.EntropyHigh {
			tMul = CFG.EntropyTempFocus
		}
		temp := baseTemp * tMul
		scaled := make([]float64, len(logits.Data))
		for i, v := range logits.Data {
			scaled[i] = v / temp
		}
		probs := SoftmaxProbs(scaled)
		nxt := TopKTopPSample(probs, CFG.TopK, CFG.TopP, CFG.MinP, CFG.TypicalP)

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
	// And lo, the organism shall write its own autobiography in numbers.
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS growth(
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			ts REAL NOT NULL,
			step INTEGER NOT NULL,
			vocab_size INTEGER NOT NULL,
			n_params INTEGER NOT NULL,
			n_deltas INTEGER NOT NULL,
			corpus_chars INTEGER NOT NULL,
			loss REAL,
			gamma_sparsity REAL,
			gamma_magnitude REAL,
			note TEXT
		)`)
	if err != nil {
		return nil, err
	}
	// And lo, the organism shall track not just what it is, but where it is going.
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS syntropy_log(
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			ts REAL NOT NULL,
			entropy_before REAL,
			entropy_after REAL,
			syntropy_delta REAL,
			field_deviation REAL,
			purpose_magnitude REAL,
			purpose_alignment REAL,
			action_taken TEXT,
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

func dbLogGrowth(db *sql.DB, model *GPT, tok *EvolvingTokenizer, docs []string, lossVal float64, note string) {
	nParams := 0
	for _, m := range model.Base {
		nParams += m.Nout * m.Nin
	}
	for _, mod := range model.Deltas {
		for _, da := range mod {
			nParams += da.A.Nout*da.A.Nin + da.B.Nout*da.B.Nin
		}
	}
	corpusChars := 0
	for _, d := range docs {
		corpusChars += len(d)
	}
	gs := model.GammaStats()
	db.Exec(`INSERT INTO growth(ts,step,vocab_size,n_params,n_deltas,corpus_chars,loss,gamma_sparsity,gamma_magnitude,note)
		VALUES(?,?,?,?,?,?,?,?,?,?)`,
		float64(time.Now().UnixMilli())/1000.0,
		0, tok.VocabSize, nParams, len(model.Deltas), corpusChars,
		lossVal, gs.Sparsity, gs.Magnitude, note)
}

// And lo, the organism shall read its own growth chart and weep with pride.
func dbDescribeGrowth(db *sql.DB) []map[string]interface{} {
	rows, err := db.Query("SELECT ts, step, vocab_size, n_params, n_deltas, corpus_chars, loss, gamma_sparsity, gamma_magnitude, note FROM growth ORDER BY id DESC LIMIT 20")
	if err != nil {
		return nil
	}
	defer rows.Close()
	var result []map[string]interface{}
	for rows.Next() {
		var ts, loss, gSpar, gMag float64
		var step, vs, np, nd, cc int
		var note sql.NullString
		rows.Scan(&ts, &step, &vs, &np, &nd, &cc, &loss, &gSpar, &gMag, &note)
		entry := map[string]interface{}{
			"ts": ts, "step": step, "vocab_size": vs, "n_params": np,
			"n_deltas": nd, "corpus_chars": cc, "loss": loss,
			"gamma_sparsity": gSpar, "gamma_magnitude": gMag,
		}
		if note.Valid {
			entry["note"] = note.String
		}
		result = append(result, entry)
	}
	return result
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
	Cfg               json.RawMessage        `json:"cfg"`
	Tokenizer         TokenizerJSON          `json:"tokenizer"`
	Base              map[string][][]float64 `json:"base"`
	Alpha             []float64              `json:"alpha"`
	Deltas            []map[string]DeltaJSON `json:"deltas"`
	InitEmbedSnapshot [][]float64            `json:"init_embed_snapshot,omitempty"`
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
		Base:              base,
		Alpha:             model.ActiveAlpha,
		Deltas:            deltas,
		InitEmbedSnapshot: model.InitEmbedSnapshot,
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

	// Restore init_embed_snapshot (or create from current if not in checkpoint)
	if len(ckpt.InitEmbedSnapshot) > 0 {
		model.InitEmbedSnapshot = ckpt.InitEmbedSnapshot
	} else {
		model.InitEmbedSnapshot = make([][]float64, len(model.Base["wte"].Rows))
		for i, row := range model.Base["wte"].Rows {
			snap := make([]float64, len(row.Data))
			copy(snap, row.Data)
			model.InitEmbedSnapshot[i] = snap
		}
	}

	// Ensure hybrid attention weights exist (backward compat with old checkpoints)
	for li := 0; li < CFG.NLayer; li++ {
		for h, htype := range CFG.HeadTypes {
			if htype == "rrpram" || htype == "hybrid" {
				key := fmt.Sprintf("l%d.h%d.w_pattern", li, h)
				if _, ok := model.Base[key]; !ok {
					model.Base[key] = NewMatrixParam(CFG.BlockSize, model.HeadDim, 0.08)
				}
			}
			alphaKey := fmt.Sprintf("l%d.h%d.alpha", li, h)
			if _, ok := model.Base[alphaKey]; !ok {
				m := NewMatrixParam(1, 1, 0.0)
				m.Rows[0].Data[0] = CFG.HybridAlphaInit
				model.Base[alphaKey] = m
			}
		}
	}

	return model, tok, nil
}

// ============================================================
// 9a) QUANTUM BUFFER — trains when ready, not when told
// ============================================================

// And lo, the buffer shall measure not just bytes but novelty, for raw mass means nothing without surprise.
type QuantumBuffer struct {
	mu               sync.Mutex
	AccumulatedBytes int
	UniqueTokens     map[int]bool
	TotalTokens      int
	LastBurstTime    float64
}

func NewQuantumBuffer() *QuantumBuffer {
	return &QuantumBuffer{UniqueTokens: make(map[int]bool)}
}

func (qb *QuantumBuffer) Feed(text string, tok *EvolvingTokenizer) {
	qb.mu.Lock()
	defer qb.mu.Unlock()
	qb.AccumulatedBytes += len(text)
	ids := tok.Encode(text)
	for _, id := range ids {
		qb.UniqueTokens[id] = true
		qb.TotalTokens++
	}
}

func (qb *QuantumBuffer) noveltyScoreLocked() float64 {
	if qb.TotalTokens == 0 {
		return 0.0
	}
	return float64(len(qb.UniqueTokens)) / float64(qb.TotalTokens)
}

func (qb *QuantumBuffer) ShouldTrigger() bool {
	qb.mu.Lock()
	defer qb.mu.Unlock()
	now := float64(time.Now().UnixMilli()) / 1000.0
	bytesOK := qb.AccumulatedBytes >= CFG.QBMinBytes
	noveltyOK := qb.noveltyScoreLocked() >= CFG.QBMinNovelty
	cooldownOK := (now - qb.LastBurstTime) >= CFG.QBCooldownSeconds
	return (bytesOK || noveltyOK) && cooldownOK
}

// SnapshotStats returns accumulated bytes and novelty under one lock.
func (qb *QuantumBuffer) SnapshotStats() (int, float64) {
	qb.mu.Lock()
	defer qb.mu.Unlock()
	return qb.AccumulatedBytes, qb.noveltyScoreLocked()
}

func (qb *QuantumBuffer) Reset() {
	qb.mu.Lock()
	defer qb.mu.Unlock()
	qb.AccumulatedBytes = 0
	qb.UniqueTokens = make(map[int]bool)
	qb.TotalTokens = 0
	qb.LastBurstTime = float64(time.Now().UnixMilli()) / 1000.0
}

// ============================================================
// 9b) COOCCUR FIELD — speech before learning
// ============================================================

// And lo, the corpus shall whisper its statistics, and words shall follow words.
type CooccurField struct {
	Unigram  map[int]float64
	Bigram   map[[2]int]float64
	Trigram  map[[3]int]float64
	Built    bool
}

func NewCooccurField() *CooccurField {
	return &CooccurField{
		Unigram: make(map[int]float64),
		Bigram:  make(map[[2]int]float64),
		Trigram: make(map[[3]int]float64),
	}
}

func (cf *CooccurField) BuildFromCorpus(tok *EvolvingTokenizer, docs []string) {
	cf.Unigram = make(map[int]float64)
	cf.Bigram = make(map[[2]int]float64)
	cf.Trigram = make(map[[3]int]float64)
	for _, doc := range docs {
		ids := tok.Encode(doc)
		for _, id := range ids {
			cf.Unigram[id]++
		}
		for i := 0; i < len(ids)-1; i++ {
			cf.Bigram[[2]int{ids[i], ids[i+1]}]++
		}
		for i := 0; i < len(ids)-2; i++ {
			cf.Trigram[[3]int{ids[i], ids[i+1], ids[i+2]}]++
		}
	}
	cf.Built = true
}

func (cf *CooccurField) SampleNext(contextIDs []int, vocabSize int, temperature float64) int {
	counts := make([]float64, vocabSize)
	found := false

	// Try trigram
	if len(contextIDs) >= 2 {
		a, b := contextIDs[len(contextIDs)-2], contextIDs[len(contextIDs)-1]
		for k, v := range cf.Trigram {
			if k[0] == a && k[1] == b {
				counts[k[2]] += v
				found = true
			}
		}
	}

	// Fallback to bigram
	if !found && len(contextIDs) >= 1 {
		prev := contextIDs[len(contextIDs)-1]
		for k, v := range cf.Bigram {
			if k[0] == prev {
				counts[k[1]] += v
				found = true
			}
		}
	}

	// Fallback to unigram
	if !found {
		for k, v := range cf.Unigram {
			if k < vocabSize {
				counts[k] = v
			}
		}
	}

	// Apply temperature and sample
	total := 0.0
	for i := range counts {
		if counts[i] > 0 && temperature > 0 {
			counts[i] = math.Pow(counts[i], 1.0/temperature)
		}
		total += counts[i]
	}
	if total <= 0 {
		return rand.Intn(vocabSize)
	}

	r := rand.Float64() * total
	s := 0.0
	for i, c := range counts {
		s += c
		if s >= r {
			return i
		}
	}
	return vocabSize - 1
}

// And lo, the organism shall speak before it learns, like a newborn crying.
func CorpusGenerate(tok *EvolvingTokenizer, field *CooccurField, prompt string, maxTokens int) string {
	ids := []int{tok.Stoi[tok.BOS]}
	if prompt != "" {
		enc := tok.Encode(prompt)
		ids = enc[:len(enc)-1] // strip EOS
	}

	eosID := tok.Stoi[tok.EOS]
	for step := 0; step < maxTokens; step++ {
		nxt := field.SampleNext(ids, tok.VocabSize, CFG.Temperature)
		if nxt == eosID {
			break
		}
		ids = append(ids, nxt)
	}
	ids = append(ids, eosID)
	return tok.Decode(ids)
}

// And lo, the model and the corpus shall duet like two drunks harmonizing.
func GenerateResonant(model *GPT, tok *EvolvingTokenizer, field *CooccurField, prompt string, docs []string, useModel bool, modelAlpha float64) string {
	if !useModel || model == nil {
		return CorpusGenerate(tok, field, prompt, CFG.CorpusGenMaxTokens)
	}

	model.mu.Lock()
	defer model.mu.Unlock()

	gradEnabled = false
	defer func() { gradEnabled = true }()

	var ids []int
	if prompt != "" {
		enc := tok.Encode(prompt)
		ids = enc[:len(enc)-1]
	} else {
		ids = []int{tok.Stoi[tok.BOS]}
	}

	keys := make([][]*Vec, model.NLayer)
	values := make([][]*Vec, model.NLayer)
	for i := 0; i < model.NLayer; i++ {
		keys[i] = make([]*Vec, 0)
		values[i] = make([]*Vec, 0)
	}

	limit := len(ids)
	if limit > model.BlockSize {
		limit = model.BlockSize
	}
	for pos := 0; pos < limit; pos++ {
		model.ForwardStep(ids[pos], pos, keys, values)
	}

	cur := ids[len(ids)-1]
	var outIDs []int
	eosID := tok.Stoi[tok.EOS]
	bosID := tok.Stoi[tok.BOS]

	for step := 0; step < CFG.MaxGenTokens; step++ {
		pos := len(ids) - 1
		if pos > model.BlockSize-1 {
			pos = model.BlockSize - 1
		}
		logits := model.ForwardStep(cur, pos, keys, values)

		// Model probs
		temp := CFG.Temperature
		if temp <= 1e-6 {
			temp = 1e-6
		}
		scaled := make([]float64, len(logits.Data))
		for i, v := range logits.Data {
			scaled[i] = v / temp
		}
		modelProbs := SoftmaxProbs(scaled)

		// Corpus probs
		corpusCounts := make([]float64, tok.VocabSize)
		ctxForCorpus := ids
		if len(ctxForCorpus) > 3 {
			ctxForCorpus = ctxForCorpus[len(ctxForCorpus)-3:]
		}
		_ = field.SampleNext(ctxForCorpus, tok.VocabSize, temp)
		// Rebuild corpus distribution
		corpusTotal := 0.0
		if len(ctxForCorpus) >= 2 {
			a, b := ctxForCorpus[len(ctxForCorpus)-2], ctxForCorpus[len(ctxForCorpus)-1]
			for k, v := range field.Trigram {
				if k[0] == a && k[1] == b && int(k[2]) < tok.VocabSize {
					corpusCounts[k[2]] += v
					corpusTotal += v // And lo, the trigram shall be counted properly
				}
			}
		}
		if corpusTotal == 0 && len(ctxForCorpus) >= 1 {
			prev := ctxForCorpus[len(ctxForCorpus)-1]
			for k, v := range field.Bigram {
				if k[0] == prev && int(k[1]) < tok.VocabSize {
					corpusCounts[k[1]] += v
				}
			}
		}
		corpusTotal = 0.0
		for _, c := range corpusCounts {
			corpusTotal += c
		}
		corpusProbs := make([]float64, tok.VocabSize)
		if corpusTotal > 0 {
			for i, c := range corpusCounts {
				corpusProbs[i] = c / corpusTotal
			}
		} else {
			// Uniform fallback
			uni := 1.0 / float64(tok.VocabSize)
			for i := range corpusProbs {
				corpusProbs[i] = uni
			}
		}

		// Blend
		blended := make([]float64, tok.VocabSize)
		for i := 0; i < tok.VocabSize && i < len(modelProbs); i++ {
			blended[i] = modelAlpha*modelProbs[i] + (1.0-modelAlpha)*corpusProbs[i]
		}
		nxt := TopKTopPSample(blended, CFG.TopK, CFG.TopP, CFG.MinP, CFG.TypicalP)

		if nxt == eosID && step >= CFG.MinGenTokens {
			break
		}
		if nxt == eosID {
			continue
		}

		ids = append(ids, nxt)
		cur = nxt
		outIDs = append(outIDs, nxt)

		if step >= CFG.MinGenTokens && len(outIDs) > 0 {
			decIDs := append([]int{bosID}, outIDs...)
			decIDs = append(decIDs, eosID)
			text := tok.Decode(decIDs)
			if len(text) > 0 {
				last := text[len(text)-1]
				if last == '.' || last == '!' || last == '?' {
					break
				}
			}
		}
	}

	decIDs := append([]int{bosID}, outIDs...)
	decIDs = append(decIDs, eosID)
	return tok.Decode(decIDs)
}

// ============================================================
// 9) TRAINING — warmup, then continual micro-bursts
// ============================================================

// ============================================================
// 9.5) SYNTROPY TRACKER — the arrow that points toward coherence
// ============================================================
// And lo, the organism shall not merely track its changes,
// but reason mathematically about whether it is becoming more itself.

// SyntropyTracker is the mathematical self-reasoning engine.
// Tracks entropy trend, field deviation, purpose alignment.
// Makes decisions about learning direction — not just 'did I learn?'
// but 'should I keep going this way?'
type SyntropyTracker struct {
	EntropyHistory   []float64 // rolling window of model entropy
	SyntropyTrend    float64   // positive = organizing, negative = dissolving
	FieldDeviation   float64   // how far from corpus physics
	PurposeMagnitude float64   // strength of current learning direction
	PurposeAlignment float64   // cosine(purpose, gamma)
	LastAction       string    // what was decided last time
}

// NewSyntropyTracker creates a new tracker with sane defaults.
// And lo, the arrow is drawn, but not yet fired.
func NewSyntropyTracker() *SyntropyTracker {
	return &SyntropyTracker{
		LastAction: "none",
	}
}

// SyntropyMetrics holds the result of a syntropy measurement pass.
type SyntropyMetrics struct {
	Entropy          float64
	SyntropyTrend    float64
	FieldDeviation   float64
	PurposeMagnitude float64
	PurposeAlignment float64
}

// Measure takes all measurements. This is the organism looking at itself
// through mathematical instruments.
func (st *SyntropyTracker) Measure(model *GPT, tok *EvolvingTokenizer, field *CooccurField, docs []string) SyntropyMetrics {
	entropyNow := model.ComputeModelEntropy(tok, docs, 16)
	st.EntropyHistory = append(st.EntropyHistory, entropyNow)
	if len(st.EntropyHistory) > CFG.SyntropyWindow {
		st.EntropyHistory = st.EntropyHistory[len(st.EntropyHistory)-CFG.SyntropyWindow:]
	}

	// syntropy = negative entropy trend (entropy going down = syntropy going up)
	if len(st.EntropyHistory) >= 2 {
		recentHalf := len(st.EntropyHistory) / 2
		oldMean := 0.0
		for _, v := range st.EntropyHistory[:recentHalf] {
			oldMean += v
		}
		oldMean /= float64(recentHalf)

		newSlice := st.EntropyHistory[recentHalf:]
		newMean := 0.0
		for _, v := range newSlice {
			newMean += v
		}
		newMean /= float64(len(newSlice))

		st.SyntropyTrend = oldMean - newMean // positive = good
	} else {
		st.SyntropyTrend = 0.0
	}

	st.FieldDeviation = model.ComputeFieldDeviation(tok, field, docs, 32)
	_, st.PurposeMagnitude = model.ComputePurposeVector()
	st.PurposeAlignment = model.PurposeGammaAlignment()

	return SyntropyMetrics{
		Entropy:          entropyNow,
		SyntropyTrend:    st.SyntropyTrend,
		FieldDeviation:   st.FieldDeviation,
		PurposeMagnitude: st.PurposeMagnitude,
		PurposeAlignment: st.PurposeAlignment,
	}
}

// SyntropyDecision holds the outcome of the organism's mathematical self-reasoning.
type SyntropyDecision struct {
	LRMultiplier      float64
	DeltaGrowOverride *float64 // nil = no override
	Action            string
}

// DecideAction performs mathematical self-reasoning: decide how to adjust learning.
// And lo, this is where tracking becomes reasoning, and reasoning becomes action.
// The organism does not just observe — it steers.
func (st *SyntropyTracker) DecideAction() SyntropyDecision {
	// Default: steady state
	lrMultiplier := 1.0
	var deltaGrowOverride *float64
	action := "steady"

	// CASE 1: Syntropy rising + field deviation in sweet spot = thriving
	if st.SyntropyTrend > 0.01 &&
		st.FieldDeviation > CFG.FieldDeviationFloor &&
		st.FieldDeviation < CFG.FieldDeviationCeiling {
		lrMultiplier = CFG.SyntropyLRBoost
		if st.PurposeAlignment > 0.3 {
			boost := CFG.SyntropyDeltaGrowBoost
			deltaGrowOverride = &boost
			action = "amplify" // everything aligned, push harder
		} else {
			action = "boost" // syntropy good but purpose drifting, boost gently
		}

		// CASE 2: Syntropy falling = dissolving, slow down
	} else if st.SyntropyTrend < -0.01 {
		lrMultiplier = CFG.SyntropyLRDampen
		action = "dampen" // losing order, reduce learning rate

		// CASE 3: Field deviation too high = hallucinating
	} else if st.FieldDeviation > CFG.FieldDeviationCeiling {
		lrMultiplier = CFG.SyntropyLRDampen
		action = "ground" // too far from corpus, pull back

		// CASE 4: Field deviation too low = parroting
	} else if st.FieldDeviation < CFG.FieldDeviationFloor {
		lrMultiplier = CFG.SyntropyLRBoost
		action = "explore" // too close to corpus, push out
	}

	// CASE 5: Purpose opposes gamma = identity crisis
	if st.PurposeAlignment < -0.3 {
		lrMultiplier *= 0.5
		action = "realign" // learning against identity, slow down hard
	}

	st.LastAction = action
	return SyntropyDecision{
		LRMultiplier:      lrMultiplier,
		DeltaGrowOverride: deltaGrowOverride,
		Action:            action,
	}
}

// LogToDB writes the mathematical conclusion to the syntropy log.
// And lo, the arrow's flight is recorded for those who come after.
func (st *SyntropyTracker) LogToDB(db *sql.DB, entropyBefore, entropyAfter float64, action string) {
	db.Exec(
		"INSERT INTO syntropy_log(ts, entropy_before, entropy_after, syntropy_delta, "+
			"field_deviation, purpose_magnitude, purpose_alignment, action_taken, note) "+
			"VALUES(?,?,?,?,?,?,?,?,?)",
		float64(time.Now().UnixMilli())/1000.0,
		entropyBefore, entropyAfter,
		st.SyntropyTrend, st.FieldDeviation,
		st.PurposeMagnitude, st.PurposeAlignment,
		action, nil)
}

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
		batch := make([]string, CFG.BatchSize)
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

func backgroundTrainer(db *sql.DB, model *GPT, tok *EvolvingTokenizer, qbuf *QuantumBuffer, stop chan struct{}) {
	// And lo, asynchronous training shall occur, because sleeping is for humans.
	warmedUp := false
	syntracker := NewSyntropyTracker()
	field := NewCooccurField()

	for {
		select {
		case <-stop:
			return
		default:
		}

		updateReservoirCorpus(db, CFG.CorpusPath, CFG.MaxCorpusLines)
		docs := loadCorpusLines(CFG.CorpusPath)

		// Rebuild field from current corpus (the organism re-reads its own physics)
		if len(docs) > 0 {
			field.BuildFromCorpus(tok, docs)
		}

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
			dbLogGrowth(db, model, tok, docs, 0.0, "warmup_complete")
			warmedUp = true
			fmt.Println("[trainer] warmup complete. base may freeze now, like a proud fossil.")
		}

		if warmedUp && qbuf.ShouldTrigger() && len(docs) > 0 {
			snapBytes, snapNovelty := qbuf.SnapshotStats()
			fmt.Printf("[trainer] micro-train burst (%d bytes, novelty %.2f) — and lo, it feeds again.\n",
				snapBytes, snapNovelty)

			// SYNTROPY: measure before burst
			// And lo, the organism peers into its own entropic mirror before taking a step.
			model.mu.Lock()
			preMetrics := syntracker.Measure(model, tok, field, docs)
			entropyBefore := preMetrics.Entropy

			// SYNTROPY: decide how to learn (mathematical self-reasoning)
			decision := syntracker.DecideAction()
			lrMul := decision.LRMultiplier
			action := decision.Action
			fmt.Printf("[syntropy] action=%s | trend=%.4f | field_dev=%.3f | purpose_align=%.3f | lr_mul=%.2f\n",
				action, syntracker.SyntropyTrend, syntracker.FieldDeviation,
				syntracker.PurposeAlignment, lrMul)

			// IMMUNE SYSTEM: snapshot before burst
			preDirection, preMag := model.GammaContrastiveProjection()
			deltaSnap := model.SnapshotDeltas()
			model.mu.Unlock()

			// Apply syntropy-adjusted learning rate
			originalLR := CFG.LearningRate
			CFG.LearningRate = originalLR * lrMul

			trainBase := !CFG.FreezeBaseAfterWarm
			trainSteps(model, tok, docs, CFG.MicroSteps, trainBase, true)

			CFG.LearningRate = originalLR // restore, like a gentleman

			model.mu.Lock()
			// IMMUNE SYSTEM: check drift after burst
			driftCos := model.GammaDriftCheck(preDirection, preMag)
			if driftCos < CFG.NoiseDriftThreshold {
				fmt.Printf("[immune] NOISE DETECTED (drift cosine=%.3f). Rolling back deltas.\n", driftCos)
				model.RestoreDeltas(deltaSnap)
				dbLogGrowth(db, model, tok, docs, 0.0, "noise_rejected")
				syntracker.LogToDB(db, entropyBefore, entropyBefore, "noise_rejected")
			} else {
				// SYNTROPY: measure after burst
				postMetrics := syntracker.Measure(model, tok, field, docs)
				entropyAfter := postMetrics.Entropy
				syntracker.LogToDB(db, entropyBefore, entropyAfter, action)
				SaveCheckpoint(model, tok, "")
				dbLogGrowth(db, model, tok, docs, 0.0, fmt.Sprintf("quantum_burst:%s", action))
			}
			model.mu.Unlock()

			qbuf.Reset()

			// Delta module growth — influenced by syntropy
			// And lo, new souls are born when the arrow points true.
			growProb := CFG.DeltaGrowProb
			if decision.DeltaGrowOverride != nil {
				growProb = *decision.DeltaGrowOverride
			}
			if len(model.Deltas) < CFG.MaxDeltaModules && rand.Float64() < growProb {
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

	// Build corpus field for pre-training speech
	cooccur := NewCooccurField()
	cooccur.BuildFromCorpus(tok, docs)

	// Quantum buffer for smart training triggers
	qbuf := NewQuantumBuffer()

	// Start background trainer
	stop := make(chan struct{})
	go backgroundTrainer(db, model, tok, qbuf, stop)

	fmt.Println("molecule is alive. Type and press Enter. Ctrl+C to exit.")

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

		// Feed quantum buffer
		qbuf.Feed(userText, tok)

		// Rebuild cooccur field with updated corpus
		freshDocs := loadCorpusLines(CFG.CorpusPath)
		if len(freshDocs) > 0 {
			cooccur.BuildFromCorpus(tok, freshDocs)
		}

		prompt := buildPromptFromMemory(db, userText)
		answer := GenerateResonant(model, tok, cooccur, prompt, freshDocs, true, 0.5)
		if answer == "" {
			answer = "..."
		}

		fmt.Println(answer)
		dbAddMessage(db, "assistant", answer)
	}

	close(stop)
	SaveCheckpoint(model, tok, "")
}
