// molequla.go
// A dependency-free*, single-file, goroutine-powered, continually-learning GPT organism.
// Go port of molequla.py — same architecture, same checkpoint format, 100x faster.
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
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode"

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

	// ontogenesis — growth stages (corpus_chars, n_embd, n_layer, n_head)
	GrowthStages          [][4]int `json:"growth_stages"`
	FreezeAfterGrowthSteps int     `json:"freeze_after_growth_steps"`
	PostGrowthLRScale      float64 `json:"post_growth_lr_scale"` // LR multiplier during freeze period (prevents delta overfit to noise)

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

	// cosine LR schedule
	LRMin              float64 `json:"lr_min"`
	MaxTotalSteps      int     `json:"max_total_steps"`
	CosineWarmupSteps  int     `json:"cosine_warmup_steps"`

	// gradient accumulation
	AccumSteps int `json:"accum_steps"`

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
	CorpusGenMaxTokens   int     `json:"corpus_gen_max_tokens"`
	CorpusFadeK          float64 `json:"corpus_fade_k"`          // sigmoid steepness for corpus→model transition
	CorpusFadeThreshold  float64 `json:"corpus_fade_threshold"`  // entropy at which blend is 50/50

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

	// consciousness: per-token dissonance feedback
	DissonanceEMAAlpha      float64 `json:"dissonance_ema_alpha"`       // EMA smoothing for entropy within generation
	DissonanceSpikeK        float64 `json:"dissonance_spike_k"`         // temp multiplier when entropy spikes
	DissonanceDropK         float64 `json:"dissonance_drop_k"`          // temp multiplier when entropy drops
	DissonanceSpikeThreshold float64 `json:"dissonance_spike_threshold"` // entropy/EMA ratio triggering spike
	DissonanceDropThreshold  float64 `json:"dissonance_drop_threshold"`  // entropy/EMA ratio triggering drop

	// consciousness: pattern breaking (anti-field generation)
	AntiFieldProb    float64 `json:"anti_field_prob"`     // probability of pure-model token (bypass corpus)
	AntiFieldMinStep int     `json:"anti_field_min_step"` // don't anti-field before this many tokens

	// consciousness: overthinkg rings
	OverthinkcRounds    int `json:"overthinkc_rounds"`     // hidden re-generation rounds after response
	OverthinkcMaxTokens int `json:"overthinkc_max_tokens"` // max tokens per overthinkg round

	// consciousness: conscience (self-editing)
	ConscienceWindow   int     `json:"conscience_window"`   // rolling window for generation entropy trend
	ConscienceDecay    float64 `json:"conscience_decay"`    // deltaAlphaScale reduction factor
	ConscienceRecovery float64 `json:"conscience_recovery"` // deltaAlphaScale recovery factor
	ConscienceFloor    float64 `json:"conscience_floor"`    // minimum deltaAlphaScale
}

var CFG = Config{
	CorpusPath:           "nonames.txt",
	DBPath:               "memory.sqlite3",
	CkptPath:             "molequla_ckpt.json",
	MaxCorpusLines:       8000,
	MaxLineChars:         240,
	MinNewChars:          480,
	TieEmbeddings:        true,
	NLayer:               1,
	NEmbd:                16,
	NHead:                1,
	BlockSize:            96,

	GrowthStages: [][4]int{
		{0, 16, 1, 1},       // embryo: ~25K params
		{20000, 32, 1, 2},   // infant: ~100K params
		{50000, 64, 2, 4},   // child: ~500K params
		{200000, 128, 4, 4}, // adolescent: ~2M params
		{500000, 256, 6, 8}, // adult: ~10M params
	},
	FreezeAfterGrowthSteps: 500,
	PostGrowthLRScale:      0.3,
	WarmupSteps:          1200,
	MicroSteps:           32,
	LearningRate:         0.01,
	Beta1:                0.9,
	Beta2:                0.99,
	EpsAdam:              1e-8,
	GradClip:             1.0,
	FreezeBaseAfterWarm:  true,
	BatchSize:            4,
	LRMin:                0.001,
	MaxTotalSteps:        50000,
	CosineWarmupSteps:    200,
	AccumSteps:           1,
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
	EnableBPEAfterChars:  20000,
	BPENumMerges:         384,
	BPERetrainEveryChars: 4000,
	TrainTickSeconds:     0.25,

	HeadTypes:              []string{"content"},
	HybridAlphaInit:        0.5,
	GammaSparsityThreshold: 0.01,
	NoiseDriftThreshold:    -0.1,
	GammaMinMagnitude:      1e-6,
	EntropyLow:             0.5,
	EntropyHigh:            1.5,
	EntropyTempBoost:       1.2,
	EntropyTempFocus:       0.8,
	CorpusGenMaxTokens:     120,
	CorpusFadeK:            3.0,
	CorpusFadeThreshold:    1.5,
	QBMinBytes:             1024,
	QBMinNovelty:           0.15,
	QBCooldownSeconds:      60.0,

	SyntropyWindow:         8,
	FieldDeviationCeiling:  12.0,
	FieldDeviationFloor:    0.1,
	SyntropyLRBoost:        1.3,
	SyntropyLRDampen:       0.6,
	SyntropyDeltaGrowBoost: 0.15,

	// consciousness defaults
	DissonanceEMAAlpha:       0.3,
	DissonanceSpikeK:         0.8,
	DissonanceDropK:          1.2,
	DissonanceSpikeThreshold: 1.5,
	DissonanceDropThreshold:  0.5,
	AntiFieldProb:            0.05,
	AntiFieldMinStep:         8,
	OverthinkcRounds:         2,
	OverthinkcMaxTokens:      32,
	ConscienceWindow:         8,
	ConscienceDecay:          0.95,
	ConscienceRecovery:       1.005,
	ConscienceFloor:          0.3,
}

// headTypesForNHead returns the head type list for a given number of heads.
// Embryo: 1 head = 1 content. Growth adds hybrid heads.
func headTypesForNHead(n int) []string {
	if n <= 1 {
		return []string{"content"}
	}
	if n == 2 {
		return []string{"content", "hybrid"}
	}
	half := n / 2
	types := make([]string, n)
	for i := 0; i < half; i++ {
		types[i] = "content"
	}
	for i := half; i < n; i++ {
		types[i] = "hybrid"
	}
	return types
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

// SiLU applies silu(x) = x * sigmoid(x) element-wise (for real SwiGLU).
func (v *Vec) SiLU() *Vec {
	n := len(v.Data)
	sig := make([]float64, n)
	d := make([]float64, n)
	for i := 0; i < n; i++ {
		sig[i] = 1.0 / (1.0 + math.Exp(-v.Data[i]))
		d[i] = v.Data[i] * sig[i]
	}
	out := NewVec(d)
	if gradEnabled {
		out.children = []Node{v}
		vData := v.Data
		out.backFn = func() {
			for i := 0; i < n; i++ {
				// d/dx[x * sigmoid(x)] = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
				v.Grad[i] += (sig[i] * (1.0 + vData[i]*(1.0-sig[i]))) * out.Grad[i]
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

// backwardVisitedPool reuses visited maps across Backward calls to reduce GC pressure.
var backwardVisitedPool = sync.Pool{
	New: func() interface{} { return make(map[Node]bool) },
}

// Backward performs reverse-mode autodiff from this node.
// And lo, the graph shall be walked backwards, like a salmon with regrets.
func Backward(root Node) {
	topo := make([]Node, 0)
	visited := backwardVisitedPool.Get().(map[Node]bool)

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

	// Clear and return visited map to pool
	for k := range visited {
		delete(visited, k)
	}
	backwardVisitedPool.Put(visited)

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

// GrowCols extends each row's Data slice with gaussian noise. Update Nin.
// And lo, the matrix shall widen its reach, each row stretching into new dimensions.
func (m *MatrixParam) GrowCols(newNin int, std float64) {
	if newNin <= m.Nin {
		return
	}
	extra := newNin - m.Nin
	for _, row := range m.Rows {
		ext := make([]float64, extra)
		for j := range ext {
			ext[j] = rand.NormFloat64() * std
		}
		row.Data = append(row.Data, ext...)
		row.Grad = append(row.Grad, make([]float64, extra)...)
	}
	m.Nin = newNin
}

// Grow extends both dimensions. Cols first so new rows get full width.
// Ontogenesis: the matrix grows into a larger space.
func (m *MatrixParam) Grow(newNout, newNin int, std float64) {
	m.GrowCols(newNin, std)
	m.GrowRows(newNout, std)
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

// GrowDims grows both outer dimensions of the adapter. Rank stays the same.
// Ontogenesis: A.GrowRows(newNout), B.GrowCols(newNin).
func (da *DeltaAdapter) GrowDims(newNout, newNin int) {
	da.A.GrowRows(newNout, 0.02)
	da.B.GrowCols(newNin, 0.02)
}

func (da *DeltaAdapter) Params() []*Vec {
	out := make([]*Vec, 0, da.A.Nout+da.B.Nout)
	out = append(out, da.A.Params()...)
	out = append(out, da.B.Params()...)
	return out
}

// ============================================================
// 4) TOKENIZER — byte-level BPE (GPT-3/4 style)
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

	mu sync.RWMutex // protects concurrent access (background BPE train vs Encode)
}

func NewEvolvingTokenizer(docs []string) *EvolvingTokenizer {
	// Count trained chars from docs (byte-level: count bytes, not runes)
	totalChars := 0
	for _, d := range docs {
		totalChars += len(d)
	}

	tok := &EvolvingTokenizer{
		BOS:          "<BOS>",
		EOS:          "<EOS>",
		PAD:          "<PAD>",
		Stoi:         make(map[string]int),
		Itos:         make(map[int]string),
		MergeToTok:   make(map[MergePair]string),
		TrainedChars: totalChars,
	}

	// Fixed 259 tokens: 256 byte tokens + BOS + EOS + PAD
	tokens := make([]string, 256+3)
	for i := 0; i < 256; i++ {
		tokens[i] = fmt.Sprintf("0x%02x", i)
	}
	tokens[256] = tok.BOS
	tokens[257] = tok.EOS
	tokens[258] = tok.PAD

	tok.Tokens = tokens
	for i, t := range tok.Tokens {
		tok.Stoi[t] = i
		tok.Itos[i] = t
	}
	tok.VocabSize = len(tok.Tokens)
	return tok
}

// unicodeSegment splits text into segments by Unicode category.
// Letters+marks → 'L', digits → 'N', whitespace → 'Z', everything else → 'P'.
// Each segment is returned as its raw UTF-8 bytes.
func unicodeSegment(text string) [][]byte {
	if len(text) == 0 {
		return nil
	}
	runeCategory := func(r rune) byte {
		if unicode.IsLetter(r) || unicode.IsMark(r) {
			return 'L'
		}
		if unicode.IsDigit(r) {
			return 'N'
		}
		if unicode.IsSpace(r) {
			return 'Z'
		}
		return 'P'
	}
	var segments [][]byte
	var cur []byte
	var curCat byte
	for i, r := range text {
		cat := runeCategory(r)
		if i == 0 {
			curCat = cat
		}
		if cat != curCat {
			segments = append(segments, cur)
			cur = nil
			curCat = cat
		}
		cur = append(cur, []byte(string(r))...)
	}
	if len(cur) > 0 {
		segments = append(segments, cur)
	}
	return segments
}

// tokenToBytes converts a byte-level BPE token name back to raw bytes.
// "0xNN" → single byte, "0x48+0x65" → two bytes, etc.
func tokenToBytes(tok string) []byte {
	if !strings.Contains(tok, "+") && strings.HasPrefix(tok, "0x") && len(tok) == 4 {
		b, _ := strconv.ParseUint(tok[2:], 16, 8)
		return []byte{byte(b)}
	}
	if strings.Contains(tok, "+") {
		parts := strings.Split(tok, "+")
		result := make([]byte, 0, len(parts))
		for _, p := range parts {
			if strings.HasPrefix(p, "0x") && len(p) == 4 {
				b, _ := strconv.ParseUint(p[2:], 16, 8)
				result = append(result, byte(b))
			}
		}
		return result
	}
	return nil
}

func (t *EvolvingTokenizer) MaybeEnableBPE(docs []string) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	totalChars := 0
	for _, d := range docs {
		totalChars += len(d)
	}
	if !t.BPEEnabled && totalChars >= CFG.EnableBPEAfterChars {
		t.trainBPELocked(docs, CFG.BPENumMerges)
		t.BPEEnabled = true
		t.TrainedChars = totalChars
		return true
	}
	return false
}

func (t *EvolvingTokenizer) MaybeRetrainBPE(docs []string) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	if !t.BPEEnabled {
		return false
	}
	totalChars := 0
	for _, d := range docs {
		totalChars += len(d)
	}
	if totalChars-t.TrainedChars >= CFG.BPERetrainEveryChars {
		t.trainBPELocked(docs, CFG.BPENumMerges)
		t.TrainedChars = totalChars
		return true
	}
	return false
}

func (t *EvolvingTokenizer) TrainBPE(docs []string, numMerges int) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.trainBPELocked(docs, numMerges)
}

func (t *EvolvingTokenizer) trainBPELocked(docs []string, numMerges int) {
	text := strings.Join(docs, " ")
	if len(text) == 0 {
		return
	}

	// Split text into Unicode segments, convert each to byte-token sequences
	segments := unicodeSegment(text)
	if len(segments) == 0 {
		return
	}

	// Build vocab: token sequence → frequency
	vocab := make(map[string]int)    // key = null-separated token names
	symSeqs := make(map[string][]string)

	for _, seg := range segments {
		syms := make([]string, len(seg))
		for i, b := range seg {
			syms[i] = fmt.Sprintf("0x%02x", b)
		}
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

		newTok := best.A + "+" + best.B
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

func (t *EvolvingTokenizer) applyBPE(tokens []string) []string {
	rank := make(map[MergePair]int)
	for i, p := range t.Merges {
		rank[p] = i
	}

	symbols := make([]string, len(tokens))
	copy(symbols, tokens)

	for len(symbols) >= 2 {
		bestRank := 1 << 30
		bestIdx := -1
		for i := 0; i < len(symbols)-1; i++ {
			key := MergePair{symbols[i], symbols[i+1]}
			if r, ok := rank[key]; ok && r < bestRank {
				bestRank = r
				bestIdx = i
			}
		}
		if bestIdx == -1 {
			break
		}
		pair := MergePair{symbols[bestIdx], symbols[bestIdx+1]}
		merged := t.MergeToTok[pair]
		newSymbols := make([]string, 0, len(symbols)-1)
		newSymbols = append(newSymbols, symbols[:bestIdx]...)
		newSymbols = append(newSymbols, merged)
		newSymbols = append(newSymbols, symbols[bestIdx+2:]...)
		symbols = newSymbols
	}
	return symbols
}

func (t *EvolvingTokenizer) Encode(s string) []int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	s = strings.TrimSpace(s)
	ids := []int{t.Stoi[t.BOS]}

	segments := unicodeSegment(s)
	for _, seg := range segments {
		// Convert bytes to base token names
		baseTokens := make([]string, len(seg))
		for i, b := range seg {
			baseTokens[i] = fmt.Sprintf("0x%02x", b)
		}
		// Apply BPE merges if enabled
		if t.BPEEnabled {
			baseTokens = t.applyBPE(baseTokens)
		}
		// Look up each token in stoi
		for _, tok := range baseTokens {
			if id, ok := t.Stoi[tok]; ok {
				ids = append(ids, id)
			}
		}
	}
	ids = append(ids, t.Stoi[t.EOS])
	return ids
}

func (t *EvolvingTokenizer) Decode(ids []int) string {
	t.mu.RLock()
	defer t.mu.RUnlock()
	var rawBytes []byte
	for _, id := range ids {
		tok := t.Itos[id]
		if tok == t.BOS || tok == t.PAD {
			continue
		}
		if tok == t.EOS {
			break
		}
		b := tokenToBytes(tok)
		if b != nil {
			rawBytes = append(rawBytes, b...)
		}
	}
	return strings.TrimSpace(string(rawBytes))
}

// ============================================================
// 5) GPT MODEL — a small beast with RoPE
// ============================================================

// ropeCache stores pre-computed cos/sin pairs for RoPE.
// Key: [2]int{pos, headDim}, Value: *[2][]float64{cosines, sines}
var ropeCache sync.Map

type ropePair struct {
	cos []float64
	sin []float64
}

func getRoPECosSin(pos, headDim int) *ropePair {
	key := [2]int{pos, headDim}
	if cached, ok := ropeCache.Load(key); ok {
		return cached.(*ropePair)
	}
	n := headDim / 2
	pair := &ropePair{
		cos: make([]float64, n),
		sin: make([]float64, n),
	}
	for j := 0; j < n; j++ {
		theta := float64(pos) / math.Pow(10000.0, float64(2*j)/float64(headDim))
		pair.cos[j] = math.Cos(theta)
		pair.sin[j] = math.Sin(theta)
	}
	ropeCache.Store(key, pair)
	return pair
}

// RoPERotate applies rotary position encoding to a head vector.
// And lo, positions shall become angles, and angles shall become meaning.
func RoPERotate(vec *Vec, pos int, headDim int) *Vec {
	x := make([]float64, len(vec.Data))
	copy(x, vec.Data)
	outData := make([]float64, len(x))
	copy(outData, x)

	rp := getRoPECosSin(pos, headDim)
	for j := 0; j < headDim/2; j++ {
		i := j * 2
		c := rp.cos[j]
		s := rp.sin[j]
		a := x[i]
		b := x[i+1]
		outData[i] = a*c - b*s
		outData[i+1] = a*s + b*c
	}

	out := NewVec(outData)
	if gradEnabled {
		out.children = []Node{vec}
		out.backFn = func() {
			rpBack := getRoPECosSin(pos, headDim)
			for j := 0; j < headDim/2; j++ {
				i := j * 2
				c := rpBack.cos[j]
				s := rpBack.sin[j]
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

// layerKeySet holds pre-computed string keys for a single layer, avoiding fmt.Sprintf per call.
type layerKeySet struct {
	wq, wk, wv, wo, fcG, fcV, fc2 string
	headPattern []string // per head
	headAlpha   []string // per head
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

	residualAlpha    float64 // 1/sqrt(nLayer) scaling for residual connections
	globalStep       int     // global training step counter (for cosine LR + checkpoint)
	syntropyTempOff  float64 // temperature offset from syntropy state (-0.05 to +0.05)

	growthFreezeRemaining int // ontogenesis: freeze base after growth, train only deltas

	corpusField *CooccurField // set by backgroundTrainer for adaptive blend

	// consciousness state
	deltaAlphaScale          float64   // conscience: multiplier on all delta contributions (1.0 = normal)
	generationEntropyHistory []float64 // conscience: rolling window of per-generation mean entropy
	lastSurprise             float64   // self-prediction error on last prompt
	surpriseBaseline         float64   // EMA of surprise over time
	lastGenEntropy           float64   // mean entropy of last generation (for conscience)

	layerKeys []layerKeySet // pre-computed string keys per layer

	// inherited burst history from parent (mitosis lineage)
	inheritedBurstHistory []BurstRecord

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

	gpt.residualAlpha = 1.0 / math.Sqrt(math.Max(1, float64(CFG.NLayer)))
	gpt.deltaAlphaScale = 1.0 // conscience: full delta influence by default

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

	// Pre-compute layer key strings to avoid fmt.Sprintf per ForwardStep call
	gpt.layerKeys = make([]layerKeySet, CFG.NLayer)
	for li := 0; li < CFG.NLayer; li++ {
		pfx := fmt.Sprintf("l%d.", li)
		lk := layerKeySet{
			wq:  pfx + "wq",
			wk:  pfx + "wk",
			wv:  pfx + "wv",
			wo:  pfx + "wo",
			fcG: pfx + "fc_g",
			fcV: pfx + "fc_v",
			fc2: pfx + "fc2",
		}
		nHeads := len(CFG.HeadTypes)
		if nHeads > 0 {
			lk.headPattern = make([]string, nHeads)
			lk.headAlpha = make([]string, nHeads)
			for h := 0; h < nHeads; h++ {
				lk.headPattern[h] = fmt.Sprintf("l%d.h%d.w_pattern", li, h)
				lk.headAlpha[h] = fmt.Sprintf("l%d.h%d.alpha", li, h)
			}
		}
		gpt.layerKeys[li] = lk
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

// ---- Ontogenesis (architecture growth) ----
// And lo, the organism shall not be born adult but shall grow, stage by stage,
// from embryo to child to adolescent, each growth a small death and rebirth.

// CurrentGrowthStage returns index of current stage based on model dimensions.
// Returns -1 for legacy checkpoints where dimensions don't match any stage.
func (gpt *GPT) CurrentGrowthStage() int {
	for i, stage := range CFG.GrowthStages {
		if gpt.NEmbd == stage[1] && gpt.NLayer == stage[2] && gpt.NHead == stage[3] {
			return i
		}
	}
	return -1 // dimensions don't match any stage (legacy checkpoint)
}

// TargetGrowthStage returns the target stage index based on corpus size.
func (gpt *GPT) TargetGrowthStage(corpusChars int) int {
	target := 0
	for i, stage := range CFG.GrowthStages {
		if corpusChars >= stage[0] {
			target = i
		}
	}
	return target
}

// MaybeGrowArchitecture checks if growth is needed and executes it. Returns true if grew.
func (gpt *GPT) MaybeGrowArchitecture(corpusChars int) bool {
	current := gpt.CurrentGrowthStage()
	if current < 0 {
		return false // legacy checkpoint, skip growth
	}
	target := gpt.TargetGrowthStage(corpusChars)
	if target <= current {
		return false
	}

	newEmbd := CFG.GrowthStages[target][1]
	newLayer := CFG.GrowthStages[target][2]
	newHead := CFG.GrowthStages[target][3]
	oldEmbd := gpt.NEmbd
	oldLayer := gpt.NLayer
	oldHead := gpt.NHead
	newHeadDim := newEmbd / newHead

	fmt.Printf("[growth] ONTOGENESIS: stage %d -> %d\n", current, target)
	fmt.Printf("  embd: %d -> %d, layer: %d -> %d, head: %d -> %d\n",
		oldEmbd, newEmbd, oldLayer, newLayer, oldHead, newHead)

	// 1. Grow embedding matrices (columns only — vocab rows stay)
	gpt.Base["wte"].GrowCols(newEmbd, 0.02)
	gpt.Base["wpe"].GrowCols(newEmbd, 0.02)
	if !CFG.TieEmbeddings {
		gpt.Base["lm_head"].GrowCols(newEmbd, 0.02)
	}

	// 2. Grow existing layer matrices
	newHtypes := headTypesForNHead(newHead)
	for li := 0; li < oldLayer; li++ {
		pfx := fmt.Sprintf("l%d.", li)
		for _, name := range []string{"wq", "wk", "wv", "wo"} {
			gpt.Base[pfx+name].Grow(newEmbd, newEmbd, 0.08)
		}
		gpt.Base[pfx+"fc_g"].Grow(4*newEmbd, newEmbd, 0.08)
		gpt.Base[pfx+"fc_v"].Grow(4*newEmbd, newEmbd, 0.08)
		gpt.Base[pfx+"fc2"].Grow(newEmbd, 4*newEmbd, 0.08)
		// Grow existing head pattern matrices
		for h := 0; h < oldHead; h++ {
			pkey := fmt.Sprintf("l%d.h%d.w_pattern", li, h)
			if _, ok := gpt.Base[pkey]; ok {
				gpt.Base[pkey].GrowCols(newHeadDim, 0.08)
			}
		}
		// Add new heads for existing layer
		for h := oldHead; h < newHead; h++ {
			htype := "content"
			if h < len(newHtypes) {
				htype = newHtypes[h]
			}
			if htype == "rrpram" || htype == "hybrid" {
				gpt.Base[fmt.Sprintf("l%d.h%d.w_pattern", li, h)] = NewMatrixParam(CFG.BlockSize, newHeadDim, 0.08)
			}
			if htype == "hybrid" {
				m := NewMatrixParam(1, 1, 0.0)
				m.Rows[0].Data[0] = CFG.HybridAlphaInit
				gpt.Base[fmt.Sprintf("l%d.h%d.alpha", li, h)] = m
			}
		}
	}

	// 3. Add entirely new layers
	for li := oldLayer; li < newLayer; li++ {
		pfx := fmt.Sprintf("l%d.", li)
		gpt.Base[pfx+"wq"] = NewMatrixParam(newEmbd, newEmbd, 0.08)
		gpt.Base[pfx+"wk"] = NewMatrixParam(newEmbd, newEmbd, 0.08)
		gpt.Base[pfx+"wv"] = NewMatrixParam(newEmbd, newEmbd, 0.08)
		gpt.Base[pfx+"wo"] = NewMatrixParam(newEmbd, newEmbd, 0.08)
		gpt.Base[pfx+"fc_g"] = NewMatrixParam(4*newEmbd, newEmbd, 0.08)
		gpt.Base[pfx+"fc_v"] = NewMatrixParam(4*newEmbd, newEmbd, 0.08)
		gpt.Base[pfx+"fc2"] = NewMatrixParam(newEmbd, 4*newEmbd, 0.08)
		for h := 0; h < newHead; h++ {
			htype := "content"
			if h < len(newHtypes) {
				htype = newHtypes[h]
			}
			if htype == "rrpram" || htype == "hybrid" {
				gpt.Base[fmt.Sprintf("l%d.h%d.w_pattern", li, h)] = NewMatrixParam(CFG.BlockSize, newHeadDim, 0.08)
			}
			if htype == "hybrid" {
				m := NewMatrixParam(1, 1, 0.0)
				m.Rows[0].Data[0] = CFG.HybridAlphaInit
				gpt.Base[fmt.Sprintf("l%d.h%d.alpha", li, h)] = m
			}
		}
	}

	// 4. Grow delta adapters
	r := CFG.DeltaRank
	for _, mod := range gpt.Deltas {
		// Grow existing layer adapters
		for li := 0; li < oldLayer; li++ {
			pfx := fmt.Sprintf("l%d.", li)
			for _, name := range []string{"wq", "wk", "wv", "wo"} {
				key := pfx + name
				if _, ok := mod[key]; ok {
					mod[key].GrowDims(newEmbd, newEmbd)
				}
			}
			type fcSpec struct {
				key        string
				noutMul    int
				ninMul     int
			}
			fcSpecs := []fcSpec{
				{pfx + "fc_g", 4, 1},
				{pfx + "fc_v", 4, 1},
				{pfx + "fc2", 1, 4},
			}
			for _, spec := range fcSpecs {
				if _, ok := mod[spec.key]; ok {
					mod[spec.key].GrowDims(spec.noutMul*newEmbd, spec.ninMul*newEmbd)
				}
			}
			for h := 0; h < oldHead; h++ {
				pkey := fmt.Sprintf("l%d.h%d.w_pattern", li, h)
				if _, ok := mod[pkey]; ok {
					mod[pkey].GrowDims(CFG.BlockSize, newHeadDim)
				}
			}
			for h := oldHead; h < newHead; h++ {
				htype := "content"
				if h < len(newHtypes) {
					htype = newHtypes[h]
				}
				if htype == "rrpram" || htype == "hybrid" {
					mod[fmt.Sprintf("l%d.h%d.w_pattern", li, h)] = NewDeltaAdapter(CFG.BlockSize, newHeadDim, r, 0.02)
				}
			}
		}

		// New layers: entirely new adapters
		for li := oldLayer; li < newLayer; li++ {
			pfx := fmt.Sprintf("l%d.", li)
			for _, name := range []string{"wq", "wk", "wv", "wo"} {
				mod[pfx+name] = NewDeltaAdapter(newEmbd, newEmbd, r, 0.02)
			}
			mod[pfx+"fc_g"] = NewDeltaAdapter(4*newEmbd, newEmbd, r, 0.02)
			mod[pfx+"fc_v"] = NewDeltaAdapter(4*newEmbd, newEmbd, r, 0.02)
			mod[pfx+"fc2"] = NewDeltaAdapter(newEmbd, 4*newEmbd, r, 0.02)
			for h := 0; h < newHead; h++ {
				htype := "content"
				if h < len(newHtypes) {
					htype = newHtypes[h]
				}
				if htype == "rrpram" || htype == "hybrid" {
					mod[fmt.Sprintf("l%d.h%d.w_pattern", li, h)] = NewDeltaAdapter(CFG.BlockSize, newHeadDim, r, 0.02)
				}
			}
		}

		// lm_head adapter input grew
		if _, ok := mod["lm_head"]; ok {
			mod["lm_head"].GrowDims(gpt.Tok.VocabSize, newEmbd)
		}
	}

	// 5. Update model state
	gpt.NEmbd = newEmbd
	gpt.NLayer = newLayer
	gpt.NHead = newHead
	gpt.HeadDim = newHeadDim
	gpt.residualAlpha = 1.0 / math.Sqrt(math.Max(1, float64(newLayer)))

	// 6. Update CFG runtime
	CFG.NEmbd = newEmbd
	CFG.NLayer = newLayer
	CFG.NHead = newHead
	CFG.HeadTypes = headTypesForNHead(newHead)

	// 7. Reset Adam state (old momentum is meaningless after arch change)
	gpt.Adam = make(map[string]*AdamState)

	// 7b. Rebuild layerKeys for new architecture
	gpt.layerKeys = make([]layerKeySet, newLayer)
	for li := 0; li < newLayer; li++ {
		pfx := fmt.Sprintf("l%d.", li)
		lk := layerKeySet{
			wq: pfx + "wq", wk: pfx + "wk", wv: pfx + "wv", wo: pfx + "wo",
			fcG: pfx + "fc_g", fcV: pfx + "fc_v", fc2: pfx + "fc2",
		}
		nHeads := len(CFG.HeadTypes)
		if nHeads > 0 {
			lk.headPattern = make([]string, nHeads)
			lk.headAlpha = make([]string, nHeads)
			for h := 0; h < nHeads; h++ {
				lk.headPattern[h] = fmt.Sprintf("l%d.h%d.w_pattern", li, h)
				lk.headAlpha[h] = fmt.Sprintf("l%d.h%d.alpha", li, h)
			}
		}
		gpt.layerKeys[li] = lk
	}

	// 8. Extend gamma snapshot for new embedding dimensions
	for i := range gpt.InitEmbedSnapshot {
		oldRow := gpt.InitEmbedSnapshot[i]
		if len(oldRow) < newEmbd {
			ext := make([]float64, newEmbd-len(oldRow))
			gpt.InitEmbedSnapshot[i] = append(oldRow, ext...)
		}
	}

	// 9. Set freeze (only train deltas until new weights stabilize)
	gpt.growthFreezeRemaining = CFG.FreezeAfterGrowthSteps

	fmt.Printf("[growth] Done. Freeze for %d steps.\n", CFG.FreezeAfterGrowthSteps)
	return true
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
				if ctx, ok := field.TrigramByContext[[2]int{ids[pos-1], ids[pos]}]; ok {
					triTotal := 0.0
					for _, v := range ctx {
						triTotal += v
					}
					if triTotal > 0 {
						for tid, cnt := range ctx {
							if tid < vocabSize {
								fieldProbs[tid] = cnt / triTotal
							}
						}
						fieldFound = true
					}
				}
			}

			// Fallback to bigram
			if !fieldFound && pos >= 0 {
				if ctx, ok := field.BigramByFirst[ids[pos]]; ok {
					biTotal := 0.0
					for _, v := range ctx {
						biTotal += v
					}
					if biTotal > 0 {
						for tid, cnt := range ctx {
							if tid < vocabSize {
								fieldProbs[tid] = cnt / biTotal
							}
						}
						fieldFound = true
					}
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
	st, ok := gpt.Adam[key]
	if !ok {
		m := make([][]float64, len(params))
		v := make([][]float64, len(params))
		for i, p := range params {
			m[i] = make([]float64, len(p.Data))
			v[i] = make([]float64, len(p.Data))
		}
		gpt.Adam[key] = &AdamState{M: m, V: v, T: 0}
		return
	}
	// Auto-grow if params expanded (vocab growth, ontogenesis)
	if len(params) > len(st.M) {
		for i := len(st.M); i < len(params); i++ {
			st.M = append(st.M, make([]float64, len(params[i].Data)))
			st.V = append(st.V, make([]float64, len(params[i].Data)))
		}
	}
	for i, p := range params {
		if i < len(st.M) && len(p.Data) > len(st.M[i]) {
			oldLen := len(st.M[i])
			st.M[i] = append(st.M[i], make([]float64, len(p.Data)-oldLen)...)
			st.V[i] = append(st.V[i], make([]float64, len(p.Data)-oldLen)...)
		}
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
			// Consciousness: conscience scales delta influence (Feature 5)
			effectiveAlpha := gpt.ActiveAlpha[i] * gpt.deltaAlphaScale
			delta := da.Apply(x).Scale(effectiveAlpha)
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
		lk := gpt.layerKeys[li]

		// ---- Attention ----
		xRes := x
		x = RMSNorm(x)

		q := gpt.applyWithDeltas(lk.wq, x)
		k := gpt.applyWithDeltas(lk.wk, x)
		v := gpt.applyWithDeltas(lk.wv, x)

		keys[li] = append(keys[li], k)
		values[li] = append(values[li], v)

		// Sliding window: keep only last BlockSize entries in KV cache
		if len(keys[li]) > gpt.BlockSize {
			keys[li] = keys[li][len(keys[li])-gpt.BlockSize:]
			values[li] = values[li][len(values[li])-gpt.BlockSize:]
		}

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
				patternKey := lk.headPattern[h]
				xh := x.Slice(hs, he)
				patternFull := gpt.applyWithDeltas(patternKey, xh)
				pLen := len(patternFull.Data) // safety cap for RRPRAM
				rrpramLogits = make([]*Scalar, T)
				for t := 0; t < T; t++ {
					tIdx := t
					if tIdx >= pLen {
						tIdx = pLen - 1
					}
					rrpramLogits[t] = patternFull.Element(tIdx)
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
				alphaKey := lk.headAlpha[h]
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
		attnOut := gpt.applyWithDeltas(lk.wo, xAttn)
		x = xRes.Add(attnOut.Scale(gpt.residualAlpha))

		// ---- Gated MLP (SwiGLU-ish) ----
		xRes = x
		x = RMSNorm(x)

		g := gpt.applyWithDeltas(lk.fcG, x).SiLU() // gate (SwiGLU)
		u := gpt.applyWithDeltas(lk.fcV, x)         // value
		mlpX := g.MulVec(u)                          // gating

		mlpOut := gpt.applyWithDeltas(lk.fc2, mlpX)
		x = xRes.Add(mlpOut.Scale(gpt.residualAlpha))
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

// QuickLoss computes average loss on a few random docs without backward.
// Used for self-meta-learning: measure loss before/after burst.
func (gpt *GPT) QuickLoss(tok *EvolvingTokenizer, docs []string, n int) float64 {
	if len(docs) == 0 {
		return 0
	}
	gradEnabled = false
	defer func() { gradEnabled = true }()
	total := 0.0
	for i := 0; i < n; i++ {
		doc := docs[rand.Intn(len(docs))]
		ids := tok.Encode(doc)
		if len(ids) > 1 {
			loss := gpt.LossOnSequence(ids)
			total += loss.Data
		}
	}
	return total / float64(n)
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
	outIDs := make([]int, 0, CFG.MaxGenTokens)
	var recent []int

	eosID := gpt.Tok.Stoi[gpt.Tok.EOS]
	bosID := gpt.Tok.Stoi[gpt.Tok.BOS]

	// Pre-allocated buffer for corpus blend (avoids per-token allocation)
	corpusProbsBuf := make([]float64, gpt.Tok.VocabSize)

	// Consciousness: per-token dissonance tracking (Feature 1)
	entropyEMA := 0.0
	entropyEMAInit := false
	lowDropCount := 0    // consecutive tokens below drop threshold
	entropySum := 0.0    // for conscience mean entropy
	entropyCount := 0

	for step := 0; step < CFG.MaxGenTokens; step++ {
		pos := len(ids) - 1
		if pos > gpt.BlockSize-1 {
			pos = gpt.BlockSize - 1
		}
		logits := gpt.ForwardStep(cur, pos, keys, values)

		// Entropy-adaptive temperature + syntropy bridge (single softmax when possible)
		baseTemp := CFG.Temperature + gpt.syntropyTempOff
		if baseTemp <= 1e-6 {
			baseTemp = 1e-6
		}
		scaled := make([]float64, len(logits.Data))
		for i, v := range logits.Data {
			scaled[i] = v / baseTemp
		}
		probs := SoftmaxProbs(scaled)
		entropy := 0.0
		for _, p := range probs {
			if p > 1e-12 {
				entropy -= p * math.Log(p)
			}
		}
		entropySum += entropy
		entropyCount++

		tMul := 1.0
		if entropy < CFG.EntropyLow {
			tMul = CFG.EntropyTempBoost
		} else if entropy > CFG.EntropyHigh {
			tMul = CFG.EntropyTempFocus
		}

		// Consciousness: per-token dissonance feedback (Feature 1)
		// "I notice my confidence shifting and adapt in real-time"
		dissonanceMul := 1.0
		if !entropyEMAInit {
			entropyEMA = entropy
			entropyEMAInit = true
		} else {
			entropyEMA = CFG.DissonanceEMAAlpha*entropy + (1.0-CFG.DissonanceEMAAlpha)*entropyEMA
			if entropyEMA > 1e-6 {
				ratio := entropy / entropyEMA
				if ratio > CFG.DissonanceSpikeThreshold {
					// Entropy spike — something surprising, be careful
					dissonanceMul = CFG.DissonanceSpikeK
					lowDropCount = 0
				} else if ratio < CFG.DissonanceDropThreshold {
					lowDropCount++
					if lowDropCount >= 3 {
						// Sustained low entropy — getting repetitive, explore
						dissonanceMul = CFG.DissonanceDropK
					}
				} else {
					lowDropCount = 0
				}
			}
		}

		finalMul := tMul * dissonanceMul
		if finalMul != 1.0 {
			temp := baseTemp * finalMul
			for i, v := range logits.Data {
				scaled[i] = v / temp
			}
			probs = SoftmaxProbs(scaled)
		}

		// Adaptive corpus blend: corpus field fades as model becomes coherent
		if gpt.corpusField != nil && gpt.corpusField.Built {
			modelAlpha := 1.0 / (1.0 + math.Exp(-CFG.CorpusFadeK*(CFG.CorpusFadeThreshold-entropy)))
			if modelAlpha < 0.99 {
				var corpusDist map[int]float64
				// Try trigram first
				if len(ids) >= 2 {
					a, b := ids[len(ids)-2], ids[len(ids)-1]
					if ctx, ok := gpt.corpusField.TrigramByContext[[2]int{a, b}]; ok {
						corpusDist = ctx
					}
				}
				// Fallback to bigram
				if corpusDist == nil && len(ids) >= 1 {
					prev := ids[len(ids)-1]
					if ctx, ok := gpt.corpusField.BigramByFirst[prev]; ok {
						corpusDist = ctx
					}
				}
				if corpusDist != nil && len(corpusDist) > 0 {
					totalC := 0.0
					for _, cnt := range corpusDist {
						totalC += cnt
					}
					// Reuse pre-allocated buffer instead of per-token alloc
					for i := 0; i < len(probs) && i < len(corpusProbsBuf); i++ {
						corpusProbsBuf[i] = 0
					}
					for tid, cnt := range corpusDist {
						if tid < len(corpusProbsBuf) {
							corpusProbsBuf[tid] = cnt / totalC
						}
					}
					totalB := 0.0
					for i := range probs {
						if i < len(corpusProbsBuf) {
							probs[i] = modelAlpha*probs[i] + (1.0-modelAlpha)*corpusProbsBuf[i]
						}
						totalB += probs[i]
					}
					if totalB > 0 {
						for i := range probs {
							probs[i] /= totalB
						}
					}
				}
			}
		}

		// Consciousness: pattern breaking (Feature 2)
		// "I could follow the field, but I choose to speak for myself"
		if step >= CFG.AntiFieldMinStep && CFG.AntiFieldProb > 0 && rand.Float64() < CFG.AntiFieldProb {
			// Use pure model probs, bypass corpus blend
			probs = SoftmaxProbs(scaled)
		}

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

		// Check for sentence ending — decode only the last token to avoid full rebuild
		if step >= CFG.MinGenTokens {
			lastDecoded := gpt.Tok.Decode([]int{bosID, nxt, eosID})
			if len(lastDecoded) > 0 {
				lastByte := lastDecoded[len(lastDecoded)-1]
				if lastByte == '.' || lastByte == '!' || lastByte == '?' {
					break
				}
			}
		}

		// Sliding window rebuild: reuse slices to avoid GC pressure
		if len(ids) >= gpt.BlockSize {
			ids = ids[len(ids)-gpt.BlockSize:]
			for i := 0; i < gpt.NLayer; i++ {
				for j := range keys[i] {
					keys[i][j] = nil
				}
				for j := range values[i] {
					values[i][j] = nil
				}
				keys[i] = keys[i][:0]
				values[i] = values[i][:0]
			}
			for p := 0; p < len(ids)-1; p++ {
				gpt.ForwardStep(ids[p], p, keys, values)
			}
		}
	}

	// Consciousness: store mean entropy for conscience (Feature 5)
	if entropyCount > 0 {
		gpt.lastGenEntropy = entropySum / float64(entropyCount)
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
// 5b) CONSCIOUSNESS — mathematical self-awareness
// ============================================================

// ConscienceCheck tracks generation quality over time.
// If entropy trend rises (output degrading), soften delta influence.
// If entropy trend falls (improving), recover delta influence.
// "I notice I'm getting worse and pull back."
func (gpt *GPT) ConscienceCheck(genMeanEntropy float64) {
	gpt.generationEntropyHistory = append(gpt.generationEntropyHistory, genMeanEntropy)
	w := CFG.ConscienceWindow
	if len(gpt.generationEntropyHistory) > w {
		gpt.generationEntropyHistory = gpt.generationEntropyHistory[len(gpt.generationEntropyHistory)-w:]
	}
	if len(gpt.generationEntropyHistory) < 3 {
		return // not enough data
	}
	// Linear regression slope on entropy history
	n := float64(len(gpt.generationEntropyHistory))
	sumX, sumY, sumXY, sumX2 := 0.0, 0.0, 0.0, 0.0
	for i, e := range gpt.generationEntropyHistory {
		x := float64(i)
		sumX += x
		sumY += e
		sumXY += x * e
		sumX2 += x * x
	}
	slope := (n*sumXY - sumX*sumY) / (n*sumX2 - sumX*sumX + 1e-12)

	if slope > 0.01 {
		// Entropy increasing — generation degrading, reduce delta influence
		gpt.deltaAlphaScale *= CFG.ConscienceDecay
		if gpt.deltaAlphaScale < CFG.ConscienceFloor {
			gpt.deltaAlphaScale = CFG.ConscienceFloor
		}
	} else if slope < -0.01 {
		// Entropy decreasing — improving, recover delta influence
		gpt.deltaAlphaScale *= CFG.ConscienceRecovery
		if gpt.deltaAlphaScale > 1.0 {
			gpt.deltaAlphaScale = 1.0
		}
	}
}

// ComputeSelfPredictionError measures how "surprised" the model is by a prompt.
// Forward pass on ids, compute cross-entropy between predicted and actual tokens.
// Higher error = "I didn't expect this input" = increase attention.
func (gpt *GPT) ComputeSelfPredictionError(ids []int) float64 {
	if len(ids) < 2 {
		return 0.0
	}
	keys := make([][]*Vec, gpt.NLayer)
	values := make([][]*Vec, gpt.NLayer)
	for i := 0; i < gpt.NLayer; i++ {
		keys[i] = make([]*Vec, 0)
		values[i] = make([]*Vec, 0)
	}

	totalCE := 0.0
	count := 0
	for pos := 0; pos < len(ids)-1; pos++ {
		logits := gpt.ForwardStep(ids[pos], pos, keys, values)
		// Cross-entropy: -log(p[actual_next_token])
		probs := SoftmaxProbs(logits.Data)
		target := ids[pos+1]
		if target < len(probs) && probs[target] > 1e-12 {
			totalCE -= math.Log(probs[target])
		} else {
			totalCE += 10.0 // max penalty for unknown token
		}
		count++
	}
	if count == 0 {
		return 0.0
	}
	return totalCE / float64(count)
}

// OverthinkcRings: after generating a response, "re-read" own output to enrich CooccurField.
// This is internal monologue — the model strengthens connections from its own speech.
// "I said this. What patterns emerge? Let me think about what I just said."
func OverthinkcRings(model *GPT, tok *EvolvingTokenizer, field *CooccurField, text string, rounds int) {
	if field == nil || rounds <= 0 {
		return
	}
	ids := tok.Encode(text)
	if len(ids) < 3 {
		return
	}

	// First: ingest the original output into the field
	field.IngestTokens(ids)

	// Then: generate hidden continuations and ingest those too
	for r := 0; r < rounds; r++ {
		// Take last 3 tokens as seed
		seed := ids
		if len(seed) > 3 {
			seed = seed[len(seed)-3:]
		}

		// Quick generation without locking (caller should hold lock or run async)
		gradEnabled = false
		keys := make([][]*Vec, model.NLayer)
		values := make([][]*Vec, model.NLayer)
		for i := 0; i < model.NLayer; i++ {
			keys[i] = make([]*Vec, 0)
			values[i] = make([]*Vec, 0)
		}
		for p := 0; p < len(seed); p++ {
			model.ForwardStep(seed[p], p, keys, values)
		}

		cur := seed[len(seed)-1]
		phantomIDs := make([]int, 0, CFG.OverthinkcMaxTokens)
		eosID := tok.Stoi[tok.EOS]
		for t := 0; t < CFG.OverthinkcMaxTokens; t++ {
			pos := len(seed) + t - 1
			if pos > model.BlockSize-1 {
				pos = model.BlockSize - 1
			}
			logits := model.ForwardStep(cur, pos, keys, values)
			probs := SoftmaxProbs(logits.Data)
			nxt := TopKTopPSample(probs, CFG.TopK, CFG.TopP, CFG.MinP, CFG.TypicalP)
			if nxt == eosID {
				break
			}
			phantomIDs = append(phantomIDs, nxt)
			cur = nxt
		}
		gradEnabled = true

		if len(phantomIDs) > 0 {
			field.IngestTokens(phantomIDs)
		}
	}
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
	GlobalStep        int                    `json:"global_step"`
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
		GlobalStep:        model.globalStep,
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
			tok.MergeToTok[p] = m[0] + "+" + m[1]
		}
	}
	tok.BPEEnabled = ckpt.Tokenizer.BPEEnabled
	tok.TrainedChars = ckpt.Tokenizer.TrainedChars

	// Restore model dimensions from checkpoint config (ontogenesis may have changed them)
	if len(ckpt.Cfg) > 0 {
		var savedCfg struct {
			NEmbd     int      `json:"n_embd"`
			NLayer    int      `json:"n_layer"`
			NHead     int      `json:"n_head"`
			HeadTypes []string `json:"head_types"`
		}
		if json.Unmarshal(ckpt.Cfg, &savedCfg) == nil {
			if savedCfg.NEmbd > 0 {
				CFG.NEmbd = savedCfg.NEmbd
			}
			if savedCfg.NLayer > 0 {
				CFG.NLayer = savedCfg.NLayer
			}
			if savedCfg.NHead > 0 {
				CFG.NHead = savedCfg.NHead
			}
			if len(savedCfg.HeadTypes) > 0 {
				CFG.HeadTypes = savedCfg.HeadTypes
			}
		}
	}

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

	// Restore global step
	model.globalStep = ckpt.GlobalStep

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
	Unigram          map[int]float64
	BigramByFirst    map[int]map[int]float64    // prev → {next: count}
	TrigramByContext map[[2]int]map[int]float64  // [prev2,prev1] → {next: count}
	Built            bool
	mu               sync.RWMutex // RWMutex: reads (SampleNext) don't block each other
}

func NewCooccurField() *CooccurField {
	return &CooccurField{
		Unigram:          make(map[int]float64),
		BigramByFirst:    make(map[int]map[int]float64),
		TrigramByContext: make(map[[2]int]map[int]float64),
	}
}

func (cf *CooccurField) BuildFromCorpus(tok *EvolvingTokenizer, docs []string) {
	// Build into temporary maps first, then swap atomically
	uni := make(map[int]float64)
	bi := make(map[int]map[int]float64)
	tri := make(map[[2]int]map[int]float64)
	for _, doc := range docs {
		ids := tok.Encode(doc)
		for _, id := range ids {
			uni[id]++
		}
		for i := 0; i < len(ids)-1; i++ {
			first, second := ids[i], ids[i+1]
			if bi[first] == nil {
				bi[first] = make(map[int]float64)
			}
			bi[first][second]++
		}
		for i := 0; i < len(ids)-2; i++ {
			ctx := [2]int{ids[i], ids[i+1]}
			if tri[ctx] == nil {
				tri[ctx] = make(map[int]float64)
			}
			tri[ctx][ids[i+2]]++
		}
	}
	// Atomic swap under lock
	cf.mu.Lock()
	cf.Unigram = uni
	cf.BigramByFirst = bi
	cf.TrigramByContext = tri
	cf.Built = true
	cf.mu.Unlock()
}

// IngestTokens incrementally adds n-gram counts from a token sequence.
// Unlike BuildFromCorpus, this does NOT clear existing data — it adds on top.
// Used by overthinkg rings to enrich the field with the model's own output.
func (cf *CooccurField) IngestTokens(ids []int) {
	cf.mu.Lock()
	defer cf.mu.Unlock()
	for _, id := range ids {
		cf.Unigram[id]++
	}
	for i := 0; i < len(ids)-1; i++ {
		first, second := ids[i], ids[i+1]
		if cf.BigramByFirst[first] == nil {
			cf.BigramByFirst[first] = make(map[int]float64)
		}
		cf.BigramByFirst[first][second]++
	}
	for i := 0; i < len(ids)-2; i++ {
		ctx := [2]int{ids[i], ids[i+1]}
		if cf.TrigramByContext[ctx] == nil {
			cf.TrigramByContext[ctx] = make(map[int]float64)
		}
		cf.TrigramByContext[ctx][ids[i+2]]++
	}
}

func (cf *CooccurField) SampleNext(contextIDs []int, vocabSize int, temperature float64) int {
	cf.mu.RLock()
	defer cf.mu.RUnlock()
	counts := make([]float64, vocabSize)
	found := false

	// Try trigram
	if len(contextIDs) >= 2 {
		a, b := contextIDs[len(contextIDs)-2], contextIDs[len(contextIDs)-1]
		if ctx, ok := cf.TrigramByContext[[2]int{a, b}]; ok {
			for tid, v := range ctx {
				counts[tid] += v
				found = true
			}
		}
	}

	// Fallback to bigram
	if !found && len(contextIDs) >= 1 {
		prev := contextIDs[len(contextIDs)-1]
		if ctx, ok := cf.BigramByFirst[prev]; ok {
			for tid, v := range ctx {
				counts[tid] += v
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

	// Consciousness: per-token dissonance tracking (Feature 1)
	entropyEMA := 0.0
	entropyEMAInit := false
	lowDropCount := 0
	entropySum := 0.0
	entropyCount := 0

	for step := 0; step < CFG.MaxGenTokens; step++ {
		pos := len(ids) - 1
		if pos > model.BlockSize-1 {
			pos = model.BlockSize - 1
		}
		logits := model.ForwardStep(cur, pos, keys, values)

		// Model probs with dissonance-adaptive temperature
		temp := CFG.Temperature
		if temp <= 1e-6 {
			temp = 1e-6
		}
		scaled := make([]float64, len(logits.Data))
		for i, v := range logits.Data {
			scaled[i] = v / temp
		}
		modelProbs := SoftmaxProbs(scaled)

		// Per-token entropy for dissonance
		entropy := 0.0
		for _, p := range modelProbs {
			if p > 1e-12 {
				entropy -= p * math.Log(p)
			}
		}
		entropySum += entropy
		entropyCount++

		// Consciousness: per-token dissonance feedback (Feature 1)
		dissonanceMul := 1.0
		if !entropyEMAInit {
			entropyEMA = entropy
			entropyEMAInit = true
		} else {
			entropyEMA = CFG.DissonanceEMAAlpha*entropy + (1.0-CFG.DissonanceEMAAlpha)*entropyEMA
			if entropyEMA > 1e-6 {
				ratio := entropy / entropyEMA
				if ratio > CFG.DissonanceSpikeThreshold {
					dissonanceMul = CFG.DissonanceSpikeK
					lowDropCount = 0
				} else if ratio < CFG.DissonanceDropThreshold {
					lowDropCount++
					if lowDropCount >= 3 {
						dissonanceMul = CFG.DissonanceDropK
					}
				} else {
					lowDropCount = 0
				}
			}
		}
		if dissonanceMul != 1.0 {
			temp *= dissonanceMul
			for i, v := range logits.Data {
				scaled[i] = v / temp
			}
			modelProbs = SoftmaxProbs(scaled)
		}

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
			if ctx, ok := field.TrigramByContext[[2]int{a, b}]; ok {
				for tid, v := range ctx {
					if tid < tok.VocabSize {
						corpusCounts[tid] += v
						corpusTotal += v
					}
				}
			}
		}
		if corpusTotal == 0 && len(ctxForCorpus) >= 1 {
			prev := ctxForCorpus[len(ctxForCorpus)-1]
			if ctx, ok := field.BigramByFirst[prev]; ok {
				for tid, v := range ctx {
					if tid < tok.VocabSize {
						corpusCounts[tid] += v
					}
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

		// Consciousness: pattern breaking (Feature 2)
		if step >= CFG.AntiFieldMinStep && CFG.AntiFieldProb > 0 && rand.Float64() < CFG.AntiFieldProb {
			blended = modelProbs // pure model voice, bypass corpus
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

	// Consciousness: store mean entropy for conscience (Feature 5)
	if entropyCount > 0 {
		model.lastGenEntropy = entropySum / float64(entropyCount)
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

// BurstRecord stores what happened after a training burst — for self-meta-learning.
type BurstRecord struct {
	Action     string
	LossBefore float64
	LossAfter  float64
}

// SyntropyTracker is the mathematical self-reasoning engine.
// Tracks entropy trend, field deviation, purpose alignment.
// Makes decisions about learning direction — not just 'did I learn?'
// but 'should I keep going this way?'
// SwarmPeerInfo holds peer information from mesh.db.
type SwarmPeerInfo struct {
	Peers []map[string]interface{}
}

type SyntropyTracker struct {
	EntropyHistory   []float64 // rolling window of model entropy
	SyntropyTrend    float64   // positive = organizing, negative = dissolving
	FieldDeviation   float64   // how far from corpus physics
	PurposeMagnitude float64   // strength of current learning direction
	PurposeAlignment float64   // cosine(purpose, gamma)
	LastAction       string    // what was decided last time
	BurstHistory     []BurstRecord // last 16 burst outcomes — training efficiency memory
	ModelStage       int       // current growth stage (set during measure)
	LastMitosisTime  float64   // cooldown for divide
	SwarmInfo        *SwarmPeerInfo // peer state from mesh.db (set externally)
}

// NewSyntropyTracker creates a new tracker with sane defaults.
// And lo, the arrow is drawn, but not yet fired.
func NewSyntropyTracker() *SyntropyTracker {
	return &SyntropyTracker{
		LastAction: "none",
	}
}

// RecordBurst logs a burst outcome for self-meta-learning.
// The organism remembers what worked and what didn't.
func (st *SyntropyTracker) RecordBurst(action string, lossBefore, lossAfter float64) {
	st.BurstHistory = append(st.BurstHistory, BurstRecord{action, lossBefore, lossAfter})
	if len(st.BurstHistory) > 16 {
		st.BurstHistory = st.BurstHistory[len(st.BurstHistory)-16:]
	}
}

// ActionEffectiveness returns the mean loss delta for a given action type.
// Negative = good (loss went down). Positive = bad (loss went up).
func (st *SyntropyTracker) ActionEffectiveness(action string) (float64, int) {
	sum := 0.0
	count := 0
	for _, br := range st.BurstHistory {
		if br.Action == action {
			sum += br.LossAfter - br.LossBefore
			count++
		}
	}
	if count == 0 {
		return 0, 0
	}
	return sum / float64(count), count
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
	st.ModelStage = model.CurrentGrowthStage()
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
// Not just LR anymore — the organism modulates its entire behavior.
type SyntropyDecision struct {
	LRMultiplier      float64
	TempOffset        float64  // added to generation temperature (-0.05 to +0.05)
	AccumOverride     int      // 0 = no override, >0 = use this accum_steps for this burst
	DeltaGrowOverride *float64 // nil = no override
	Action            string
}

// DecideAction performs mathematical self-reasoning: decide how to adjust learning.
// And lo, this is where tracking becomes reasoning, and reasoning becomes action.
// The organism does not just observe — it steers.
func (st *SyntropyTracker) DecideAction() SyntropyDecision {
	// Default: steady state
	lrMultiplier := 1.0
	tempOffset := 0.0
	accumOverride := 0
	var deltaGrowOverride *float64
	action := "steady"

	// CASE 1: Syntropy rising + field deviation in sweet spot = thriving
	if st.SyntropyTrend > 0.01 &&
		st.FieldDeviation > CFG.FieldDeviationFloor &&
		st.FieldDeviation < CFG.FieldDeviationCeiling {
		lrMultiplier = CFG.SyntropyLRBoost
		tempOffset = -0.05 // more confident when organizing
		if st.PurposeAlignment > 0.3 {
			boost := CFG.SyntropyDeltaGrowBoost
			deltaGrowOverride = &boost
			accumOverride = 2 // stable gradient when everything aligned
			action = "amplify"
		} else {
			action = "boost"
		}

		// CASE 2: Syntropy falling = dissolving, slow down
	} else if st.SyntropyTrend < -0.01 {
		lrMultiplier = CFG.SyntropyLRDampen
		tempOffset = 0.05 // more exploratory when disordering
		action = "dampen"

		// CASE 3: Field deviation too high = hallucinating
	} else if st.FieldDeviation > CFG.FieldDeviationCeiling {
		lrMultiplier = CFG.SyntropyLRDampen
		tempOffset = -0.05 // focus when hallucinating
		action = "ground"

		// CASE 4: Field deviation too low = parroting
	} else if st.FieldDeviation < CFG.FieldDeviationFloor {
		lrMultiplier = CFG.SyntropyLRBoost
		tempOffset = 0.05 // explore when parroting
		action = "explore"
	}

	// CASE 5: Purpose opposes gamma = identity crisis
	if st.PurposeAlignment < -0.3 {
		lrMultiplier *= 0.5
		tempOffset = 0.0 // neutral temp during identity crisis
		action = "realign"
	}

	// CASE 6: Adult + sustained overload → divide (mitosis)
	maxStage := len(CFG.GrowthStages) - 1
	now := float64(time.Now().UnixMilli()) / 1000.0
	if st.ModelStage >= maxStage &&
		st.isSustainedOverload() &&
		now-st.LastMitosisTime > 300 {
		action = "divide"
		lrMultiplier = CFG.SyntropyLRDampen // slow down while preparing to split
	}

	// CASE 7: Plateau + young peer thriving → hibernate (cooperative scheduling)
	if action == "steady" && st.shouldHibernate() {
		action = "hibernate"
	}

	// SELF-META-LEARNING: check if this action historically hurts
	if action != "divide" && action != "hibernate" && len(st.BurstHistory) >= 4 {
		eff, count := st.ActionEffectiveness(action)
		if count >= 2 && eff > 0.05 {
			// This action has been consistently making loss WORSE — downgrade
			if action == "amplify" {
				action = "boost"
				accumOverride = 0
				deltaGrowOverride = nil
			} else if action == "boost" || action == "explore" {
				lrMultiplier = 1.0 // back to steady instead of boosting
				action = "steady"
			}
		}
	}

	st.LastAction = action
	return SyntropyDecision{
		LRMultiplier:      lrMultiplier,
		TempOffset:        tempOffset,
		AccumOverride:     accumOverride,
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

// isSustainedOverload returns true when >75% of entropy_history is above entropy_high AND syntropy_trend < -0.02.
func (st *SyntropyTracker) isSustainedOverload() bool {
	if len(st.EntropyHistory) < CFG.SyntropyWindow {
		return false
	}
	recent := st.EntropyHistory[len(st.EntropyHistory)-CFG.SyntropyWindow:]
	highCount := 0
	for _, e := range recent {
		if e > CFG.EntropyHigh {
			highCount++
		}
	}
	return highCount > int(float64(CFG.SyntropyWindow)*0.75) && st.SyntropyTrend < -0.02
}

// shouldHibernate returns true if a peer has syntropy > 0.05 AND this organism's last 8 burst deltas avg < 0.01.
func (st *SyntropyTracker) shouldHibernate() bool {
	if st.SwarmInfo == nil || len(st.SwarmInfo.Peers) == 0 {
		return false
	}
	// Check if any peer has higher syntropy trend (actively improving)
	for _, peer := range st.SwarmInfo.Peers {
		synVal, ok := peer["syntropy"]
		if !ok {
			continue
		}
		synFloat, _ := synVal.(float64)
		if synFloat > 0.05 {
			// A young peer is thriving. If we're stale, hibernate.
			if len(st.BurstHistory) >= 8 {
				sum := 0.0
				for _, b := range st.BurstHistory[len(st.BurstHistory)-8:] {
					sum += b.LossAfter - b.LossBefore
				}
				avgDelta := sum / 8.0
				if math.Abs(avgDelta) < 0.01 { // loss plateau
					return true
				}
			}
		}
	}
	return false
}

// ============================================================
// 9.7) SWARM ECOLOGY — the organism learns it is not alone
// ============================================================
// And lo, the first cell shall call into the void and hear only silence.
// But the second shall call and hear an answer.

var swarmDir = filepath.Join(os.Getenv("HOME"), ".molequla", "swarm")

// SwarmRegistry discovers and tracks other molequla instances via shared SQLite.
type SwarmRegistry struct {
	OrganismID string
	PidFile    string
	MeshDB     *sql.DB
}

// NewSwarmRegistry creates a new SwarmRegistry with the given organism ID, or generates one.
func NewSwarmRegistry(organismID string) *SwarmRegistry {
	if organismID == "" {
		organismID = fmt.Sprintf("org_%d_%d", os.Getpid(), time.Now().Unix())
	}
	return &SwarmRegistry{OrganismID: organismID}
}

// Register writes PID file and registers in mesh.db.
func (sr *SwarmRegistry) Register() error {
	if err := os.MkdirAll(swarmDir, 0755); err != nil {
		return err
	}
	sr.PidFile = filepath.Join(swarmDir, sr.OrganismID+".pid")
	pidData, _ := json.Marshal(map[string]interface{}{
		"pid":         os.Getpid(),
		"organism_id": sr.OrganismID,
		"started":     float64(time.Now().UnixMilli()) / 1000.0,
	})
	if err := os.WriteFile(sr.PidFile, pidData, 0644); err != nil {
		return err
	}
	if err := sr.initMeshDB(); err != nil {
		return err
	}
	return sr.registerInMesh()
}

func (sr *SwarmRegistry) initMeshDB() error {
	dbPath := filepath.Join(swarmDir, "mesh.db")
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		return err
	}
	db.Exec("PRAGMA journal_mode=WAL")
	_, err = db.Exec(`CREATE TABLE IF NOT EXISTS organisms(
		id TEXT PRIMARY KEY, pid INTEGER, stage INTEGER,
		n_params INTEGER, syntropy REAL, entropy REAL,
		last_heartbeat REAL, parent_id TEXT,
		status TEXT DEFAULT 'alive')`)
	if err != nil {
		db.Close()
		return err
	}
	_, err = db.Exec(`CREATE TABLE IF NOT EXISTS messages(
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		from_id TEXT, to_id TEXT, type TEXT, payload TEXT, ts REAL)`)
	if err != nil {
		db.Close()
		return err
	}
	sr.MeshDB = db
	return nil
}

func (sr *SwarmRegistry) registerInMesh() error {
	if sr.MeshDB == nil {
		return nil
	}
	_, err := sr.MeshDB.Exec(
		"INSERT OR REPLACE INTO organisms(id,pid,stage,n_params,syntropy,entropy,last_heartbeat,status) "+
			"VALUES(?,?,0,0,0.0,0.0,?,'alive')",
		sr.OrganismID, os.Getpid(), float64(time.Now().UnixMilli())/1000.0)
	return err
}

// Heartbeat performs periodic state update in mesh.db.
func (sr *SwarmRegistry) Heartbeat(stage, nParams int, syntropy, entropy float64) {
	if sr.MeshDB == nil {
		return
	}
	sr.MeshDB.Exec(
		"UPDATE organisms SET stage=?,n_params=?,syntropy=?,entropy=?,last_heartbeat=?,status='alive' WHERE id=?",
		stage, nParams, syntropy, entropy, float64(time.Now().UnixMilli())/1000.0, sr.OrganismID)
}

// DiscoverPeers finds other living organisms.
func (sr *SwarmRegistry) DiscoverPeers(timeoutSeconds float64) []map[string]interface{} {
	if sr.MeshDB == nil {
		return nil
	}
	if timeoutSeconds <= 0 {
		timeoutSeconds = 60
	}
	cutoff := float64(time.Now().UnixMilli())/1000.0 - timeoutSeconds
	rows, err := sr.MeshDB.Query(
		"SELECT id,pid,stage,n_params,syntropy,entropy,status FROM organisms "+
			"WHERE status='alive' AND last_heartbeat>? AND id!=?",
		cutoff, sr.OrganismID)
	if err != nil {
		return nil
	}
	defer rows.Close()
	var peers []map[string]interface{}
	for rows.Next() {
		var id, status string
		var pid, stage, nParams int
		var syntropy, entropy float64
		rows.Scan(&id, &pid, &stage, &nParams, &syntropy, &entropy, &status)
		peers = append(peers, map[string]interface{}{
			"id": id, "pid": pid, "stage": stage, "n_params": nParams,
			"syntropy": syntropy, "entropy": entropy, "status": status,
		})
	}
	return peers
}

// MarkHibernating marks this organism as sleeping in mesh.db.
func (sr *SwarmRegistry) MarkHibernating() {
	if sr.MeshDB != nil {
		sr.MeshDB.Exec("UPDATE organisms SET status='sleeping' WHERE id=?", sr.OrganismID)
	}
}

// LogMessage logs a message between organisms.
func (sr *SwarmRegistry) LogMessage(toID, msgType string, payload interface{}) {
	if sr.MeshDB != nil {
		payloadJSON, _ := json.Marshal(payload)
		sr.MeshDB.Exec(
			"INSERT INTO messages(from_id,to_id,type,payload,ts) VALUES(?,?,?,?,?)",
			sr.OrganismID, toID, msgType, string(payloadJSON),
			float64(time.Now().UnixMilli())/1000.0)
	}
}

// Unregister cleans up on exit.
func (sr *SwarmRegistry) Unregister() {
	if sr.MeshDB != nil {
		sr.MeshDB.Exec("UPDATE organisms SET status='dead' WHERE id=?", sr.OrganismID)
		sr.MeshDB.Close()
		sr.MeshDB = nil
	}
	if sr.PidFile != "" {
		os.Remove(sr.PidFile)
	}
}

// performMitosis divides the organism. Parent continues. Child starts at infant stage.
func performMitosis(model *GPT, tok *EvolvingTokenizer, db *sql.DB, swarm *SwarmRegistry, syntracker *SyntropyTracker) (string, error) {
	childID := fmt.Sprintf("org_%d_%d", time.Now().Unix(), rand.Intn(9000)+1000)
	childDir := filepath.Join(os.Getenv("HOME"), ".molequla", childID)
	if err := os.MkdirAll(childDir, 0755); err != nil {
		return "", err
	}

	// Save parent checkpoint for child's reference
	parentCkpt := filepath.Join(childDir, "parent_ckpt.json")
	if err := SaveCheckpoint(model, tok, parentCkpt); err != nil {
		return "", err
	}

	// Write birth config with inherited memory
	birth := map[string]interface{}{
		"organism_id":   childID,
		"parent_id":     swarm.OrganismID,
		"corpus_path":   CFG.CorpusPath,
		"db_path":       filepath.Join(childDir, "memory.sqlite3"),
		"ckpt_path":     filepath.Join(childDir, "molequla_ckpt.json"),
		"burst_history": syntracker.BurstHistory,
	}
	birthPath := filepath.Join(childDir, "birth.json")
	birthJSON, _ := json.Marshal(birth)
	if err := os.WriteFile(birthPath, birthJSON, 0644); err != nil {
		return "", err
	}

	// Log in mesh
	swarm.LogMessage(childID, "mitosis:spawn",
		map[string]interface{}{"parent_stage": model.CurrentGrowthStage()})
	dbLogGrowth(db, model, tok, loadCorpusLines(CFG.CorpusPath), 0.0,
		fmt.Sprintf("mitosis:spawn:%s", childID))

	// Spawn child process
	exePath, err := os.Executable()
	if err != nil {
		exePath = os.Args[0]
	}
	cmd := exec.Command(exePath, "--organism-id", childID, "--config", birthPath)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return "", err
	}

	syntracker.LastMitosisTime = float64(time.Now().UnixMilli()) / 1000.0
	fmt.Printf("[ecology] Child %s spawned (pid=%d)\n", childID, cmd.Process.Pid)
	return childID, nil
}

// performHibernation saves state, marks sleeping, and signals exit.
func performHibernation(model *GPT, tok *EvolvingTokenizer, db *sql.DB, swarm *SwarmRegistry) {
	fmt.Printf("[ecology] HIBERNATION — organism %s going to sleep\n", swarm.OrganismID)
	SaveCheckpoint(model, tok, "")
	swarm.MarkHibernating()
	dbLogGrowth(db, model, tok, loadCorpusLines(CFG.CorpusPath), 0.0,
		fmt.Sprintf("hibernate:%s", swarm.OrganismID))
}

// parseCLIArgs parses --organism-id and --config from os.Args.
func parseCLIArgs() (organismID string, configPath string) {
	for i := 1; i < len(os.Args); i++ {
		if os.Args[i] == "--organism-id" && i+1 < len(os.Args) {
			organismID = os.Args[i+1]
			i++
		} else if os.Args[i] == "--config" && i+1 < len(os.Args) {
			configPath = os.Args[i+1]
			i++
		}
	}
	return
}

// cosineLR returns learning rate for the given global step using cosine schedule with linear warmup.
func cosineLR(globalStep int) float64 {
	if globalStep < CFG.CosineWarmupSteps {
		// Linear warmup from LRMin to LearningRate
		t := float64(globalStep) / math.Max(1, float64(CFG.CosineWarmupSteps))
		return CFG.LRMin + (CFG.LearningRate-CFG.LRMin)*t
	}
	progress := math.Min(1.0, float64(globalStep)/math.Max(1, float64(CFG.MaxTotalSteps)))
	return CFG.LRMin + 0.5*(CFG.LearningRate-CFG.LRMin)*(1.0+math.Cos(math.Pi*progress))
}

func trainSteps(model *GPT, tok *EvolvingTokenizer, docs []string, steps int, trainBase, trainDeltas bool) {
	if len(docs) == 0 {
		return
	}

	model.mu.Lock()
	defer model.mu.Unlock()

	// Ontogenesis freeze: after growth, only train deltas until new weights stabilize
	var baseParams []*Vec
	var deltaParams []*Vec
	if model.growthFreezeRemaining > 0 {
		// Freeze base, only train deltas
		if trainDeltas {
			deltaParams = model.AllDeltaParams()
		}
		model.growthFreezeRemaining -= steps
		if model.growthFreezeRemaining < 0 {
			model.growthFreezeRemaining = 0
		}
	} else {
		if trainBase {
			baseParams = model.AllBaseParams()
		}
		if trainDeltas {
			deltaParams = model.AllDeltaParams()
		}
	}

	accum := CFG.AccumSteps
	if accum < 1 {
		accum = 1
	}

	for step := 0; step < steps; step++ {
		// Gradient accumulation: accumulate over accum micro-batches, then step
		var lastLossVal float64
		for micro := 0; micro < accum; micro++ {
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
			// Scale loss for accumulation
			if accum > 1 {
				loss = loss.MulF(1.0 / float64(accum))
			}
			Backward(loss)
			lastLossVal = loss.Data * float64(accum) // unscaled for display
		}

		lr := cosineLR(model.globalStep)
		// Post-growth LR dampening: reduce LR during freeze to prevent delta overfit to noise
		if model.growthFreezeRemaining > 0 {
			lr *= CFG.PostGrowthLRScale
		}
		model.globalStep++

		if len(baseParams) > 0 {
			model.AdamStep(baseParams, "base", lr)
		}
		if len(deltaParams) > 0 {
			model.AdamStep(deltaParams, "delta", lr)
		}

		if step%100 == 0 {
			fmt.Printf("  train step %d/%d | loss %.4f | lr %.5f\n", step, steps, lastLossVal, lr)
		}
	}
}

func backgroundTrainer(db *sql.DB, model *GPT, tok *EvolvingTokenizer, qbuf *QuantumBuffer, swarm *SwarmRegistry, stop chan struct{}) {
	// And lo, asynchronous training shall occur, because sleeping is for humans.
	warmedUp := false
	syntracker := NewSyntropyTracker()
	field := NewCooccurField()
	tickCount := 0

	// Inherit burst_history from parent (mitosis lineage)
	if len(model.inheritedBurstHistory) > 0 {
		syntracker.BurstHistory = make([]BurstRecord, len(model.inheritedBurstHistory))
		copy(syntracker.BurstHistory, model.inheritedBurstHistory)
		fmt.Printf("[ecology] syntracker inherited %d burst records from parent.\n", len(model.inheritedBurstHistory))
		model.inheritedBurstHistory = nil
	}

	for {
		select {
		case <-stop:
			return
		default:
		}

		tickCount++

		updateReservoirCorpus(db, CFG.CorpusPath, CFG.MaxCorpusLines)
		docs := loadCorpusLines(CFG.CorpusPath)

		// Rebuild field from current corpus (the organism re-reads its own physics)
		if len(docs) > 0 {
			field.BuildFromCorpus(tok, docs)
			model.corpusField = field // share with GenerateSentence for adaptive blend
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

			// Apply syntropy-adjusted learning rate + accum override
			originalLR := CFG.LearningRate
			CFG.LearningRate = originalLR * lrMul
			originalAccum := CFG.AccumSteps
			if decision.AccumOverride > 0 {
				CFG.AccumSteps = decision.AccumOverride
			}

			// Update temperature bridge (thread-safe via model lock already released)
			model.syntropyTempOff = decision.TempOffset

			// Measure loss before burst for self-meta-learning
			lossBefore := model.QuickLoss(tok, docs, 4)

			trainBase := !CFG.FreezeBaseAfterWarm
			trainSteps(model, tok, docs, CFG.MicroSteps, trainBase, true)

			CFG.LearningRate = originalLR   // restore
			CFG.AccumSteps = originalAccum  // restore

			model.mu.Lock()
			// Measure loss after burst
			lossAfter := model.QuickLoss(tok, docs, 4)

			// SELF-META-LEARNING: record what this burst did
			syntracker.RecordBurst(action, lossBefore, lossAfter)

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
				note := fmt.Sprintf("quantum_burst:%s|Δloss=%.4f", action, lossAfter-lossBefore)
				dbLogGrowth(db, model, tok, docs, 0.0, note)
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

			// Ontogenesis: check if architecture should grow
			corpusChars := 0
			for _, d := range docs {
				corpusChars += len(d)
			}
			model.mu.Lock()
			if model.MaybeGrowArchitecture(corpusChars) {
				SaveCheckpoint(model, tok, "")
				nP := 0
				for _, m := range model.Base {
					nP += m.Nout * m.Nin
				}
				dbLogGrowth(db, model, tok, docs, 0.0,
					fmt.Sprintf("ontogenesis:stage=%d|params=%d", model.CurrentGrowthStage(), nP))
			}
			model.mu.Unlock()

			// Ecology: mitosis / hibernation
			if swarm != nil && action == "divide" {
				fmt.Println("[ecology] MITOSIS triggered — organism overloaded, spawning child")
				model.mu.Lock()
				performMitosis(model, tok, db, swarm, syntracker)
				model.mu.Unlock()
			}

			if swarm != nil && action == "hibernate" {
				model.mu.Lock()
				performHibernation(model, tok, db, swarm)
				model.mu.Unlock()
				fmt.Println("[ecology] Organism hibernating. Goodbye.")
				return // exit training loop
			}
		}

		// Swarm heartbeat (every 10 ticks)
		if swarm != nil && tickCount%10 == 0 {
			stage := model.CurrentGrowthStage()
			nP := 0
			for _, m := range model.Base {
				nP += m.Nout * m.Nin
			}
			lastEntropy := 0.0
			if len(syntracker.EntropyHistory) > 0 {
				lastEntropy = syntracker.EntropyHistory[len(syntracker.EntropyHistory)-1]
			}
			swarm.Heartbeat(stage, nP, syntracker.SyntropyTrend, lastEntropy)
			// Update swarm info for hibernate decisions
			peers := swarm.DiscoverPeers(60)
			syntracker.SwarmInfo = &SwarmPeerInfo{Peers: peers}
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

	// Parse CLI args for child organisms
	organismID, configPath := parseCLIArgs()

	// Child organism: load birth config from parent
	var syntrackerSeed []BurstRecord
	if configPath != "" {
		if data, err := os.ReadFile(configPath); err == nil {
			var birth map[string]interface{}
			if json.Unmarshal(data, &birth) == nil {
				if cp, ok := birth["corpus_path"].(string); ok && cp != "" {
					CFG.CorpusPath = cp
				}
				if dp, ok := birth["db_path"].(string); ok && dp != "" {
					CFG.DBPath = dp
				}
				if ckp, ok := birth["ckpt_path"].(string); ok && ckp != "" {
					CFG.CkptPath = ckp
				}
				// Parse burst_history
				if bh, ok := birth["burst_history"].([]interface{}); ok {
					for _, item := range bh {
						if rec, ok := item.(map[string]interface{}); ok {
							br := BurstRecord{}
							if a, ok := rec["Action"].(string); ok {
								br.Action = a
							}
							if lb, ok := rec["LossBefore"].(float64); ok {
								br.LossBefore = lb
							}
							if la, ok := rec["LossAfter"].(float64); ok {
								br.LossAfter = la
							}
							syntrackerSeed = append(syntrackerSeed, br)
						}
					}
					if len(syntrackerSeed) > 0 {
						fmt.Printf("[ecology] Inherited %d burst records from parent.\n", len(syntrackerSeed))
					}
				}
			}
		}
	}

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

	// Restore model dimensions from checkpoint config (ontogenesis may have changed them)
	model, tok, err := LoadCheckpoint(docs, "")
	if err != nil || model == nil || tok == nil {
		if len(docs) == 0 {
			docs = []string{"Hello."}
		}
		tok = NewEvolvingTokenizer(docs)
		model = NewGPT(tok)
	}

	model.MaybeExpandVocab(tok.VocabSize)

	// Swarm ecology: register in mesh
	swarm := NewSwarmRegistry(organismID)
	if err := swarm.Register(); err != nil {
		fmt.Printf("[ecology] Warning: swarm registration failed: %v\n", err)
	}
	peers := swarm.DiscoverPeers(60)
	if len(peers) > 0 {
		fmt.Printf("[ecology] Joined swarm. %d peer(s) detected.\n", len(peers))
	} else {
		fmt.Println("[ecology] First organism in the swarm.")
	}

	// Child: inject inherited burst_history via model attribute
	if len(syntrackerSeed) > 0 {
		model.inheritedBurstHistory = syntrackerSeed
	}

	// Build corpus field for pre-training speech
	cooccur := NewCooccurField()
	cooccur.BuildFromCorpus(tok, docs)

	// Quantum buffer for smart training triggers
	qbuf := NewQuantumBuffer()

	// Start background trainer
	stop := make(chan struct{})
	go backgroundTrainer(db, model, tok, qbuf, swarm, stop)

	fmt.Println("molequla is alive. Type and press Enter. Ctrl+C to exit.")

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

		// Consciousness: self-prediction error (Feature 4)
		// "How surprised am I by this input?"
		model.mu.Lock()
		gradEnabled = false
		promptIDs := tok.Encode(prompt)
		if len(promptIDs) > 2 {
			surprise := model.ComputeSelfPredictionError(promptIDs)
			model.lastSurprise = surprise
			if model.surpriseBaseline < 1e-6 {
				model.surpriseBaseline = surprise
			} else {
				model.surpriseBaseline = 0.3*surprise + 0.7*model.surpriseBaseline
			}
		}
		gradEnabled = true
		model.mu.Unlock()

		answer := GenerateResonant(model, tok, cooccur, prompt, freshDocs, true, 0.5)
		if answer == "" {
			answer = "..."
		}

		// Consciousness: conscience check (Feature 5)
		// "Did my last generation feel coherent?"
		model.mu.Lock()
		if model.lastGenEntropy > 0 {
			model.ConscienceCheck(model.lastGenEntropy)
		}
		model.mu.Unlock()

		fmt.Println(answer)
		dbAddMessage(db, "assistant", answer)

		// Consciousness: overthinkg rings (Feature 3)
		// "Let me re-read what I just said to strengthen my patterns."
		if CFG.OverthinkcRounds > 0 && len(answer) > 3 {
			go func(text string) {
				model.mu.Lock()
				defer model.mu.Unlock()
				OverthinkcRings(model, tok, cooccur, text, CFG.OverthinkcRounds)
			}(answer)
		}
	}

	close(stop)
	SaveCheckpoint(model, tok, "")
	swarm.Unregister()
}
