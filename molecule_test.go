package main

import (
	"math"
	"testing"
)

// TestTopKTopPSample tests the basic top-k/top-p sampling.
func TestTopKTopPSample(t *testing.T) {
	probs := []float64{0.1, 0.2, 0.05, 0.65}

	// With k=1, should always pick argmax (index 3)
	for i := 0; i < 10; i++ {
		result := TopKTopPSample(probs, 1, 1.0, 0.0, 1.0)
		if result != 3 {
			t.Errorf("With k=1, expected index 3, got %d", result)
		}
	}
}

// TestTopKTopPSampleTopKLimits tests that top-k limits candidates.
func TestTopKTopPSampleTopKLimits(t *testing.T) {
	probs := []float64{0.1, 0.2, 0.3, 0.4}
	// With k=2, only indices 2 and 3 should be possible
	for i := 0; i < 100; i++ {
		result := TopKTopPSample(probs, 2, 1.0, 0.0, 1.0)
		if result != 2 && result != 3 {
			t.Errorf("With k=2, expected index 2 or 3, got %d", result)
		}
	}
}

// TestMinPFiltersLowProbs tests that min_p filters tokens with low probabilities.
func TestMinPFiltersLowProbs(t *testing.T) {
	probs := []float64{0.01, 0.02, 0.07, 0.9} // max = 0.9
	// With min_p=0.1, threshold = 0.09
	// Only index 3 (0.9 >= 0.09) should remain
	for i := 0; i < 20; i++ {
		result := TopKTopPSample(probs, 0, 1.0, 0.1, 1.0)
		if result != 3 {
			t.Errorf("With min_p=0.1, expected index 3, got %d", result)
		}
	}
}

// TestMinPKeepsProportional tests that min_p keeps tokens proportionally above threshold.
func TestMinPKeepsProportional(t *testing.T) {
	probs := []float64{0.05, 0.15, 0.30, 0.50} // max = 0.5
	// With min_p=0.2, threshold = 0.1
	// Indices 1 (0.15), 2 (0.30), 3 (0.50) should remain
	for i := 0; i < 100; i++ {
		result := TopKTopPSample(probs, 0, 1.0, 0.2, 1.0)
		// Index 0 (0.05) should never be sampled
		if result == 0 {
			t.Errorf("With min_p=0.2, index 0 should never be sampled, but got it")
		}
	}
}

// TestTypicalPPrefersTypical tests that typical_p prefers tokens with typical information content.
func TestTypicalPPrefersTypical(t *testing.T) {
	// With uniform probs, all tokens are equally typical
	probs := []float64{0.25, 0.25, 0.25, 0.25}
	seen := make(map[int]bool)
	for i := 0; i < 100; i++ {
		result := TopKTopPSample(probs, 0, 1.0, 0.0, 0.9)
		seen[result] = true
	}
	// Should see at least 3 different indices
	if len(seen) < 3 {
		t.Errorf("With uniform probs and typical_p=0.9, expected at least 3 different indices, got %d", len(seen))
	}
}

// TestTypicalPWithVariedProbs tests that typical_p works with varied probability distributions.
func TestTypicalPWithVariedProbs(t *testing.T) {
	probs := []float64{0.01, 0.09, 0.30, 0.60}
	// Should still produce valid samples
	for i := 0; i < 50; i++ {
		result := TopKTopPSample(probs, 0, 1.0, 0.0, 0.8)
		if result < 0 || result >= len(probs) {
			t.Errorf("Sample out of range: %d", result)
		}
	}
}

// TestCombinedMinPTypicalP tests that min_p and typical_p work together.
func TestCombinedMinPTypicalP(t *testing.T) {
	probs := []float64{0.02, 0.08, 0.20, 0.70}
	// min_p=0.1 with max=0.7 => threshold=0.07
	// Keeps indices 1, 2, 3 (0.08, 0.20, 0.70)
	for i := 0; i < 50; i++ {
		result := TopKTopPSample(probs, 0, 1.0, 0.1, 0.9)
		// Index 0 should never be sampled (below min_p threshold)
		if result == 0 {
			t.Errorf("With min_p=0.1, index 0 should never be sampled")
		}
	}
}

// TestSoftmaxProbs tests the softmax function.
func TestSoftmaxProbs(t *testing.T) {
	logits := []float64{1.0, 2.0, 3.0, 4.0}
	probs := SoftmaxProbs(logits)

	// Sum should be ~1
	sum := 0.0
	for _, p := range probs {
		sum += p
	}
	if math.Abs(sum-1.0) > 1e-6 {
		t.Errorf("Softmax should sum to 1, got %f", sum)
	}

	// All should be positive
	for i, p := range probs {
		if p < 0 {
			t.Errorf("Softmax[%d] should be positive, got %f", i, p)
		}
	}

	// Higher logits should give higher probs
	for i := 0; i < len(probs)-1; i++ {
		if probs[i] >= probs[i+1] {
			t.Errorf("Higher logit should give higher prob: probs[%d]=%f >= probs[%d]=%f", i, probs[i], i+1, probs[i+1])
		}
	}
}
