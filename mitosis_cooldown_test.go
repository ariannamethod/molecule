package main

import (
	"testing"
	"time"
)

// Regression guard for the mitosis cooldown seed (deep-core audit bug 1+6).
// NewSyntropyTracker must seed LastMitosisTime to birth time so the 300s divide
// cooldown (CASE 6: now-LastMitosisTime>300) also guards the FIRST division.
// Zero-init made now(epoch-sec)-0 always >300, so seed organisms and every
// mitosis child could divide instantly. This must never regress.
func TestMitosisCooldownSeededAtBirth(t *testing.T) {
	st := NewSyntropyTracker()
	now := float64(time.Now().UnixMilli()) / 1000.0
	if st.LastMitosisTime <= 0 {
		t.Fatalf("LastMitosisTime not seeded: got %v (zero-init bug back)", st.LastMitosisTime)
	}
	if d := now - st.LastMitosisTime; d < 0 || d > 5 {
		t.Fatalf("LastMitosisTime not ~= now: now-LMT=%v s (want 0..5)", d)
	}
	// The divide gate (now-LastMitosisTime > 300) MUST be false at birth.
	if now-st.LastMitosisTime > 300 {
		t.Fatalf("divide cooldown not enforced at birth: now-LMT=%v > 300", now-st.LastMitosisTime)
	}
}
