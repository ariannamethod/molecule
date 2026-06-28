package main

import "testing"

// Phase-0 cascade-governor regression tests (deep-core audit design items 1/6/8).

// (a) population cap — capAllowsDivide gates colony growth before divide.
func TestPopulationCap(t *testing.T) {
	if capAllowsDivide(16, 16) {
		t.Fatal("at cap (16/16) must refuse divide")
	}
	if capAllowsDivide(17, 16) {
		t.Fatal("over cap (17/16) must refuse divide")
	}
	if !capAllowsDivide(15, 16) {
		t.Fatal("below cap (15/16) must allow divide")
	}
	if !capAllowsDivide(9999, 0) {
		t.Fatal("max=0 must be uncapped (always allow)")
	}
}

// (c) divide relieves the parent — after relieveOverload the triggering high-loss
// bursts are gone, so lossOverload() (and thus the divide gate) reads false until
// NEW overload accumulates.
func TestDivideRelievesParent(t *testing.T) {
	st := NewSyntropyTracker()
	high := CFG.OverloadLossHigh + 2.0 // above threshold, delta 0 (not improving) -> overload
	for i := 0; i < CFG.OverloadLossWindow+1; i++ {
		st.RecordBurst("boost", high, high)
	}
	if !st.lossOverload() {
		t.Fatal("setup: expected lossOverload before relieve")
	}
	st.relieveOverload()
	if st.lossOverload() {
		t.Fatal("after relieve: lossOverload must be false (triggering bursts dropped)")
	}
}

// (#2) write-storm throttle — debouncer coalesces rapid periodic checkpoints.
func TestCheckpointDebounce(t *testing.T) {
	var d debouncer
	if !d.allow(100, 30) {
		t.Fatal("first call must allow")
	}
	if d.allow(120, 30) {
		t.Fatal("20s < 30s window must be blocked")
	}
	if !d.allow(131, 30) {
		t.Fatal("31s >= 30s window must allow")
	}
	if !d.allow(131.0001, 0) {
		t.Fatal("minInterval<=0 must always allow (throttle disabled)")
	}
}

