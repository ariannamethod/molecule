package main

import (
	"database/sql"
	"testing"
	"time"
)

// Phase-0 cascade-governor regression tests (deep-core audit design items 1/6/8 + C2/C3).

// (a, audit C3) AcquireMitosisSlot atomically admits a divide only below the cap
// AND only one divider at a time, across the shared mesh.db — TOCTOU-safe, not the
// old check-then-act that let concurrent dividers overshoot.
func TestMitosisSlotAtomicCap(t *testing.T) {
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Skipf("sqlite unavailable: %v", err)
	}
	defer db.Close()
	if _, err := db.Exec(`CREATE TABLE organisms(id TEXT PRIMARY KEY, status TEXT, last_heartbeat REAL)`); err != nil {
		t.Skipf("sqlite exec: %v", err)
	}
	db.Exec(`CREATE TABLE mitosis_lock(organism_id TEXT PRIMARY KEY, acquired_at REAL)`)
	now := float64(time.Now().UnixMilli()) / 1000.0
	for _, id := range []string{"a", "b", "c"} { // 3 live organisms
		db.Exec(`INSERT INTO organisms VALUES(?,?,?)`, id, "alive", now)
	}
	srA := &SwarmRegistry{OrganismID: "a", MeshDB: db}
	srB := &SwarmRegistry{OrganismID: "b", MeshDB: db}

	if !srA.AcquireMitosisSlot(4) {
		t.Fatal("3 live < cap 4 must admit")
	}
	if srB.AcquireMitosisSlot(4) {
		t.Fatal("sibling must be refused while another holds the mitosis lock (no concurrent overshoot)")
	}
	srA.ReleaseMitosisLock()
	if srA.AcquireMitosisSlot(3) {
		t.Fatal("3 live at cap 3 must refuse (hard ceiling)")
	}
	if !srA.AcquireMitosisSlot(0) {
		t.Fatal("max<=0 must be uncapped")
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

// (c, audit C2) divide relieves the ENTROPY path too — §9 Air divided on entropy
// (e=true l=false); relieving only loss left it re-firing. After relieveOverload,
// entropyOverload() must read false.
func TestDivideRelievesParentEntropy(t *testing.T) {
	st := NewSyntropyTracker()
	high := CFG.EntropyHigh + 1.0
	for i := 0; i < CFG.SyntropyWindow; i++ {
		st.EntropyHistory = append(st.EntropyHistory, high)
	}
	st.SyntropyTrend = -0.1 // dissolving field → entropyOverload condition met
	if !st.entropyOverload() {
		t.Fatal("setup: expected entropyOverload before relieve")
	}
	st.relieveOverload()
	if st.entropyOverload() {
		t.Fatal("after relieve: entropyOverload must be false (high-entropy samples dropped)")
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

