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

// CPU thread-cap (GPU multi-process stall fix): the 4-element colony shares the
// cores without oversubscription. 9-core cgroup → 2/org (the RunPod case, verified
// util 99% vs the 96-thread/proc stall).
func TestColonyThreadsFor(t *testing.T) {
	if g := colonyThreadsFor(9); g != 2 {
		t.Fatalf("9 cores → want 2 threads/org, got %d", g)
	}
	if g := colonyThreadsFor(96); g != 24 {
		t.Fatalf("96 cores → 24, got %d", g)
	}
	if g := colonyThreadsFor(2); g != 1 {
		t.Fatalf("2 cores → floor 1, got %d", g)
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

// (M-GOV-001) A reserved child slot counts toward the population cap immediately,
// so a sibling cannot overshoot the cap in the window before the child boots and
// self-registers. Before the fix the cap only saw self-registered organisms, so a
// spawned-but-not-yet-registered child let the next AcquireMitosisSlot admit an
// over-cap divide.
func TestReservedChildCountsTowardCap(t *testing.T) {
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Skipf("sqlite unavailable: %v", err)
	}
	defer db.Close()
	if _, err := db.Exec(`CREATE TABLE organisms(id TEXT PRIMARY KEY, pid INTEGER, stage INTEGER, n_params INTEGER, syntropy REAL, entropy REAL, last_heartbeat REAL, parent_id TEXT, status TEXT, element TEXT)`); err != nil {
		t.Skipf("sqlite exec: %v", err)
	}
	db.Exec(`CREATE TABLE mitosis_lock(organism_id TEXT PRIMARY KEY, acquired_at REAL)`)
	now := float64(time.Now().UnixMilli()) / 1000.0
	for _, id := range []string{"a", "b", "c"} { // 3 live
		db.Exec(`INSERT INTO organisms(id,status,last_heartbeat) VALUES(?,?,?)`, id, "alive", now)
	}
	parent := &SwarmRegistry{OrganismID: "a", MeshDB: db}

	if !parent.AcquireMitosisSlot(4) {
		t.Fatal("3 live < cap 4 must admit the divide")
	}
	// The parent reserves the child's slot immediately on a successful spawn.
	parent.ReserveChildSlot("a_child", 12345, "earth")
	parent.ReleaseMitosisLock()

	// Now 4 live (3 originals + the reserved child) at cap 4: the next divide MUST
	// be refused even though the child has not yet self-registered on its own.
	if parent.AcquireMitosisSlot(4) {
		t.Fatal("reserved child must count toward the cap: 4 live at cap 4 must refuse")
	}
}

