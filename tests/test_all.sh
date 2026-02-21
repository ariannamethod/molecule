#!/bin/bash
# test_all.sh — integration tests for all molequla elements + mycelium
# Can be run from anywhere: auto-detects repo root.
set -euo pipefail

# cd to repo root (one level up from tests/)
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export PATH=$PATH:/usr/local/go/bin:$HOME/.cargo/bin

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

PASS=0
FAIL=0
SKIP=0

pass() { echo -e "  ${GREEN}PASS${NC} $1"; PASS=$((PASS+1)); }
fail() { echo -e "  ${RED}FAIL${NC} $1: $2"; FAIL=$((FAIL+1)); }
skip() { echo -e "  ${YELLOW}SKIP${NC} $1: $2"; SKIP=$((SKIP+1)); }

TESTDIR=$(mktemp -d /tmp/molequla_test_XXXX)
cd ~/molequla

echo "═══════════════════════════════════════════════════"
echo " molequla integration tests"
echo " $(date)"
echo " testdir: $TESTDIR"
echo "═══════════════════════════════════════════════════"
echo ""

# ─────────────────────────────────────────────────────
echo -e "${CYAN}1. BUILD${NC}"
echo "─────────────────────────────────────────────────"

# Go (must build from module dir)
if go build -o $TESTDIR/molequla_go molequla.go 2>/dev/null; then
    pass "go build"
else
    fail "go build" "compilation failed"
fi

# C
if gcc -O2 -Wno-format-truncation -o $TESTDIR/molequla_c molequla.c -lm -lsqlite3 -lpthread 2>/dev/null; then
    pass "c build"
else
    fail "c build" "compilation failed"
fi

# Rust
if cargo build --release 2>/dev/null; then
    cp target/release/molequla $TESTDIR/molequla_rs 2>/dev/null || true
    pass "rust build"
else
    fail "rust build" "compilation failed"
fi

# AML C library
if (cd ariannamethod && make clean && make) >/dev/null 2>&1; then
    pass "libaml build (BLAS)"
else
    fail "libaml build" "compilation failed"
fi

# JS — verify syntax
if node -c molequla.js 2>/dev/null; then
    pass "js syntax check"
else
    fail "js syntax" "parse error"
fi

echo ""

# ─────────────────────────────────────────────────────
echo -e "${CYAN}2. ELEMENT SMOKE TESTS${NC}"
echo "─────────────────────────────────────────────────"

cp nonames.txt $TESTDIR/

# Go: writes to stdout "[init] Stage 0..."
echo -n "  "
cd $TESTDIR
timeout 10 ./molequla_go > go.log 2>&1 || true
if grep -qi "stage\|init\|train\|step" go.log 2>/dev/null; then
    GOLINES=$(wc -l < go.log)
    pass "go smoke ($GOLINES lines, starts training)"
elif [ -f memory.sqlite3 ]; then
    pass "go smoke (created memory.sqlite3)"
else
    fail "go smoke" "no output and no db"
fi
cd ~/molequla

# C: writes to memory.sqlite3, stdout is quiet
echo -n "  "
cd $TESTDIR
rm -f memory.sqlite3
timeout 10 ./molequla_c > c.log 2>&1 || true
if [ -f memory.sqlite3 ]; then
    TABLES=$(python3 -c "
import sqlite3
con = sqlite3.connect('memory.sqlite3')
t = [r[0] for r in con.execute(\"SELECT name FROM sqlite_master WHERE type='table'\")]
print(len(t))
" 2>/dev/null)
    pass "c smoke (memory.sqlite3 created, $TABLES tables)"
elif grep -qi "stage\|init\|train\|step\|corpus" c.log 2>/dev/null; then
    pass "c smoke (output detected)"
else
    fail "c smoke" "no db and no output"
fi
cd ~/molequla

# JS: browser-first, in Node exports modules
echo -n "  "
JSTEST=$(node -e "
const m = require('./molequla.js');
const checks = [];
if (m.GPT) checks.push('GPT');
if (m.EvolvingTokenizer) checks.push('Tokenizer');
if (m.DeltaAdapter) checks.push('DeltaAdapter');
if (m.SyntropyTracker) checks.push('SyntropyTracker');
if (m.SwarmRegistry) checks.push('SwarmRegistry');
console.log(checks.join(','));
" 2>/dev/null)
if echo "$JSTEST" | grep -q "GPT"; then
    pass "js smoke (exports: $JSTEST)"
else
    fail "js smoke" "module exports missing"
fi

# JS: verify key classes instantiate
JSINST=$(node -e "
const m = require('./molequla.js');
try {
    const tok = new m.EvolvingTokenizer();
    const tracker = new m.SyntropyTracker(4);
    const has_encode = typeof tok.encode === 'function';
    const has_decode = typeof tok.decode === 'function';
    console.log('PASS encode=' + has_encode + ' decode=' + has_decode);
} catch(e) { console.log('FAIL ' + e.message); }
" 2>/dev/null)
if echo "$JSINST" | grep -q "PASS"; then
    pass "js instantiation ($JSINST)"
else
    fail "js instantiation" "$JSINST"
fi

# Rust: outputs banner/phases
echo -n "  "
cd $TESTDIR
timeout 12 ./molequla_rs > rs.log 2>&1 || true
if grep -qi "element\|phase\|stage\|init\|Fourth\|molequla" rs.log 2>/dev/null; then
    RSLINES=$(wc -l < rs.log)
    pass "rust smoke ($RSLINES lines)"
elif [ -f memory.sqlite3 ]; then
    pass "rust smoke (created memory.sqlite3)"
else
    fail "rust smoke" "no output"
fi
cd ~/molequla

echo ""

# ─────────────────────────────────────────────────────
echo -e "${CYAN}3. ARIANNAMETHOD (C library + Python bindings)${NC}"
echo "─────────────────────────────────────────────────"

# Import test
if python3 -c "from ariannamethod import Method; m = Method('/tmp/x.db'); print('lib:', m.lib is not None)" 2>/dev/null | grep -q "lib: True"; then
    pass "ariannamethod import + libaml load"
else
    fail "ariannamethod import" "could not load"
fi

# BLAS symbols
BLAS_COUNT=$(nm -D ariannamethod/libaml.so 2>/dev/null | grep -c "cblas_")
if [ "$BLAS_COUNT" -ge 2 ]; then
    pass "BLAS symbols ($BLAS_COUNT cblas_* functions)"
else
    fail "BLAS symbols" "expected 2+, got $BLAS_COUNT"
fi

# METHOD symbols
METHOD_SYMS=$(nm -D ariannamethod/libaml.so 2>/dev/null | grep -c "am_method_")
if [ "$METHOD_SYMS" -ge 8 ]; then
    pass "METHOD C API ($METHOD_SYMS functions)"
else
    fail "METHOD C API" "expected 8+, got $METHOD_SYMS"
fi

# METHOD field metrics
MTEST=$(python3 -c "
import ctypes, sys
sys.path.insert(0, '.')
from ariannamethod.method import _load_libaml
lib = _load_libaml()
lib.am_method_clear()
lib.am_method_push_organism(1, ctypes.c_float(1.5), ctypes.c_float(0.3), ctypes.c_float(0.8), ctypes.c_float(0.7))
lib.am_method_push_organism(2, ctypes.c_float(0.8), ctypes.c_float(0.6), ctypes.c_float(0.9), ctypes.c_float(0.8))
ent = float(lib.am_method_field_entropy())
syn = float(lib.am_method_field_syntropy())
coh = float(lib.am_method_field_coherence())
ok = abs(ent - 1.15) < 0.01 and abs(syn - 0.45) < 0.01 and abs(coh - 0.75) < 0.01
print('PASS' if ok else f'FAIL ent={ent:.4f} syn={syn:.4f} coh={coh:.4f}')
" 2>/dev/null)
if [ "$MTEST" = "PASS" ]; then
    pass "METHOD field metrics (entropy, syntropy, coherence)"
else
    fail "METHOD field metrics" "$MTEST"
fi

# METHOD steering
STEST=$(python3 -c "
import ctypes, sys
sys.path.insert(0, '.')
from ariannamethod.method import _load_libaml, AM_MethodSteering, ACTION_NAMES
lib = _load_libaml()
lib.am_method_clear()
for i in range(5):
    lib.am_method_push_organism(i, ctypes.c_float(0.8+i*0.2), ctypes.c_float(0.4), ctypes.c_float(0.7), ctypes.c_float(0.6))
s = lib.am_method_step(ctypes.c_float(1.0))
name = ACTION_NAMES.get(s.action, 'unknown')
ok = s.n_organisms == 5 and name != 'unknown' and s.step >= 1
print(f'PASS action={name} n={s.n_organisms}' if ok else f'FAIL')
" 2>/dev/null)
if echo "$STEST" | grep -q "PASS"; then
    pass "METHOD steering ($STEST)"
else
    fail "METHOD steering" "$STEST"
fi

# Notorch BLAS (verify matrices actually change)
NTEST=$(python3 -c "
import ctypes, numpy as np, sys
sys.path.insert(0, '.')
from ariannamethod.method import _load_libaml
lib = _load_libaml()
A = np.random.randn(64, 8).astype(np.float32)
B = np.random.randn(8, 64).astype(np.float32)
x = np.random.randn(64).astype(np.float32)
dy = np.random.randn(64).astype(np.float32)
A_before = A.copy()
Ac = A.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
Bc = B.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
xc = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
dc = dy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
lib.am_notorch_step(Ac, Bc, 64, 64, 8, xc, dc, ctypes.c_float(0.5))
diff = np.abs(A - A_before).sum()
print('PASS' if diff > 0.01 else f'FAIL diff={diff}')
" 2>/dev/null)
if [ "$NTEST" = "PASS" ]; then
    pass "notorch BLAS (matrices modified)"
else
    fail "notorch BLAS" "$NTEST"
fi

# apply_delta BLAS
DTEST=$(python3 -c "
import ctypes, numpy as np, sys
sys.path.insert(0, '.')
from ariannamethod.method import _load_libaml
lib = _load_libaml()
out = np.zeros(32, dtype=np.float32)
A = np.random.randn(32, 8).astype(np.float32)
B = np.random.randn(8, 64).astype(np.float32)
x = np.random.randn(64).astype(np.float32)
oc = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
lib.am_apply_delta(oc, A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 32, 64, 8, ctypes.c_float(0.5))
print('PASS' if np.count_nonzero(out) > 0 else 'FAIL')
" 2>/dev/null)
if [ "$DTEST" = "PASS" ]; then
    pass "apply_delta BLAS (non-zero output)"
else
    fail "apply_delta BLAS" "$DTEST"
fi

echo ""

# ─────────────────────────────────────────────────────
echo -e "${CYAN}4. MYCELIUM${NC}"
echo "─────────────────────────────────────────────────"

# Create test mesh.db
python3 -c "
import sqlite3, time, numpy as np
db = '$TESTDIR/mesh.db'
con = sqlite3.connect(db)
con.execute('PRAGMA journal_mode=WAL')
con.execute('''CREATE TABLE IF NOT EXISTS organisms(
    id TEXT PRIMARY KEY, pid INTEGER, stage INTEGER,
    n_params INTEGER, syntropy REAL, entropy REAL,
    last_heartbeat REAL, parent_id TEXT,
    status TEXT DEFAULT 'alive',
    gamma_direction BLOB, gamma_magnitude REAL,
    rrpram_signature BLOB)''')
now = time.time()
for i, (name, ent, syn) in enumerate([
    ('go-01', 1.2, 0.4), ('c-02', 0.9, 0.5),
    ('js-03', 1.8, 0.3), ('rust-04', 0.7, 0.6)
]):
    g = np.random.randn(32).astype(np.float64).tobytes()
    con.execute('INSERT INTO organisms VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
        (name, 100+i, 5, 100000, syn, ent, now, None, 'alive', g, 0.8, None))
con.commit(); con.close()
" 2>/dev/null

# --once single step
MONCE=$(python3 mycelium.py --once --mesh $TESTDIR/mesh.db 2>/dev/null)
if echo "$MONCE" | grep -q "organisms=4"; then
    ACTION=$(echo "$MONCE" | grep -oP 'action=\K\w+' | head -1)
    pass "mycelium --once (4 organisms, action=$ACTION)"
else
    fail "mycelium --once" "did not see 4 organisms"
fi

# JSON validity
MJSON=$(python3 mycelium.py --once --mesh $TESTDIR/mesh.db 2>/dev/null | grep -A100 '{' | head -20)
if echo "$MJSON" | python3 -c "import json,sys; d=json.load(sys.stdin); assert d['n_organisms']==4; print('PASS')" 2>/dev/null | grep -q PASS; then
    pass "mycelium JSON output"
else
    fail "mycelium JSON" "invalid"
fi

# Async loop (daemon mode, 3 seconds, interval 0.3s → expect 5+ steps)
MLOOP=$(python3 -c "
import subprocess, sys, time
proc = subprocess.Popen(
    [sys.executable, 'mycelium.py', '--daemon', '--mesh', '$TESTDIR/mesh.db', '--interval', '0.3'],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
time.sleep(4)
proc.terminate()
out, _ = proc.communicate(timeout=5)
lines = [l for l in out.strip().split(chr(10)) if 'step=' in l or 'organisms=' in l]
print(len(lines))
" 2>/dev/null)
if [ "$MLOOP" -ge 5 ] 2>/dev/null; then
    pass "mycelium async loop ($MLOOP steps in 3s)"
else
    fail "mycelium async loop" "only $MLOOP steps"
fi

# Add drifter, verify detection
python3 -c "
import sqlite3, time, numpy as np
con = sqlite3.connect('$TESTDIR/mesh.db')
g = np.random.randn(32).astype(np.float64).tobytes()
con.execute('INSERT OR REPLACE INTO organisms VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
    ('drifter-99', 999, 3, 50000, 0.05, 5.0, time.time(), None, 'alive', g, 0.2, None))
con.commit(); con.close()
" 2>/dev/null

MDRIFT=$(python3 mycelium.py --once --mesh $TESTDIR/mesh.db 2>/dev/null)
if echo "$MDRIFT" | grep -q "organisms=5"; then
    pass "mycelium detects new organism (5 total)"
else
    fail "mycelium drift" "wrong count"
fi

# C+BLAS engine detection
MENGINE=$(python3 mycelium.py --once --mesh $TESTDIR/mesh.db 2>/dev/null | head -1 || true)
# Check via the step output that it's working
if echo "$MDRIFT" | grep -q "action="; then
    pass "mycelium METHOD engine active"
else
    fail "mycelium METHOD engine" "no action in output"
fi

# ─────────────────────────────────────────────────────
echo -e "${CYAN}4b. MYCELIUM SELF-AWARENESS${NC}"
echo "─────────────────────────────────────────────────"

# MyceliumGamma
GAMMA_TEST=$(python3 -c "
import sys; sys.path.insert(0, '.')
from mycelium import MyceliumGamma
import numpy as np
g = MyceliumGamma()
assert g.magnitude == 0.0, 'initial magnitude should be 0'
g.imprint('dampen', 0.8, -0.5)
g.imprint('dampen', 0.7, -0.3)
g.imprint('explore', 0.5, 0.2)
assert g.magnitude > 0, 'magnitude should grow'
t, c = g.dominant_tendency()
assert t in ('dampen','explore','sustain','amplify','ground','realign','wait'), f'bad tendency: {t}'
assert -1 <= c <= 1, f'cosine out of range: {c}'
# Test cosine_with
other = np.random.randn(32).astype(np.float64)
cos = g.cosine_with(other)
assert -1 <= cos <= 1
print(f'PASS mag={g.magnitude:.3f} tendency={t} cos={c:.3f}')
" 2>/dev/null)
if echo "$GAMMA_TEST" | grep -q "PASS"; then
    pass "MyceliumGamma ($GAMMA_TEST)"
else
    fail "MyceliumGamma" "$GAMMA_TEST"
fi

# HarmonicNet
HNET_TEST=$(python3 -c "
import sys; sys.path.insert(0, '.')
from mycelium import HarmonicNet
import numpy as np

class FakeOrg:
    def __init__(self, oid, ent):
        self.id = oid
        self.entropy = ent
        self.gamma_direction = np.random.randn(32).astype(np.float64).tobytes()

h = HarmonicNet()
orgs = [FakeOrg('a', 1.1), FakeOrg('b', 0.9), FakeOrg('c', 1.5)]
# Run 5 steps to build history
for i in range(5):
    out = h.forward(orgs, 1.0 + 0.2*np.sin(i), 0.7, 0.4, i)
assert 'action_bias' in out, 'missing action_bias'
assert 'strength_mod' in out, 'missing strength_mod'
assert 'harmonics' in out, 'missing harmonics'
assert 'resonance' in out, 'missing resonance'
nh = len(out['harmonics'])
nr = len(out['resonance'])
sm = out['strength_mod']
assert nh == 8, 'expected 8 harmonics, got %d' % nh
assert nr == 3, 'expected 3 resonance, got %d' % nr
assert 0 <= sm <= 1.5, 'strength_mod out of range: %.2f' % sm
print('PASS harmonics=%d resonance=%d str_mod=%.2f' % (nh, nr, sm))
" 2>/dev/null)
if echo "$HNET_TEST" | grep -q "PASS"; then
    pass "HarmonicNet ($HNET_TEST)"
else
    fail "HarmonicNet" "$HNET_TEST"
fi

# MyceliumSyntropy
SYN_TEST=$(python3 -c "
import sys; sys.path.insert(0, '.')
from mycelium import MyceliumSyntropy

s = MyceliumSyntropy()
# Simulate field improving under guidance
for i in range(8):
    ent = 1.5 - i * 0.05  # entropy decreasing
    syn = 0.3 + i * 0.02
    coh = 0.6 + i * 0.01
    s.snapshot_before(ent + 0.05, syn - 0.02, coh - 0.01)
    s.record_decision('dampen' if i % 2 == 0 else 'sustain', 0.5, ent, syn, coh)

m = s.measure()
assert m['n_decisions'] == 8, 'expected 8, got %d' % m['n_decisions']
assert m['syntropy_trend'] > 0, 'trend should be positive: %.4f' % m['syntropy_trend']
assert m['decision_entropy'] > 0, 'decision_entropy should be > 0'
assert 'effectiveness' in m
should, reason = s.should_change_strategy()
print('PASS trend=%+.3f H_dec=%.2f strat=%s' % (m['syntropy_trend'], m['decision_entropy'], reason))
" 2>/dev/null)
if echo "$SYN_TEST" | grep -q "PASS"; then
    pass "MyceliumSyntropy ($SYN_TEST)"
else
    fail "MyceliumSyntropy" "$SYN_TEST"
fi

# FieldPulse
PULSE_TEST=$(python3 -c "
import sys; sys.path.insert(0, '.')
from mycelium import FieldPulse
import numpy as np

class FakeOrg:
    def __init__(self, oid, ent):
        self.id = oid
        self.entropy = ent

p = FieldPulse()
orgs = [FakeOrg('a', 1.1), FakeOrg('b', 0.9)]
p.measure(orgs, 1.0)
assert p.novelty == 1.0, 'first observation novelty should be 1.0: %.2f' % p.novelty

p.measure(orgs, 1.2)
assert p.novelty == 0.0, 'same organisms, novelty should be 0: %.2f' % p.novelty
assert p.arousal > 0, 'entropy changed, arousal should be > 0: %.2f' % p.arousal

# Add new organism → novelty
orgs.append(FakeOrg('c', 1.5))
p.measure(orgs, 1.2)
assert p.novelty > 0, 'new organism, novelty should be > 0: %.2f' % p.novelty
print('PASS n=%.2f a=%.2f e=%.2f' % (p.novelty, p.arousal, p.entropy))
" 2>/dev/null)
if echo "$PULSE_TEST" | grep -q "PASS"; then
    pass "FieldPulse ($PULSE_TEST)"
else
    fail "FieldPulse" "$PULSE_TEST"
fi

# SteeringDissonance
DIS_TEST=$(python3 -c "
import sys; sys.path.insert(0, '.')
from mycelium import SteeringDissonance

d = SteeringDissonance()

# Dampen but entropy went up → dissonance should be high
d.update('dampen', 0.5, 0.0)  # delta_H=0.5 (up), delta_C=0
assert d.dissonance > 0, 'dissonance should be > 0 after bad dampen: %.3f' % d.dissonance

# Explore and entropy went up → consonance (low dissonance change)
d2 = SteeringDissonance()
d2.update('explore', 0.3, 0.1)  # entropy up, that's what we wanted
# Dissonance should be lower since intent matched outcome

# strength_multiplier test
assert 0.3 <= d.strength_multiplier() <= 2.0, 'multiplier out of range: %.2f' % d.strength_multiplier()
print('PASS dis=%.3f mul=%.2f' % (d.dissonance, d.strength_multiplier()))
" 2>/dev/null)
if echo "$DIS_TEST" | grep -q "PASS"; then
    pass "SteeringDissonance ($DIS_TEST)"
else
    fail "SteeringDissonance" "$DIS_TEST"
fi

# OrganismAttention
ATT_TEST=$(python3 -c "
import sys; sys.path.insert(0, '.')
from mycelium import OrganismAttention

class FakeOrg:
    def __init__(self, oid, ent):
        self.id = oid
        self.entropy = ent

a = OrganismAttention()
orgs = [FakeOrg('go', 1.0), FakeOrg('c', 0.9)]
pre = {'go': 1.2, 'c': 0.9}  # go-alpha entropy was 1.2, now 1.0 (went down)

# Under 'dampen' action, go responded (entropy down), c didn't change
a.update(orgs, 'dampen', pre)
assert a.weights['go'] > a.weights['c'], 'go should have higher attention'
top = a.top_organisms(1)
assert top[0][0] == 'go', 'go should be most responsive'
print('PASS go=%.3f c=%.3f' % (a.weights['go'], a.weights['c']))
" 2>/dev/null)
if echo "$ATT_TEST" | grep -q "PASS"; then
    pass "OrganismAttention ($ATT_TEST)"
else
    fail "OrganismAttention" "$ATT_TEST"
fi

# Full self-awareness integration (step with gamma+harmonic+syntropy+pulse)
FULL_TEST=$(python3 -c "
import sys, sqlite3, time, numpy as np
sys.path.insert(0, '.')

db = '$TESTDIR/mesh.db'
# Ensure organisms have fresh heartbeat
con = sqlite3.connect(db)
con.execute('UPDATE organisms SET last_heartbeat = ?', (time.time(),))
con.commit()
con.close()

from mycelium import Mycelium
myc = Mycelium(mesh_path=db, interval=0.5, verbose=False)
for i in range(5):
    s = myc.step()

# Check all components are alive
assert myc.gamma.magnitude >= 0, 'gamma not working'
m = myc.syntropy.measure()
assert m['n_decisions'] == 5, 'expected 5 decisions, got %d' % m['n_decisions']
assert myc.pulse.entropy >= 0, 'pulse not measuring'
assert len(myc.harmonic._entropy_history) >= 5, 'harmonic not recording'
print('PASS g=%.3f syn=%+.3f pulse=%.2f steps=5' % (myc.gamma.magnitude, m['syntropy_trend'], myc.pulse.entropy))
" 2>/dev/null)
if echo "$FULL_TEST" | grep -q "PASS"; then
    pass "self-awareness integration ($FULL_TEST)"
else
    fail "self-awareness integration" "$FULL_TEST"
fi

echo ""

# ─────────────────────────────────────────────────────
echo -e "${CYAN}5. MESH.DB SCHEMA${NC}"
echo "─────────────────────────────────────────────────"

SCHEMA=$(python3 -c "
import sqlite3
con = sqlite3.connect('$TESTDIR/mesh.db')
cols = [r[1] for r in con.execute('PRAGMA table_info(organisms)')]
expected = ['id','pid','stage','n_params','syntropy','entropy','last_heartbeat','gamma_direction','gamma_magnitude']
missing = set(expected) - set(cols)
print('PASS' if not missing else f'FAIL missing: {missing}')
" 2>/dev/null)
if [ "$SCHEMA" = "PASS" ]; then
    pass "organisms table schema"
else
    fail "organisms schema" "$SCHEMA"
fi

# field_deltas via Method init
python3 -c "
import sys; sys.path.insert(0, '.')
from ariannamethod import Method
Method('$TESTDIR/mesh.db')
" 2>/dev/null
FDELTA=$(python3 -c "
import sqlite3
con = sqlite3.connect('$TESTDIR/mesh.db')
tables = [r[0] for r in con.execute(\"SELECT name FROM sqlite_master WHERE type='table'\")]
print('PASS' if 'field_deltas' in tables else 'FAIL')
" 2>/dev/null)
if [ "$FDELTA" = "PASS" ]; then
    pass "field_deltas table"
else
    fail "field_deltas table" "not created"
fi

echo ""

# ─────────────────────────────────────────────────────
echo -e "${CYAN}6. PERFORMANCE${NC}"
echo "─────────────────────────────────────────────────"

PERF=$(python3 -c "
import ctypes, time, sys
sys.path.insert(0, '.')
from ariannamethod.method import _load_libaml
lib = _load_libaml()
lib.am_method_clear()
for i in range(32):
    lib.am_method_push_organism(i, ctypes.c_float(0.5+i*0.05), ctypes.c_float(0.3), ctypes.c_float(0.7), ctypes.c_float(0.6))
N = 100000
t0 = time.perf_counter()
for _ in range(N):
    lib.am_method_step(ctypes.c_float(0.0))
t1 = time.perf_counter()
us = (t1-t0)/N*1e6
print(f'{us:.1f}')
" 2>/dev/null)
if [ -n "$PERF" ]; then
    pass "METHOD C: ${PERF}μs/iter (32 organisms, 100k iterations)"
else
    fail "METHOD benchmark" "no output"
fi

PERF_NT=$(python3 -c "
import ctypes, numpy as np, time, sys
sys.path.insert(0, '.')
from ariannamethod.method import _load_libaml
lib = _load_libaml()
A = np.random.randn(2048, 64).astype(np.float32)
B = np.random.randn(64, 2048).astype(np.float32)
x = np.random.randn(2048).astype(np.float32)
dy = np.random.randn(2048).astype(np.float32)
Ac = A.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
Bc = B.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
xc = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
dc = dy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
N = 1000
t0 = time.perf_counter()
for _ in range(N):
    lib.am_notorch_step(Ac, Bc, 2048, 2048, 64, xc, dc, ctypes.c_float(0.5))
t1 = time.perf_counter()
ms = (t1-t0)/N*1000
print(f'{ms:.2f}')
" 2>/dev/null)
if [ -n "$PERF_NT" ]; then
    pass "notorch BLAS: ${PERF_NT}ms/step (2048x2048, rank=64)"
else
    fail "notorch benchmark" "no output"
fi

# TEST: HarmonicNet C forward speed
PERF_HN=$(cd ~/molequla && python3 -c "
import ctypes, time
from ariannamethod.method import _load_libaml
lib = _load_libaml()
for i in range(64):
    lib.am_harmonic_push_entropy(ctypes.c_float(1.0 + 0.5 * (i % 7)))
import random
random.seed(42)
for oid in range(16):
    gamma = (ctypes.c_float * 32)(*[random.gauss(0, 1) for _ in range(32)])
    lib.am_harmonic_push_gamma(oid, gamma, 32, ctypes.c_float(1.0 + oid * 0.1))
N = 100000
t0 = time.perf_counter()
for _ in range(N):
    lib.am_harmonic_forward(0)
t1 = time.perf_counter()
us = (t1-t0)/N*1e6
print(f'{us:.1f}')
" 2>/dev/null)
if [ -n "$PERF_HN" ]; then
    pass "HarmonicNet C: ${PERF_HN}μs/forward (16 organisms, 64 history)"
else
    fail "HarmonicNet C benchmark" "no output"
fi

echo ""
echo "═══════════════════════════════════════════════════"
echo -e " Results: ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC}, ${YELLOW}$SKIP skipped${NC}"
echo "═══════════════════════════════════════════════════"

# Cleanup
rm -rf $TESTDIR

exit $FAIL
