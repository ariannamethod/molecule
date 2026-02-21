"""
method.py — the METHOD operator for distributed cognition orchestration.

all AML operators (DESTINY, PAIN, GAMMA, TUNNEL...) work on a single organism.
METHOD works on the field — it reads collective metrics from all organisms
and computes steering deltas for the mouth (Rust).

usage:
    from ariannamethod import Method

    m = Method("mesh.db")
    m.step()              # read field, compute, write deltas
    m.field_entropy()     # system-level entropy
    m.field_coherence()   # pairwise gamma cosine across organisms
    m.field_syntropy()    # is the system organizing or dissolving?
"""

import ctypes
import ctypes.util
import os
import struct
import sqlite3
import time
import math
import numpy as np
from pathlib import Path


def _find_libaml():
    """find libaml.so/dylib next to this file."""
    here = Path(__file__).parent
    for name in ("libaml.dylib", "libaml.so"):
        p = here / name
        if p.exists():
            return str(p)
    return None


def _load_libaml():
    """load AML C library and bind functions."""
    path = _find_libaml()
    if path is None:
        return None

    lib = ctypes.CDLL(path)

    # void am_init(void)
    lib.am_init.restype = None
    lib.am_init.argtypes = []

    # void am_step(float dt)
    lib.am_step.restype = None
    lib.am_step.argtypes = [ctypes.c_float]

    # void am_apply_field_to_logits(float* logits, int n)
    lib.am_apply_field_to_logits.restype = None
    lib.am_apply_field_to_logits.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]

    # void am_apply_delta(float* out, const float* A, const float* B,
    #                     const float* x, int out_dim, int in_dim, int rank, float alpha)
    lib.am_apply_delta.restype = None
    lib.am_apply_delta.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # out
        ctypes.POINTER(ctypes.c_float),  # A
        ctypes.POINTER(ctypes.c_float),  # B
        ctypes.POINTER(ctypes.c_float),  # x
        ctypes.c_int, ctypes.c_int, ctypes.c_int,  # out_dim, in_dim, rank
        ctypes.c_float,  # alpha
    ]

    # void am_notorch_step(float* A, float* B, int out_dim, int in_dim, int rank,
    #                      const float* x, const float* dy, float signal)
    lib.am_notorch_step.restype = None
    lib.am_notorch_step.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # A
        ctypes.POINTER(ctypes.c_float),  # B
        ctypes.c_int, ctypes.c_int, ctypes.c_int,  # out_dim, in_dim, rank
        ctypes.POINTER(ctypes.c_float),  # x
        ctypes.POINTER(ctypes.c_float),  # dy
        ctypes.c_float,  # signal
    ]

    # int am_exec(const char* script)
    lib.am_exec.restype = ctypes.c_int
    lib.am_exec.argtypes = [ctypes.c_char_p]

    # AM_State* am_get_state(void)
    lib.am_get_state.restype = ctypes.c_void_p
    lib.am_get_state.argtypes = []

    lib.am_init()
    return lib


class Organism:
    """a single organism's snapshot from mesh.db."""
    __slots__ = ("id", "pid", "stage", "n_params", "syntropy", "entropy",
                 "gamma_direction", "gamma_magnitude", "last_seen")

    def __init__(self, row):
        self.id = row[0]
        self.pid = row[1]
        self.stage = row[2]
        self.n_params = row[3]
        self.syntropy = row[4]
        self.entropy = row[5]
        self.gamma_direction = row[6]  # BLOB or None
        self.gamma_magnitude = row[7] if len(row) > 7 else 0.0
        self.last_seen = row[8] if len(row) > 8 else 0.0


class Method:
    """
    METHOD — the distributed cognition operator.

    reads all organisms from mesh.db.
    computes system-level awareness (entropy, coherence, syntropy).
    writes steering deltas for the mouth (Rust).
    """

    def __init__(self, mesh_path="mesh.db", rank=8):
        self.mesh_path = mesh_path
        self.rank = rank
        self.organisms = []
        self.lib = _load_libaml()

        # system-level tracking
        self._entropy_history = []
        self._coherence_history = []
        self._step_count = 0

        # steering deltas (computed by METHOD, consumed by Rust)
        self.deltas = {}  # layer_name -> (A, B, alpha)

        # ensure field_deltas table exists
        self._init_db()

    def _init_db(self):
        """create field_deltas table if not exists."""
        try:
            con = sqlite3.connect(self.mesh_path)
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("""
                CREATE TABLE IF NOT EXISTS field_deltas (
                    layer TEXT PRIMARY KEY,
                    A BLOB,
                    B BLOB,
                    out_dim INTEGER,
                    in_dim INTEGER,
                    rank INTEGER,
                    alpha REAL,
                    updated_at REAL
                )
            """)
            con.commit()
            con.close()
        except Exception:
            pass  # mesh.db might not exist yet

    def read_field(self):
        """read all organisms from mesh.db."""
        self.organisms = []
        try:
            con = sqlite3.connect(self.mesh_path)
            con.row_factory = sqlite3.Row
            cur = con.execute("""
                SELECT id, pid, stage, n_params, syntropy, entropy,
                       gamma_direction, gamma_magnitude, last_heartbeat
                FROM organisms
                WHERE status = 'alive'
                  AND last_heartbeat > ?
            """, (time.time() - 120,))
            for row in cur:
                self.organisms.append(Organism(tuple(row)))
            con.close()
        except Exception:
            pass
        return self.organisms

    def field_entropy(self):
        """system-level entropy: mean of all organisms' entropy."""
        if not self.organisms:
            return 0.0
        return sum(o.entropy for o in self.organisms) / len(self.organisms)

    def field_syntropy(self):
        """system-level syntropy: mean of all organisms' syntropy."""
        if not self.organisms:
            return 0.0
        return sum(o.syntropy for o in self.organisms) / len(self.organisms)

    def field_coherence(self):
        """pairwise gamma cosine similarity across all organisms."""
        gammas = []
        for o in self.organisms:
            if o.gamma_direction and len(o.gamma_direction) > 0:
                arr = np.frombuffer(o.gamma_direction, dtype=np.float64)
                if len(arr) > 0 and np.linalg.norm(arr) > 1e-12:
                    gammas.append(arr / np.linalg.norm(arr))

        if len(gammas) < 2:
            return 1.0  # single organism = perfectly coherent with itself

        # mean pairwise cosine
        total = 0.0
        count = 0
        for i in range(len(gammas)):
            for j in range(i + 1, len(gammas)):
                # pad to same length
                a, b = gammas[i], gammas[j]
                min_len = min(len(a), len(b))
                cos = np.dot(a[:min_len], b[:min_len])
                total += cos
                count += 1

        return total / count if count > 0 else 1.0

    def field_drift(self):
        """detect which organisms are drifting from the field mean."""
        if len(self.organisms) < 2:
            return {}

        mean_entropy = self.field_entropy()
        drifters = {}
        for o in self.organisms:
            deviation = abs(o.entropy - mean_entropy)
            if deviation > 0.5:
                drifters[o.id] = deviation
        return drifters

    def compute_steering(self):
        """
        METHOD operator: compute system-level steering signal.

        returns a dict of signals that can drive delta injection:
        - action: "amplify" | "dampen" | "ground" | "explore" | "realign"
        - strength: 0..1
        - target_organism: which organism to amplify (lowest entropy)
        """
        if not self.organisms:
            return {"action": "wait", "strength": 0.0}

        entropy = self.field_entropy()
        syntropy = self.field_syntropy()
        coherence = self.field_coherence()

        self._entropy_history.append(entropy)
        self._coherence_history.append(coherence)

        # keep window of 16
        if len(self._entropy_history) > 16:
            self._entropy_history = self._entropy_history[-16:]
        if len(self._coherence_history) > 16:
            self._coherence_history = self._coherence_history[-16:]

        # entropy trend (syntropy of the field)
        trend = 0.0
        if len(self._entropy_history) >= 4:
            recent = self._entropy_history[-4:]
            earlier = self._entropy_history[-8:-4] if len(self._entropy_history) >= 8 else self._entropy_history[:4]
            trend = sum(earlier) / len(earlier) - sum(recent) / len(recent)

        # find the most confident organism (lowest entropy)
        best = min(self.organisms, key=lambda o: o.entropy)

        # decide action
        if coherence < 0.3:
            # organisms diverging — realign
            action = "realign"
            strength = 1.0 - coherence
        elif trend > 0.05:
            # entropy falling = system organizing = amplify
            action = "amplify"
            strength = min(1.0, trend * 5)
        elif trend < -0.05:
            # entropy rising = system dissolving = dampen
            action = "dampen"
            strength = min(1.0, abs(trend) * 5)
        elif entropy > 2.0:
            # high system entropy = ground to best organism
            action = "ground"
            strength = min(1.0, (entropy - 1.5) * 0.5)
        elif entropy < 0.5:
            # low system entropy = explore
            action = "explore"
            strength = min(1.0, (1.0 - entropy) * 0.5)
        else:
            # stable
            action = "sustain"
            strength = 0.1

        self._step_count += 1

        return {
            "action": action,
            "strength": strength,
            "target": best.id,
            "entropy": entropy,
            "syntropy": syntropy,
            "coherence": coherence,
            "trend": trend,
            "n_organisms": len(self.organisms),
            "step": self._step_count,
        }

    def write_deltas(self, deltas):
        """write steering deltas to mesh.db for Rust to consume."""
        try:
            con = sqlite3.connect(self.mesh_path)
            con.execute("PRAGMA journal_mode=WAL")
            now = time.time()
            for layer, (A, B, alpha) in deltas.items():
                A_blob = A.astype(np.float64).tobytes()
                B_blob = B.astype(np.float64).tobytes()
                con.execute("""
                    INSERT OR REPLACE INTO field_deltas
                    (layer, A, B, out_dim, in_dim, rank, alpha, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (layer, A_blob, B_blob,
                      A.shape[0], A.shape[1] if A.ndim > 1 else 1,
                      self.rank, alpha, now))
            con.commit()
            con.close()
        except Exception as e:
            print(f"[method] write_deltas error: {e}")

    def step(self, dt=1.0):
        """
        one tick of the METHOD operator.

        1. read field (all organisms from mesh.db)
        2. compute system awareness (entropy, coherence, syntropy)
        3. compute steering signal
        4. advance AML physics (if C library loaded)
        5. return steering decision
        """
        self.read_field()
        steering = self.compute_steering()

        # advance AML field physics
        if self.lib is not None:
            self.lib.am_step(ctypes.c_float(dt))

            # translate steering to AML state
            if steering["action"] == "dampen":
                self.lib.am_exec(b"PAIN 0.3")
                self.lib.am_exec(b"VELOCITY WALK")
            elif steering["action"] == "amplify":
                self.lib.am_exec(b"VELOCITY RUN")
                self.lib.am_exec(b"DESTINY 0.6")
            elif steering["action"] == "ground":
                self.lib.am_exec(b"ATTEND_FOCUS 0.9")
                self.lib.am_exec(b"VELOCITY NOMOVE")
            elif steering["action"] == "explore":
                self.lib.am_exec(b"TUNNEL_CHANCE 0.3")
                self.lib.am_exec(b"VELOCITY RUN")
            elif steering["action"] == "realign":
                self.lib.am_exec(b"PAIN 0.5")
                self.lib.am_exec(b"ATTEND_FOCUS 0.8")

        return steering

    def apply_to_logits(self, logits_np):
        """apply AML field to logits array (numpy float32)."""
        if self.lib is None:
            return logits_np
        n = len(logits_np)
        c_arr = (ctypes.c_float * n)(*logits_np)
        self.lib.am_apply_field_to_logits(c_arr, n)
        return np.array(c_arr[:], dtype=np.float32)

    def notorch_update(self, layer, A, B, x, dy, signal):
        """
        run one notorch plasticity step on a delta pair.
        A: (out_dim, rank), B: (rank, in_dim), x: (in_dim,), dy: (out_dim,)
        """
        if self.lib is None:
            return A, B

        out_dim, rank = A.shape
        _, in_dim = B.shape

        A_c = A.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        B_c = B.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        x_c = x.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        dy_c = dy.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        self.lib.am_notorch_step(A_c, B_c, out_dim, in_dim, rank, x_c, dy_c,
                                 ctypes.c_float(signal))

        return A, B  # modified in-place via ctypes


# convenience: from ariannamethod import method
method = Method
