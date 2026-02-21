"""
method.py — thin Python wrapper around the METHOD operator (C implementation).

METHOD is a field operator for distributed cognition orchestration.
All heavy computation is done in C (ariannamethod.c).
Python handles SQLite I/O and async orchestration.

usage:
    from ariannamethod import Method

    m = Method("mesh.db")
    m.step()              # read field, compute in C, write deltas
    m.field_entropy()     # system-level entropy (C)
    m.field_coherence()   # pairwise gamma cosine (C)
    m.field_syntropy()    # is the system organizing or dissolving? (C)
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


# C struct mirrors for METHOD
class AM_MethodSteering(ctypes.Structure):
    _fields_ = [
        ("action", ctypes.c_int),
        ("strength", ctypes.c_float),
        ("target_id", ctypes.c_int),
        ("entropy", ctypes.c_float),
        ("syntropy", ctypes.c_float),
        ("coherence", ctypes.c_float),
        ("trend", ctypes.c_float),
        ("n_organisms", ctypes.c_int),
        ("step", ctypes.c_int),
    ]


# Action constants (match C defines)
METHOD_WAIT = 0
METHOD_AMPLIFY = 1
METHOD_DAMPEN = 2
METHOD_GROUND = 3
METHOD_EXPLORE = 4
METHOD_REALIGN = 5
METHOD_SUSTAIN = 6

ACTION_NAMES = {
    METHOD_WAIT: "wait",
    METHOD_AMPLIFY: "amplify",
    METHOD_DAMPEN: "dampen",
    METHOD_GROUND: "ground",
    METHOD_EXPLORE: "explore",
    METHOD_REALIGN: "realign",
    METHOD_SUSTAIN: "sustain",
}


def _load_libaml():
    """load AML C library and bind functions."""
    path = _find_libaml()
    if path is None:
        return None

    lib = ctypes.CDLL(path)

    # === Core AML API ===
    lib.am_init.restype = None
    lib.am_init.argtypes = []

    lib.am_step.restype = None
    lib.am_step.argtypes = [ctypes.c_float]

    lib.am_apply_field_to_logits.restype = None
    lib.am_apply_field_to_logits.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]

    lib.am_apply_delta.restype = None
    lib.am_apply_delta.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # out
        ctypes.POINTER(ctypes.c_float),  # A
        ctypes.POINTER(ctypes.c_float),  # B
        ctypes.POINTER(ctypes.c_float),  # x
        ctypes.c_int, ctypes.c_int, ctypes.c_int,  # out_dim, in_dim, rank
        ctypes.c_float,  # alpha
    ]

    lib.am_notorch_step.restype = None
    lib.am_notorch_step.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # A
        ctypes.POINTER(ctypes.c_float),  # B
        ctypes.c_int, ctypes.c_int, ctypes.c_int,  # out_dim, in_dim, rank
        ctypes.POINTER(ctypes.c_float),  # x
        ctypes.POINTER(ctypes.c_float),  # dy
        ctypes.c_float,  # signal
    ]

    lib.am_exec.restype = ctypes.c_int
    lib.am_exec.argtypes = [ctypes.c_char_p]

    lib.am_get_state.restype = ctypes.c_void_p
    lib.am_get_state.argtypes = []

    # === METHOD API (new — C-native field operator) ===
    lib.am_method_init.restype = None
    lib.am_method_init.argtypes = []

    lib.am_method_clear.restype = None
    lib.am_method_clear.argtypes = []

    lib.am_method_push_organism.restype = None
    lib.am_method_push_organism.argtypes = [
        ctypes.c_int,    # id
        ctypes.c_float,  # entropy
        ctypes.c_float,  # syntropy
        ctypes.c_float,  # gamma_mag
        ctypes.c_float,  # gamma_cos
    ]

    lib.am_method_field_entropy.restype = ctypes.c_float
    lib.am_method_field_entropy.argtypes = []

    lib.am_method_field_syntropy.restype = ctypes.c_float
    lib.am_method_field_syntropy.argtypes = []

    lib.am_method_field_coherence.restype = ctypes.c_float
    lib.am_method_field_coherence.argtypes = []

    lib.am_method_step.restype = AM_MethodSteering
    lib.am_method_step.argtypes = [ctypes.c_float]

    lib.am_method_get_state.restype = ctypes.c_void_p
    lib.am_method_get_state.argtypes = []

    # Initialize both AML core and METHOD
    lib.am_init()
    lib.am_method_init()
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
    pushes their metrics into C METHOD operator.
    C computes awareness (entropy, coherence, syntropy) and steering.
    writes steering deltas for the mouth (Rust).
    """

    def __init__(self, mesh_path="mesh.db", rank=8):
        self.mesh_path = mesh_path
        self.rank = rank
        self.organisms = []
        self.lib = _load_libaml()

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
                CREATE TABLE IF NOT EXISTS field_steering (
                    id INTEGER PRIMARY KEY DEFAULT 1,
                    action TEXT,
                    strength REAL,
                    target_id TEXT,
                    entropy REAL,
                    syntropy REAL,
                    coherence REAL,
                    trend REAL,
                    n_organisms INTEGER,
                    updated_at REAL
                )
            """)
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
        """read all organisms from mesh.db and push into C METHOD."""
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

        # Push into C METHOD operator
        if self.lib is not None:
            # Pre-compute gamma vectors and mean for coherence
            gammas = {}
            for o in self.organisms:
                if o.gamma_direction and len(o.gamma_direction) > 0:
                    try:
                        arr = np.frombuffer(o.gamma_direction, dtype=np.float64)
                        norm = np.linalg.norm(arr)
                        if norm > 1e-12:
                            gammas[o.id] = arr / norm
                    except Exception:
                        pass

            # Compute mean gamma direction
            if len(gammas) >= 2:
                vecs = list(gammas.values())
                min_len = min(len(v) for v in vecs)
                mean_gamma = np.mean([v[:min_len] for v in vecs], axis=0)
                mean_norm = np.linalg.norm(mean_gamma)
                if mean_norm > 1e-12:
                    mean_gamma /= mean_norm
                else:
                    mean_gamma = None
            else:
                mean_gamma = None

            self.lib.am_method_clear()
            for o in self.organisms:
                gamma_mag = 0.0
                gamma_cos = 0.0
                if o.id in gammas:
                    g = gammas[o.id]
                    gamma_mag = float(np.linalg.norm(g))  # ~1.0 (normalized)
                    if mean_gamma is not None:
                        min_len = min(len(g), len(mean_gamma))
                        gamma_cos = float(np.dot(g[:min_len], mean_gamma[:min_len]))

                oid = hash(o.id) & 0x7FFFFFFF if isinstance(o.id, str) else int(o.id)
                self.lib.am_method_push_organism(
                    oid,
                    ctypes.c_float(o.entropy),
                    ctypes.c_float(o.syntropy),
                    ctypes.c_float(gamma_mag),
                    ctypes.c_float(gamma_cos),
                )

        return self.organisms

    def field_entropy(self):
        """system-level entropy (C implementation)."""
        if self.lib is not None:
            return float(self.lib.am_method_field_entropy())
        # fallback: Python
        if not self.organisms:
            return 0.0
        return sum(o.entropy for o in self.organisms) / len(self.organisms)

    def field_syntropy(self):
        """system-level syntropy (C implementation)."""
        if self.lib is not None:
            return float(self.lib.am_method_field_syntropy())
        if not self.organisms:
            return 0.0
        return sum(o.syntropy for o in self.organisms) / len(self.organisms)

    def field_coherence(self):
        """pairwise gamma cosine similarity (C implementation)."""
        if self.lib is not None:
            return float(self.lib.am_method_field_coherence())
        # fallback: Python (expensive)
        gammas = []
        for o in self.organisms:
            if o.gamma_direction and len(o.gamma_direction) > 0:
                arr = np.frombuffer(o.gamma_direction, dtype=np.float64)
                if len(arr) > 0 and np.linalg.norm(arr) > 1e-12:
                    gammas.append(arr / np.linalg.norm(arr))
        if len(gammas) < 2:
            return 1.0
        total = 0.0
        count = 0
        for i in range(len(gammas)):
            for j in range(i + 1, len(gammas)):
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
        METHOD step: compute system-level steering signal via C.
        Returns dict compatible with previous Python API.
        """
        if self.lib is not None:
            s = self.lib.am_method_step(ctypes.c_float(0.0))  # dt=0, we call am_step separately
            return {
                "action": ACTION_NAMES.get(s.action, "unknown"),
                "strength": s.strength,
                "target": s.target_id,
                "entropy": s.entropy,
                "syntropy": s.syntropy,
                "coherence": s.coherence,
                "trend": s.trend,
                "n_organisms": s.n_organisms,
                "step": s.step,
            }

        # Fallback: pure Python (same logic as before)
        if not self.organisms:
            return {"action": "wait", "strength": 0.0}
        entropy = self.field_entropy()
        syntropy = self.field_syntropy()
        coherence = self.field_coherence()
        return {
            "action": "sustain",
            "strength": 0.1,
            "entropy": entropy,
            "syntropy": syntropy,
            "coherence": coherence,
            "n_organisms": len(self.organisms),
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

    def write_steering(self, steering):
        """write steering decision to mesh.db for Rust to read."""
        try:
            con = sqlite3.connect(self.mesh_path)
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("""
                INSERT OR REPLACE INTO field_steering
                (id, action, strength, target_id, entropy, syntropy,
                 coherence, trend, n_organisms, updated_at)
                VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                steering.get("action", "wait"),
                steering.get("strength", 0.0),
                str(steering.get("target", "")),
                steering.get("entropy", 0.0),
                steering.get("syntropy", 0.0),
                steering.get("coherence", 0.0),
                steering.get("trend", 0.0),
                steering.get("n_organisms", 0),
                time.time(),
            ))
            con.commit()
            con.close()
        except Exception as e:
            print(f"[method] write_steering error: {e}")

    def step(self, dt=1.0):
        """
        one tick of the METHOD operator.

        1. read field (all organisms from mesh.db → push to C)
        2. C computes awareness + steering + advances physics
        3. return steering decision
        """
        self.read_field()

        if self.lib is not None:
            # Full C step: computes metrics, steering, advances am_step(dt)
            s = self.lib.am_method_step(ctypes.c_float(dt))
            return {
                "action": ACTION_NAMES.get(s.action, "unknown"),
                "strength": s.strength,
                "target": s.target_id,
                "entropy": s.entropy,
                "syntropy": s.syntropy,
                "coherence": s.coherence,
                "trend": s.trend,
                "n_organisms": s.n_organisms,
                "step": s.step,
            }

        return self.compute_steering()

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
        BLAS-accelerated in C when USE_BLAS is defined.
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
