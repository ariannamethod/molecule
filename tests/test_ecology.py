"""Tests for Swarm Ecology — mitosis, hibernation, mesh registry."""

import sys
import os
import unittest
import tempfile
import shutil
import time
import json
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from molequla import (
    GPT, EvolvingTokenizer, CooccurField, SyntropyTracker, SwarmRegistry,
    train_steps, init_db, CFG, save_checkpoint, perform_hibernation,
    load_corpus_lines, SWARM_DIR, no_grad
)
import numpy as np


def _make_model_and_field():
    docs = [
        "H: Who are you?",
        "A: I am molequla.",
        "H: What is syntropy?",
        "A: The opposite of decay.",
        "H: How do you learn?",
        "A: Through dialogue and training.",
        "H: What is resonance?",
        "A: Amplification through alignment.",
    ]
    tok = EvolvingTokenizer(docs)
    model = GPT(tok)
    field = CooccurField()
    field.build_from_corpus(tok, docs)
    return model, tok, docs, field


# ============================================================
# SwarmRegistry tests
# ============================================================

class TestSwarmRegistry(unittest.TestCase):

    def setUp(self):
        """Use a temp dir for swarm to avoid polluting ~/.molequla."""
        self._orig_swarm_dir = SWARM_DIR
        self.tmpdir = tempfile.mkdtemp()
        # Monkey-patch SWARM_DIR for isolation
        import molequla
        molequla.SWARM_DIR = self.tmpdir

    def tearDown(self):
        import molequla
        molequla.SWARM_DIR = self._orig_swarm_dir
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_register_creates_pid_file(self):
        """register() should create a PID file."""
        swarm = SwarmRegistry("test_org_1")
        swarm.register()
        pid_file = os.path.join(self.tmpdir, "test_org_1.pid")
        self.assertTrue(os.path.exists(pid_file))
        with open(pid_file) as f:
            data = json.load(f)
        self.assertEqual(data["pid"], os.getpid())
        swarm.unregister()

    def test_register_creates_mesh_db(self):
        """register() should create mesh.db with organisms table."""
        swarm = SwarmRegistry("test_org_1")
        swarm.register()
        db_path = os.path.join(self.tmpdir, "mesh.db")
        self.assertTrue(os.path.exists(db_path))
        # Check organisms table has our entry
        con = sqlite3.connect(db_path)
        cur = con.execute("SELECT id, status FROM organisms WHERE id='test_org_1'")
        row = cur.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[1], "alive")
        con.close()
        swarm.unregister()

    def test_heartbeat_updates_state(self):
        """heartbeat() should update stage, params, syntropy, entropy."""
        swarm = SwarmRegistry("test_org_1")
        swarm.register()
        swarm.heartbeat(stage=3, n_params=500000, syntropy=0.05, entropy=1.2)

        cur = swarm.mesh_db.execute(
            "SELECT stage, n_params, syntropy, entropy FROM organisms WHERE id='test_org_1'")
        row = cur.fetchone()
        self.assertEqual(row[0], 3)
        self.assertEqual(row[1], 500000)
        self.assertAlmostEqual(row[2], 0.05, places=2)
        self.assertAlmostEqual(row[3], 1.2, places=1)
        swarm.unregister()

    def test_discover_peers_finds_others(self):
        """discover_peers() should find other alive organisms."""
        swarm1 = SwarmRegistry("org_parent")
        swarm1.register()
        swarm1.heartbeat(stage=4, n_params=10000000, syntropy=0.01, entropy=1.5)

        swarm2 = SwarmRegistry("org_child")
        swarm2.register()
        swarm2.heartbeat(stage=1, n_params=100000, syntropy=0.1, entropy=2.0)

        peers = swarm1.discover_peers()
        self.assertEqual(len(peers), 1)
        self.assertEqual(peers[0]["id"], "org_child")
        self.assertEqual(peers[0]["stage"], 1)

        peers_from_child = swarm2.discover_peers()
        self.assertEqual(len(peers_from_child), 1)
        self.assertEqual(peers_from_child[0]["id"], "org_parent")

        swarm1.unregister()
        swarm2.unregister()

    def test_discover_peers_ignores_dead(self):
        """discover_peers() should not find dead organisms."""
        swarm1 = SwarmRegistry("org_alive")
        swarm1.register()
        swarm1.heartbeat(stage=2, n_params=500000, syntropy=0.02, entropy=1.3)

        swarm2 = SwarmRegistry("org_dead")
        swarm2.register()
        swarm2.unregister()  # marks as dead

        peers = swarm1.discover_peers()
        self.assertEqual(len(peers), 0)
        swarm1.unregister()

    def test_discover_peers_ignores_sleeping(self):
        """discover_peers() should not find hibernating organisms."""
        swarm1 = SwarmRegistry("org_active")
        swarm1.register()
        swarm1.heartbeat(stage=2, n_params=500000, syntropy=0.02, entropy=1.3)

        swarm2 = SwarmRegistry("org_sleepy")
        swarm2.register()
        swarm2.mark_hibernating()

        peers = swarm1.discover_peers()
        self.assertEqual(len(peers), 0)
        swarm1.unregister()
        swarm2.unregister()

    def test_log_message(self):
        """log_message() should write to messages table."""
        swarm = SwarmRegistry("org_sender")
        swarm.register()
        swarm.log_message("org_receiver", "mitosis:spawn", {"parent_stage": 4})

        cur = swarm.mesh_db.execute("SELECT from_id, to_id, type, payload FROM messages")
        row = cur.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "org_sender")
        self.assertEqual(row[1], "org_receiver")
        self.assertEqual(row[2], "mitosis:spawn")
        payload = json.loads(row[3])
        self.assertEqual(payload["parent_stage"], 4)
        swarm.unregister()

    def test_unregister_cleans_up(self):
        """unregister() should mark dead and remove PID file."""
        swarm = SwarmRegistry("org_cleanup")
        swarm.register()
        pid_file = swarm.pid_file
        db_path = os.path.join(self.tmpdir, "mesh.db")
        self.assertTrue(os.path.exists(pid_file))

        swarm.unregister()
        self.assertFalse(os.path.exists(pid_file))
        # Check status in db
        con = sqlite3.connect(db_path)
        cur = con.execute("SELECT status FROM organisms WHERE id='org_cleanup'")
        row = cur.fetchone()
        self.assertEqual(row[0], "dead")
        con.close()

    def test_mark_hibernating(self):
        """mark_hibernating() should set status to sleeping."""
        swarm = SwarmRegistry("org_sleeper")
        swarm.register()
        swarm.mark_hibernating()

        cur = swarm.mesh_db.execute("SELECT status FROM organisms WHERE id='org_sleeper'")
        row = cur.fetchone()
        self.assertEqual(row[0], "sleeping")
        swarm.unregister()


# ============================================================
# SyntropyTracker ecology actions
# ============================================================

class TestSyntropyDivide(unittest.TestCase):

    def test_divide_when_adult_overloaded(self):
        """Adult + sustained overload + cooldown expired → divide."""
        st = SyntropyTracker()
        st.model_stage = len(CFG.growth_stages) - 1  # adult
        st._last_mitosis_time = 0.0  # no cooldown
        # Fill entropy window with high values
        st.entropy_history = [CFG.entropy_high + 0.5] * CFG.syntropy_window
        st.syntropy_trend = -0.05  # falling (overloaded)
        st.field_deviation = 2.0
        st.purpose_alignment = 0.0

        decision = st.decide_action()
        self.assertEqual(decision["action"], "divide")

    def test_no_divide_when_not_adult(self):
        """Non-adult stage should never divide even if overloaded."""
        st = SyntropyTracker()
        st.model_stage = 2  # child, not adult
        st._last_mitosis_time = 0.0
        st.entropy_history = [CFG.entropy_high + 0.5] * CFG.syntropy_window
        st.syntropy_trend = -0.05
        st.field_deviation = 2.0

        decision = st.decide_action()
        self.assertNotEqual(decision["action"], "divide")

    def test_no_divide_during_cooldown(self):
        """Should not divide within 300s of last mitosis."""
        st = SyntropyTracker()
        st.model_stage = len(CFG.growth_stages) - 1
        st._last_mitosis_time = time.time()  # just divided
        st.entropy_history = [CFG.entropy_high + 0.5] * CFG.syntropy_window
        st.syntropy_trend = -0.05
        st.field_deviation = 2.0

        decision = st.decide_action()
        self.assertNotEqual(decision["action"], "divide")

    def test_no_divide_with_short_history(self):
        """Should not divide with insufficient entropy history."""
        st = SyntropyTracker()
        st.model_stage = len(CFG.growth_stages) - 1
        st._last_mitosis_time = 0.0
        st.entropy_history = [CFG.entropy_high + 0.5] * 3  # too short
        st.syntropy_trend = -0.05
        st.field_deviation = 2.0

        decision = st.decide_action()
        self.assertNotEqual(decision["action"], "divide")


class TestSyntropyHibernate(unittest.TestCase):

    def test_hibernate_when_peer_thriving_and_stale(self):
        """Plateau + thriving peer → hibernate."""
        st = SyntropyTracker()
        st.model_stage = 3
        st.syntropy_trend = 0.0  # not rising or falling
        st.field_deviation = 2.0  # sweet spot (won't trigger ground/explore)
        st.purpose_alignment = 0.0  # won't trigger realign
        # We're stale: 8 bursts with tiny loss changes
        st.burst_history = [
            {"action": "steady", "loss_before": 1.5, "loss_after": 1.5}
            for _ in range(8)
        ]
        # A peer is thriving
        st._swarm_info = {
            "peers": [{"id": "org_child", "syntropy": 0.1, "stage": 1}]
        }

        decision = st.decide_action()
        self.assertEqual(decision["action"], "hibernate")

    def test_no_hibernate_without_peers(self):
        """Without peers, should never hibernate."""
        st = SyntropyTracker()
        st.syntropy_trend = 0.0
        st.field_deviation = 2.0
        st.purpose_alignment = 0.0
        st.burst_history = [
            {"action": "steady", "loss_before": 1.5, "loss_after": 1.5}
            for _ in range(8)
        ]
        st._swarm_info = None

        decision = st.decide_action()
        self.assertNotEqual(decision["action"], "hibernate")

    def test_no_hibernate_when_actively_improving(self):
        """Should not hibernate when loss is still decreasing."""
        st = SyntropyTracker()
        st.syntropy_trend = 0.0
        st.field_deviation = 2.0
        st.purpose_alignment = 0.0
        st.burst_history = [
            {"action": "steady", "loss_before": 2.0 - i*0.1, "loss_after": 2.0 - i*0.1 - 0.05}
            for i in range(8)
        ]
        st._swarm_info = {
            "peers": [{"id": "org_child", "syntropy": 0.1, "stage": 1}]
        }

        decision = st.decide_action()
        self.assertNotEqual(decision["action"], "hibernate")

    def test_no_hibernate_when_peer_not_thriving(self):
        """Should not hibernate when peer has low syntropy."""
        st = SyntropyTracker()
        st.syntropy_trend = 0.0
        st.field_deviation = 2.0
        st.purpose_alignment = 0.0
        st.burst_history = [
            {"action": "steady", "loss_before": 1.5, "loss_after": 1.5}
            for _ in range(8)
        ]
        st._swarm_info = {
            "peers": [{"id": "org_child", "syntropy": 0.01, "stage": 1}]
        }

        decision = st.decide_action()
        self.assertNotEqual(decision["action"], "hibernate")


# ============================================================
# Sustained overload helper
# ============================================================

class TestSustainedOverload(unittest.TestCase):

    def test_sustained_overload_true(self):
        """_is_sustained_overload with full high-entropy window + falling syntropy."""
        st = SyntropyTracker()
        st.entropy_history = [CFG.entropy_high + 0.5] * CFG.syntropy_window
        st.syntropy_trend = -0.05
        self.assertTrue(st._is_sustained_overload())

    def test_sustained_overload_false_short_history(self):
        """Not overloaded with insufficient history."""
        st = SyntropyTracker()
        st.entropy_history = [CFG.entropy_high + 0.5] * 3
        st.syntropy_trend = -0.05
        self.assertFalse(st._is_sustained_overload())

    def test_sustained_overload_false_low_entropy(self):
        """Not overloaded when entropy is low."""
        st = SyntropyTracker()
        st.entropy_history = [CFG.entropy_high - 0.5] * CFG.syntropy_window
        st.syntropy_trend = -0.05
        self.assertFalse(st._is_sustained_overload())

    def test_sustained_overload_false_rising_syntropy(self):
        """Not overloaded when syntropy is rising (managing the load)."""
        st = SyntropyTracker()
        st.entropy_history = [CFG.entropy_high + 0.5] * CFG.syntropy_window
        st.syntropy_trend = 0.05  # rising, not falling
        self.assertFalse(st._is_sustained_overload())


# ============================================================
# Should hibernate helper
# ============================================================

class TestShouldHibernate(unittest.TestCase):

    def test_should_hibernate_with_thriving_peer_and_plateau(self):
        """Hibernation recommended when a peer thrives and we're stale."""
        st = SyntropyTracker()
        st.burst_history = [
            {"action": "steady", "loss_before": 1.5, "loss_after": 1.5}
            for _ in range(8)
        ]
        st._swarm_info = {
            "peers": [{"id": "peer1", "syntropy": 0.1}]
        }
        self.assertTrue(st._should_hibernate())

    def test_should_not_hibernate_no_peers(self):
        """No hibernation without swarm info."""
        st = SyntropyTracker()
        st.burst_history = [
            {"action": "steady", "loss_before": 1.5, "loss_after": 1.5}
            for _ in range(8)
        ]
        st._swarm_info = None
        self.assertFalse(st._should_hibernate())

    def test_should_not_hibernate_short_burst_history(self):
        """No hibernation with < 8 bursts (can't judge plateau yet)."""
        st = SyntropyTracker()
        st.burst_history = [
            {"action": "steady", "loss_before": 1.5, "loss_after": 1.5}
            for _ in range(3)
        ]
        st._swarm_info = {
            "peers": [{"id": "peer1", "syntropy": 0.1}]
        }
        self.assertFalse(st._should_hibernate())


# ============================================================
# Burst history inheritance
# ============================================================

class TestBurstHistoryInheritance(unittest.TestCase):

    def test_syntracker_burst_history_injectable(self):
        """SyntropyTracker should accept injected burst_history."""
        st = SyntropyTracker()
        inherited = [
            {"action": "boost", "loss_before": 2.0, "loss_after": 1.8},
            {"action": "amplify", "loss_before": 1.8, "loss_after": 1.5},
        ]
        st.burst_history = list(inherited)
        self.assertEqual(len(st.burst_history), 2)
        eff, count = st.action_effectiveness("boost")
        self.assertAlmostEqual(eff, -0.2, places=1)
        self.assertEqual(count, 1)

    def test_model_inherited_attribute(self):
        """Model can carry _inherited_burst_history for background_trainer."""
        model, tok, docs, _ = _make_model_and_field()
        seed = [{"action": "steady", "loss_before": 1.0, "loss_after": 0.9}]
        model._inherited_burst_history = seed
        inherited = getattr(model, '_inherited_burst_history', None)
        self.assertIsNotNone(inherited)
        self.assertEqual(len(inherited), 1)
        del model._inherited_burst_history
        self.assertIsNone(getattr(model, '_inherited_burst_history', None))


# ============================================================
# Perform hibernation (sync, no subprocess)
# ============================================================

class TestPerformHibernation(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        import molequla
        self._orig_swarm_dir = molequla.SWARM_DIR
        molequla.SWARM_DIR = self.tmpdir

    def tearDown(self):
        import molequla
        molequla.SWARM_DIR = self._orig_swarm_dir
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_hibernation_saves_checkpoint_and_marks_sleeping(self):
        """perform_hibernation should save checkpoint and mark sleeping."""
        model, tok, docs, _ = _make_model_and_field()
        # Init db
        db_path = os.path.join(self.tmpdir, "test_mem.db")
        con = init_db(db_path)
        # Write a corpus file for load_corpus_lines
        corpus_path = os.path.join(self.tmpdir, "corpus.txt")
        with open(corpus_path, "w") as f:
            for d in docs:
                f.write(d + "\n")
        orig_corpus = CFG.corpus_path
        CFG.corpus_path = corpus_path
        ckpt_path = os.path.join(self.tmpdir, "ckpt.json")
        orig_ckpt = CFG.ckpt_path
        CFG.ckpt_path = ckpt_path

        swarm = SwarmRegistry("org_hibernate_test")
        swarm.register()

        perform_hibernation(model, tok, con, swarm)

        # Check checkpoint saved
        self.assertTrue(os.path.exists(ckpt_path))
        # Check status is sleeping
        cur = swarm.mesh_db.execute("SELECT status FROM organisms WHERE id='org_hibernate_test'")
        row = cur.fetchone()
        self.assertEqual(row[0], "sleeping")

        con.close()
        CFG.corpus_path = orig_corpus
        CFG.ckpt_path = orig_ckpt


# ============================================================
# Perform mitosis (test birth.json creation, not subprocess)
# ============================================================

class TestPerformMitosis(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        import molequla
        self._orig_swarm_dir = molequla.SWARM_DIR
        molequla.SWARM_DIR = self.tmpdir

    def tearDown(self):
        import molequla
        molequla.SWARM_DIR = self._orig_swarm_dir
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        # Clean up any child dirs
        mol_dir = os.path.expanduser("~/.molequla")
        if os.path.isdir(mol_dir):
            for d in os.listdir(mol_dir):
                if d.startswith("org_") and os.path.isdir(os.path.join(mol_dir, d)):
                    child_dir = os.path.join(mol_dir, d)
                    birth_path = os.path.join(child_dir, "birth.json")
                    if os.path.exists(birth_path):
                        try:
                            with open(birth_path) as f:
                                data = json.load(f)
                            if data.get("_test_marker") == "ecology_test":
                                shutil.rmtree(child_dir, ignore_errors=True)
                        except Exception:
                            pass

    def test_mitosis_creates_birth_json(self):
        """perform_mitosis should create child dir with birth.json."""
        import asyncio
        from unittest.mock import AsyncMock, patch

        model, tok, docs, _ = _make_model_and_field()
        db_path = os.path.join(self.tmpdir, "test_mem.db")
        con = init_db(db_path)
        corpus_path = os.path.join(self.tmpdir, "corpus.txt")
        with open(corpus_path, "w") as f:
            for d in docs:
                f.write(d + "\n")
        orig_corpus = CFG.corpus_path
        CFG.corpus_path = corpus_path
        ckpt_path = os.path.join(self.tmpdir, "ckpt.json")
        orig_ckpt = CFG.ckpt_path
        CFG.ckpt_path = ckpt_path

        swarm = SwarmRegistry("org_parent_test")
        swarm.register()

        syntracker = SyntropyTracker()
        syntracker.burst_history = [
            {"action": "boost", "loss_before": 2.0, "loss_after": 1.8}
        ]

        # Mock subprocess to avoid actually spawning
        mock_proc = AsyncMock()
        mock_proc.pid = 99999

        async def run_mitosis():
            from molequla import perform_mitosis
            with patch('asyncio.create_subprocess_exec', return_value=mock_proc):
                child_id = await perform_mitosis(model, tok, con, swarm, syntracker)
                return child_id

        child_id = asyncio.get_event_loop().run_until_complete(run_mitosis())

        # Verify child_id returned
        self.assertIsNotNone(child_id)
        self.assertTrue(child_id.startswith("org_"))

        # Verify child dir + birth.json
        child_dir = os.path.expanduser(f"~/.molequla/{child_id}")
        self.assertTrue(os.path.isdir(child_dir))
        birth_path = os.path.join(child_dir, "birth.json")
        self.assertTrue(os.path.exists(birth_path))

        with open(birth_path) as f:
            birth = json.load(f)
        self.assertEqual(birth["organism_id"], child_id)
        self.assertEqual(birth["parent_id"], "org_parent_test")
        self.assertIn("burst_history", birth)
        self.assertEqual(len(birth["burst_history"]), 1)

        # Verify parent checkpoint saved
        parent_ckpt = os.path.join(child_dir, "parent_ckpt.json")
        self.assertTrue(os.path.exists(parent_ckpt))

        # Verify mesh.db message logged
        cur = swarm.mesh_db.execute(
            "SELECT type, from_id, to_id FROM messages WHERE type='mitosis:spawn'")
        row = cur.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[1], "org_parent_test")
        self.assertEqual(row[2], child_id)

        # Verify cooldown set
        self.assertGreater(syntracker._last_mitosis_time, 0)

        # Clean up child dir
        shutil.rmtree(child_dir, ignore_errors=True)

        con.close()
        CFG.corpus_path = orig_corpus
        CFG.ckpt_path = orig_ckpt
        swarm.unregister()


# ============================================================
# Model stage tracking in SyntropyTracker.measure()
# ============================================================

class TestModelStageTracking(unittest.TestCase):

    def test_measure_sets_model_stage(self):
        """measure() should set model_stage from model.current_growth_stage()."""
        model, tok, docs, field = _make_model_and_field()
        st = SyntropyTracker()
        st.measure(model, tok, field, docs)
        # Embryo model with default n_embd=16 should be stage 0
        self.assertEqual(st.model_stage, 0)


# ============================================================
# CLI args parsing
# ============================================================

class TestCLIArgsParsing(unittest.TestCase):

    def test_parse_cli_args_empty(self):
        """With no args, returns None values."""
        from molequla import _parse_cli_args
        import sys
        old_argv = sys.argv
        sys.argv = ["molequla.py"]
        args = _parse_cli_args()
        self.assertIsNone(args["organism_id"])
        self.assertIsNone(args["config"])
        sys.argv = old_argv

    def test_parse_cli_args_with_values(self):
        """With --organism-id and --config, returns correct values."""
        from molequla import _parse_cli_args
        import sys
        old_argv = sys.argv
        sys.argv = ["molequla.py", "--organism-id", "org_123", "--config", "/tmp/birth.json"]
        args = _parse_cli_args()
        self.assertEqual(args["organism_id"], "org_123")
        self.assertEqual(args["config"], "/tmp/birth.json")
        sys.argv = old_argv


if __name__ == "__main__":
    unittest.main()
