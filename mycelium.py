#!/usr/bin/env python3
"""
mycelium.py — the orchestrator. the connective tissue between organisms.

molequla has four elements:
    molequla.go   — Go
    molequla.c    — C
    molequla.js   — JavaScript
    molequla.rs   — Rust (the mouth)

mycelium connects them. it reads the field (mesh.db), computes
system-level awareness via METHOD (C-native, BLAS-accelerated),
and writes steering deltas for the mouth to consume.

usage:
    python3 mycelium.py                    # interactive REPL (default)
    python3 mycelium.py --mesh ./mesh.db   # explicit path
    python3 mycelium.py --daemon           # background daemon, no REPL
    python3 mycelium.py --interval 2.0     # step every 2 seconds
    python3 mycelium.py --once             # single step, print, exit

part of molequla. the method that connects organisms.
"""

import argparse
import asyncio
import json
import os
import signal
import sqlite3
import sys
import threading
import time
from pathlib import Path

# Add parent dir so ariannamethod is importable
sys.path.insert(0, str(Path(__file__).parent))

from ariannamethod import Method


# ═══════════════════════════════════════════════════════════════════════════════
# field monitor — watches organism health
# ═══════════════════════════════════════════════════════════════════════════════

class FieldMonitor:
    """tracks field health over time, detects anomalies."""

    def __init__(self):
        self.history = []       # list of steering dicts
        self.max_history = 64
        self.alerts = []

    def record(self, steering):
        self.history.append(steering)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        self._check_alerts(steering)

    def _check_alerts(self, s):
        self.alerts.clear()

        if s.get("n_organisms", 0) == 0:
            self.alerts.append("no organisms alive")
            return

        if s.get("coherence", 1.0) < 0.3:
            self.alerts.append(f"coherence low: {s['coherence']:.3f}")

        if s.get("entropy", 0) > 2.5:
            self.alerts.append(f"entropy high: {s['entropy']:.3f}")

        action = s.get("action", "")
        if len(self.history) >= 8:
            recent_actions = [h.get("action") for h in self.history[-8:]]
            if all(a == "dampen" for a in recent_actions):
                self.alerts.append("stuck in dampen loop")
            if all(a == "realign" for a in recent_actions):
                self.alerts.append("persistent incoherence")

    def status_line(self, steering):
        n = steering.get("n_organisms", 0)
        action = steering.get("action", "?")
        strength = steering.get("strength", 0)
        ent = steering.get("entropy", 0)
        syn = steering.get("syntropy", 0)
        coh = steering.get("coherence", 0)
        step = steering.get("step", 0)

        line = (f"[mycelium] step={step} organisms={n} "
                f"action={action}({strength:.2f}) "
                f"H={ent:.3f} S={syn:.3f} C={coh:.3f}")

        if self.alerts:
            line += f"  !! {'; '.join(self.alerts)}"

        return line


# ═══════════════════════════════════════════════════════════════════════════════
# drift tracker — which organisms are diverging
# ═══════════════════════════════════════════════════════════════════════════════

class DriftTracker:
    """identifies organisms drifting from the field mean."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.drifters = {}

    def update(self, method):
        self.drifters = method.field_drift()
        return self.drifters

    def report(self):
        if not self.drifters:
            return None
        lines = [f"  organism {oid}: deviation={dev:.3f}"
                 for oid, dev in sorted(self.drifters.items(), key=lambda x: -x[1])]
        return "[mycelium] drifters detected:\n" + "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# voice — how mycelium speaks about the field
# ═══════════════════════════════════════════════════════════════════════════════

class MyceliumVoice:
    """translates field state into words. not generation — awareness."""

    ACTION_SPEECH = {
        "wait":     "the field is empty. i am listening to silence.",
        "sustain":  "the field breathes. all organisms in rhythm.",
        "amplify":  "entropy falling. the field is organizing. i amplify the signal.",
        "dampen":   "entropy rising. the field dissolves. i slow the pulse.",
        "ground":   "entropy too high. i ground the field to the strongest organism.",
        "explore":  "entropy too low. the field is rigid. i open tunnels.",
        "realign":  "organisms diverge. coherence breaking. i pull them together.",
    }

    def speak(self, steering, organisms, drifters=None):
        """compose a field report in mycelium's voice."""
        n = steering.get("n_organisms", 0)
        action = steering.get("action", "wait")
        ent = steering.get("entropy", 0)
        syn = steering.get("syntropy", 0)
        coh = steering.get("coherence", 0)
        strength = steering.get("strength", 0)
        trend = steering.get("trend", 0)

        lines = []

        # opening — what am i doing
        base = self.ACTION_SPEECH.get(action, f"action: {action}.")
        if strength > 0.7:
            base = base.upper()
        lines.append(base)

        # field state
        if n == 0:
            lines.append("no organisms. the mesh is dark.")
        elif n == 1:
            lines.append(f"one organism alone. entropy {ent:.2f}.")
        else:
            # name the organisms
            names = [o.id for o in organisms[:8]]
            lines.append(f"{n} organisms: {', '.join(str(x) for x in names)}.")

            # coherence reading
            if coh > 0.8:
                lines.append(f"coherence {coh:.2f} — they think as one.")
            elif coh > 0.5:
                lines.append(f"coherence {coh:.2f} — aligned but individual.")
            elif coh > 0.3:
                lines.append(f"coherence {coh:.2f} — drifting apart.")
            else:
                lines.append(f"coherence {coh:.2f} — fragmented. they don't see each other.")

            # entropy reading
            if ent < 0.3:
                lines.append(f"entropy {ent:.2f} — crystallized. too certain.")
            elif ent < 1.0:
                lines.append(f"entropy {ent:.2f} — focused. good.")
            elif ent < 2.0:
                lines.append(f"entropy {ent:.2f} — searching.")
            else:
                lines.append(f"entropy {ent:.2f} — chaotic. losing shape.")

            # syntropy
            if syn > 0.5:
                lines.append(f"syntropy {syn:.2f} — the field has purpose.")
            elif syn > 0.2:
                lines.append(f"syntropy {syn:.2f} — some direction, not yet clear.")
            else:
                lines.append(f"syntropy {syn:.2f} — no direction. wandering.")

            # trend
            if trend > 0.1:
                lines.append("trend: organizing. entropy falling.")
            elif trend < -0.1:
                lines.append("trend: dissolving. entropy rising.")

        # drifters
        if drifters:
            for oid, dev in sorted(drifters.items(), key=lambda x: -x[1])[:3]:
                lines.append(f"  drifter: {oid} (deviation {dev:.2f})")

        return "\n".join(lines)

    def greet(self, lib_loaded, mesh_path, n_organisms):
        """startup message."""
        engine = "C+BLAS" if lib_loaded else "Python"
        lines = [
            "mycelium awakens.",
            f"METHOD engine: {engine}.",
            f"mesh: {mesh_path}.",
        ]
        if n_organisms > 0:
            lines.append(f"i see {n_organisms} organisms.")
        else:
            lines.append("the field is empty. waiting.")
        lines.append("type /field, /who, /drift, /help — or just talk.\n")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# mycelium core
# ═══════════════════════════════════════════════════════════════════════════════

class Mycelium:
    """the orchestrator. connects organisms through METHOD."""

    def __init__(self, mesh_path="mesh.db", interval=1.0, verbose=True):
        self.method = Method(mesh_path)
        self.interval = interval
        self.verbose = verbose
        self.monitor = FieldMonitor()
        self.drift = DriftTracker()
        self.voice = MyceliumVoice()
        self.running = False
        self._step_count = 0
        self._last_steering = None

    def step(self):
        """one tick: read field -> METHOD -> monitor -> write steering -> report."""
        steering = self.method.step(dt=self.interval)
        self.method.write_steering(steering)
        self.monitor.record(steering)
        self._step_count += 1
        self._last_steering = steering

        # Check for drifters every 4 steps
        drift_report = None
        if self._step_count % 4 == 0:
            self.drift.update(self.method)
            drift_report = self.drift.report()

        if self.verbose:
            print(self.monitor.status_line(steering))
            if drift_report:
                print(drift_report)

        return steering

    def speak(self):
        """field report in mycelium's voice."""
        if self._last_steering is None:
            self.step()
        drifters = self.drift.drifters if self.drift.drifters else None
        return self.voice.speak(
            self._last_steering,
            self.method.organisms,
            drifters,
        )

    # ── REPL commands ──

    def cmd_field(self):
        """full field report."""
        self.step()
        self.drift.update(self.method)
        return self.speak()

    def cmd_who(self):
        """list all organisms."""
        self.method.read_field()
        if not self.method.organisms:
            return "no organisms alive."
        lines = []
        for o in self.method.organisms:
            age = time.time() - o.last_seen if o.last_seen else 0
            lines.append(
                f"  {o.id}: stage={o.stage} entropy={o.entropy:.2f} "
                f"syntropy={o.syntropy:.2f} params={o.n_params} "
                f"last_seen={age:.0f}s ago"
            )
        return f"{len(self.method.organisms)} organisms:\n" + "\n".join(lines)

    def cmd_drift(self):
        """show drifters."""
        self.method.read_field()
        self.drift.update(self.method)
        report = self.drift.report()
        return report if report else "no drifters. field is stable."

    def cmd_status(self):
        """one-line status."""
        self.step()
        return self.monitor.status_line(self._last_steering)

    def cmd_entropy(self):
        """field entropy details."""
        self.method.read_field()
        if not self.method.organisms:
            return "no organisms."
        ent = self.method.field_entropy()
        lines = [f"field entropy: {ent:.4f}"]
        for o in self.method.organisms:
            bar = "#" * int(o.entropy * 10)
            lines.append(f"  {o.id}: {o.entropy:.3f} {bar}")
        return "\n".join(lines)

    def cmd_coherence(self):
        """field coherence details."""
        self.method.read_field()
        coh = self.method.field_coherence()
        return f"field coherence: {coh:.4f}"

    def cmd_help(self):
        return (
            "mycelium commands:\n"
            "  /field     — full field report (voice)\n"
            "  /who       — list organisms\n"
            "  /drift     — show drifters\n"
            "  /status    — one-line status\n"
            "  /entropy   — entropy per organism\n"
            "  /coherence — pairwise gamma coherence\n"
            "  /step      — force one METHOD step\n"
            "  /json      — last steering as JSON\n"
            "  /quit      — exit\n"
            "\n"
            "or just type anything — mycelium will read the field and respond."
        )

    # ── daemon mode ──

    async def run_daemon(self):
        """background loop: step every interval, no REPL."""
        self.running = True
        lib_status = "C+BLAS" if self.method.lib else "Python fallback"
        print(f"[mycelium] daemon started. METHOD engine: {lib_status}")
        print(f"[mycelium] mesh: {self.method.mesh_path}")
        print(f"[mycelium] interval: {self.interval}s")
        print()

        while self.running:
            try:
                self.step()
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[mycelium] error: {e}")

            await asyncio.sleep(self.interval)

        print("\n[mycelium] stopped.")

    def stop(self):
        self.running = False

    # ── REPL ──

    def repl(self):
        """interactive REPL."""
        # Initial field read
        self.method.read_field()
        n = len(self.method.organisms)
        print(self.voice.greet(
            self.method.lib is not None,
            self.method.mesh_path,
            n,
        ))

        # Background stepper thread
        self.running = True
        def bg_step():
            while self.running:
                try:
                    self.method.read_field()
                    if self.method.organisms:
                        steering = self.method.step(dt=self.interval)
                        self.method.write_steering(steering)
                        self.monitor.record(steering)
                        self._last_steering = steering
                        self._step_count += 1
                except Exception:
                    pass
                time.sleep(self.interval)

        bg = threading.Thread(target=bg_step, daemon=True)
        bg.start()

        try:
            while True:
                try:
                    line = input("mycelium> ").strip()
                except EOFError:
                    break

                if not line:
                    continue

                if line in ("/quit", "/exit", "quit", "exit"):
                    break
                elif line == "/field":
                    print(self.cmd_field())
                elif line == "/who":
                    print(self.cmd_who())
                elif line == "/drift":
                    print(self.cmd_drift())
                elif line == "/status":
                    print(self.cmd_status())
                elif line == "/entropy":
                    print(self.cmd_entropy())
                elif line == "/coherence":
                    print(self.cmd_coherence())
                elif line == "/step":
                    s = self.step()
                    print(json.dumps(s, indent=2))
                elif line == "/json":
                    if self._last_steering:
                        print(json.dumps(self._last_steering, indent=2))
                    else:
                        print("no data yet. run /step first.")
                elif line == "/help":
                    print(self.cmd_help())
                else:
                    # Any other input: read field, speak
                    self.method.read_field()
                    self.drift.update(self.method)
                    if self._last_steering is None:
                        self.step()
                    print(self.speak())

                print()

        except KeyboardInterrupt:
            pass

        self.running = False
        print("\nmycelium sleeps.")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="mycelium — distributed cognition orchestrator")
    parser.add_argument("--mesh", default="mesh.db",
                        help="path to mesh.db (default: ./mesh.db)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="seconds between METHOD steps (default: 1.0)")
    parser.add_argument("--once", action="store_true",
                        help="single step, print, exit")
    parser.add_argument("--daemon", action="store_true",
                        help="background daemon (no REPL)")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress per-step output in daemon mode")
    args = parser.parse_args()

    myc = Mycelium(
        mesh_path=args.mesh,
        interval=args.interval,
        verbose=not args.quiet,
    )

    if args.once:
        steering = myc.step()
        print(json.dumps(steering, indent=2))
        return

    if args.daemon:
        def handle_signal(sig, frame):
            myc.stop()
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
        asyncio.run(myc.run_daemon())
        return

    # Default: interactive REPL
    myc.repl()


if __name__ == "__main__":
    main()
