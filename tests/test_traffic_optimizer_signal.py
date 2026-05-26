"""
Unit tests for tools/traffic_optimizer_signal.py.

Covers the SUMO physics floor enforcement added in Week 2-D (commit
e57cb6937): clamp_phase_duration must respect minimum yellow / all-red /
green durations even when the optimizer's bounds would allow shorter ones.

Runs with stdlib only:
    python -m unittest tests.test_traffic_optimizer_signal
"""
import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.join(REPO_ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)


class TestClassifyPhaseKind(unittest.TestCase):
    """SUMO state string → {yellow, all_red, green, other, unknown}."""

    def setUp(self):
        from traffic_optimizer_signal import classify_phase_kind
        self._classify = classify_phase_kind

    def test_yellow_takes_priority(self):
        # 'y' anywhere in the state string makes it a yellow (clearance) phase
        self.assertEqual(self._classify("GGGyrrr"), "yellow")
        self.assertEqual(self._classify("Y"), "yellow")
        self.assertEqual(self._classify("yyyy"), "yellow")

    def test_all_red_when_only_red(self):
        self.assertEqual(self._classify("rrrr"), "all_red")
        self.assertEqual(self._classify("rrrRR"), "all_red")  # uppercase R also red
        # s = red with right-turn-on-red allowed; per the helper this still counts as red-family
        self.assertEqual(self._classify("rrs"), "all_red")

    def test_green_when_no_yellow_and_some_green(self):
        self.assertEqual(self._classify("GGGrrr"), "green")
        self.assertEqual(self._classify("gggGGG"), "green")
        self.assertEqual(self._classify("rrgrrG"), "green")

    def test_empty_or_unrecognized(self):
        self.assertEqual(self._classify(""), "unknown")
        self.assertEqual(self._classify(None), "unknown")
        # Only off-states are "other" (no g/G, no y/Y, not all r/s)
        self.assertEqual(self._classify("oOoO"), "other")


class TestPhysicsMinDuration(unittest.TestCase):
    """_physics_min_duration returns the SUMO physical floor per phase kind."""

    def setUp(self):
        from traffic_optimizer_signal import _physics_min_duration
        self._floor = _physics_min_duration

    def test_yellow_floor_3s(self):
        # Default; env-overridable via TRAFFICVISION_MIN_YELLOW
        self.assertEqual(self._floor("yyy"), 3.0)
        self.assertEqual(self._floor("GGGy"), 3.0)  # any 'y' → yellow kind

    def test_all_red_floor_1s(self):
        self.assertEqual(self._floor("rrr"), 1.0)

    def test_green_floor_5s(self):
        self.assertEqual(self._floor("GGG"), 5.0)

    def test_unknown_floor_zero(self):
        self.assertEqual(self._floor(""), 0.0)
        self.assertEqual(self._floor("oo"), 0.0)


class TestClampPhaseDuration(unittest.TestCase):
    """clamp_phase_duration enforces [bounded_min, bounded_max] AND physics floor."""

    def setUp(self):
        from traffic_optimizer_signal import clamp_phase_duration
        self._clamp = clamp_phase_duration

    def _phase(self, duration, state, minDur=None, maxDur=None):
        return {
            "duration": duration,
            "state": state,
            "minDur": minDur if minDur is not None else duration,
            "maxDur": maxDur if maxDur is not None else duration,
            "hasMinDur": minDur is not None,
            "hasMaxDur": maxDur is not None,
        }

    def test_yellow_promoted_to_physics_floor(self):
        """An optimizer trying to set a yellow phase to 1.5s should be lifted
        to 3.0s (MIN_YELLOW), regardless of declared minDur=1.5s."""
        phase = self._phase(duration=3.0, state="yyy", minDur=1.5, maxDur=4.0)
        self.assertEqual(self._clamp(phase, 1.5), 3.0)

    def test_green_promoted_to_physics_floor(self):
        """A green phase clamp below MIN_GREEN (5s) is lifted to 5s."""
        phase = self._phase(duration=10.0, state="GGG", minDur=2.0, maxDur=30.0)
        self.assertEqual(self._clamp(phase, 3.0), 5.0)

    def test_all_red_floor_1s_respected(self):
        phase = self._phase(duration=2.0, state="rrr", minDur=0.5, maxDur=3.0)
        self.assertEqual(self._clamp(phase, 0.5), 1.0)

    def test_value_within_bounds_passes_through(self):
        """No clamping needed when requested duration sits inside the window
        and above the physics floor."""
        phase = self._phase(duration=10.0, state="GGG", minDur=5.0, maxDur=30.0)
        self.assertEqual(self._clamp(phase, 12.0), 12.0)

    def test_value_above_max_clamped_down(self):
        phase = self._phase(duration=10.0, state="GGG", minDur=5.0, maxDur=20.0)
        self.assertEqual(self._clamp(phase, 50.0), 20.0)

    def test_physics_floor_can_exceed_declared_max(self):
        """If a misconfigured phase declares maxDur=2s on a yellow, the
        physics floor still wins (we'd rather over-shoot SUMO bounds than
        emit a sub-3s yellow that violates road engineering norms)."""
        phase = self._phase(duration=3.0, state="yyy", minDur=1.0, maxDur=2.0)
        # effective_min = max(declared_min=1, floor=3) = 3
        # effective_max = max(effective_min=3, declared_max=2) = 3
        self.assertEqual(self._clamp(phase, 5.0), 3.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
