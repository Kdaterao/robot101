"""Movement-triggered segmentation ignores small nudges and retains quick reversals."""
import sys
from pathlib import Path
from unittest import TestCase, main

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tapnet"))
try:
    import tapnet.torch
except ModuleNotFoundError as exc:
    if exc.name != "tapnet.torch":
        raise
    import tapnet.tapir_torch
    sys.modules["tapnet.torch"] = tapnet.tapir_torch

from robotap import events_from_gripper_thresholds, stages_from_events
from hf_preprocess_smolvla_grounded import build_parser


class MovementTests(TestCase):
    def test_small_nudge_ignored_partial_grasp_detected(self):
        signal = np.repeat([.8, .9, .4, 1., 0., .8], 6)
        events = events_from_gripper_thresholds(signal, smooth_window=1, min_change_frac=.25)
        self.assertEqual([e["type"] for e in events], ["grasp", "release", "grasp", "release"])
        self.assertGreater(events[0]["g_smooth"], events[0]["closed_thresh"])
        self.assertEqual(events[0]["movement_start_frame"], 11)
        self.assertTrue(all(e["position_change"] >= e["required_position_change"] for e in events))

    def test_quick_open_close_retains_short_stage(self):
        signal = np.r_[np.zeros(6), np.ones(2), np.full(20, .4)]
        events = events_from_gripper_thresholds(signal, smooth_window=1, min_dwell_frames=1,
                                               min_change_frac=.25)
        self.assertEqual([e["frame"] for e in events], [7, 9])
        stages = stages_from_events(events, len(signal), min_stage_frames=build_parser().get_default("min_stage_frames"))
        self.assertEqual([(s.start, s.end, s.primitive) for s in stages],
                         [(0, 7, "release"), (7, 9, "grasp"), (9, 27, "none")])

    def test_absolute_floor_rejects_small_range_episode(self):
        signal = np.repeat([0., .01, 0.], 10)
        self.assertEqual(events_from_gripper_thresholds(signal, smooth_window=1,
                         min_change_frac=.25, min_change_abs=.1), [])

    def test_constant_episode_has_no_events(self):
        self.assertEqual(events_from_gripper_thresholds(np.ones(20), min_change_frac=.25), [])

    def test_legacy_bands_are_available(self):
        signal = np.repeat([.8, .9, .4, 1., 0., .8], 6)
        legacy = events_from_gripper_thresholds(signal, smooth_window=1)
        self.assertEqual(legacy[0]["type"], "release")
        self.assertNotIn("position_change", legacy[0])


if __name__ == "__main__":
    main()
