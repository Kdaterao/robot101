"""CPU regression checks with fake models; no checkpoint downloads."""
import ast
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from robot101.perception.gripper_masks import filter_candidates, GripperMaskWorker
from robot101.data.utilities.gripper_filtering import prepare_gripper_filter


def functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *selected], type_ignores=[])), str(path), 'exec'), namespace)
    return namespace


endpoint = functions(ROOT/'src/robot101/perception/tracking.py', ['track_tail_endpoint'], {'np': np})['track_tail_endpoint']
tracking_stub = SimpleNamespace(track_tail_endpoint=endpoint)
ns = functions(ROOT/'src/robot101/data/preprocess.py',
               ['_ground_and_track_camera', '_valid_points', '_third_person_sample_indices',
                '_expand_sparse_tracks', '_bridge_visibility_gaps', '_stage_cluster_bounds', '_select_pov_points'],
               {'np': np, 'GRIPPER_QUERY': 'robot gripper'})


class FilterTests(unittest.TestCase):
    def setUp(self):
        self.patch = patch.dict(sys.modules, {'robot101.perception.tracking': tracking_stub})
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_exclusion_weights_and_masked_observations(self):
        masks = np.zeros((3, 20, 20), bool)
        masks[:, 5:10, 5:10] = True
        tracks = np.array([[[6, 6]]*3, [[10, 7]]*3, [[19, 19]]*3,
                           [[11, 7], [6, 6], [11, 7]]], np.float32)
        filled, vis, weights, eligible, ends = filter_candidates(tracks, np.ones((4, 3), bool), masks)
        self.assertEqual(eligible.tolist(), [False, True, True, True])
        self.assertFalse(vis[3, 1])
        self.assertEqual(weights[3, 1], 0)
        self.assertTrue(np.all(filled[3] == [11, 7]))
        self.assertGreater(weights[1, 0], 1)
        self.assertLessEqual(weights[1, 0], 2)
        self.assertEqual(weights[2, 0], 1)

    def test_missing_mask_does_not_mean_background(self):
        masks = np.zeros((1, 20, 20), bool)
        _, vis, weights, eligible, _ = filter_candidates(np.array([[[1, 1]]], np.float32), np.ones((1, 1), bool), masks)
        self.assertFalse(vis.any())
        self.assertFalse(eligible.any())
        self.assertFalse(weights.any())

    def test_queue_and_exact_frame_correction(self):
        frames = [np.zeros((20, 20, 3), np.uint8) for _ in range(4)]
        def entry():
            return {'tail_start': 1, 'tracks': np.array([[[11, 7]]*3], np.float32),
                    'visible': np.ones((1, 3), bool)}
        class Sam:
            def propagate(self, clip, seed, points, labels):
                self.seed = seed
                mask = np.zeros((len(clip), 20, 20), bool)
                mask[:, 5:10, 5:10] = True
                return mask
        sam = Sam()
        molmo = SimpleNamespace(ground=lambda *args: {'points_xy': []})
        with TemporaryDirectory() as temp:
            root = Path(temp)
            e = entry()
            prepare_gripper_filter(e, frames, np.ones(4, bool), molmo, sam, root, 4, 1, {}, 2, .05, 1)
            self.assertTrue(e['gripper_filter']['manual_review_required'])
            review = root/'mask_review/ep000004_stage001.json'
            self.assertTrue(review.exists())
            correction = {'4:1': {'frame_index': 2, 'points_xy': [[6, 6]], 'labels': [1]}}
            e = entry()
            prepare_gripper_filter(e, frames, np.ones(4, bool), molmo, sam, root, 4, 1, correction, 2, .05, 1)
            self.assertEqual(sam.seed, 1)
            self.assertEqual(e['gripper_filter']['status'], 'ok')
            self.assertFalse(review.exists())
            with np.load(root/e['gripper_filter']['mask_file']) as saved:
                shape = saved['shape']
                mask = np.unpackbits(saved['packed_masks'], axis=-1)[..., :shape[2]]
                self.assertEqual(mask.shape, (3, 20, 20))
                self.assertEqual(saved['frame_indices'].tolist(), [1, 2, 3])
            correction['4:1']['frame_index'] = 0
            prepare_gripper_filter(e, frames, np.ones(4, bool), molmo, sam, root, 4, 1, correction, 2, .05, 1)
            self.assertTrue(e['gripper_filter']['manual_review_required'])

    def test_windows_include_extension_and_truncate(self):
        stage = lambda a, b: SimpleNamespace(start=a, end=b)
        stages = [stage(0, 4), stage(5, 14)]
        self.assertEqual(ns['_stage_cluster_bounds'](stages, 0, 15, 30, 10), (0, 14))
        self.assertEqual(ns['_stage_cluster_bounds'](stages, 1, 15, 30, 10), (5, 14))


class GroundingTests(unittest.TestCase):
    def run_ground(self, results, missing_end=False, start=2, end=6):
        frames = [np.full((20, 20, 3), i, np.uint8) for i in range(10)]
        calls, seen = [], []
        def ground(frame, prompt):
            calls.append((int(frame[0, 0, 0]), prompt))
            return {'points_xy': results.get(prompt, [])}
        def track(clip, points, query_frame_index):
            seen.extend(int(f[0, 0, 0]) for f in clip)
            tracks = np.repeat(points[:, None], len(clip), axis=1)
            return tracks, np.ones(tracks.shape[:2], bool), None, None
        masks = np.ones(10, bool)
        if missing_end:
            masks[end] = False
        out = ns['_ground_and_track_camera'](SimpleNamespace(track_video=track), SimpleNamespace(ground=ground),
                                             frames, masks, SimpleNamespace(start=start, end=end),
                                             ['robot gripper', 'battery'], .08, .01, 'test', fps=30)
        return out, calls, seen

    def test_start_candidates_track_forward_and_match_endpoint_gripper(self):
        out, calls, seen = self.run_ground({'Point to the battery.': [[12, 12]], 'Point to the robot gripper.': [[11, 11]]})
        self.assertEqual(out['source'], 'semantic_object')
        self.assertEqual(calls, [(2, 'Point to the robot gripper.'), (2, 'Point to the battery.')])
        self.assertEqual(out['tracking_direction'], 'forward')
        self.assertGreater(out['distance_to_gripper'], 0)
        self.assertEqual(seen[0], 2)
        self.assertEqual(seen[-1], 6)
        self.assertEqual(np.asarray(out['tracks']).shape, (1, 5, 2))

    def test_gripper_fallback_exact_intermediate_and_final_endpoint(self):
        for end in [6, 9]:
            out, calls, seen = self.run_ground({'Point to the robot gripper.': [[5, 5]]}, end=end)
            self.assertEqual(out['source'], 'gripper_fallback')
            self.assertEqual(calls[-1], (end, 'Point to the robot gripper.'))
            self.assertEqual(seen[0], end)
            self.assertEqual(out['seed_frame'], end)

    def test_missing_endpoint_or_detection_requires_review(self):
        out, calls, seen = self.run_ground({}, missing_end=True)
        self.assertTrue(out['manual_review_required'])
        self.assertFalse(calls)
        out, calls, seen = self.run_ground({})
        self.assertTrue(out['manual_review_required'])
        self.assertFalse(seen)

    def test_malformed_object_uses_endpoint_fallback(self):
        out, _, _ = self.run_ground({'Point to the battery.': [3], 'Point to the robot gripper.': [[5, 5]]})
        self.assertEqual(out['source'], 'gripper_fallback')


class SamAdapterTests(unittest.TestCase):
    def test_video_state_frame_order_and_cleanup(self):
        import torch
        seen = []
        class Predictor:
            def init_state(self, directory, **kwargs):
                self.files = sorted(Path(directory).glob('*.jpg'))
                self.directory = Path(directory)
                return {}
            def add_new_points_or_box(self, state, **kwargs):
                seen.append(('seed', kwargs['frame_idx']))
                return kwargs['frame_idx'], [1], torch.ones((1, 1, 10, 10))
            def propagate_in_video(self, state, start_frame_idx, reverse):
                seen.append(('reverse', reverse))
                indices = range(start_frame_idx, -1, -1) if reverse else range(start_frame_idx, 3)
                for i in indices:
                    yield i, [1], torch.ones((1, 1, 10, 10))
            def reset_state(self, state):
                seen.append(('reset', True))
        worker = GripperMaskWorker.__new__(GripperMaskWorker)
        worker.device = torch.device('cpu')
        worker.predictor = Predictor()
        masks = worker.propagate([np.zeros((10, 10, 3), np.uint8)]*3, 1, [[2, 2]])
        self.assertTrue(masks.all())
        self.assertEqual(len(worker.predictor.files), 3)
        self.assertFalse(worker.predictor.directory.exists())
        self.assertEqual(seen, [('seed', 1), ('reverse', False), ('reverse', True), ('reset', True)])

    def test_negative_only_prompt_rejected(self):
        worker = GripperMaskWorker.__new__(GripperMaskWorker)
        with self.assertRaisesRegex(ValueError, 'positive points'):
            worker.propagate([np.zeros((10, 10, 3), np.uint8)], 0, [[2, 2]], [0])

    def test_tracking_failure_uses_exact_endpoint_gripper(self):
        calls = []
        def ground(frame, prompt):
            calls.append((int(frame[0, 0, 0]), prompt))
            return {'points_xy': [[8, 8]] if 'battery' in prompt else [[4, 4]]}
        def track(clip, points, query_frame_index):
            if points[0, 0] == 8:
                raise RuntimeError('lost object')
            return np.repeat(points[:, None], len(clip), 1), np.ones((1, len(clip)), bool), None, None
        frames = [np.full((10, 10, 3), i, np.uint8) for i in range(4)]
        out = ns['_ground_and_track_camera'](SimpleNamespace(track_video=track), SimpleNamespace(ground=ground),
                frames, np.ones(4, bool), SimpleNamespace(start=0, end=3), ['battery'], .08, .01, 'test')
        self.assertEqual(out['source'], 'gripper_fallback')
        self.assertEqual([frame for frame, _ in calls], [0, 0, 3])


class WeightedSelectionTests(unittest.TestCase):
    def test_proximity_votes_change_cluster_winner(self):
        motion_ns = functions(ROOT/'src/robot101/perception/motion_plan.py',
                              ['select_stage_active', '_first_visible', '_resample'],
                              {'np': np, 'track_tail_endpoint': endpoint,
                               'resolve_torch_device': lambda x: x,
                               '_kmeans_torch': lambda *args, **kwargs: np.array([0, 0, 1, 1])})
        tr = np.array([[[0, i], [10, i]] for i in range(4)], np.float32)
        out = motion_ns['select_stage_active']([tr], [np.ones((4, 2), bool)], 20, 20,
                                                n_clusters=2, goal_tail_frames=1, merge_frac=.9,
                                                point_weights=np.array([1, 1, 2, 2]))
        self.assertEqual(out['winners'], [1])
        self.assertEqual(out['point_ids'].tolist(), [2, 3])

    def test_shared_selection_excludes_unresolved_and_masked_endpoints(self):
        calls = []
        ns['select_stage_active'] = lambda tracks, visible, **kwargs: (calls.append(kwargs) or {'point_ids': np.arange(len(tracks[0]))})
        e = {'tracks': np.zeros((3, 2, 2)), 'visible': np.ones((3, 2), bool),
             'eligible': np.array([False, True, True]), 'observation_weights': np.ones((3, 2)),
             'gripper_filter': {'manual_review_required': False}}
        failed = {**e, 'gripper_filter': {'manual_review_required': True}}
        args = SimpleNamespace(num_poi_points=2, n_clusters=2, static_thresh=.03,
                               funnel_quantile=.4, goal_tail_frames=1, seed=0)
        selected, _ = ns['_select_pov_points']([[e], [failed]], [[1], [1]], args, None)
        self.assertEqual(selected[0, 0].tolist(), [1, 2])
        self.assertEqual(selected[1, 0].tolist(), [])
        self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
