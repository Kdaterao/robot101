"""Regression checks for sparse decoding and reuse of semantic trajectories."""
import sys
from pathlib import Path
from types import SimpleNamespace, ModuleType
from unittest import TestCase, main
from unittest.mock import patch
from tempfile import TemporaryDirectory
import json

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tapnet'))
# The developer checkout uses the newer module spelling; the VM pin uses torch.
try:
    import tapnet.torch
except ModuleNotFoundError as exc:
    if exc.name != 'tapnet.torch':
        raise
    import tapnet.tapir_torch
    sys.modules['tapnet.torch'] = tapnet.tapir_torch

from grounded_episode_io import load_episode_metadata, decode_episode_cameras
from hf_preprocess_smolvla_grounded import (
    _ground_and_track_camera, _stage_cache_entry, _tail_query_frames, GRIPPER_QUERY)
from robotap import Stage


class DecodeTests(TestCase):
    def test_sparse_decode_preserves_offset_masks_and_skips_gaps(self):
        cam = 'wrist'
        key = 'observation.images.wrist'
        rows = [dict(timestamp=torch.tensor(i/30), episode_index=torch.tensor(7),
                     task_index=torch.tensor(0), source_dataset_index=torch.tensor(i), **{
                         'observation.state': torch.arange(8), 'action': torch.arange(8),
                         'observation.images.wrist_padding_mask': torch.tensor([i != 2])})
                for i in range(8)]
        meta = SimpleNamespace(features={key: {}, 'source_dataset_index': {'dtype': 'int64', 'shape': [1]},
                                             'timestamp': {'dtype': 'float32', 'shape': [1]}}, video_keys=[key],
                               tasks=SimpleNamespace(iloc=[SimpleNamespace(name='pick cup')]),
                               episodes={7: {f'videos/{key}/from_timestamp': 10.0}},
                               get_video_file_path=lambda episode, camera: Path('video.mp4'))
        ds = SimpleNamespace(hf_dataset=rows, meta=meta, root=Path('/tmp'), tolerance_s=.01)
        data = load_episode_metadata(ds, [cam, 'side'])
        self.assertEqual(data['tasks'], ['pick cup'] * 8)
        self.assertEqual(set(data['extras'][0]), {'source_dataset_index'})
        self.assertEqual(data['extras'][6]['source_dataset_index'].dtype, np.dtype('int64'))
        self.assertEqual(data['extras'][6]['source_dataset_index'].shape, (1,))
        self.assertEqual(data['extras'][6]['source_dataset_index'][0], 6)
        self.assertFalse(data['masks'][cam][2])
        self.assertFalse(data['masks']['side'].any())
        calls = []
        def decode(path, timestamps, tolerance, **kwargs):
            self.assertTrue(kwargs['return_uint8'])
            calls.append(timestamps)
            return torch.stack([torch.full((3, 2, 3), round((t-10)*30), dtype=torch.uint8)
                                for t in timestamps])
        module = ModuleType('lerobot.datasets.video_utils')
        module.decode_video_frames = decode
        with patch.dict(sys.modules, {'lerobot.datasets.video_utils': module}):
            decode_episode_cameras(ds, data, [cam, 'side'], indices=[1, 2, 6], batch_size=64)
        self.assertEqual([len(c) for c in calls], [2, 1])
        self.assertAlmostEqual(calls[0][0], 10 + 1/30, places=6)
        self.assertIsNone(data['frames'][cam][0])
        self.assertEqual(data['frames'][cam][6].shape, (2, 3, 3))
        self.assertTrue((data['frames'][cam][6] == 6).all())
        self.assertTrue((data['frames']['side'][6] == 0).all())

    def test_full_decode_obeys_batch_limit_and_keeps_all_frames(self):
        key = 'observation.images.wrist'
        meta = SimpleNamespace(features={key: {}}, video_keys=[key],
                               episodes={1: {f'videos/{key}/from_timestamp': 0}},
                               get_video_file_path=lambda *args: Path('video.mp4'))
        ds = SimpleNamespace(root=Path('/tmp'), meta=meta, tolerance_s=.01)
        data = dict(n=5, episode_index=1, timestamps=list(range(5)), frames={})
        calls = []
        def decode(path, timestamps, *args, **kwargs):
            calls.append(timestamps)
            return torch.stack([torch.full((3, 2, 2), t, dtype=torch.uint8) for t in timestamps])
        module = ModuleType('lerobot.datasets.video_utils')
        module.decode_video_frames = decode
        with patch.dict(sys.modules, {'lerobot.datasets.video_utils': module}):
            decode_episode_cameras(ds, data, ['wrist'], batch_size=2)
        self.assertEqual(calls, [[0, 1], [2, 3], [4]])
        self.assertEqual([int(f[0, 0, 0]) for f in data['frames']['wrist']], list(range(5)))


class TrackingTests(TestCase):
    def test_shared_features_only_track_tail_and_mask_invalid_frames(self):
        bank = object()
        calls = []
        class Tapir:
            def track_with_features(self, clip, features, **kwargs):
                self.assert_bank = features
                calls.append([int(f[0, 0, 0]) for f in clip])
                return np.zeros((2, len(clip), 2)), np.ones((2, len(clip)), bool)
            def track_video(self, *args, **kwargs):
                raise AssertionError('Candidate descriptors must not be re-seeded')
        tracker = Tapir()
        frames = [np.full((2, 2, 3), i, np.uint8) for i in range(10)]
        masks = np.ones(10, bool)
        masks[8] = False
        result = _stage_cache_entry(tracker, frames, masks, Stage(0, 9, 'grasp'), bank, 2, 3, 'test')
        self.assertIs(tracker.assert_bank, bank)
        self.assertEqual(calls, [[7, 8, 9]])
        self.assertEqual(result['tail_start'], 7)
        self.assertFalse(result['visible'][:, 1].any())
        short = _stage_cache_entry(tracker, frames, masks, Stage(0, 1, 'grasp'), bank, 2, 30, 'short')
        self.assertEqual(calls[-1], [0, 1])
        self.assertEqual(short['tail_start'], 0)
        empty = _stage_cache_entry(tracker, frames, np.zeros(10, bool), Stage(0, 9, 'grasp'), bank, 2, 3, 'empty')
        self.assertEqual(empty['tracks'].shape, (2, 3, 2))
        self.assertFalse(empty['visible'].any())

    def test_seed_frames_cover_each_tail_and_skip_missing_frames(self):
        stages = [Stage(0, 9, 'grasp'), Stage(10, 11, 'release')]
        masks = np.ones(12, bool)
        self.assertEqual(_tail_query_frames(stages, 4, 5, 1, masks), [6, 7, 8, 9, 10, 11])
        masks[7] = False
        self.assertEqual(_tail_query_frames(stages, 4, 5, 1, masks), [6, 8, 9, 10, 11])

    def test_semantic_selection_reuses_four_value_tapir_result(self):
        class Molmo:
            def ground(self, frame, prompt):
                return {'points_xy': [[10, 10]] if 'gripper' in prompt else [[12, 10]]}
        class Tapir:
            calls = 0
            def track_video(self, clip, queries, query_frame_index):
                self.calls += 1
                tracks = np.repeat(queries[:, None, :], len(clip), axis=1)
                return tracks, np.ones(tracks.shape[:2], bool), object(), object()
        tracker = Tapir()
        result = _ground_and_track_camera(
            tracker, Molmo(), [np.zeros((100, 100, 3), np.uint8) for _ in range(3)],
            np.ones(3, bool), Stage(0, 2, 'grasp'), [GRIPPER_QUERY, 'cup'], .08, .01, 'test')
        self.assertEqual(tracker.calls, 1)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['entity'], 'cup')
        self.assertEqual(result['initial_points'], [[12., 10.]])
        self.assertEqual(np.asarray(result['tracks']).shape, (1, 3, 2))


class SkipFlowTests(TestCase):
    def test_bad_metadata_and_camera_shapes_do_not_block_valid_episode(self):
        import hf_preprocess_smolvla_grounded as pipeline
        from grounded_validation import EpisodeShapeError
        features = {
            'observation.state': {'dtype': 'float32', 'shape': [8]},
            'action': {'dtype': 'float32', 'shape': [8]},
            'observation.images.wrist': {'dtype': 'video', 'shape': [3, 10, 10]},
            'observation.images.top': {'dtype': 'video', 'shape': [3, 10, 10]},
            'source_dataset_index': {'dtype': 'int64', 'shape': [1]},
        }
        class Meta:
            fps = 30
            total_episodes = 3
            robot_type = 'so101'
            def __init__(self, *args, **kwargs):
                self.features = features
        class Source:
            writer = None
            def __init__(self, repo, episodes, **kwargs):
                self.episode = episodes[0]
            @classmethod
            def create(cls, **kwargs):
                cls.writer = Writer(kwargs['features'])
                return cls.writer
        class Writer:
            def __init__(self, schema):
                self.meta = SimpleNamespace(features=schema, total_episodes=0)
                self.written = []
                self.frames = []
            def add_frame(self, frame):
                self.frames.append(frame)
            def save_episode(self):
                self.written.append(self.frames)
                self.frames = []
                self.meta.total_episodes += 1
            def finalize(self):
                pass
        def load(source, cameras):
            if source.episode == 0:
                raise EpisodeShapeError('wrong state shape')
            return dict(n=4, gripper=np.zeros(4), action_gripper=np.zeros(4),
                        states=np.zeros((4, 8), np.float32), actions=np.zeros((4, 8), np.float32),
                        tasks=['pick cup']*4, masks={c: np.ones(4, bool) for c in cameras},
                        frames={}, extras=[{'source_dataset_index': np.array([source.episode], np.int64)}]*4)
        def decode(source, data, cameras, **kwargs):
            for camera in cameras:
                shape = (9, 10, 3) if source.episode == 2 and camera == 'top' else (10, 10, 3)
                data['frames'][camera] = [np.zeros(shape, np.uint8) for _ in range(4)]
            return data
        bank = object()
        tail_banks = []
        backward_lengths = []
        def backward(clip, endpoints, **kwargs):
            backward_lengths.append(len(clip))
            return np.zeros((len(endpoints), len(clip), 2)), np.ones((len(endpoints), len(clip)), bool)
        def tail(*args, **kwargs):
            tail_banks.append(args[4])
            return dict(tracks=np.zeros((1, 4, 2)), visible=np.ones((1, 4), bool))
        with TemporaryDirectory() as temporary, patch.multiple(pipeline,
                _load_lerobot=lambda: (Source, Meta, {}, Path(temporary)),
                load_episode_metadata=load, decode_episode_cameras=decode,
                install_statistics_validation=lambda *args: None,
                BootsTAPIR=lambda **kwargs: SimpleNamespace(device='cpu', init_features=lambda *args: object(),
                                                           track_segment_backward=backward),
                concat_query_features=lambda parts: bank,
                Molmo2Worker=lambda *args, **kwargs: object(),
                events_from_gripper_thresholds=lambda *args, **kwargs: [],
                stages_from_events=lambda *args: [Stage(0, 3, 'grasp')],
                _stage_cache_entry=tail,
                _select_pov_points=lambda *args: ({(0, 0): np.array([0]), (1, 0): np.array([0])}, 1),
                _extract_noun_phrases=lambda *args: ['cup'],
                _ground_and_track_camera=lambda *args: dict(status='ok', tracks=[], visibility=[])), \
                patch.dict(sys.modules, {'spacy': SimpleNamespace(load=lambda *args: object())}), \
                patch.object(sys, 'argv', ['preprocess', '--episodes', '0-2', '--dst-repo-id', 'test/output']):
            pipeline.main()
            self.assertEqual(len(tail_banks), 2)
            self.assertTrue(all(features is bank for features in tail_banks))
            self.assertEqual(backward_lengths, [4])
            provenance = json.loads((Path(temporary)/'test/output/point_tracks/pov_query_bank.json').read_text())
            self.assertEqual({row[0] for row in provenance['source_episode_frame']}, {1, 2})
            self.assertEqual(len(Source.writer.written), 1)
            self.assertEqual(len(Source.writer.written[0]), 4)
            self.assertEqual(Source.writer.written[0][0]['source_dataset_index'][0], 1)
            records = [json.loads(line) for line in
                       (Path(temporary)/'test/output/point_tracks/skipped_episodes.jsonl').read_text().splitlines()]
            self.assertEqual([r['episode'] for r in records], [0, 2])
            self.assertEqual(records[0]['phase'], 'tail_pass')
            self.assertEqual(records[1]['phase'], 'full_episode')


if __name__ == '__main__':
    main()
