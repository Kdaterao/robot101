"""Catch genuine source shape mismatches without rejecting scalar camera masks."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from grounded_validation import (EpisodeShapeError, validate_episode_shapes,
                                validate_stat_value, record_skipped_episode)


class ShapeTests(unittest.TestCase):
    def data(self):
        return dict(n=2, states=np.zeros((2, 8)), actions=np.zeros((2, 8)),
                    frames={'wrist': [np.zeros((48, 64, 3)), None]})

    def features(self):
        return {'observation.state': {'shape': [8]}, 'action': {'shape': [8]},
                'observation.images.wrist': {'dtype': 'video', 'shape': [3, 48, 64]}}

    def test_valid_sparse_frames_and_invalid_resolution(self):
        data = self.data()
        validate_episode_shapes(data, self.features(), ['wrist'])
        data['frames']['wrist'][1] = np.zeros((40, 64, 3))
        with self.assertRaisesRegex(EpisodeShapeError, 'wrist frame 1'):
            validate_episode_shapes(data, self.features(), ['wrist'])

    def test_invalid_action_shape(self):
        data = self.data()
        data['actions'] = np.zeros((2, 7))
        with self.assertRaisesRegex(EpisodeShapeError, 'action'):
            validate_episode_shapes(data, self.features(), ['wrist'])

    def test_padding_mask_scalar_stats_are_not_image_stats(self):
        features = {'observation.images.wrist_padding_mask': {'dtype': 'bool', 'shape': [1]},
                    'observation.images.wrist': {'dtype': 'video', 'shape': [3, 48, 64]}}
        # The exact statistic shape that crashed at second-episode aggregation.
        for statistic in ('min', 'max', 'mean', 'std', 'q01', 'count'):
            validate_stat_value(np.ones(1), statistic, 'observation.images.wrist_padding_mask', features)
        validate_stat_value(np.zeros((3, 1, 1)), 'min', 'observation.images.wrist', features)
        with self.assertRaises(ValueError):
            validate_stat_value(np.ones(1), 'min', 'observation.images.wrist', features)
        with self.assertRaises(ValueError):
            validate_stat_value(np.ones(2), 'count', 'observation.images.wrist_padding_mask', features)

    def test_skip_report_preserves_source_episode_id_and_reason(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / 'skipped.jsonl'
            record_skipped_episode(path, 7, 'tail_pass', 'bad wrist resolution')
            record_skipped_episode(path, 12, 'full_episode', 'bad action shape')
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([r['episode'] for r in records], [7, 12])
            self.assertEqual(records[1]['reason'], 'bad action shape')


if __name__ == '__main__':
    unittest.main()
