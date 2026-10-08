import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from grounded_destination import recover_empty_destination, check_resume_metadata

class DestinationTests(unittest.TestCase):
    def setup_root(self, folder, episodes=0):
        root = Path(folder)/'output'
        (root/'meta').mkdir(parents=True)
        (root/'meta/info.json').write_text(json.dumps(dict(total_episodes=episodes, total_frames=episodes*10)))
        return root
    def test_empty_skipped_run_is_preserved_for_fresh_creation(self):
        with TemporaryDirectory() as folder:
            root = self.setup_root(folder)
            (root/'point_tracks').mkdir()
            (root/'point_tracks/skipped_episodes.jsonl').write_text('skip report')
            backup = recover_empty_destination(root)
            self.assertFalse(root.exists())
            self.assertEqual((backup/'point_tracks/skipped_episodes.jsonl').read_text(), 'skip report')
    def test_written_or_uncommitted_data_is_never_moved(self):
        with TemporaryDirectory() as folder:
            root = self.setup_root(folder)
            (root/'data').mkdir()
            (root/'data/episode.parquet').write_bytes(b'data')
            self.assertIsNone(recover_empty_destination(root))
            self.assertTrue(root.exists())
        with TemporaryDirectory() as folder:
            root = self.setup_root(folder, episodes=1)
            self.assertIsNone(recover_empty_destination(root))
            with self.assertRaisesRegex(ValueError, 'tasks.parquet'):
                check_resume_metadata(root)
if __name__ == '__main__':
    unittest.main()
