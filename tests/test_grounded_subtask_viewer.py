"""Local fixtures for saved stage mapping and seekable video serving."""
import importlib.util
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import Mock, patch

import pandas as pd

spec = importlib.util.spec_from_file_location("subtask_viewer", Path(__file__).resolve().parents[1] /
                                            "scripts/view_grounded_subtasks.py")
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)


class ViewerTests(TestCase):
    def test_raw_source_samples_distinct_repeatable_episodes_without_reports(self):
        import numpy as np
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "meta/episodes/chunk-000").mkdir(parents=True)
            (root / "data/chunk-000").mkdir(parents=True)
            key = "observation.images.wrist"
            (root / "meta/info.json").write_text(json.dumps(dict(fps=30, total_episodes=10,
                features={key: {"dtype": "video"}},
                data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")))
            for chunk in range(2):
                pd.DataFrame([dict(episode_index=i, tasks=["pick battery"], **{"data/chunk_index": 0,
                    "data/file_index": 0, f"videos/{key}/chunk_index": 0,
                    f"videos/{key}/file_index": i, f"videos/{key}/from_timestamp": 0.})
                    for i in range(chunk*5, chunk*5+5)]).to_parquet(root / f"meta/episodes/chunk-000/file-{chunk:03d}.parquet")
            signal = np.r_[np.zeros(6), np.ones(2), np.full(20, .4)]
            pd.DataFrame([dict(episode_index=ep, frame_index=i, **{"observation.state": [0.] * 7 + [float(value)]})
                          for ep in range(10) for i, value in enumerate(signal)]).to_parquet(root / "data/chunk-000/file-000.parquet")
            settings = viewer.segmentation_helpers().parser().parse_args([
                "--gripper-smooth-window", "1", "--gripper-min-dwell-frames", "1"])
            a = viewer.Viewer("test/repo", "main", root, recompute_settings=settings, seed=42)
            b = viewer.Viewer("test/repo", "main", root, recompute_settings=settings, seed=42)
            self.assertEqual([e["episode"] for e in a.episodes], [0, 1, 4])
            self.assertEqual([e["episode"] for e in a.episodes], [e["episode"] for e in b.episodes])
            self.assertEqual(len(a.episodes[0]["subtasks"]), 3)
            self.assertEqual(a.episodes[0]["task"], "pick battery")
            self.assertIn("Raw source videos", a.warning)
            self.assertEqual(a.cached_videos, {})
            self.assertFalse((root / "point_tracks").exists())

    def test_random_metadata_lookup_does_not_read_every_shard(self):
        calls = []
        def read(i):
            calls.append(i)
            return pd.DataFrame({"episode_index": list(range(i*10, i*10+10))})
        rows = viewer.episode_metadata(list(range(100)), {1, 501, 999}, read)
        self.assertEqual(set(rows.episode_index), {1, 501, 999})
        self.assertLess(len(calls), 22)

    def test_preview_recomputes_splits_without_modifying_saved_report(self):
        import numpy as np
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "meta/episodes/chunk-000").mkdir(parents=True)
            (root / "point_tracks").mkdir()
            (root / "data/chunk-000").mkdir(parents=True)
            key = "observation.images.wrist"
            (root / "meta/info.json").write_text(json.dumps(dict(fps=30, total_episodes=1,
                features={key: {"dtype": "video"}},
                data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")))
            pd.DataFrame([dict(episode_index=0, **{"data/chunk_index": 0, "data/file_index": 0,
                f"videos/{key}/chunk_index": 0, f"videos/{key}/file_index": 0,
                f"videos/{key}/from_timestamp": 3.5})]).to_parquet(root / "meta/episodes/chunk-000/file-000.parquet")
            signal = np.r_[np.zeros(6), np.ones(2), np.full(20, .4)]
            pd.DataFrame([dict(episode_index=0, frame_index=i,
                              **{"observation.state": [0.] * 7 + [float(value)]})
                          for i, value in enumerate(signal)]).to_parquet(root / "data/chunk-000/file-000.parquet")
            report = root / "point_tracks/ep000000.json"
            report.write_text(json.dumps(dict(episode=0, destination_episode=0, task="test",
                subtasks=[dict(subtask=0, start_frame=0, end_frame=27, primitive="none", pov={}, third_person={})])))
            before = report.read_bytes()
            helper = viewer.segmentation_helpers()
            settings = helper.parser().parse_args(["--gripper-smooth-window", "1", "--gripper-min-dwell-frames", "1"])
            instance = viewer.Viewer("test/repo", "main", root, recompute_settings=settings)
            stages = instance.episodes[0]["subtasks"]
            self.assertEqual([(s["start_frame"], s["end_frame"], s["primitive"]) for s in stages],
                             [(0, 7, "release"), (7, 9, "grasp"), (9, 27, "none")])
            self.assertEqual(report.read_bytes(), before)
            self.assertIn("does not regenerate", instance.warning)
            self.assertEqual(instance.episodes[0]["saved_subtask_count"], 1)

    def test_default_limit_downloads_only_three_reports_and_needed_metadata(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "meta/episodes/chunk-000").mkdir(parents=True)
            (root / "point_tracks").mkdir()
            key = "observation.images.wrist"
            (root / "meta/info.json").write_text(json.dumps(dict(fps=30, total_episodes=10,
                features={key: {"dtype": "video"}},
                video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")))
            pd.DataFrame([dict(episode_index=i, **{f"videos/{key}/chunk_index": 0,
                f"videos/{key}/file_index": 0, f"videos/{key}/from_timestamp": i}) for i in range(5)]).to_parquet(
                root / "meta/episodes/chunk-000/file-000.parquet")
            # These files must never be opened for the first three episodes.
            (root / "meta/episodes/chunk-000/file-001.parquet").write_bytes(b"not parquet")
            for i in range(10):
                report = dict(episode=i + 7, task="test", subtasks=[dict(subtask=0, start_frame=0,
                    end_frame=29, primitive="release", pov={}, third_person={})])
                (root / f"point_tracks/ep{i+7:06d}.json").write_text(json.dumps(report) if i < 9 else "bad JSON")
            files = [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()]
            downloads = []
            def download(repo, name, **kwargs):
                downloads.append(name)
                return str(root / name)
            api = Mock()
            api.repo_info.return_value.sha = "pinned"
            api.list_repo_files.return_value = files
            with patch("huggingface_hub.HfApi", return_value=api), \
                 patch("huggingface_hub.hf_hub_download", side_effect=download):
                instance = viewer.Viewer("test/repo", "main")
            self.assertEqual([e["episode"] for e in instance.episodes], [0, 1, 2])
            self.assertEqual(set(downloads), {"meta/info.json", "meta/episodes/chunk-000/file-000.parquet",
                "point_tracks/ep000007.json", "point_tracks/ep000008.json", "point_tracks/ep000009.json"})
            next_one = viewer.Viewer("test/repo", "main", root, max_episodes=1, episode_offset=3)
            self.assertEqual([e["episode"] for e in next_one.episodes], [3])
            self.assertEqual(next_one.episodes[0]["source_episode"], 10)

    def test_destination_mapping_and_camera_offset(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "meta/episodes/chunk-000").mkdir(parents=True)
            (root / "point_tracks").mkdir()
            key = "observation.images.wrist"
            (root / "meta/info.json").write_text(json.dumps(dict(fps=30, features={key: {"dtype": "video"}},
                video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")))
            pd.DataFrame([dict(episode_index=0, **{f"videos/{key}/chunk_index": 0,
                f"videos/{key}/file_index": 1, f"videos/{key}/from_timestamp": 12.5})]).to_parquet(
                root / "meta/episodes/chunk-000/file-000.parquet")
            (root / "point_tracks/ep000007.json").write_text(json.dumps(dict(episode=7, destination_episode=0,
                task="drop battery", subtasks=[dict(subtask=0, start_frame=0, end_frame=29, primitive="release",
                pov={}, third_person={"top": dict(source="gripper_fallback", snapshot_frame=29)})])))
            instance = viewer.Viewer("test/repo", "main", root)
            self.assertEqual(instance.episodes[0]["source_episode"], 7)
            self.assertEqual(instance.episodes[0]["cameras"][0]["offset"], 12.5)
            self.assertEqual(instance.episodes[0]["subtasks"][0]["third_person"]["top"]["snapshot_frame"], 29)
            self.assertEqual(instance.videos[(0, "wrist")], "videos/observation.images.wrist/chunk-000/file-001.mp4")
            self.assertEqual(instance.warning, "")

    def test_video_byte_range(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "video.mp4"
            path.write_bytes(b"0123456789")
            handler = object.__new__(viewer.Handler)
            handler.headers = {"Range": "bytes=3-6"}
            handler.wfile = BytesIO()
            handler.send_response = Mock()
            handler.send_header = Mock()
            handler.end_headers = Mock()
            handler.stream(path)
            handler.send_response.assert_called_once_with(206)
            handler.send_header.assert_any_call("Content-Range", "bytes 3-6/10")
            self.assertEqual(handler.wfile.getvalue(), b"3456")


if __name__ == "__main__":
    main()
