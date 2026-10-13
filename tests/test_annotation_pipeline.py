import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from robot101.data.utilities.annotation_tree import (
    atomic_json, average_observations, build_tree, cached_json, descendants,
    fingerprint, midpoint, pool_tokens, sampled_frames, uniform_frames, windows,
)
from robot101.data.aggregate_annotations import (
    ClaudeClient, aggregate_node, build_evidence, main as aggregate_main,
    validate_annotation, validate_tree,
)
from robot101.data.annotate import build_parser, caption_records, segment_episode


def episode():
    ids = sampled_frames(120, 4)
    features = np.zeros((len(ids), 2))
    features[len(ids) // 2:] = 10
    tree = build_tree(features, ids, 120, 30)
    tree.update(schema_version=1, fingerprint="segment-key", episode_index=0,
                camera="top", source_repo_id="test/source", source_revision="abc",
                fps=30, bounds="half_open_source_frames", n_frames=120,
                task="put object in bucket", timestamps=[i / 30 for i in range(120)])
    for node in tree["nodes"]:
        node["caption"] = dict(status="ok", text=f'caption {node["id"]}', truncated=False)
    return tree


def annotation(node):
    return dict(node_id=node["id"], start_frame=node["start_frame"], end_frame=node["end_frame"],
                brief_action="move object", detailed_action="The robot moves an object.", actor="robot",
                brief_caption="A robot moves.", detailed_caption="A robot moves an object toward a bucket.",
                critique="", robotics=dict(manipulated_object="object", destination="bucket",
                    completion_condition="object in bucket", evidence=[node["id"]], conflicts=[],
                    review_status="accepted", suggested_changes=None))


class FakeClient:
    def __init__(self, node, responses=None):
        self.node, self.calls, self.responses = node, 0, list(responses or [])

    def complete(self, prompt, log):
        self.calls += 1
        text = self.responses.pop(0) if self.responses else json.dumps(annotation(self.node))
        log(dict(status="ok", text=text, usage=dict(input_tokens=20, output_tokens=10)))
        return text


class TreeTests(unittest.TestCase):
    def test_defaults_and_stepped_selection(self):
        args = build_parser().parse_args([])
        self.assertEqual((args.sample_step, args.window_frames, args.window_stride), (4, 64, 8))
        self.assertEqual((args.video_frames, args.max_caption_tokens), (32, 1024))
        from robot101.data.utilities.episode_helpers import _parse_episodes
        self.assertEqual(_parse_episodes("0-25:5"), [0, 5, 10, 15, 20, 25])

    def test_window_endpoint_padding_and_tubelet_mapping(self):
        for count in (1, 3, 64, 65, 131):
            bank = windows(count)
            observations = [(positions, valid, np.arange(32)[:, None] + start)
                            for start, (positions, valid) in enumerate(bank)]
            features, counts = average_observations(count, observations)
            self.assertEqual(features.shape, (count, 1))
            self.assertTrue(np.all(counts > 0))
            self.assertEqual(sum(counts), sum(valid.sum() for _, valid in bank))
        # A padded observation must not increment the final real frame.
        features, counts = average_observations(3, [(np.array([0, 1, 2, 2]),
                                    np.array([1, 1, 1, 0], bool), np.array([[2], [8]]))])
        np.testing.assert_array_equal(features[:, 0], [2, 2, 8])
        np.testing.assert_array_equal(counts, [1, 1, 1])
        with self.assertRaises(ValueError):
            windows(10, stride=3)

    def test_pool_and_overlap(self):
        tokens = np.arange(16).reshape(1, 8, 2)
        pooled = pool_tokens(tokens, 2, 4)
        np.testing.assert_array_equal(pooled[0], [[3, 4], [11, 12]])
        features, counts = average_observations(4, [
            (np.arange(4), np.ones(4, bool), np.array([[1], [3]])),
            (np.array([2, 3, 3, 3]), np.array([1, 1, 0, 0], bool), np.array([[7], [100]]))])
        np.testing.assert_array_equal(features[:, 0], [1, 1, 5, 5])
        np.testing.assert_array_equal(counts, [1, 1, 2, 2])
        with self.assertRaises(ValueError):
            pool_tokens(tokens, 3, 4)

    def test_hierarchy_and_cut_coverage(self):
        record = episode()
        validate_tree(record)
        self.assertEqual(len(record["hierarchy"]), 2 * len(sampled_frames(120)) - 1)
        self.assertGreater(len(record["nodes"]), len(record["selected_node_ids"]))
        self.assertEqual(len(record["selected_node_ids"]), 2)
        for node in record["nodes"]:
            self.assertGreater((node["end_frame"] - node["start_frame"]) / record["fps"], .5)
            children = [n for n in record["nodes"] if n["parent"] == node["id"]]
            self.assertEqual({c["id"] for c in children}, set(node["children"]))
        constant = build_tree(np.ones((10, 2)), np.arange(10), 10, 10, min_duration=0)
        self.assertEqual(constant["selected_node_ids"], [constant["root_id"]])

    def test_short_episode_and_strict_threshold(self):
        tree = build_tree(np.ones((2, 1)), [0, 14], 15, 30)
        self.assertEqual(tree["nodes"], [])
        self.assertTrue(tree["review_required"])
        self.assertIsNone(tree["root_id"])

    def test_midpoint_and_uniform_sampling(self):
        node = dict(start_frame=10, end_frame=21)
        self.assertEqual(midpoint(node), 15)
        self.assertEqual(uniform_frames(node, 3), [10, 15, 20])
        self.assertEqual(uniform_frames(dict(start_frame=4, end_frame=5), 32), [4] * 32)

    def test_cache_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            atomic_json(path, dict(fingerprint="one"))
            self.assertEqual(cached_json(path, "one", True)["fingerprint"], "one")
            for resume, key in ((False, "one"), (True, "two")):
                with self.assertRaises(ValueError):
                    cached_json(path, key, resume)

    def test_tree_cycle_and_immutable_bounds(self):
        record = episode()
        root = next(n for n in record["nodes"] if n["id"] == record["root_id"])
        bad = annotation(root)
        bad["end_frame"] += 1
        with self.assertRaises(ValueError):
            validate_annotation(json.dumps(bad), root)
        broken = copy.deepcopy(record)
        broken["nodes"][-1]["children"].append(broken["root_id"])
        with self.assertRaises(ValueError):
            validate_tree(broken)


class AdapterTests(unittest.TestCase):
    def test_caption_adapter_preserves_sparse_video_and_trims_prompt(self):
        import torch
        from robot101.perception.annotation_models import Captioner
        class Inputs(dict):
            def to(self, device):
                return self
        worker = object.__new__(Captioner)
        worker.torch, worker.kind, worker.device, worker.dtype = torch, "video", "cpu", torch.float32
        worker.processor = Mock()
        worker.processor.apply_chat_template.return_value = ["video prompt"]
        worker.processor.return_value = Inputs(input_ids=torch.tensor([[3, 4]]),
                                               pixel_values_videos=torch.zeros((1, 32, 3, 320, 320)))
        worker.processor.batch_decode.return_value = ["observed action"]
        worker.model = Mock()
        worker.model.generate.return_value = torch.tensor([[3, 4, 7, 8]])
        worker.model.generation_config.eos_token_id = 8
        frames = np.zeros((32, 320, 320, 3), np.uint8)
        result = worker.caption([dict(frames=frames)], 1024)
        kwargs = worker.processor.call_args.kwargs
        self.assertFalse(kwargs["do_sample_frames"])
        self.assertIs(kwargs["videos"][0], frames)
        self.assertEqual(kwargs["size"], dict(height=320, width=320))
        np.testing.assert_array_equal(worker.processor.batch_decode.call_args.args[0], [[7, 8]])
        self.assertFalse(result[0]["truncated"])

    def test_mock_encoder_feature_cache_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = build_parser().parse_args(["--output-dir", tmp])
            run = dict(requested=vars(args), resolved=dict(source="abc", jepa="def"), fps=30)
            data = dict(n=120, tasks=["task"] * 120, timestamps=[i / 30 for i in range(120)],
                        gripper=np.zeros(120), masks={"top": np.ones(120, bool)},
                        frames={"top": [np.zeros((8, 8, 3), np.uint8)] * 120})
            worker = Mock(tubelet=2)
            worker.encode.side_effect = lambda clips: np.zeros((len(clips), 32, 2))
            factory = Mock(return_value=worker)
            with patch("robot101.data.annotate.load_episode_metadata", return_value=data), \
                 patch("robot101.data.annotate.decode_episode_cameras"):
                first = segment_episode(args, run, 0, Mock(), factory)
                args.resume = True
                second = segment_episode(args, run, 0, Mock(side_effect=AssertionError), Mock(side_effect=AssertionError))
            self.assertEqual(first["fingerprint"], second["fingerprint"])
            self.assertEqual(factory.call_count, 1)

    def test_mock_captioning_hierarchy_sampling_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = build_parser().parse_args(["--output-dir", tmp])
            run = dict(resolved=dict(video="video-sha", image="image-sha"))
            record = episode()
            for node in record["nodes"]:
                node.pop("caption")
            data = dict(n=120, timestamps=[i / 30 for i in range(120)],
                        masks={"top": np.ones(120, bool)},
                        frames={"top": [np.zeros((10, 20, 3), np.uint8)] * 120})
            worker = Mock()
            worker.caption.side_effect = lambda requests, tokens: [dict(text="caption", truncated=False) for r in requests]
            with patch("robot101.data.annotate.load_episode_metadata", return_value=data), \
                 patch("robot101.data.annotate.decode_episode_cameras"), \
                 patch("robot101.perception.annotation_models.release"):
                for kind in ("video", "image"):
                    caption_records(args, run, [record], Mock(), kind, Mock(return_value=worker))
                for node in record["nodes"]:
                    indices = node["caption"]["sample_frames"]
                    if node["caption_kind"] == "video":
                        self.assertEqual(indices, uniform_frames(node, 32))
                    else:
                        self.assertEqual(indices, [midpoint(node)])
                # Reload clean segment records, as the real command does.
                for node in record["nodes"]:
                    node.pop("caption")
                args.resume = True
                for kind in ("video", "image"):
                    caption_records(args, run, [record], Mock(side_effect=AssertionError), kind,
                                    Mock(side_effect=AssertionError))


class AggregationTests(unittest.TestCase):
    def test_descendant_and_root_evidence(self):
        record = episode()
        root = next(n for n in record["nodes"] if n["id"] == record["root_id"])
        evidence = build_evidence(record, root)
        self.assertEqual([n["node_id"] for n in evidence["descendants_depth_first"]],
                         [n["id"] for n in descendants(record, root["id"])])
        self.assertEqual(evidence["global_context"][0]["node_id"], root["id"])
        self.assertEqual(evidence["task_context"], record["task"])

    def test_rounds_repair_usage_and_completed_resume(self):
        record = episode()
        node = record["nodes"][-1]
        config = dict(model="mock", rounds=3, max_tokens=2048, prompt_version="test")
        client = FakeClient(node, ["not json"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "node.json"
            result = aggregate_node(record, node, path, config, client)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(len(result["rounds"]), 3)
            self.assertEqual(len(result["requests"]), 4)
            self.assertEqual(result["rounds"][-1]["annotation"]["end_frame"], node["end_frame"])
            resumed = aggregate_node(record, node, path, config, FakeClient(node), True)
            self.assertEqual(resumed, result)
            with self.assertRaises(ValueError):
                aggregate_node(record, node, path, dict(config, rounds=4), client, True)

    def test_partial_request_checkpoint_prevents_duplicate_call(self):
        record = episode()
        node = record["nodes"][-1]
        config = dict(model="mock", rounds=1)
        client = FakeClient(node)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "node.json"
            result = aggregate_node(record, node, path, config, client)
            # Simulate interruption after response persistence, before round completion.
            result.update(rounds=[], status="running")
            result.pop("annotation")
            atomic_json(path, result)
            again = FakeClient(node)
            resumed = aggregate_node(record, node, path, config, again, True)
            self.assertEqual(again.calls, 0)
            self.assertEqual(resumed["status"], "ok")

    def test_invalid_output_failed_then_resume(self):
        record = episode()
        node = record["nodes"][-1]
        config = dict(model="mock", rounds=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "node.json"
            result = aggregate_node(record, node, path, config, FakeClient(node, ["bad", "also bad"]))
            self.assertEqual(result["status"], "failed")
            self.assertEqual(len(result["rounds"]), 0)
            resumed = aggregate_node(record, node, path, config, FakeClient(node), True)
            self.assertEqual(resumed["status"], "ok")

    def test_mid_round_interruption_reuses_previous_rounds(self):
        record = episode()
        node = record["nodes"][-1]
        config = dict(model="mock", rounds=3)
        client = FakeClient(node)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "node.json"
            original = aggregate_node(record, node, path, config, client)
            original.update(rounds=original["rounds"][:1], requests=original["requests"][:1], status="running")
            atomic_json(path, original)
            again = FakeClient(node)
            result = aggregate_node(record, node, path, config, again, True)
            self.assertEqual(len(result["rounds"]), 3)
            self.assertEqual(again.calls, 2)

    def test_missing_captions_and_suggested_changes_require_review(self):
        record = episode()
        node = record["nodes"][-1]
        node["caption"]["status"] = "failed"
        changed = annotation(node)
        changed["robotics"]["suggested_changes"] = dict(destination="table")
        self.assertEqual(validate_annotation(json.dumps(changed), node)["robotics"]["review_status"], "needs_review")
        with tempfile.TemporaryDirectory() as tmp:
            result = aggregate_node(record, node, Path(tmp) / "node.json", dict(model="mock", rounds=1), FakeClient(node))
            self.assertEqual(result["annotation"]["robotics"]["review_status"], "needs_review")

    def test_dry_run_needs_no_api_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atomic_json(root / "input" / "episodes" / "ep000000.json", episode())
            with patch.dict("os.environ", {}, clear=True):
                aggregate_main(["--input-dir", str(root / "input"), "--output-dir", str(root / "output"), "--dry-run"])
            self.assertTrue(list((root / "output" / "request_previews").glob("*.json")))
            self.assertFalse((root / "output" / "nodes").exists())

    def test_aggregation_command_exports_and_resumes_without_api(self):
        class TreeClient:
            calls = 0
            def complete(self, prompt, log):
                evidence = json.loads(prompt.split("EVIDENCE:\n", 1)[1].split("\nPREVIOUS ANNOTATION:", 1)[0])
                current = evidence["current"]
                node = dict(id=current["node_id"], start_frame=current["start_frame"], end_frame=current["end_frame"])
                self.calls += 1
                text = json.dumps(annotation(node))
                log(dict(status="ok", text=text, usage=dict(input_tokens=5, output_tokens=3)))
                return text
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = episode()
            atomic_json(root / "input" / "episodes" / "ep000000.json", record)
            client = TreeClient()
            argv = ["--input-dir", str(root / "input"), "--output-dir", str(root / "output"), "--rounds", "1"]
            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "mock"}), \
                 patch("robot101.data.aggregate_annotations.ClaudeClient", return_value=client):
                aggregate_main(argv)
            self.assertEqual(client.calls, len(record["nodes"]))
            summary = json.loads((root / "output" / "summary.json").read_text())
            self.assertEqual(summary["failures"], 0)
            self.assertEqual(summary["usage"]["input_tokens"], 5 * len(record["nodes"]))
            selected = json.loads((root / "output" / "annotations_subtasks.jsonl").read_text())
            self.assertEqual(len(selected["nodes"]), len(record["selected_node_ids"]))
            with patch.dict("os.environ", {}, clear=True), \
                 patch("robot101.data.aggregate_annotations.ClaudeClient", side_effect=AssertionError):
                aggregate_main(argv + ["--resume"])

    def test_transient_retry_logs_attempts(self):
        class RateLimit(Exception):
            status_code = 429
            response = Mock(headers={"retry-after": "3"})
        client = object.__new__(ClaudeClient)
        client.model, client.max_tokens, client.attempts = "mock", 2048, 5
        response = Mock(content=[Mock(type="text", text="{}")], model="mock", id="id", stop_reason="end_turn")
        response.usage.model_dump.return_value = dict(input_tokens=1, output_tokens=2)
        client.client = Mock()
        client.client.messages.create.side_effect = [RateLimit(), response]
        log, sleep = [], Mock()
        self.assertEqual(client.complete("prompt", log.append, sleep), "{}")
        self.assertEqual([r["status"] for r in log], ["failed", "ok"])
        sleep.assert_called_once_with(3.0)


if __name__ == "__main__":
    unittest.main()
