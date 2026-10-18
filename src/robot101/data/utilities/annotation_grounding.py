"""Map annotation artifacts back to pinned LeRobot episodes and exact stages."""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np

from robot101.data.utilities.annotation_tree import atomic_json, fingerprint
from robot101.data.utilities.stages import Stage


def _label(node, field, prefer_aggregation):
    state = node.get("aggregation", {})
    annotation = state.get("annotation", {}) if state.get("status") == "ok" else {}
    if field == "auto":
        if prefer_aggregation and str(annotation.get("brief_action") or "").strip():
            return str(annotation["brief_action"]).strip(), "aggregation.brief_action"
        field = "caption"
    if field == "caption":
        caption = node.get("caption", {})
        text = str(caption.get("text") or "").strip() if caption.get("status") == "ok" else ""
        if text:
            return text, "caption.text"
    elif str(annotation.get(field) or "").strip():
        return str(annotation[field]).strip(), f"aggregation.{field}"
    raise ValueError(f"Node {node.get('id')} has no completed {field} text for grounding")


class AnnotationGroundingInput:
    def __init__(self, records, origins, text_field="auto"):
        self.records, self.origins, self.text_field = records, origins, text_field
        sources = {(r["source_repo_id"], r["source_revision"]) for r in records.values()}
        if len(sources) != 1:
            raise ValueError("One grounding destination requires one source dataset/revision; split mixed sources into separate runs")
        self.source_repo_id, self.source_revision = next(iter(sources))
        if not re.fullmatch(r"[0-9a-fA-F]{40,64}", str(self.source_revision)):
            raise ValueError("Annotations must reference an immutable source revision SHA")
        self.stages = {}
        for episode, record in records.items():
            lookup = {node["id"]: node for node in record["nodes"]}
            if len(lookup) != len(record["nodes"]):
                raise ValueError(f"Duplicate node IDs in episode {episode}")
            ids = record["selected_node_ids"]
            if not ids or len(ids) != len(set(ids)) or any(i not in lookup for i in ids):
                raise ValueError(f"Invalid selected subtask IDs in episode {episode}")
            nodes = sorted((lookup[i] for i in ids), key=lambda n: n["start_frame"])
            cursor, stages = 0, []
            for node in nodes:
                start, end = node["start_frame"], node["end_frame"]
                if (type(start) is not int or type(end) is not int or start != cursor
                        or not start < end <= record["n_frames"]):
                    raise ValueError(f"Episode {episode}: selected intervals must cover the episode contiguously without overlaps/gaps")
                text, text_source = _label(node, text_field, origins[episode]["stage"] == "aggregation")
                state = node.get("aggregation", {})
                robotics = state.get("annotation", {}).get("robotics", {}) if state.get("status") == "ok" else {}
                entities = [robotics[key] for key in ("manipulated_object", "destination")
                            if isinstance(robotics.get(key), str) and robotics[key].strip()]
                stages.append(dict(node_id=node["id"], start_frame=start, end_frame=end,
                    text=text, text_source=text_source, entities=entities,
                    review_status=robotics.get("review_status", "unreviewed"),
                    aggregation_status=state.get("status", "not_available"),
                    caption_status=node.get("caption", {}).get("status", "missing"),
                    caption_truncated=node.get("caption", {}).get("truncated", False),
                    annotation=node))
                cursor = end
            if cursor != record["n_frames"]:
                raise ValueError(f"Episode {episode}: selected intervals do not cover all frames")
            self.stages[episode] = stages

    def manifest(self):
        value = dict(mode="annotation_selected_subtasks", source_repo_id=self.source_repo_id,
                     source_revision=self.source_revision, text_field=self.text_field,
                     annotation_origins=self.origins,
                     annotation_record_fingerprints={ep: fingerprint(record) for ep, record in self.records.items()},
                     selected_subtasks={ep: [{k: v for k, v in stage.items() if k != "annotation"}
                                            for stage in stages] for ep, stages in self.stages.items()})
        return dict(fingerprint=fingerprint(value), **value)

    def validate_source(self, episode, data, fps):
        record = self.records[episode]
        if data["episode_index"] != episode or data["n"] != record["n_frames"]:
            raise ValueError(f"Episode {episode}: source identity/frame count differs from annotations")
        if not np.isclose(float(record["fps"]), fps, atol=1e-6, rtol=0):
            raise ValueError(f"Episode {episode}: source FPS differs from annotations")
        if record.get("timestamps") is not None and not np.allclose(
                data["timestamps"], record["timestamps"], atol=1e-4, rtol=0):
            raise ValueError(f"Episode {episode}: source timestamps differ from annotations")

    def stage_list(self, episode, data, fps, max_stages, *, events=None, min_stage_frames=1):
        self.validate_source(episode, data, fps)
        if len(self.stages[episode]) > max_stages:
            raise ValueError("Annotation stage count exceeds --max-stages; raise the limit instead of merging boundaries")
        # Grounding uses inclusive Stage ends; annotation bounds remain half-open.
        # No gripper primitive is guessed from free-text action descriptions.
        result = []
        for parent in self.stages[episode]:
            start, stop = parent["start_frame"], parent["end_frame"]
            for event in sorted(events or [], key=lambda e: int(e["frame"])):
                frame = int(event["frame"])
                if not start <= frame < stop or frame - start + 1 < min_stage_frames:
                    continue
                # Event frame belongs to the segment reaching that gripper state.
                # The next segment starts one frame later, avoiding duplicate ownership.
                if frame < stop - 1 and stop - frame - 1 < min_stage_frames:
                    continue
                result.append(Stage(start, frame, event["type"]))
                start = frame + 1
            if start < stop:
                result.append(Stage(start, stop - 1, "none"))
        if len(result) > max_stages:
            raise ValueError("Grounding subdivisions exceed --max-stages; raise the limit instead of merging annotation bounds")
        return result

    def parent_stage(self, episode, stage):
        return next(s for s in self.stages[episode]
                    if s["start_frame"] <= stage.start <= stage.end < s["end_frame"])

    def write_episode(self, root, episode):
        atomic_json(Path(root) / "annotations" / f"ep{episode:06d}.json",
                    dict(origin=self.origins[episode], record=self.records[episode]))


def load_annotation_grounding(args, episodes):
    """Download only annotation JSON sidecars; support disjoint worker repositories."""
    records, origins = {}, {}
    wanted = None if episodes is None else set(episodes)
    def add(record, origin, expected_episode):
        episode = record.get("episode_index")
        if type(episode) is not int or episode != expected_episode:
            raise ValueError("Annotation filename and episode identity disagree")
        if record.get("bounds") != "half_open_source_frames":
            raise ValueError(f"Episode {episode}: expected half_open_source_frames annotation bounds")
        if episode in records:
            raise ValueError(f"Episode {episode} appears in multiple annotation inputs; use disjoint worker outputs")
        records[episode], origins[episode] = record, origin
    for directory in args.annotation_input_dirs or []:
        root = Path(directory)
        stage = args.annotation_stage
        if (root / "episodes").is_dir():
            base = root
        else:
            if stage == "auto":
                stage = "aggregation" if (root / "aggregation" / "episodes").is_dir() else "captioning"
            base = root / stage
        for path in sorted((base / "episodes").glob("ep*.json")):
            match = re.fullmatch(r"ep(\d+)\.json", path.name)
            if not match or (wanted is not None and int(match[1]) not in wanted):
                continue
            record = json.loads(path.read_text())
            actual_stage = stage
            if stage == "auto":
                actual_stage = "aggregation" if any("aggregation" in n for n in record["nodes"]) else "captioning"
            add(record, dict(kind="local", path=str(path.resolve()), stage=actual_stage,
                             content_fingerprint=fingerprint(record)), int(match[1]))
    if args.annotation_repo_ids:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()
        for repo in args.annotation_repo_ids:
            revision = api.repo_info(repo, repo_type="dataset", revision=args.annotation_revision).sha
            files = api.list_repo_files(repo, repo_type="dataset", revision=revision)
            stage = args.annotation_stage
            if stage == "auto":
                stage = "aggregation" if any(re.fullmatch(r"aggregation/episodes/ep\d+\.json", n) for n in files) else "captioning"
            for name in sorted(files):
                match = re.fullmatch(rf"{stage}/episodes/ep(\d+)\.json", name)
                if not match or (wanted is not None and int(match[1]) not in wanted):
                    continue
                path = hf_hub_download(repo, name, repo_type="dataset", revision=revision)
                add(json.loads(Path(path).read_text()), dict(kind="hub", repo_id=repo,
                    revision=revision, path=name, stage=stage), int(match[1]))
    if not records:
        raise ValueError("No selected episode JSON sidecars found in annotation inputs")
    if wanted is not None and wanted - records.keys():
        raise ValueError(f"Requested annotation episodes missing: {sorted(wanted - records.keys())}")
    return AnnotationGroundingInput(records, origins, args.annotation_text)


def check_grounding_annotations(root, manifest):
    """Prevent mixing incompatible annotation boundaries or source revisions on resume."""
    root = Path(root)
    marker = root / "meta" / "grounding_annotations.json"
    if marker.exists():
        if manifest is None or json.loads(marker.read_text())["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Destination annotation input differs; choose a new --dst-repo-id")
    elif manifest is not None and (root / "meta" / "info.json").exists():
        if json.loads((root / "meta" / "info.json").read_text()).get("total_episodes", 0):
            raise ValueError("Cannot resume raw gripper-stage output with annotation stages; choose a new destination")
