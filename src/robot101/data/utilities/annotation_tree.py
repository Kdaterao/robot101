"""CPU-only temporal hierarchy, sampling, and atomic annotation artifacts.

Bounds everywhere are half-open source-frame intervals. No model imports here.
"""
from __future__ import annotations

import hashlib
import heapq
import html
import json
import os
from pathlib import Path
import tempfile

import numpy as np

SCHEMA_VERSION = 1


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as f:
        temporary = Path(f.name)
        try:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def cached_json(path, expected, resume):
    path = Path(path)
    if not path.exists():
        return None
    if not resume:
        raise ValueError(f"Artifact exists: {path}; use --resume or a new output directory")
    value = json.loads(path.read_text())
    if value.get("fingerprint") != expected:
        raise ValueError(f"Incompatible cached artifact: {path}; use a new output directory")
    return value


def sampled_frames(n, step=4):
    if n < 1 or step < 1:
        raise ValueError("Frame count and sampling step must be positive")
    return np.unique([*range(0, n, step), n - 1]).astype(np.int64)


def windows(count, size=64, stride=8, tubelet=2):
    if count < 1 or size < tubelet or stride < 1 or size % tubelet or stride % tubelet or stride > size:
        raise ValueError("Window/stride must be positive tubelet multiples, with stride <= window")
    starts = list(range(0, max(1, count - size + 1), stride))
    if starts[-1] + size < count:
        starts.append(starts[-1] + stride)
    result = []
    for start in starts:
        real = min(size, count - start)
        positions = np.minimum(np.arange(start, start + size), count - 1)
        result.append((positions, np.arange(size) < real))
    return result


def pool_tokens(tokens, temporal_count, spatial_count):
    tokens = np.asarray(tokens)
    if tokens.ndim != 3 or tokens.shape[1] != temporal_count * spatial_count:
        raise ValueError(f"Unexpected JEPA token layout {tokens.shape}; expected B,{temporal_count * spatial_count},D")
    return tokens.reshape(tokens.shape[0], temporal_count, spatial_count, tokens.shape[-1]).mean(axis=2)


def average_observations(count, observations, tubelet=2):
    """Assign each tubelet feature to its real sampled frames; ignore padding."""
    sums = counts = None
    for positions, valid, features in observations:
        features = np.asarray(features, dtype=np.float64)
        if len(positions) != len(features) * tubelet or len(valid) != len(positions):
            raise ValueError("Tubelet/window alignment mismatch")
        if sums is None:
            sums = np.zeros((count, features.shape[-1]), np.float64)
            counts = np.zeros(count, np.int64)
        for offset, (index, real) in enumerate(zip(positions, valid, strict=True)):
            if real:
                sums[index] += features[offset // tubelet]
                counts[index] += 1
    if sums is None or np.any(counts == 0) or not np.isfinite(sums).all():
        raise ValueError("Missing or nonfinite temporal observations")
    return (sums / counts[:, None]).astype(np.float32), counts


def midpoint(node):
    return (node["start_frame"] + node["end_frame"] - 1) // 2


def uniform_frames(node, count=32):
    if count < 1:
        raise ValueError("Video sample count must be positive")
    # Repeated indices in very short spans are intentional and recorded.
    return np.rint(np.linspace(node["start_frame"], node["end_frame"] - 1, count)).astype(int).tolist()


def silhouette(features, labels):
    labels = np.asarray(labels)
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        return -1.0
    x = np.asarray(features, np.float64)
    squared = np.sum(x * x, axis=1)
    distance = np.sqrt(np.maximum(0, squared[:, None] + squared[None, :] - 2 * (x @ x.T)))
    scores = []
    for i, label in enumerate(labels):
        same = labels == label
        if same.sum() == 1:
            scores.append(0.0)
            continue
        a = distance[i, same].sum() / (same.sum() - 1)
        b = min(distance[i, labels == other].mean() for other in unique if other != label)
        scores.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return float(np.mean(scores))


def build_tree(features, frame_ids, n_frames, fps, *, min_duration=0.5, max_segments=8, threshold=0.15):
    """Full adjacency-constrained Ward tree, followed by retention and a cut.

    Ward weights count sampled-frame observations, not source-frame durations.
    The heap holds only neighboring pairs; stale entries are ignored.
    """
    x = np.asarray(features, np.float64)
    ids = np.asarray(frame_ids, np.int64)
    if x.ndim != 2 or len(x) != len(ids) or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected finite N x D features aligned with sampled frames")
    if (ids[0] != 0 or ids[-1] != n_frames - 1 or np.any(np.diff(ids) <= 0) or
            not np.isfinite(fps) or fps <= 0 or min_duration < 0 or max_segments < 1):
        raise ValueError("Invalid source-frame coverage or FPS")
    nodes, active, heap = {}, {}, []
    for i, frame in enumerate(ids):
        nodes[i] = dict(id=f"n{i:06d}", start_frame=int(frame),
                        end_frame=int(ids[i + 1]) if i + 1 < len(ids) else n_frames,
                        children=[], parent=None, sample_start=i, sample_end=i + 1, merge_cost=0.0)
        active[i] = dict(mean=x[i], weight=1, left=i - 1 if i else None,
                         right=i + 1 if i + 1 < len(ids) else None)

    def offer(left, right):
        if left is None or right is None:
            return
        a, b = active[left], active[right]
        cost = a["weight"] * b["weight"] / (a["weight"] + b["weight"]) * np.sum((a["mean"] - b["mean"]) ** 2)
        heapq.heappush(heap, (float(cost), left, right))

    for i in range(len(ids) - 1):
        offer(i, i + 1)
    cuts = {}
    if len(active) <= max_segments:
        cuts[len(active)] = list(active)
    while len(active) > 1:
        cost, left, right = heapq.heappop(heap)
        if left not in active or right not in active or active[left]["right"] != right:
            continue
        a, b = active.pop(left), active.pop(right)
        index = len(nodes)
        weight = a["weight"] + b["weight"]
        active[index] = dict(mean=(a["mean"] * a["weight"] + b["mean"] * b["weight"]) / weight,
                             weight=weight, left=a["left"], right=b["right"])
        nodes[index] = dict(id=f"n{index:06d}", start_frame=nodes[left]["start_frame"],
                            end_frame=nodes[right]["end_frame"], children=[nodes[left]["id"], nodes[right]["id"]],
                            parent=None, sample_start=nodes[left]["sample_start"],
                            sample_end=nodes[right]["sample_end"], merge_cost=cost)
        nodes[left]["parent"] = nodes[right]["parent"] = nodes[index]["id"]
        if a["left"] is not None:
            active[a["left"]]["right"] = index
        if b["right"] is not None:
            active[b["right"]]["left"] = index
        offer(a["left"], index)
        offer(index, b["right"])
        if len(active) <= max_segments:
            cuts[len(active)] = sorted(active, key=lambda j: nodes[j]["start_frame"])
    root = next(iter(active))
    retained = {i for i, node in nodes.items() if (node["end_frame"] - node["start_frame"]) / fps > min_duration}
    full = {node["id"]: node for node in nodes.values()}
    kept = []
    for i in sorted(retained):
        node = dict(nodes[i], children=[])
        parent = node["parent"]
        while parent and int(parent[1:]) not in retained:
            parent = full[parent]["parent"]
        node["parent"] = parent
        kept.append(node)
    by_id = {node["id"]: node for node in kept}
    for node in kept:
        if node["parent"]:
            by_id[node["parent"]]["children"].append(node["id"])
    for node in kept:
        node["children"].sort(key=lambda child: by_id[child]["start_frame"])
        node["caption_kind"] = "video" if node["children"] else "image"
    selected, best, scores = [root] if root in retained else [], -1.0, {}
    for k in sorted(cuts):
        if k < 2 or k >= len(x) or not all(i in retained for i in cuts[k]):
            continue
        labels = np.zeros(len(x), int)
        for label, index in enumerate(cuts[k]):
            labels[nodes[index]["sample_start"]:nodes[index]["sample_end"]] = label
        score = silhouette(x, labels)
        scores[str(k)] = score
        if score >= threshold and score > best + 1e-9:
            selected, best = cuts[k], score
    return dict(root_id=nodes[root]["id"] if root in retained else None,
                hierarchy=list(nodes.values()), nodes=kept,
                selected_node_ids=[nodes[i]["id"] for i in selected],
                cut_scores=scores, review_required=not bool(kept),
                review_reason="episode_below_duration_threshold" if not kept else None)


def descendants(tree, node_id):
    lookup = {node["id"]: node for node in tree["nodes"]}
    result = []
    def visit(index):
        for child in lookup[index]["children"]:
            result.append(lookup[child])
            visit(child)
    visit(node_id)
    return result


def write_exports(root, records, prefix="captions"):
    root = Path(root)
    for name, rows in ((f"{prefix}.jsonl", records),
                       (f"{prefix}_subtasks.jsonl", [dict(record, nodes=[n for n in record["nodes"]
                         if n["id"] in record["selected_node_ids"]]) for record in records])):
        target = root / name
        with tempfile.NamedTemporaryFile(mode="w", dir=root, delete=False) as f:
            for row in rows:
                f.write(json.dumps(row, allow_nan=False) + "\n")
            temporary = f.name
        os.replace(temporary, target)


def write_report(root, record):
    root = Path(root)
    rows = []
    for node in sorted(record["nodes"], key=lambda n: (n["start_frame"], n["end_frame"])):
        caption = node.get("caption", {})
        sample = caption.get("sample_frames", [])
        views = caption.get("views") or [dict(camera=record.get("camera", "primary"), image_subdir="")]
        images = ""
        for view in views:
            subdir = str(view.get("image_subdir", ""))
            if subdir and (not subdir.replace("_", "").replace("-", "").isalnum()):
                continue
            images += f'<p>{html.escape(str(view.get("camera", "primary")))}</p>'
            images += "".join(f'<img width="160" src="../images/ep{record["episode_index"]:06d}/'
                              f'{html.escape(subdir + "/" if subdir else "", quote=True)}f{i:06d}.jpg">'
                              for i in dict.fromkeys(sample))
        rows.append(f'<tr><td>{node["id"]}</td><td>{node["start_frame"]}–{node["end_frame"]}</td>'
                    f'<td>{"selected" if node["id"] in record["selected_node_ids"] else "context"}</td>'
                    f'<td>{html.escape(caption.get("text", caption.get("error", "pending")))}</td><td>{images}</td></tr>')
    report = root / "reports" / f'ep{record["episode_index"]:06d}.html'
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text('<!doctype html><meta charset="utf-8"><title>Annotation inspection</title>'
                      '<style>td{border:1px solid #aaa;padding:6px}body{font-family:sans-serif}</style>'
                      f'<h1>Episode {record["episode_index"]}</h1><p>Bounds: [start,end). '
                      f'Task: {html.escape(record.get("task", ""))}</p><table>' + "".join(rows) + '</table>')
