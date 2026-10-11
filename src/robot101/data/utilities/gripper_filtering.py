"""Persist mask sidecars and exact-frame human correction requests."""
from pathlib import Path
import json
from time import perf_counter
import cv2
import numpy as np
from robot101.perception.gripper_masks import BODY_PROMPT, valid_points, filter_candidates


def prepare_gripper_filter(entry, frames, real, molmo, sam, root, episode, subtask,
                           corrections, max_weight, radius_fraction, goal_tail_frames, prompt_result=None):
    start = entry['tail_start']
    length = entry['tracks'].shape[1]
    clip = frames[start:start+length]
    real = np.asarray(real[start:start+length], bool)
    key = f'{episode}:{subtask}'
    stem = f'ep{episode:06d}_stage{subtask:03d}'
    review_dir = root / 'mask_review'
    review_dir.mkdir(parents=True, exist_ok=True)
    valid = np.flatnonzero(real)
    seed = int(valid[0]) if len(valid) else 0
    points, labels, source = [], [], 'molmo_body_prompt'
    metadata = {'status': 'failed', 'manual_review_required': True,
                'camera': 'wrist', 'episode': int(episode), 'subtask': int(subtask),
                'window': [start, start+length-1], 'model': getattr(sam, 'model_id', None),
                'mask_validation': {'min_pixels': 8, 'max_area_fraction': 0.8},
                'max_weight': max_weight,
                'radius_fraction': radius_fraction, 'correction_key': key}
    try:
        if not len(valid) or not real.all():
            raise ValueError('missing_wrist_mask_frame')
        h, w = clip[0].shape[:2]
        correction = corrections.get(key)
        if correction is not None:
            seed = int(correction['frame_index']) - start
            if seed < 0 or seed >= length or not real[seed]:
                raise ValueError('correction_frame_outside_clustering_window')
            points = valid_points({'points_xy': correction['points_xy']}, w, h)
            if len(points) != len(correction['points_xy']):
                raise ValueError('invalid_correction_coordinates')
            labels = correction.get('labels', [1]*len(points))
            source = 'human_mask_prompt'
        else:
            points = valid_points(prompt_result if prompt_result is not None else molmo.ground(clip[seed], BODY_PROMPT), w, h)
            labels = [1]*len(points)
        if not len(points):
            raise ValueError('no_visible_gripper_body_points')
        started = perf_counter()
        print(f"  SAM 2 ep{episode}/stage{subtask}: {length} clustering frames", flush=True)
        masks = sam.propagate(clip, seed, points, labels)
        metadata['mask_seconds'] = perf_counter()-started
        if masks.shape != (length, h, w):
            raise ValueError('invalid_mask_shape')
        area = masks.mean(axis=(1, 2))
        # Reject empty or near-full-frame segmentations; these checks do not
        # assert semantic correctness. Human inspection is still necessary.
        if (masks.sum(axis=(1, 2)) < 8).any() or (area > 0.8).any():
            raise ValueError('empty_or_excessive_gripper_mask')
        positive = np.asarray(labels) == 1
        px = np.rint(points[positive]).astype(int)
        px[:, 0] = np.clip(px[:, 0], 0, w-1)
        px[:, 1] = np.clip(px[:, 1], 0, h-1)
        if not len(px) or not masks[seed, px[:, 1], px[:, 0]].all():
            raise ValueError('mask_misses_positive_prompt')
        filtered, visibility, weights, eligible, endpoints = filter_candidates(
            entry['tracks'], entry['visible'], masks, max_weight, radius_fraction, goal_tail_frames)
        entry.update(tracks=filtered, visible=visibility, observation_weights=weights,
                     eligible=eligible, endpoints=endpoints)
        directory = root / 'gripper_masks'
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f'{stem}.npz'
        np.savez_compressed(path, packed_masks=np.packbits(masks, axis=-1),
                            shape=np.asarray(masks.shape), frame_indices=np.arange(start, start+length))
        previews = []
        for t in dict.fromkeys([seed, length-1]):
            preview = clip[t].copy()
            preview[masks[t]] = (0.6*preview[masks[t]] + 0.4*np.array([255, 0, 255])).astype(np.uint8)
            cv2.putText(preview, f"frame {start+t}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            previews.append(preview)
        preview_path = directory / f'{stem}_preview.jpg'
        cv2.imwrite(str(preview_path), cv2.cvtColor(np.concatenate(previews, axis=1), cv2.COLOR_RGB2BGR))
        metadata.update(status='ok', manual_review_required=False,
                        preview_file=preview_path.relative_to(root).as_posix(),
                        mask_file=path.relative_to(root).as_posix(), retained_candidates=int(eligible.sum()),
                        area_fraction=area.tolist())
        pending = review_dir / f'{stem}.json'
        if pending.exists():
            pending.unlink()
    except Exception as exc:
        metadata['reason'] = str(exc)
        # Do not let unresolved masks contribute to shared clustering.
        entry['visible'][:] = False
        entry['eligible'] = np.zeros(len(entry['tracks']), bool)
        review = review_dir / f'{stem}.json'
        if clip and 0 <= seed < length and isinstance(clip[seed], np.ndarray):
            image = review_dir / f'{stem}.png'
            cv2.imwrite(str(image), cv2.cvtColor(clip[seed], cv2.COLOR_RGB2BGR))
            metadata['review_image'] = image.name
        review.write_text(json.dumps({**metadata, 'frame_index': start+seed,
                                     'prompt': BODY_PROMPT}, indent=2)+'\n')
    metadata.update(seed_frame=start+seed, prompt=BODY_PROMPT, prompt_source=source,
                    points_xy=np.asarray(points).reshape(-1, 2).tolist(), labels=list(labels))
    print(f"  SAM 2 ep{episode}/stage{subtask}: {metadata['status']} {metadata.get('reason', '')}", flush=True)
    entry['gripper_filter'] = metadata
