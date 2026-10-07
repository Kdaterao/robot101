"""Batched LeRobot episode decoding for offline point preprocessing."""
from __future__ import annotations

from time import perf_counter
import numpy as np

from hf_preprocess_smolvla import (
    GRIPPER_INDEX, _as_numpy, _cam_key, _mask_is_real,
)


def _frame_batches(indices, batch_size):
    """Keep sparse tail windows apart so PyAV need not decode gaps."""
    batch = []
    for index in sorted(set(indices)):
        if batch and (index != batch[-1] + 1 or len(batch) >= batch_size):
            yield batch
            batch = []
        batch.append(index)
    if batch:
        yield batch


def load_episode_metadata(ds, cameras):
    """Read state/actions/task text without triggering per-frame video seeks."""
    start = perf_counter()
    rows = ds.hf_dataset
    n = len(rows)
    if not n:
        raise ValueError('empty episode')
    states, actions, tasks, timestamps, episodes = [], [], [], [], []
    masks = {cam: [] for cam in cameras}
    for row in rows:
        states.append(_as_numpy(row['observation.state']).astype(np.float32).reshape(-1))
        actions.append(_as_numpy(row['action']).astype(np.float32).reshape(-1))
        timestamps.append(float(_as_numpy(row['timestamp']).item()))
        episodes.append(int(_as_numpy(row['episode_index']).item()))
        if 'task' in row:
            tasks.append(str(row['task']))
        else:
            task_index = int(_as_numpy(row['task_index']).item())
            tasks.append(str(ds.meta.tasks.iloc[task_index].name))
        for cam in cameras:
            masks[cam].append(_cam_key(cam) in ds.meta.features and _mask_is_real(row, cam))
    if len(set(episodes)) != 1:
        raise ValueError('Expected a dataset filtered to one episode')
    states, actions = np.stack(states), np.stack(actions)
    result = dict(n=n, states=states, actions=actions, tasks=tasks,
                  gripper=states[:, GRIPPER_INDEX] if states.shape[1] > GRIPPER_INDEX else np.zeros(n, np.float32),
                  action_gripper=actions[:, GRIPPER_INDEX] if actions.shape[1] > GRIPPER_INDEX else np.zeros(n, np.float32),
                  masks={cam: np.asarray(mask, dtype=bool) for cam, mask in masks.items()},
                  frames={}, timestamps=timestamps, episode_index=episodes[0])
    print(f'  metadata: {n} frames in {perf_counter()-start:.1f}s', flush=True)
    return result


def decode_episode_cameras(ds, data, cameras, *, indices=None, batch_size=64, backend='pyav'):
    """Decode timestamp batches as uint8, preserving episode offsets and masks.

    Unrequested slots are None for sparse decoding; callers may only access the
    requested tail windows. Full decoding returns the original list layout.
    """
    from lerobot.datasets.video_utils import decode_video_frames
    if batch_size < 1:
        raise ValueError('decode batch size must be positive')
    requested = list(range(data['n'])) if indices is None else sorted(set(indices))
    if any(i < 0 or i >= data['n'] for i in requested):
        raise ValueError('Frame index outside episode')
    episode = data['episode_index']
    for cam in cameras:
        start = perf_counter()
        key = _cam_key(cam)
        frames = [None] * data['n']
        if key not in ds.meta.features:
            missing = np.zeros((480, 640, 3), dtype=np.uint8)
            for i in requested:
                frames[i] = missing
        elif key not in ds.meta.video_keys:
            # Image-backed datasets keep images directly in the HF rows.
            from hf_preprocess_smolvla import _tensor_to_rgb
            for i in requested:
                frames[i] = _tensor_to_rgb(ds.hf_dataset[i][key])
        else:
            path = ds.root / ds.meta.get_video_file_path(episode, key)
            offset = float(ds.meta.episodes[episode][f'videos/{key}/from_timestamp'])
            for batch in _frame_batches(requested, batch_size):
                timestamps = [offset + data['timestamps'][i] for i in batch]
                decoded = decode_video_frames(
                    path, timestamps, ds.tolerance_s, backend=backend, return_uint8=True)
                # Copies release the entire batch tensor when it goes out of scope.
                for i, image in zip(batch, decoded, strict=True):
                    frames[i] = image.permute(1, 2, 0).cpu().numpy().copy()
                del decoded
        data['frames'][cam] = frames
        print(f'  decode {cam}: {len(requested)} frames in {perf_counter()-start:.1f}s', flush=True)
    return data
