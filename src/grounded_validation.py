"""Shape checks and schema-aware LeRobot statistics validation."""
import json
from pathlib import Path
import numpy as np


class EpisodeShapeError(ValueError):
    """An episode cannot be represented by the declared output schema."""


def validate_episode_shapes(data, features, cameras):
    n = data['n']
    for name, values in [('observation.state', data['states']), ('action', data['actions'])]:
        if name in features:
            expected = (n, *features[name]['shape'])
            if tuple(values.shape) != expected:
                raise EpisodeShapeError(f'{name}: expected {expected}, got {values.shape}')
    for camera in cameras:
        key = f'observation.images.{camera}'
        if key not in features or camera not in data['frames']:
            continue
        shape = tuple(features[key]['shape'])
        if len(shape) != 3:
            raise EpisodeShapeError(f'{key}: expected declared CHW or HWC image shape, got {shape}')
        # Decoded frames are HWC; source schemas may declare either layout.
        expected = []
        if shape[-1] in (1, 3, 4):
            expected.append(shape)
        if shape[0] in (1, 3, 4):
            expected.append((shape[1], shape[2], shape[0]))
        if not expected:
            raise EpisodeShapeError(f'{key}: cannot identify channel axis in declared shape {shape}')
        for index, image in enumerate(data['frames'][camera]):
            if image is not None and tuple(image.shape) not in expected:
                raise EpisodeShapeError(f'{key} frame {index}: expected HWC {expected}, got {image.shape}')


def record_skipped_episode(path, episode, phase, reason):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(episode=int(episode), phase=phase, reason=str(reason))
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record)+'\n')
    print(f"[skip ep {episode}] {phase}: {reason}", flush=True)


def validate_stat_value(value, key, feature_key, features):
    """A padding mask is numeric, even when its name contains 'images'."""
    if not isinstance(value, np.ndarray):
        raise ValueError(f'{feature_key}/{key}: statistic must be a numpy array')
    if value.ndim == 0:
        raise ValueError(f'{feature_key}/{key}: statistic must have at least one dimension')
    if key == 'count' and value.shape != (1,):
        raise ValueError(f'{feature_key}/count: expected (1,), got {value.shape}')
    is_visual = features.get(feature_key, {}).get('dtype') in {'image', 'video'}
    if is_visual and key != 'count' and value.shape not in ((3, 1, 1), (1, 1, 1)):
        raise ValueError(f'{feature_key}/{key}: invalid image statistic shape {value.shape}')


def install_statistics_validation(features):
    """Use feature dtypes to validate stats in this preprocessing process only."""
    from lerobot.datasets import compute_stats
    original = compute_stats._validate_stat_value
    original = getattr(original, '_original_validator', original)
    def validate(value, key, feature_key):
        if feature_key in features:
            validate_stat_value(value, key, feature_key, features)
        else:
            original(value, key, feature_key)
    validate._original_validator = original
    compute_stats._validate_stat_value = validate
