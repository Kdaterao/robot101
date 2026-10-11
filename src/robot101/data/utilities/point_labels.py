"""Single- and multiple-point label compatibility and coordinate validation."""
from __future__ import annotations
import math
from numbers import Real


def point_pairs(value) -> list[list[float]]:
    """Normalize a legacy [x,y] or an N-by-2 array to a nonempty point list."""
    if isinstance(value, (str, bytes)) or value is None:
        raise ValueError('Expected a point or list of points')
    try:
        rows = list(value)
        if len(rows) == 2 and all(isinstance(v, Real) for v in rows):
            rows = [rows]
        if not rows:
            raise ValueError('At least one point is required')
        result = []
        for pair in rows:
            if isinstance(pair, (str, bytes)) or len(pair) != 2:
                raise ValueError('Each point must have exactly two coordinates')
            x, y = float(pair[0]), float(pair[1])
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError('Point coordinates must be finite')
            result.append([x, y])
        return result
    except (TypeError, OverflowError) as exc:
        raise ValueError('Expected an N-by-2 point array') from exc


def _normalized_pixels(value, width, height):
    points = point_pairs(value)
    if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in points):
        raise ValueError('Normalized points must be in [0, 1]')
    return [[x*max(0, width-1), y*max(0, height-1)] for x, y in points]


def pixel_points(row, width: int, height: int) -> list[list[float]]:
    """Read current plural fields or legacy singleton fields without truncation."""
    if width <= 0 or height <= 0:
        raise ValueError('Image dimensions must be positive')
    if 'points_xy' in row:
        points = point_pairs(row['points_xy'])
    elif 'points_xy_norm' in row:
        points = _normalized_pixels(row['points_xy_norm'], width, height)
    elif row.get('point_xy') is not None:
        points = point_pairs(row['point_xy'])
    else:
        points = _normalized_pixels(row['point_xy_norm'], width, height)
    if any(not (0 <= x < width and 0 <= y < height) for x, y in points):
        raise ValueError('Point lies outside the source image')
    return points
