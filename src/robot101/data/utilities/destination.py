"""Resume locally complete datasets and preserve empty interrupted outputs."""
from datetime import datetime, timezone
import json
from pathlib import Path


def recover_empty_destination(root):
    """Move an initialized output aside only when no episode data was written."""
    root = Path(root)
    info_path = root / 'meta/info.json'
    if not info_path.is_file():
        return None
    info = json.loads(info_path.read_text())
    progress = root / '_preprocess_done.json'
    completed = json.loads(progress.read_text()).get('episodes', []) if progress.is_file() else []
    has_payload = any((root / name).exists() and any(p.is_file() for p in (root / name).rglob('*'))
                      for name in ('data', 'videos', 'images'))
    if info.get('total_episodes') != 0 or info.get('total_frames') != 0 or completed or has_payload:
        return None
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    backup = root.with_name(root.name + '.empty-' + stamp)
    root.rename(backup)
    print(f'Empty destination preserved at {backup}; creating a fresh dataset.', flush=True)
    return backup


def check_resume_metadata(root):
    root = Path(root)
    missing = [name for name in ('info.json', 'tasks.parquet', 'stats.json')
               if not (root / 'meta' / name).is_file()]
    if missing:
        raise ValueError(f'Local destination has incomplete metadata ({", ".join(missing)}). '
                         'Choose a fresh destination; existing episode data has been preserved.')
