"""Date-based training archives and a validated latest-completed-run pointer.

This module deliberately uses only the Python standard library so run
selection and configuration checks do not initialize TensorFlow.
"""

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import tempfile


SHANGHAI_TIMEZONE = timezone(timedelta(hours=8), name="Asia/Shanghai")


def local_timestamp():
    """Return an ISO timestamp in the user's Asia/Shanghai time zone."""
    return datetime.now(SHANGHAI_TIMEZONE).isoformat(timespec="seconds")


def create_run(output_root):
    """Create a fresh date/run directory without reusing an existing run."""
    output_root = Path(output_root).expanduser().resolve()
    while True:
        now = datetime.now(SHANGHAI_TIMEZONE)
        date_dir = output_root / now.strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)
        run_dir = date_dir / now.strftime("%Y%m%d_%H%M%S_%f")
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            # A concurrent process may have claimed this microsecond.
            continue


def write_json(path, data):
    """Atomically replace a UTF-8 JSON file after writing it completely."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=str(path.parent),
                prefix="." + path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary_path = stream.name
            json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, str(path))
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _read_json(path):
    try:
        with path.open("r", encoding="utf-8-sig") as stream:
            data = json.load(stream)
    except FileNotFoundError:
        raise FileNotFoundError("Required run metadata does not exist: {}".format(path))
    except (ValueError, UnicodeError) as error:
        raise ValueError("Invalid JSON in {}: {}".format(path, error)) from error
    if not isinstance(data, dict):
        raise ValueError("Run metadata must be a JSON object: {}".format(path))
    return data


def _resolve_path(value, base, description):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a nonempty path string.".format(description))
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _validate_checkpoint(prefix, phase):
    """Check that a TensorFlow checkpoint has its index and all data shards."""
    index = Path(str(prefix) + ".index")
    if not index.is_file() or index.stat().st_size == 0:
        raise FileNotFoundError("Missing or empty {} checkpoint index: {}".format(phase, index))
    shards = list(prefix.parent.glob(prefix.name + ".data-*"))
    if not shards:
        raise FileNotFoundError("Missing {} checkpoint data: {}.data-*".format(phase, prefix))
    shard_pattern = re.compile(re.escape(prefix.name) + r"\.data-(\d+)-of-(\d+)$")
    numbers = set()
    totals = set()
    for shard in shards:
        match = shard_pattern.fullmatch(shard.name)
        if match is None or not shard.is_file() or shard.stat().st_size == 0:
            raise ValueError("Invalid or empty {} checkpoint shard: {}".format(phase, shard))
        numbers.add(int(match.group(1)))
        totals.add(int(match.group(2)))
    if (len(totals) != 1 or next(iter(totals)) < 1
            or numbers != set(range(next(iter(totals))))):
        raise ValueError("Incomplete {} checkpoint data shards: {}".format(phase, prefix))


def load_run(output_root, run_dir=None):
    """Load a complete run and validate both saved network checkpoints.

    Relative pointer paths and explicit run directories are resolved under
    output_root. Relative paths inside config.json are resolved under the run
    directory. Returned checkpoint paths are absolute. An absent/incomplete
    latest run never falls back to the repository's bundled pretrained model.
    """
    output_root = Path(output_root).expanduser().resolve()
    if run_dir is None:
        pointer_path = output_root / "latest.json"
        if not pointer_path.is_file():
            raise FileNotFoundError(
                "No completed training run is registered at {}. "
                "Run main.py to train first, or provide an explicit run directory."
                .format(pointer_path))
        pointer = _read_json(pointer_path)
        run_path = _resolve_path(pointer.get("run_dir"), output_root, "latest.json run_dir")
    else:
        run_path = _resolve_path(os.fspath(run_dir), output_root, "run_dir")

    config = _read_json(run_path / "config.json")
    if config.get("training_complete") is not True:
        raise ValueError("Training has not completed for this run: {}".format(run_path))
    paths = config.get("paths")
    checkpoints = config.get("checkpoints")
    if not isinstance(paths, dict) or not isinstance(checkpoints, dict):
        raise ValueError("Run configuration must contain paths and checkpoints objects: {}"
                         .format(run_path / "config.json"))
    checkpoint_dir = _resolve_path(paths.get("checkpoint_dir"), run_path,
                                   "paths.checkpoint_dir")
    resolved_checkpoints = {}
    for phase in ("Decom", "Relight"):
        prefix = _resolve_path(checkpoints.get(phase), run_path,
                               "checkpoints.{}".format(phase))
        try:
            prefix.relative_to(checkpoint_dir)
        except ValueError:
            raise ValueError("{} checkpoint is outside checkpoint_dir: {}".format(phase, prefix))
        _validate_checkpoint(prefix, phase)
        resolved_checkpoints[phase] = str(prefix)

    # Normalize only the required fields, leaving recorded training options intact.
    config["paths"] = dict(paths, checkpoint_dir=str(checkpoint_dir))
    config["checkpoints"] = dict(checkpoints, **resolved_checkpoints)
    return run_path, config


def publish_latest(output_root, run_dir):
    """Publish a completed, validated run using an atomic latest.json update."""
    output_root = Path(output_root).expanduser().resolve()
    run_path, _ = load_run(output_root, run_dir)
    try:
        stored_path = str(run_path.relative_to(output_root))
    except ValueError:
        stored_path = str(run_path)
    write_json(output_root / "latest.json", {
        "run_dir": stored_path,
        "published_at": local_timestamp(),
    })
