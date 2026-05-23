import os
from pathlib import Path


def load_project_env(start_path=None):
    env_path = find_project_env(Path(start_path or Path.cwd()).resolve())

    if not env_path:
        return None

    load_env_file(env_path)
    return env_path


def find_project_env(start_path):
    current = start_path if start_path.is_dir() else start_path.parent

    for directory in [current, *current.parents]:
        env_path = directory / ".env"

        if env_path.exists():
            return env_path

        if (directory / "pyproject.toml").exists() or (directory / ".git").exists():
            return None

    return None


def load_env_file(path):
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = parse_env_line(line)

        if not parsed:
            continue

        key, value = parsed
        os.environ.setdefault(key, value)


def parse_env_line(line):
    stripped = line.strip()

    if not stripped or stripped.startswith("#"):
        return None

    if stripped.startswith("export "):
        stripped = stripped[7:].strip()

    if "=" not in stripped:
        return None

    key, value = stripped.split("=", 1)
    key = key.strip()

    if not key:
        return None

    return key, clean_env_value(value)


def clean_env_value(value):
    value = value.strip()

    if len(value) < 2:
        return value

    if value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]

    return value
