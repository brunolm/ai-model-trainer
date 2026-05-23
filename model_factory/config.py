from pathlib import Path

import yaml


def load_config(config_path):
    path = Path(config_path).resolve()

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML object")

    config["_config_path"] = str(path)
    config["_base_dir"] = str(path.parent)
    validate_config(config)
    return config


def validate_config(config):
    task = config.get("task")

    if task not in {"text_classification", "tabular_classification", "tabular_regression"}:
        raise ValueError("task must be text_classification, tabular_classification, or tabular_regression")

    data = config.get("data")

    if not isinstance(data, dict):
        raise ValueError("Config must contain a data section")

    if not data.get("path"):
        raise ValueError("data.path is required")

    if not data.get("target"):
        raise ValueError("data.target is required")

    if task == "text_classification" and not data.get("text_column"):
        raise ValueError("data.text_column is required for text_classification")


def public_config(config):
    return {key: value for key, value in config.items() if not key.startswith("_")}


def project_name(config):
    project = config.get("project") or {}
    name = project.get("name") or config.get("name")

    if name:
        return str(name)

    return Path(config["_config_path"]).stem


def resolve_path(config, path_value):
    path = Path(path_value)

    if path.is_absolute():
        return path

    return (Path(config["_base_dir"]) / path).resolve()
