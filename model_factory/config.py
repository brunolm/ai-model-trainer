from pathlib import Path

import yaml


CLASSIC_TASKS = {"text_classification", "tabular_classification", "tabular_regression"}
IMAGE_GENERATION_TASKS = {"image_generation_lora", "paired_image_translation"}
SUPPORTED_TASKS = CLASSIC_TASKS | IMAGE_GENERATION_TASKS


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

    if task not in SUPPORTED_TASKS:
        raise ValueError(
            "task must be text_classification, tabular_classification, tabular_regression, "
            "image_generation_lora, or paired_image_translation"
        )

    if task in IMAGE_GENERATION_TASKS:
        validate_image_generation_config(config)
        return

    validate_classic_config(config)


def validate_classic_config(config):
    task = config["task"]

    data = config.get("data")

    if not isinstance(data, dict):
        raise ValueError("Config must contain a data section")

    if not data.get("path"):
        raise ValueError("data.path is required")

    if not data.get("target"):
        raise ValueError("data.target is required")

    if task == "text_classification" and not data.get("text_column"):
        raise ValueError("data.text_column is required for text_classification")


def validate_image_generation_config(config):
    task = config["task"]

    if task == "paired_image_translation":
        validate_paired_image_translation_config(config)
        return

    validate_text_to_image_config(config)


def validate_text_to_image_config(config):
    data = config.get("data")

    if not isinstance(data, dict):
        raise ValueError("Config must contain a data section")

    if not data.get("image_dir"):
        raise ValueError("data.image_dir is required for image_generation_lora")

    model = config.get("model")

    if not isinstance(model, dict):
        raise ValueError("Config must contain a model section")

    if not model.get("base_model"):
        raise ValueError("model.base_model is required for image_generation_lora")

    model_type = model.get("type", "stable_diffusion_lora")

    if model_type != "stable_diffusion_lora":
        raise ValueError("Only stable_diffusion_lora is currently supported for image_generation_lora")


def validate_paired_image_translation_config(config):
    data = config.get("data")

    if not isinstance(data, dict):
        raise ValueError("Config must contain a data section")

    if not data.get("pairs_path"):
        raise ValueError("data.pairs_path is required for paired_image_translation")

    model = config.get("model")

    if not isinstance(model, dict):
        raise ValueError("Config must contain a model section")

    if not model.get("base_model"):
        raise ValueError("model.base_model is required for paired_image_translation")

    model_type = model.get("type", "instruct_pix2pix_lora")

    if model_type != "instruct_pix2pix_lora":
        raise ValueError("Only instruct_pix2pix_lora is currently supported for paired_image_translation")


def is_image_generation_task(config):
    return config.get("task") in IMAGE_GENERATION_TASKS


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
