from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn

from model_factory.config import project_name, public_config, resolve_path


def build_artifact(config, pipeline, data_path, feature_columns, metrics):
    labels = config.get("labels") or {}

    return {
        "artifact_format": "ai-model-trainer-sklearn-v1",
        "backend": "sklearn",
        "pipeline": pipeline,
        "config": public_config(config),
        "metrics": metrics,
        "metadata": {
            "project_name": project_name(config),
            "backend": "sklearn",
            "task": config["task"],
            "target": config["data"]["target"],
            "text_column": config["data"].get("text_column"),
            "feature_columns": feature_columns,
            "positive_label": labels.get("positive_label"),
            "data_path": str(data_path),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "sklearn_version": sklearn.__version__,
        },
    }


def output_model_path(config):
    output = config.get("output") or {}
    configured_path = output.get("model_path")

    if configured_path:
        return resolve_path(config, configured_path)

    if training_backend(config) == "pytorch":
        return Path("classic-ml") / "models" / f"{project_name(config)}.bin"

    return Path("classic-ml") / "models" / f"{project_name(config)}.joblib"


def output_adapter_path(config):
    output = config.get("output") or {}
    configured_path = output.get("adapter_path")

    if configured_path:
        return resolve_path(config, configured_path)

    return Path("image-generation") / "models" / project_name(config)


def output_metrics_path(config):
    output = config.get("output") or {}
    configured_path = output.get("metrics_path")

    if not configured_path:
        return None

    return resolve_path(config, configured_path)


def output_onnx_path(config):
    output = config.get("output") or {}
    configured_path = output.get("onnx_path")

    if configured_path:
        return resolve_path(config, configured_path)

    return output_model_path(config).with_suffix(".onnx")


def output_vectorizer_path(config):
    output = config.get("output") or {}
    configured_path = output.get("vectorizer_path")

    if configured_path:
        return resolve_path(config, configured_path)

    model_path = output_model_path(config)
    return model_path.with_name(f"{model_path.stem}-vectorizer.json")


def save_artifact(artifact, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    if artifact.get("backend") == "pytorch":
        import torch

        torch.save(artifact, path)
        return

    joblib.dump(artifact, path)


def load_artifact(path):
    artifact_path = Path(path).resolve()

    if not artifact_path.exists():
        raise FileNotFoundError(f"Model file not found: {artifact_path}")

    artifact = load_torch_artifact(artifact_path)

    if not artifact:
        artifact = joblib.load(artifact_path)

    if not isinstance(artifact, dict):
        raise ValueError("Model file is not an AI Model Trainer artifact")

    if "pipeline" not in artifact and artifact.get("backend") != "pytorch":
        raise ValueError("Model file is not an AI Model Trainer artifact")

    return artifact


def training_backend(config):
    if config.get("backend"):
        return config["backend"]

    model = config.get("model") or {}
    return model.get("backend", "sklearn")


def load_torch_artifact(path):
    try:
        import torch
    except ImportError:
        return None

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        try:
            return torch.load(path, map_location="cpu")
        except Exception:
            return None
    except Exception:
        return None
