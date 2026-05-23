import pandas as pd
from sklearn.model_selection import train_test_split

from model_factory.config import resolve_path


def load_training_frame(config):
    data_path = resolve_path(config, config["data"]["path"])

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    frame = pd.read_csv(data_path)
    validate_training_frame(frame, config)
    return frame, data_path


def validate_training_frame(frame, config):
    if frame.empty:
        raise ValueError("Training data has no rows")

    target = config["data"]["target"]

    if target not in frame.columns:
        raise ValueError(f"Target column not found: {target}")

    if frame[target].isna().any():
        raise ValueError(f"Target column contains empty values: {target}")

    if config["task"] == "text_classification":
        text_column = config["data"]["text_column"]

        if text_column not in frame.columns:
            raise ValueError(f"Text column not found: {text_column}")

        if frame[text_column].isna().any():
            raise ValueError(f"Text column contains empty values: {text_column}")


def split_features_and_target(frame, config):
    target = config["data"]["target"]
    task = config["task"]

    if task == "text_classification":
        text_column = config["data"]["text_column"]
        return frame[text_column].astype(str), frame[target], [text_column]

    feature_columns = configured_feature_columns(frame, config)
    return frame[feature_columns].copy(), frame[target], feature_columns


def split_train_test(features, target, config):
    split_config = config.get("split") or {}
    test_size = split_config.get("test_size", 0.2)
    random_state = split_config.get("random_state", 42)
    stratify_target = maybe_stratify_target(target, config)

    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )


def configured_feature_columns(frame, config):
    features_config = config.get("features") or {}
    explicit_columns = features_config.get("columns")
    target = config["data"]["target"]

    if explicit_columns:
        missing_columns = [column for column in explicit_columns if column not in frame.columns]

        if missing_columns:
            raise ValueError(f"Configured feature columns not found: {missing_columns}")

        if target in explicit_columns:
            raise ValueError("features.columns must not include the target column")

        return explicit_columns

    return [column for column in frame.columns if column != target]


def maybe_stratify_target(target, config):
    task = config["task"]

    if task not in {"text_classification", "tabular_classification"}:
        return None

    split_config = config.get("split") or {}

    if split_config.get("stratify", True) is False:
        return None

    class_counts = target.value_counts()

    if len(class_counts) < 2:
        return None

    if class_counts.min() < 2:
        return None

    return target
