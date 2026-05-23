import argparse
import json
from pathlib import Path

import pandas as pd

from model_factory.artifacts import (
    build_artifact,
    load_artifact,
    output_metrics_path,
    output_model_path,
    output_onnx_path,
    output_vectorizer_path,
    save_artifact,
    training_backend,
)
from model_factory.config import is_image_generation_task, load_config
from model_factory.data import load_training_frame, split_features_and_target, split_train_test
from model_factory.env import load_project_env
from model_factory.metrics import evaluate_model, make_jsonable, save_metrics
from model_factory.pipelines import build_pipeline


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        train(args)
        return

    if args.command == "predict":
        predict(args)
        return

    parser.print_help()


def build_parser():
    parser = argparse.ArgumentParser(prog="ai-model-trainer", description="Train and use small machine learning models.")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train a model from a YAML config.")
    train_parser.add_argument("--config", required=True, help="Path to a training config YAML file.")

    predict_parser = subparsers.add_parser("predict", help="Run prediction with a trained model artifact.")
    predict_parser.add_argument("--model", required=True, help="Path to a saved .joblib model artifact.")
    predict_parser.add_argument("--text", help="Single text input for text models.")
    predict_parser.add_argument("--input", help="CSV file for batch prediction.")
    predict_parser.add_argument("--output", help="Optional CSV output path for batch prediction.")
    predict_parser.add_argument("--threshold", type=float, help="Binary classification threshold for positive label.")

    return parser


def train(args):
    load_project_env(Path(args.config))
    config = load_config(args.config)

    if is_image_generation_task(config):
        train_image_model(config)
        return

    frame, data_path = load_training_frame(config)
    features, target, feature_columns = split_features_and_target(frame, config)
    train_features, test_features, train_target, test_target = split_train_test(features, target, config)

    if training_backend(config) == "pytorch":
        train_pytorch(config, train_features, test_features, train_target, test_target, data_path, feature_columns)
        return

    train_sklearn(config, frame, train_features, test_features, train_target, test_target, data_path, feature_columns)


def train_image_model(config):
    if config["task"] == "paired_image_translation":
        from model_factory.image_pair_lora import train_paired_image_translation

        train_paired_image_translation(config)
        return

    from model_factory.image_lora import train_image_generation_lora

    train_image_generation_lora(config)


def train_pytorch(config, train_features, test_features, train_target, test_target, data_path, feature_columns):
    if config["task"] != "text_classification":
        raise ValueError("The PyTorch backend currently supports text_classification")

    from model_factory.torch_text import export_runtime_files, train_text_classifier

    artifact = train_text_classifier(
        config,
        train_features,
        test_features,
        train_target,
        test_target,
        data_path,
        feature_columns,
    )
    model_path = output_model_path(config)
    onnx_path = output_onnx_path(config)
    vectorizer_path = output_vectorizer_path(config)

    save_artifact(artifact, model_path)
    export_runtime_files(artifact, onnx_path, vectorizer_path)
    maybe_save_metrics(config, artifact["metrics"])

    print(json.dumps(make_jsonable({
        "model_path": str(model_path),
        "onnx_path": str(onnx_path),
        "vectorizer_path": str(vectorizer_path),
        "backend": "pytorch",
        "task": config["task"],
        "training_rows": len(train_features),
        "test_rows": len(test_features),
        "metrics": artifact["metrics"],
    }), indent=2))


def train_sklearn(config, frame, train_features, test_features, train_target, test_target, data_path, feature_columns):
    pipeline = build_pipeline(config, frame, feature_columns)
    pipeline.fit(train_features, train_target)

    metrics = evaluate_model(pipeline, test_features, test_target, config["task"])
    artifact = build_artifact(config, pipeline, data_path, feature_columns, metrics)
    model_path = output_model_path(config)

    save_artifact(artifact, model_path)
    maybe_save_metrics(config, metrics)

    print(json.dumps(make_jsonable({
        "model_path": str(model_path),
        "backend": "sklearn",
        "task": config["task"],
        "training_rows": len(train_features),
        "test_rows": len(test_features),
        "metrics": metrics,
    }), indent=2))


def predict(args):
    if not args.text and not args.input:
        raise ValueError("Provide --text for one text value or --input for a CSV file")

    if args.text and args.input:
        raise ValueError("Use either --text or --input, not both")

    artifact = load_artifact(args.model)
    metadata = artifact["metadata"]

    if args.text:
        result = predict_text(artifact, args.text, args.threshold)
        print(json.dumps(make_jsonable(result), indent=2))
        return

    result_frame = predict_csv(artifact, args.input, args.threshold)

    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_frame.to_csv(output_path, index=False)
        print(json.dumps({"output_path": str(output_path), "rows": len(result_frame)}, indent=2))
        return

    print(json.dumps(make_jsonable({
        "task": metadata["task"],
        "rows": result_frame.to_dict(orient="records"),
    }), indent=2))


def maybe_save_metrics(config, metrics):
    metrics_path = output_metrics_path(config)

    if not metrics_path:
        return

    save_metrics(metrics, metrics_path)


def predict_text(artifact, text, threshold):
    metadata = artifact["metadata"]

    if metadata["task"] != "text_classification":
        raise ValueError("--text can only be used with text_classification models")

    prediction, probabilities = predict_values(artifact, [text], threshold)

    return {
        "text": text,
        "prediction": prediction[0],
        "probabilities": probabilities[0] if probabilities else None,
    }


def predict_csv(artifact, input_path, threshold):
    frame = pd.read_csv(input_path)
    metadata = artifact["metadata"]
    features = prediction_features(frame, metadata)
    predictions, probabilities = predict_values(artifact, features, threshold)
    result = frame.copy()
    result["prediction"] = predictions

    if probabilities:
        probability_frame = pd.DataFrame(probabilities)
        return pd.concat([result, probability_frame], axis=1)

    return result


def prediction_features(frame, metadata):
    task = metadata["task"]

    if task == "text_classification":
        text_column = metadata["text_column"]

        if text_column not in frame.columns:
            raise ValueError(f"Input CSV must contain text column: {text_column}")

        return frame[text_column].fillna("").astype(str)

    feature_columns = metadata["feature_columns"]
    missing_columns = [column for column in feature_columns if column not in frame.columns]

    if missing_columns:
        raise ValueError(f"Input CSV is missing feature columns: {missing_columns}")

    return frame[feature_columns].copy()


def predict_values(artifact, features, threshold):
    if artifact.get("backend") == "pytorch":
        from model_factory.torch_text import predict_text_values

        return predict_text_values(artifact, features, threshold)

    pipeline = artifact["pipeline"]
    probabilities = predict_probabilities(pipeline, features)

    if threshold is not None and probabilities:
        predictions = threshold_predictions(artifact, probabilities, threshold)
        return predictions, probabilities

    return pipeline.predict(features).tolist(), probabilities


def predict_probabilities(pipeline, features):
    estimator = pipeline.named_steps["model"]

    if not hasattr(pipeline, "predict_proba"):
        return None

    classes = [str(label) for label in estimator.classes_]
    probability_rows = pipeline.predict_proba(features)

    return [
        {f"probability_{label}": float(probability) for label, probability in zip(classes, row)}
        for row in probability_rows
    ]


def threshold_predictions(artifact, probabilities, threshold):
    metadata = artifact["metadata"]
    pipeline = artifact["pipeline"]
    estimator = pipeline.named_steps["model"]
    classes = list(estimator.classes_)

    if len(classes) != 2:
        raise ValueError("--threshold is only supported for binary classification models")

    positive_label = metadata.get("positive_label")

    if positive_label not in classes:
        positive_label = classes[-1]

    negative_label = next(label for label in classes if label != positive_label)
    positive_key = f"probability_{positive_label}"

    return [
        positive_label if row[positive_key] >= threshold else negative_label
        for row in probabilities
    ]
