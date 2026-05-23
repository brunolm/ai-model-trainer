import json
from math import sqrt

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate_model(pipeline, features, target, task):
    predictions = pipeline.predict(features)

    if task in {"text_classification", "tabular_classification"}:
        return evaluate_classification(pipeline, target, predictions)

    return evaluate_regression(target, predictions)


def evaluate_classification(pipeline, target, predictions):
    return classification_metrics(target, predictions, model_labels(pipeline))


def classification_metrics(target, predictions, labels):
    if not labels:
        labels = sorted(set(target).union(set(predictions)))

    return make_jsonable({
        "accuracy": accuracy_score(target, predictions),
        "labels": labels,
        "confusion_matrix": confusion_matrix(target, predictions, labels=labels).tolist(),
        "classification_report": classification_report(
            target,
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    })


def evaluate_regression(target, predictions):
    mse = mean_squared_error(target, predictions)

    return make_jsonable({
        "mean_absolute_error": mean_absolute_error(target, predictions),
        "mean_squared_error": mse,
        "root_mean_squared_error": sqrt(mse),
        "r2_score": r2_score(target, predictions),
    })


def save_metrics(metrics, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def model_labels(pipeline):
    estimator = pipeline.named_steps["model"]

    if hasattr(estimator, "classes_"):
        return list(estimator.classes_)

    return None


def make_jsonable(value):
    if isinstance(value, dict):
        return {str(key): make_jsonable(item) for key, item in value.items()}

    if isinstance(value, list):
        return [make_jsonable(item) for item in value]

    if isinstance(value, tuple):
        return [make_jsonable(item) for item in value]

    if isinstance(value, np.ndarray):
        return make_jsonable(value.tolist())

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    return value
