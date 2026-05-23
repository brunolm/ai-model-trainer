from datetime import datetime, timezone
import json

import numpy as np
import sklearn
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model_factory.config import project_name, public_config
from model_factory.metrics import classification_metrics, make_jsonable
from model_factory.pipelines import build_text_vectorizer


class TorchTextClassifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=None, dropout=0.0):
        super().__init__()
        self.network = build_network(input_size, output_size, hidden_sizes or [], dropout)

    def forward(self, inputs):
        return self.network(inputs)


def train_text_classifier(config, train_features, test_features, train_target, test_target, data_path, feature_columns):
    training_config = config.get("training") or {}
    set_seed(training_config.get("random_state", config.get("split", {}).get("random_state", 42)))

    vectorizer = build_text_vectorizer(config.get("features") or {})
    train_matrix = vectorizer.fit_transform(train_features)
    test_matrix = vectorizer.transform(test_features)
    classes, train_indices = encode_labels(train_target, test_target)
    model_config = resolved_model_config(config, train_matrix.shape[1], len(classes))
    device = resolve_device(training_config.get("device", "auto"))
    model = TorchTextClassifier(**model_config).to(device)
    final_loss = train_model(model, train_matrix, train_indices, training_config, device)
    predictions, _ = predict_matrix(model, test_matrix, classes, device)
    metrics = classification_metrics(test_target, predictions, classes)
    metrics["training"] = {
        "backend": "pytorch",
        "epochs": training_config.get("epochs", 80),
        "batch_size": training_config.get("batch_size", 16),
        "learning_rate": training_config.get("learning_rate", 0.01),
        "device": device.type,
        "final_loss": final_loss,
    }

    return build_torch_artifact(
        config,
        vectorizer,
        model,
        model_config,
        classes,
        data_path,
        feature_columns,
        metrics,
    )


def predict_text_values(artifact, features, threshold):
    device = torch.device("cpu")
    model = restore_model(artifact).to(device)
    matrix = artifact["vectorizer"].transform(features)
    predictions, probabilities = predict_matrix(model, matrix, artifact["classes"], device, threshold, artifact)
    return predictions, probabilities


def export_runtime_files(artifact, onnx_path, vectorizer_path):
    export_onnx_model(artifact, onnx_path)
    export_vectorizer_metadata(artifact, vectorizer_path)


def build_network(input_size, output_size, hidden_sizes, dropout):
    layers = []
    current_size = input_size

    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(current_size, int(hidden_size)))
        layers.append(nn.ReLU())

        if dropout:
            layers.append(nn.Dropout(float(dropout)))

        current_size = int(hidden_size)

    layers.append(nn.Linear(current_size, output_size))
    return nn.Sequential(*layers)


def set_seed(seed):
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))


def encode_labels(train_target, test_target):
    classes = ordered_unique(list(train_target) + list(test_target))
    index_by_label = {label: index for index, label in enumerate(classes)}
    train_indices = [index_by_label[label] for label in train_target]
    return classes, train_indices


def resolved_model_config(config, input_size, output_size):
    model_config = config.get("model") or {}
    params = model_config.get("params") or {}

    return {
        "input_size": input_size,
        "output_size": output_size,
        "hidden_sizes": params.get("hidden_sizes", [64]),
        "dropout": params.get("dropout", 0.2),
    }


def resolve_device(configured_device):
    if configured_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(configured_device)


def train_model(model, matrix, target_indices, training_config, device):
    inputs = tensor_from_matrix(matrix, device)
    targets = torch.tensor(target_indices, dtype=torch.long, device=device)
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(
        dataset,
        batch_size=int(training_config.get("batch_size", 16)),
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_config.get("learning_rate", 0.01)))
    loss_function = nn.CrossEntropyLoss()
    final_loss = 0.0

    for _ in range(int(training_config.get("epochs", 80))):
        final_loss = train_epoch(model, loader, optimizer, loss_function)

    return float(final_loss)


def train_epoch(model, loader, optimizer, loss_function):
    model.train()
    total_loss = 0.0
    row_count = 0

    for inputs, targets in loader:
        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_function(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(inputs)
        row_count += len(inputs)

    if not row_count:
        return 0.0

    return total_loss / row_count


def predict_matrix(model, matrix, classes, device, threshold=None, artifact=None):
    model.eval()

    with torch.no_grad():
        inputs = tensor_from_matrix(matrix, device)
        logits = model(inputs)
        probability_array = torch.softmax(logits, dim=1).cpu().numpy()

    prediction_indices = predicted_indices(probability_array, classes, threshold, artifact)
    predictions = [classes[index] for index in prediction_indices]
    probabilities = probability_rows(probability_array, classes)
    return predictions, probabilities


def predicted_indices(probability_array, classes, threshold, artifact):
    if threshold is None:
        return probability_array.argmax(axis=1).tolist()

    if len(classes) != 2:
        raise ValueError("--threshold is only supported for binary classification models")

    positive_label = artifact["metadata"].get("positive_label") if artifact else classes[-1]

    if positive_label not in classes:
        positive_label = classes[-1]

    positive_index = classes.index(positive_label)
    negative_index = next(index for index, label in enumerate(classes) if label != positive_label)
    return [
        positive_index if row[positive_index] >= threshold else negative_index
        for row in probability_array
    ]


def probability_rows(probability_array, classes):
    return [
        {f"probability_{label}": float(probability) for label, probability in zip(classes, row)}
        for row in probability_array
    ]


def export_onnx_model(artifact, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    model = restore_model(artifact)
    model.eval()
    dummy_input = torch.zeros(1, artifact["model_config"]["input_size"], dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        str(path),
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={
            "features": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )


def export_vectorizer_metadata(artifact, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    vectorizer = artifact["vectorizer"]
    metadata = {
        "artifact_format": "ai-model-trainer-tfidf-vectorizer-v1",
        "analyzer": vectorizer.analyzer,
        "lowercase": vectorizer.lowercase,
        "ngram_range": list(vectorizer.ngram_range),
        "norm": vectorizer.norm,
        "use_idf": vectorizer.use_idf,
        "smooth_idf": vectorizer.smooth_idf,
        "sublinear_tf": vectorizer.sublinear_tf,
        "strip_accents": vectorizer.strip_accents,
        "token_pattern": vectorizer.token_pattern,
        "vocabulary": vectorizer.vocabulary_,
        "idf": vectorizer.idf_.tolist() if hasattr(vectorizer, "idf_") else None,
        "feature_count": artifact["model_config"]["input_size"],
        "classes": artifact["classes"],
        "positive_label": artifact["metadata"].get("positive_label"),
        "onnx": {
            "input_name": "features",
            "output_name": "logits",
        },
    }

    with path.open("w", encoding="utf-8") as file:
        json.dump(make_jsonable(metadata), file, indent=2)


def build_torch_artifact(config, vectorizer, model, model_config, classes, data_path, feature_columns, metrics):
    labels = config.get("labels") or {}

    return {
        "artifact_format": "ai-model-trainer-pytorch-text-classifier-v1",
        "backend": "pytorch",
        "vectorizer": vectorizer,
        "classes": classes,
        "model_state_dict": model.cpu().state_dict(),
        "model_config": model_config,
        "config": public_config(config),
        "metrics": metrics,
        "metadata": {
            "project_name": project_name(config),
            "backend": "pytorch",
            "task": config["task"],
            "target": config["data"]["target"],
            "text_column": config["data"].get("text_column"),
            "feature_columns": feature_columns,
            "positive_label": labels.get("positive_label"),
            "data_path": str(data_path),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "pytorch_version": torch.__version__,
            "sklearn_version": sklearn.__version__,
        },
    }


def restore_model(artifact):
    model = TorchTextClassifier(**artifact["model_config"])
    model.load_state_dict(artifact["model_state_dict"])
    return model


def tensor_from_matrix(matrix, device):
    return torch.tensor(matrix.astype(np.float32).toarray(), dtype=torch.float32, device=device)


def ordered_unique(values):
    unique_values = []

    for value in values:
        if value in unique_values:
            continue

        unique_values.append(value)

    return unique_values
