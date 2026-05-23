import argparse
import json
import sys
from pathlib import Path


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(root))

    from model_factory.artifacts import load_artifact
    from model_factory.metrics import make_jsonable
    from model_factory.torch_text import predict_text_values

    model_path = root / "classic-ml" / "models" / "profanity-classifier.bin"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found. Run `mise run train-profanity` from the repository root: {model_path}")

    artifact = load_artifact(model_path)
    predictions, probabilities = predict_text_values(artifact, [args.text], args.threshold)
    result = {
        "text": args.text,
        "prediction": predictions[0],
        "probabilities": probabilities[0],
    }

    print(json.dumps(make_jsonable(result), indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Run the AI Model Trainer profanity classifier .bin artifact.")
    parser.add_argument("text", nargs="*", help="Text to classify.")
    parser.add_argument("--threshold", type=float, help="Optional binary classification threshold.")
    args = parser.parse_args()
    args.text = " ".join(args.text).strip() or "hello friend"
    return args


if __name__ == "__main__":
    main()
