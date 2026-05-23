# AI Model Trainer

AI Model Trainer is a small config-driven project for training your own machine learning models.

The first included example is an English profanity classifier, but the same training and prediction CLI can also be used for:

- text classification, such as profanity, spam, sentiment, or topic detection
- tabular classification, such as churn, fraud, or pass/fail prediction
- tabular regression, such as price, duration, or score prediction

## Setup

From this folder in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\requirements.txt
```

## Mise

This project includes a local `.mise.toml` that pins Python `3.13.13` and defines common tasks.

```powershell
mise trust .\.mise.toml
mise install
mise run setup
mise run train-profanity
mise run predict-profanity
mise run check
```

Optional: install the local command name:

```powershell
pip install -e .
```

After that, you can use either:

```powershell
python -m model_factory --help
```

or:

```powershell
ai-model-trainer --help
```

## Train the profanity classifier example

```powershell
python -m model_factory train --config .\examples\profanity\config.yaml
```

This reads `examples\profanity\data.csv`, trains a TF-IDF + PyTorch feed-forward text classifier, and writes:

- `models\profanity-classifier.bin`
- `models\profanity-classifier.onnx`
- `models\profanity-classifier-vectorizer.json`
- `models\profanity-classifier-metrics.json`

The `.bin` file is a binary AI Model Trainer artifact saved with `torch.save`. AI Model Trainer can load it again with the `predict` command. The `.onnx` file is the exported neural network for runtimes like `onnxruntime-node`. The vectorizer JSON is required because the ONNX model expects numeric TF-IDF features, not raw text.

## Try prediction

```powershell
python -m model_factory predict --model .\models\profanity-classifier.bin --text "hello friend"
python -m model_factory predict --model .\models\profanity-classifier.bin --text "fuck off"
```

You can make the detector stricter or looser with a threshold:

```powershell
python -m model_factory predict --model .\models\profanity-classifier.bin --text "damn" --threshold 0.70
```

Lower thresholds flag more text. Higher thresholds flag less text.

## Batch prediction from CSV

For text models, the input CSV must contain the configured text column.

```powershell
python -m model_factory predict --model .\models\profanity-classifier.bin --input .\examples\profanity\data.csv --output .\models\profanity-predictions.csv
```

## Node Example

There is a small Bun + TypeScript example in `examples\node`. It uses `onnxruntime-node` to run `models\profanity-classifier.onnx` directly, without invoking Python.

```powershell
Set-Location .\examples\node
mise trust .\.mise.toml
mise install
mise run setup
mise run predict
bun run src\index.ts "hello friend"
```

## Python Example

There is also a small Python example in `examples\python`. It loads the native PyTorch `.bin` artifact directly.

```powershell
Set-Location .\examples\python
mise trust .\.mise.toml
mise install
mise run predict
..\..\.venv\Scripts\python.exe .\src\index.py "hello friend"
```

## How configs work

A training config tells AI Model Trainer:

- what task you are training
- where the CSV data is
- which column is the target label
- which model algorithm to use
- where to save the trained model and metrics

The profanity example is:

```yaml
project:
  name: profanity-classifier

task: text_classification
backend: pytorch

data:
  path: data.csv
  text_column: text
  target: label

features:
  type: tfidf
  lowercase: true
  analyzer: char_wb
  ngram_range: [3, 5]
  max_features: 5000

model:
  type: feedforward_text_classifier
  params:
    hidden_sizes: [64]
    dropout: 0.2

training:
  epochs: 120
  batch_size: 16
  learning_rate: 0.01
  random_state: 42
  device: auto

split:
  test_size: 0.25
  random_state: 42
  stratify: true

labels:
  positive_label: 1

output:
  model_path: ../../models/profanity-classifier.bin
  onnx_path: ../../models/profanity-classifier.onnx
  vectorizer_path: ../../models/profanity-classifier-vectorizer.json
  metrics_path: ../../models/profanity-classifier-metrics.json
```

Relative paths are resolved from the folder containing the config file.

## Supported tasks

### `text_classification`

Use this when each row has one text column and one label column.

Good examples:

- profanity detection
- spam detection
- sentiment detection
- ticket category detection

Supported feature type:

- `tfidf`

Supported models:

- `feedforward_text_classifier` with `backend: pytorch`
- `logistic_regression` with the default scikit-learn backend
- `linear_svc` with the default scikit-learn backend
- `multinomial_nb` with the default scikit-learn backend
- `random_forest_classifier` with the default scikit-learn backend
- `gradient_boosting_classifier` with the default scikit-learn backend

For new text classifiers, prefer `backend: pytorch`. The scikit-learn models remain useful for quick baselines. `linear_svc` is often strong for text but does not provide probabilities by default.

### `tabular_classification`

Use this when each row has structured columns and one label column.

Good examples:

- customer churn prediction
- fraud/not fraud
- pass/fail prediction
- lead quality prediction

Supported models:

- `logistic_regression`
- `random_forest_classifier`
- `gradient_boosting_classifier`

Numeric columns are imputed with median values and scaled. Categorical columns are imputed with the most common value and one-hot encoded.

### `tabular_regression`

Use this when each row has structured columns and the target is a number.

Good examples:

- price prediction
- time-to-complete prediction
- score prediction

Supported models:

- `linear_regression`
- `random_forest_regressor`
- `gradient_boosting_regressor`

## Creating your own model

1. Create a folder under `examples`, such as `examples\spam`.
2. Add a CSV file with your examples.
3. Copy the closest config template from `configs\templates`.
4. Edit `data.path`, `data.target`, and feature columns.
5. Choose a model type.
6. Run `python -m model_factory train --config .\examples\your-example\config.yaml`.
7. Test predictions with `python -m model_factory predict`.
8. Add model mistakes back into your dataset and retrain.

The main skill is dataset quality. A simple model with good examples usually beats a complex model with unclear labels.
