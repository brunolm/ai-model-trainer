# Python Example

This example loads the native PyTorch `.bin` artifact directly.

It uses the repository's `model_factory` code to load:

```text
models\profanity-classifier.bin
```

## Prerequisites

From the repository root, generate the model first:

```powershell
mise run setup
mise run train-profanity
```

## Run With Mise

From `examples\python`:

```powershell
mise trust .\.mise.toml
mise install
mise run predict
```

## Run Directly

From `examples\python`:

```powershell
..\..\.venv\Scripts\python.exe .\src\index.py "hello friend"
..\..\.venv\Scripts\python.exe .\src\index.py "fuck off" --threshold 0.70
```
