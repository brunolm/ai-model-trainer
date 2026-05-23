# Node Example

This example uses Bun and TypeScript to run the AI Model Trainer profanity classifier with ONNX Runtime.

The Node script loads:

```text
classic-ml\models\profanity-classifier.onnx
classic-ml\models\profanity-classifier-vectorizer.json
```

It does not invoke Python. The vectorizer JSON is required because the ONNX model expects numeric TF-IDF features, not raw text.

Package installs use `bunfig.toml` with a minimum release age of 3 days. This means Bun will only install registry packages that were published at least `259200` seconds ago.

## Prerequisites

From the repository root, generate the model first:

```powershell
mise run setup
mise run train-profanity
```

## Run With Mise

From `classic-ml\examples\node`:

```powershell
mise trust .\.mise.toml
mise install
mise run setup
mise run predict
```

## Run With Bun Directly

From `classic-ml\examples\node`:

```powershell
bun run src\index.ts "hello friend"
bun run src\index.ts "fuck off"
bun run src\index.ts "damn" --threshold 0.70
```

The script prints a friendly label and the raw JSON response from the ONNX runtime path.
