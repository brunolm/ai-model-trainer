# AI Model Trainer

AI Model Trainer is a config-driven training workspace with separate root folders for each training style.

Current styles:

- `classic-ml`: text classification, tabular classification, and tabular regression
- `image-generation`: Stable Diffusion LoRA training for image-generation adapters

The root `model_factory` package provides the shared CLI.

## Setup

From this folder in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\requirements.txt
```

Optional image-generation dependencies are separate because they pull in heavier Hugging Face packages:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements-image.txt
```

## Hugging Face Token

Image training downloads base models from Hugging Face. Public downloads can work without a token, but `HF_TOKEN` raises rate limits and is required for gated models.

Create a Hugging Face read token at:

```text
https://huggingface.co/settings/tokens
```

Save it in a root `.env` file:

```powershell
Copy-Item .\.env.example .\.env
notepad .\.env
```

Set the value:

```text
HF_TOKEN=hf_your_actual_token_here
```

The training CLI loads `.env` automatically before model downloads. `.env` is ignored by git.

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

Optional image setup:

```powershell
mise run setup-image
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

## Folder Layout

```text
classic-ml\
  configs\templates\
  examples\
  models\

image-generation\
  configs\templates\
  examples\
  models\

model_factory\
```

Add future training styles as new root folders with their own `configs`, `examples`, and `models` subfolders.

## Classic ML

Classic ML trains small supervised models from CSV files.

Supported tasks:

- `text_classification`
- `tabular_classification`
- `tabular_regression`

Train the included profanity classifier:

```powershell
python -m model_factory train --config .\classic-ml\examples\profanity\config.yaml
```

This reads `classic-ml\examples\profanity\data.csv` and writes:

- `classic-ml\models\profanity-classifier.bin`
- `classic-ml\models\profanity-classifier.onnx`
- `classic-ml\models\profanity-classifier-vectorizer.json`
- `classic-ml\models\profanity-classifier-metrics.json`

Try prediction:

```powershell
python -m model_factory predict --model .\classic-ml\models\profanity-classifier.bin --text "hello friend"
python -m model_factory predict --model .\classic-ml\models\profanity-classifier.bin --text "fuck off"
```

You can make binary classifiers stricter or looser with a threshold:

```powershell
python -m model_factory predict --model .\classic-ml\models\profanity-classifier.bin --text "damn" --threshold 0.70
```

Lower thresholds flag more text. Higher thresholds flag less text.

Batch prediction from CSV:

```powershell
python -m model_factory predict --model .\classic-ml\models\profanity-classifier.bin --input .\classic-ml\examples\profanity\data.csv --output .\classic-ml\models\profanity-predictions.csv
```

### Classic Configs

Templates live under `classic-ml\configs\templates`.

For new classic models:

1. Create a folder under `classic-ml\examples`, such as `classic-ml\examples\spam`.
2. Add a CSV file with your examples.
3. Copy the closest template from `classic-ml\configs\templates`.
4. Edit `data.path`, `data.target`, and feature columns.
5. Choose a supported model type.
6. Run `python -m model_factory train --config .\classic-ml\examples\your-example\config.yaml`.
7. Test predictions with `python -m model_factory predict`.
8. Add model mistakes back into your dataset and retrain.

Supported classic model types:

- Text: `feedforward_text_classifier`, `logistic_regression`, `linear_svc`, `multinomial_nb`, `random_forest_classifier`, `gradient_boosting_classifier`
- Tabular classification: `logistic_regression`, `random_forest_classifier`, `gradient_boosting_classifier`
- Tabular regression: `linear_regression`, `random_forest_regressor`, `gradient_boosting_regressor`

## Image Generation

Image generation currently supports:

- `image_generation_lora`: prompt-to-image LoRA training
- `paired_image_translation`: before-image plus instruction to after-image LoRA training

This trains a LoRA adapter for a Stable Diffusion 1.x-style base model using Hugging Face Diffusers. It does not train a full image generator from scratch.

Prepare dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements-image.txt
```

Create or copy an image-generation config:

```powershell
New-Item -ItemType Directory -Path .\image-generation\examples\my-style -Force | Out-Null
Copy-Item .\image-generation\configs\templates\stable-diffusion-lora.yaml .\image-generation\examples\my-style\config.yaml
```

Your example folder should contain:

```text
image-generation\examples\my-style\
  config.yaml
  captions.csv
  images\
    sample-001.png
    sample-002.png
```

The captions CSV uses one row per image:

```csv
image,caption
sample-001.png,a photo in the custom training style
sample-002.png,a detailed scene in the custom training style
```

Train:

```powershell
python -m model_factory train --config .\image-generation\examples\my-style\config.yaml
```

The image trainer writes LoRA weights and a summary JSON under `image-generation\models`.

### Image Config

```yaml
project:
  name: my-style-lora

task: image_generation_lora
backend: diffusers

data:
  image_dir: images
  captions_path: captions.csv
  image_column: image
  caption_column: caption

model:
  type: stable_diffusion_lora
  base_model: runwayml/stable-diffusion-v1-5
  params:
    rank: 16
    alpha: 16
    dropout: 0.0

training:
  resolution: 512
  batch_size: 1
  epochs: 1
  max_train_steps: 1000
  learning_rate: 0.0001
  mixed_precision: fp16
  gradient_checkpointing: true
  seed: 42
  device: auto
  num_workers: 0

output:
  adapter_path: ../../models/my-style-lora
  metrics_path: ../../models/my-style-lora-summary.json
```

If you do not want a captions CSV, omit `captions_path` and place a `.txt` file beside each image with the same basename.

### Before/After Example

Use the before/after example when the input is an image and the output should be a transformed image:

```text
before image + instruction -> after image
```

Example layout:

```text
image-generation\examples\before-after-transform\
  config.yaml
  sample-pairs.csv
  before\
    sample-001.png
  after\
    sample-001.png
```

The pairs CSV maps the source image to the desired result:

```csv
before,after,instruction
before/sample-001.png,after/sample-001.png,turn this sketch into a polished product render
before/sample-002.png,after/sample-002.png,clean up the screenshot and make it look professional
```

The matching config template is `image-generation\configs\templates\paired-image-translation.yaml`.

Train the included before/after sample:

```powershell
python -m model_factory train --config .\image-generation\examples\before-after-transform\config.yaml
```

The paired trainer saves LoRA weights under `image-generation\models\before-after-transform`. It expects an InstructPix2Pix-style base model such as `timbrooks/instruct-pix2pix`, because LoRA weights alone cannot add the before-image conditioning channel to a plain text-to-image base model.

## Runtime Examples

The classic profanity model includes small runtime examples:

```powershell
Set-Location .\classic-ml\examples\node
mise trust .\.mise.toml
mise install
mise run setup
mise run predict
```

```powershell
Set-Location .\classic-ml\examples\python
mise trust .\.mise.toml
mise install
mise run predict
```

The before/after image adapter includes matching Python and Node examples:

```powershell
Set-Location .\image-generation\examples\python
mise trust .\.mise.toml
mise install
mise run setup
mise run predict
```

```powershell
Set-Location .\image-generation\examples\node
mise trust .\.mise.toml
mise install
mise run setup
mise run predict
```

The main skill is dataset quality. A simple model with good examples usually beats a complex model with unclear labels.
