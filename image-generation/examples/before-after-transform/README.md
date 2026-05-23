# Before/After Transform Example

This example shows the dataset shape for paired image-to-image training:

```text
before image + instruction -> after image
```

The CLI trains this with `task: paired_image_translation`, which fine-tunes LoRA weights for an InstructPix2Pix-style image-to-image model.

## Files

```text
before-after-transform\
  config.yaml
  sample-pairs.csv
  pairs.csv
  before\
    sample-001.png
    sample-002.png
  after\
    sample-001.png
    sample-002.png
```

`sample-pairs.csv` maps each source image to the desired result:

```csv
before,after,instruction
before/sample-001.png,after/sample-001.png,turn this sketch into a polished product render
before/sample-002.png,after/sample-002.png,clean up the screenshot and make it look professional
```

Use this when you want the model to learn a transformation, such as sketch to render, rough screenshot to polished UI, photo cleanup, restoration, or color/style conversion.

## Train

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements-image.txt
.\.venv\Scripts\python.exe -m model_factory train --config .\image-generation\examples\before-after-transform\config.yaml
```

The trainer writes LoRA weights to `image-generation\models\before-after-transform`.

The configured base model must already be InstructPix2Pix-style. The example uses `timbrooks/instruct-pix2pix`.
