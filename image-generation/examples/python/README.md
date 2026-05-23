# Python Image Example

This example loads the generated Diffusers LoRA adapter for the before/after transform model.

It uses the repository's image inference dependencies to load:

```text
image-generation\models\before-after-transform\pytorch_lora_weights.safetensors
image-generation\models\before-after-transform-summary.json
```

The LoRA file is not a standalone model. The script loads the base InstructPix2Pix model from the training summary, then applies the local LoRA adapter.

## Prerequisites

From the repository root, install the base and image dependencies and train the adapter first:

```powershell
mise run setup
mise run setup-image
mise run train-before-after
```

## Run With Mise

From `image-generation\examples\python`:

```powershell
mise trust .\.mise.toml
mise install
mise run setup
mise run predict
```

## Run Directly

From `image-generation\examples\python`:

```powershell
..\..\..\.venv\Scripts\python.exe .\src\index.py --before ..\before-after-transform\before\0001_before.jpg --instruction "apply the learned transformation" --output .\outputs\result.png
```

Useful options:

```powershell
..\..\..\.venv\Scripts\python.exe .\src\index.py --steps 30 --resolution 512 --resize-mode center-crop --guidance-scale 7.5 --image-guidance-scale 1.5 --seed 42
```

The script resizes the input to `512x512` with a center crop by default, matching the training resolution and avoiding CUDA out-of-memory errors from large source images. Use `--resize-mode fit` to preserve the whole image aspect ratio, or `--resize-mode none` only when you know the image is already small enough for your GPU.

If Diffusers detects potential NSFW content, it returns a black image. The runner checks that flag and fails with a readable error instead of saving the black placeholder.

The script prints the resolved model settings and writes the generated image to `outputs\result.png` by default.
