# Node Image Example

This example uses Bun and TypeScript to test the generated before/after transform LoRA adapter from Node.

The local artifact is a Diffusers LoRA adapter:

```text
image-generation\models\before-after-transform\pytorch_lora_weights.safetensors
```

Node does not have the same local Diffusers LoRA image-to-image runtime as Python, so this project is a small Node command-line wrapper around the Python image example. It still gives you a Node entry point for app integration tests while keeping inference on the runtime that can load the local safetensors adapter.

## Prerequisites

From the repository root, install dependencies and train the adapter first:

```powershell
mise run setup
mise run setup-image
mise run train-before-after
```

## Run With Mise

From `image-generation\examples\node`:

```powershell
mise trust .\.mise.toml
mise install
mise run setup
mise run predict
```

## Run With Bun Directly

From `image-generation\examples\node`:

```powershell
bun run src\index.ts --before ..\before-after-transform\before\0001_before.jpg --instruction "apply the learned transformation" --output .\outputs\result.png
```

All inference flags except `--python` are passed through to the Python runner:

```powershell
bun run src\index.ts --steps 30 --resolution 512 --resize-mode center-crop --guidance-scale 7.5 --image-guidance-scale 1.5 --seed 42
```

The Python runner resizes the input to `512x512` with a center crop by default, matching the training resolution and avoiding CUDA out-of-memory errors from large source images. Use `--resize-mode fit` to preserve the whole image aspect ratio, or `--resize-mode none` only when you know the image is already small enough for your GPU.

If Diffusers detects potential NSFW content, it returns a black image. The wrapper now surfaces that as a readable Python runner error instead of silently leaving the black placeholder.

When `--output` is omitted, the Node wrapper writes to `outputs\result.png`.

Use `--python` if you want to point the wrapper at a different Python executable:

```powershell
bun run src\index.ts --python C:\Python313\python.exe --steps 20
```
