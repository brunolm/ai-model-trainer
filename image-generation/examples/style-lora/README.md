# Style LoRA Example

This example is a template for training a Stable Diffusion LoRA adapter.

Add your training images to `images`, then update `captions.csv` so every image file has a prompt-style caption. The starter rows are placeholders and will not run until matching image files exist.

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements-image.txt
.\.venv\Scripts\python.exe -m model_factory train --config .\image-generation\examples\style-lora\config.yaml
```

The trainer writes LoRA weights to `image-generation\models\style-lora`.
