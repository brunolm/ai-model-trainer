import argparse
import json
import sys
from pathlib import Path

DEFAULT_BASE_MODEL = "timbrooks/instruct-pix2pix"
DEFAULT_WEIGHT_NAME = "pytorch_lora_weights.safetensors"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def main():
    root = Path(__file__).resolve().parents[4]
    args = parse_args(root)
    settings = resolve_model_settings(args, root)

    assert_file(args.before, "Input image not found. Pass a file with --before.")
    assert_file(
        settings["adapter_path"] / args.weight_name,
        "LoRA weights not found. Run `mise run train-before-after` from the repository root.",
    )

    result, nsfw_content_detected = run_inference(args, settings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)

    print(
        json.dumps(
            {
                "base_model": settings["base_model"],
                "adapter_path": str(settings["adapter_path"]),
                "before": str(args.before),
                "output": str(args.output),
                "instruction": args.instruction,
                "device": settings["device"],
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "image_guidance_scale": args.image_guidance_scale,
                "nsfw_content_detected": nsfw_content_detected,
            },
            indent=2,
        )
    )


def parse_args(root):
    parser = argparse.ArgumentParser(
        description="Run the before-after-transform Diffusers LoRA adapter."
    )
    parser.add_argument("--before", type=Path, help="Input image to transform.")
    parser.add_argument(
        "--instruction",
        default="apply the learned transformation",
        help="Instruction prompt for InstructPix2Pix.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=example_dir(root) / "python" / "outputs" / "result.png",
        help="Path for the generated image.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=root
        / "image-generation"
        / "models"
        / "before-after-transform-summary.json",
        help="Training summary JSON to read base model settings from.",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        help="Directory containing the LoRA safetensors file.",
    )
    parser.add_argument("--base-model", help="Override the base InstructPix2Pix model.")
    parser.add_argument(
        "--weight-name",
        default=DEFAULT_WEIGHT_NAME,
        help="LoRA safetensors file name inside --adapter-path.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device.",
    )
    parser.add_argument(
        "--steps", type=int, default=20, help="Diffusion inference steps."
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=7.5, help="Text prompt guidance scale."
    )
    parser.add_argument(
        "--image-guidance-scale",
        type=float,
        default=1.5,
        help="Input image guidance scale.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Torch generator seed.")
    args = parser.parse_args()

    if not args.before:
        args.before = default_before_image(root)

    args.before = resolve_path(args.before)
    args.output = resolve_path(args.output)
    args.summary_path = resolve_path(args.summary_path)

    if args.adapter_path:
        args.adapter_path = resolve_path(args.adapter_path)

    return args


def default_before_image(root):
    before_dir = (
        root / "image-generation" / "examples" / "before-after-transform" / "before"
    )

    if before_dir.exists():
        for path in sorted(before_dir.iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                return path

    return before_dir / "sample-001.png"


def resolve_model_settings(args, root):
    summary = load_summary(args.summary_path)
    base_model = args.base_model or summary.get("base_model") or DEFAULT_BASE_MODEL
    adapter_path = (
        args.adapter_path
        or summary_adapter_path(summary)
        or root / "image-generation" / "models" / "before-after-transform"
    )

    return {
        "base_model": base_model,
        "adapter_path": adapter_path,
        "device": resolve_device(args.device),
    }


def load_summary(path):
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def summary_adapter_path(summary):
    value = summary.get("adapter_path")

    if not value:
        return None

    path = Path(value)

    if path.exists():
        return path

    return None


def resolve_device(value):
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "Missing image inference dependencies. Run `mise run setup-image` from the repository root."
        ) from error

    if value != "auto":
        return value

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def assert_file(path, message):
    if path.exists():
        return

    raise FileNotFoundError(f"{message}\nMissing path: {path}")


def run_inference(args, settings):
    dependencies = load_dependencies()
    torch = dependencies["torch"]
    image_module = dependencies["Image"]
    pipeline_class = dependencies["StableDiffusionInstructPix2PixPipeline"]
    scheduler_class = dependencies["EulerAncestralDiscreteScheduler"]

    dtype = torch.float16 if settings["device"] == "cuda" else torch.float32

    kwargs = {"torch_dtype": dtype}
    kwargs["safety_checker"] = None
    kwargs["requires_safety_checker"] = False

    pipe = pipeline_class.from_pretrained(settings["base_model"], **kwargs)
    pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(str(settings["adapter_path"]), weight_name=args.weight_name)
    pipe.to(settings["device"])

    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    image = image_module.open(args.before).convert("RGB")

    generator = torch.Generator(device=settings["device"]).manual_seed(args.seed)
    output = pipe(
        prompt=args.instruction,
        image=image,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        generator=generator,
    )

    return output.images[0], None


def load_dependencies():
    try:
        import torch
        from diffusers import (
            EulerAncestralDiscreteScheduler,
            StableDiffusionInstructPix2PixPipeline,
        )
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "Missing image inference dependencies. Run `mise run setup-image` from the repository root."
        ) from error

    return {
        "torch": torch,
        "Image": Image,
        "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
        "StableDiffusionInstructPix2PixPipeline": StableDiffusionInstructPix2PixPipeline,
    }


def example_dir(root):
    return root / "image-generation" / "examples"


def resolve_path(path):
    if path.is_absolute():
        return path

    return (Path.cwd() / path).resolve()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(error, file=sys.stderr)
        sys.exit(1)
