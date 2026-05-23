from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

from model_factory.artifacts import output_adapter_path, output_metrics_path
from model_factory.config import project_name, public_config, resolve_path
from model_factory.metrics import make_jsonable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_TARGET_MODULES = ["to_k", "to_q", "to_v", "to_out.0"]


@dataclass
class ImageRecord:
    image_path: Path
    caption: str


@dataclass
class TrainingOptions:
    resolution: int
    batch_size: int
    epochs: int
    max_train_steps: int | None
    learning_rate: float
    seed: int
    device: torch.device
    weight_dtype: torch.dtype
    num_workers: int
    gradient_checkpointing: bool
    weight_name: str


@dataclass
class LoraComponents:
    scheduler: object
    tokenizer: object
    text_encoder: object
    vae: object
    unet: object


class ImageCaptionDataset(Dataset):
    def __init__(self, records, tokenizer, image_module, resolution):
        self.records = records
        self.tokenizer = tokenizer
        self.image_module = image_module
        self.resolution = resolution

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = self.image_module.open(record.image_path).convert("RGB")
        image = resize_center_crop(image, self.resolution)
        image_array = np.asarray(image).astype(np.float32) / 127.5 - 1.0
        pixel_values = torch.from_numpy(image_array.transpose(2, 0, 1))
        tokenized = self.tokenizer(
            record.caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized.input_ids[0],
        }


def train_image_generation_lora(config):
    dependencies = load_image_dependencies()
    options = resolve_training_options(config)
    set_seed(options.seed)

    records = load_image_records(config)
    components = load_lora_components(config, dependencies, options)
    loader = build_data_loader(records, components.tokenizer, dependencies["Image"], options)
    summary = train_lora_model(components, loader, options)
    adapter_path = output_adapter_path(config)
    save_lora_adapter(components.unet, dependencies, adapter_path, options.weight_name)
    summary_path = save_training_summary(config, adapter_path, records, options, summary)

    print(json.dumps(make_jsonable({
        "adapter_path": str(adapter_path),
        "summary_path": str(summary_path),
        "backend": "diffusers",
        "task": config["task"],
        "training_images": len(records),
        "steps": summary["steps"],
        "final_loss": summary["final_loss"],
        "device": options.device.type,
    }), indent=2))


def load_image_dependencies():
    missing = []
    dependencies = {}

    try:
        from PIL import Image

        dependencies["Image"] = Image
    except ImportError:
        missing.append("Pillow")

    try:
        from diffusers import (
            AutoencoderKL,
            DDPMScheduler,
            StableDiffusionInstructPix2PixPipeline,
            StableDiffusionPipeline,
            UNet2DConditionModel,
        )
        from diffusers.utils import convert_state_dict_to_diffusers

        dependencies["AutoencoderKL"] = AutoencoderKL
        dependencies["DDPMScheduler"] = DDPMScheduler
        dependencies["StableDiffusionInstructPix2PixPipeline"] = StableDiffusionInstructPix2PixPipeline
        dependencies["StableDiffusionPipeline"] = StableDiffusionPipeline
        dependencies["UNet2DConditionModel"] = UNet2DConditionModel
        dependencies["convert_state_dict_to_diffusers"] = convert_state_dict_to_diffusers
    except ImportError:
        missing.append("diffusers")

    try:
        from peft import LoraConfig, get_peft_model_state_dict

        dependencies["LoraConfig"] = LoraConfig
        dependencies["get_peft_model_state_dict"] = get_peft_model_state_dict
    except ImportError:
        missing.append("peft")

    try:
        from transformers import CLIPTextModel, CLIPTokenizer

        dependencies["CLIPTextModel"] = CLIPTextModel
        dependencies["CLIPTokenizer"] = CLIPTokenizer
    except ImportError:
        missing.append("transformers")

    if missing:
        raise ImportError(
            "Image generation training requires optional dependencies. "
            "Install them with `.\\.venv\\Scripts\\python.exe -m pip install -r .\\requirements-image.txt`. "
            f"Missing: {', '.join(sorted(set(missing)))}"
        )

    return dependencies


def resolve_training_options(config):
    training_config = config.get("training") or {}
    device = resolve_device(training_config.get("device", "auto"))
    mixed_precision = training_config.get("mixed_precision", "no")

    return TrainingOptions(
        resolution=int(training_config.get("resolution", 512)),
        batch_size=int(training_config.get("batch_size", training_config.get("train_batch_size", 1))),
        epochs=int(training_config.get("epochs", 1)),
        max_train_steps=optional_int(training_config.get("max_train_steps")),
        learning_rate=float(training_config.get("learning_rate", 0.0001)),
        seed=int(training_config.get("seed", training_config.get("random_state", 42))),
        device=device,
        weight_dtype=resolve_weight_dtype(device, mixed_precision),
        num_workers=int(training_config.get("num_workers", 0)),
        gradient_checkpointing=bool(training_config.get("gradient_checkpointing", False)),
        weight_name=training_config.get("weight_name", "pytorch_lora_weights.safetensors"),
    )


def resolve_device(configured_device):
    if configured_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(configured_device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is false")

    return device


def resolve_weight_dtype(device, mixed_precision):
    if device.type != "cuda":
        return torch.float32

    if mixed_precision == "fp16":
        return torch.float16

    if mixed_precision == "bf16":
        return torch.bfloat16

    return torch.float32


def optional_int(value):
    if value is None:
        return None

    return int(value)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_image_records(config):
    data_config = config["data"]
    image_dir = resolve_path(config, data_config["image_dir"])

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    if data_config.get("captions_path"):
        records = load_caption_csv_records(config, image_dir)
    else:
        records = load_sidecar_caption_records(image_dir, data_config)

    return validate_image_records(records, image_dir)


def load_caption_csv_records(config, image_dir):
    data_config = config["data"]
    captions_path = resolve_path(config, data_config["captions_path"])

    if not captions_path.exists():
        raise FileNotFoundError(f"Captions CSV not found: {captions_path}")

    frame = pd.read_csv(captions_path)
    image_column = data_config.get("image_column", "image")
    caption_column = data_config.get("caption_column", "caption")
    missing_columns = [column for column in [image_column, caption_column] if column not in frame.columns]

    if missing_columns:
        raise ValueError(f"Captions CSV is missing columns: {missing_columns}")

    records = []

    for _, row in frame.iterrows():
        caption = row[caption_column]

        if pd.isna(caption) or not str(caption).strip():
            raise ValueError(f"Captions CSV contains an empty caption in column: {caption_column}")

        records.append(ImageRecord(
            resolve_image_path(image_dir, row[image_column]),
            apply_caption_prefix(str(caption), data_config),
        ))

    return records


def validate_image_records(records, image_dir):
    if not records:
        raise ValueError(f"No training images found in {image_dir}")

    missing_images = [str(record.image_path) for record in records if not record.image_path.exists()]

    if missing_images:
        raise FileNotFoundError(f"Training images not found: {missing_images}")

    return records


def resolve_image_path(image_dir, image_value):
    image_path = Path(str(image_value))

    if image_path.is_absolute():
        return image_path

    return (image_dir / image_path).resolve()


def apply_caption_prefix(caption, data_config):
    prefix = data_config.get("caption_prefix")

    if not prefix:
        return caption.strip()

    return f"{prefix.strip()} {caption.strip()}".strip()


def load_sidecar_caption_records(image_dir, data_config):
    sidecar_extension = data_config.get("sidecar_extension", ".txt")
    instance_prompt = data_config.get("instance_prompt")
    records = []

    for image_path in iter_image_paths(image_dir, data_config.get("recursive", False)):
        caption = read_caption_sidecar(image_path, sidecar_extension)

        if not caption and instance_prompt:
            caption = instance_prompt

        if not caption:
            raise ValueError(f"Missing caption sidecar for image: {image_path}")

        records.append(ImageRecord(image_path, apply_caption_prefix(caption, data_config)))

    return records


def iter_image_paths(image_dir, recursive):
    iterator = image_dir.rglob("*") if recursive else image_dir.glob("*")

    return sorted(
        path for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_caption_sidecar(image_path, sidecar_extension):
    if not sidecar_extension.startswith("."):
        sidecar_extension = f".{sidecar_extension}"

    caption_path = image_path.with_suffix(sidecar_extension)

    if not caption_path.exists():
        return None

    return caption_path.read_text(encoding="utf-8").strip()


def load_lora_components(config, dependencies, options):
    model_config = config["model"]
    base_model = model_config["base_model"]
    base_kwargs, weight_kwargs = huggingface_load_kwargs(model_config)
    scheduler = dependencies["DDPMScheduler"].from_pretrained(base_model, subfolder="scheduler", **base_kwargs)
    tokenizer = dependencies["CLIPTokenizer"].from_pretrained(base_model, subfolder="tokenizer", **base_kwargs)
    text_encoder = dependencies["CLIPTextModel"].from_pretrained(base_model, subfolder="text_encoder", **base_kwargs)
    vae = dependencies["AutoencoderKL"].from_pretrained(base_model, subfolder="vae", **weight_kwargs)
    unet = dependencies["UNet2DConditionModel"].from_pretrained(base_model, subfolder="unet", **weight_kwargs)

    freeze_module(text_encoder)
    freeze_module(vae)
    freeze_module(unet)
    unet.add_adapter(build_lora_config(model_config, dependencies["LoraConfig"]))

    text_encoder.to(options.device, dtype=options.weight_dtype)
    vae.to(options.device, dtype=options.weight_dtype)
    unet.to(options.device, dtype=options.weight_dtype)
    cast_trainable_params(unet)

    if options.gradient_checkpointing and hasattr(unet, "enable_gradient_checkpointing"):
        unet.enable_gradient_checkpointing()

    return LoraComponents(
        scheduler=scheduler,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
    )


def huggingface_load_kwargs(model_config):
    base_kwargs = {}

    for key in ["revision", "local_files_only", "token"]:
        if key in model_config:
            base_kwargs[key] = model_config[key]

    weight_kwargs = dict(base_kwargs)

    if model_config.get("variant"):
        weight_kwargs["variant"] = model_config["variant"]

    return base_kwargs, weight_kwargs


def freeze_module(module):
    module.requires_grad_(False)
    module.eval()


def build_lora_config(model_config, lora_config_class):
    params = model_config.get("params") or {}
    rank = int(params.get("rank", 16))

    return lora_config_class(
        r=rank,
        lora_alpha=int(params.get("alpha", rank)),
        lora_dropout=float(params.get("dropout", 0.0)),
        init_lora_weights=params.get("init_lora_weights", "gaussian"),
        target_modules=params.get("target_modules", DEFAULT_TARGET_MODULES),
    )


def cast_trainable_params(module):
    for parameter in module.parameters():
        if not parameter.requires_grad:
            continue

        parameter.data = parameter.data.to(torch.float32)


def build_data_loader(records, tokenizer, image_module, options):
    dataset = ImageCaptionDataset(records, tokenizer, image_module, options.resolution)

    return DataLoader(
        dataset,
        batch_size=options.batch_size,
        shuffle=True,
        num_workers=options.num_workers,
        collate_fn=collate_batch,
    )


def collate_batch(batch):
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
    }


def train_lora_model(components, loader, options):
    optimizer = torch.optim.AdamW(
        [parameter for parameter in components.unet.parameters() if parameter.requires_grad],
        lr=options.learning_rate,
    )
    final_loss = 0.0
    global_step = 0
    epoch = 0

    components.unet.train()

    while should_continue_training(epoch, global_step, options):
        epoch += 1

        for batch in loader:
            final_loss = train_lora_step(components, batch, optimizer, options)
            global_step += 1

            if options.max_train_steps and global_step >= options.max_train_steps:
                break

    return {
        "epochs_completed": epoch,
        "steps": global_step,
        "final_loss": final_loss,
    }


def should_continue_training(epoch, global_step, options):
    if options.max_train_steps:
        return global_step < options.max_train_steps

    return epoch < options.epochs


def train_lora_step(components, batch, optimizer, options):
    pixel_values = batch["pixel_values"].to(options.device, dtype=options.weight_dtype)
    input_ids = batch["input_ids"].to(options.device)

    with torch.no_grad():
        latents = components.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * components.vae.config.scaling_factor
        encoder_hidden_states = components.text_encoder(input_ids)[0]

    noise = torch.randn_like(latents)
    timesteps = random_timesteps(components.scheduler, latents, options.device)
    noisy_latents = components.scheduler.add_noise(latents, noise, timesteps)

    with autocast_context(options):
        model_prediction = components.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        target = diffusion_target(components.scheduler, latents, noise, timesteps)
        loss = functional.mse_loss(model_prediction.float(), target.float(), reduction="mean")

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return float(loss.detach().cpu())


def random_timesteps(scheduler, latents, device):
    return torch.randint(
        0,
        scheduler.config.num_train_timesteps,
        (latents.shape[0],),
        device=device,
        dtype=torch.long,
    )


def diffusion_target(scheduler, latents, noise, timesteps):
    if getattr(scheduler.config, "prediction_type", None) == "v_prediction":
        return scheduler.get_velocity(latents, noise, timesteps)

    return noise


def autocast_context(options):
    if options.device.type == "cuda" and options.weight_dtype != torch.float32:
        return torch.autocast(device_type="cuda", dtype=options.weight_dtype)

    return nullcontext()


def save_lora_adapter(unet, dependencies, adapter_path, weight_name):
    adapter_path.mkdir(parents=True, exist_ok=True)
    unet = unet.to(torch.float32)
    lora_state = dependencies["convert_state_dict_to_diffusers"](
        dependencies["get_peft_model_state_dict"](unet)
    )
    dependencies["StableDiffusionPipeline"].save_lora_weights(
        save_directory=adapter_path,
        unet_lora_layers=lora_state,
        weight_name=weight_name,
        safe_serialization=True,
    )


def save_training_summary(config, adapter_path, records, options, training_summary):
    summary_path = output_metrics_path(config) or adapter_path / "training-summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "artifact_format": "ai-model-trainer-image-lora-summary-v1",
        "project_name": project_name(config),
        "backend": "diffusers",
        "task": config["task"],
        "base_model": config["model"]["base_model"],
        "adapter_path": str(adapter_path),
        "training_images": len(records),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "config": public_config(config),
        "training": {
            "resolution": options.resolution,
            "batch_size": options.batch_size,
            "epochs_completed": training_summary["epochs_completed"],
            "steps": training_summary["steps"],
            "learning_rate": options.learning_rate,
            "device": options.device.type,
            "weight_dtype": str(options.weight_dtype).replace("torch.", ""),
            "final_loss": training_summary["final_loss"],
        },
    }

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(make_jsonable(summary), file, indent=2)

    return summary_path


def resize_center_crop(image, resolution):
    width, height = image.size
    scale = resolution / min(width, height)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    image = image.resize((resized_width, resized_height), resampling_filter())
    left = (resized_width - resolution) // 2
    top = (resized_height - resolution) // 2
    return image.crop((left, top, left + resolution, top + resolution))


def resampling_filter():
    from PIL import Image

    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS

    return Image.LANCZOS
