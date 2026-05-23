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
from model_factory.image_lora import (
    autocast_context,
    build_lora_config,
    cast_trainable_params,
    diffusion_target,
    freeze_module,
    huggingface_load_kwargs,
    load_image_dependencies,
    random_timesteps,
    resize_center_crop,
    resolve_training_options,
    set_seed,
)
from model_factory.metrics import make_jsonable


@dataclass
class PairRecord:
    before_path: Path
    after_path: Path
    instruction: str


@dataclass
class PairComponents:
    scheduler: object
    tokenizer: object
    text_encoder: object
    vae: object
    unet: object


class PairedImageDataset(Dataset):
    def __init__(self, records, tokenizer, image_module, resolution):
        self.records = records
        self.tokenizer = tokenizer
        self.image_module = image_module
        self.resolution = resolution

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        before_image = self.image_module.open(record.before_path).convert("RGB")
        after_image = self.image_module.open(record.after_path).convert("RGB")
        tokenized = self.tokenizer(
            record.instruction,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "before_pixel_values": image_tensor(before_image, self.resolution),
            "after_pixel_values": image_tensor(after_image, self.resolution),
            "input_ids": tokenized.input_ids[0],
        }


def train_paired_image_translation(config):
    dependencies = load_image_dependencies()
    options = resolve_training_options(config)
    set_seed(options.seed)

    records = load_pair_records(config)
    components = load_pair_components(config, dependencies, options)
    loader = build_pair_loader(records, components.tokenizer, dependencies["Image"], options)
    summary = train_pair_model(components, loader, options, config)
    adapter_path = output_adapter_path(config)
    save_pair_lora_adapter(components.unet, dependencies, adapter_path, options.weight_name)
    summary_path = save_pair_training_summary(config, adapter_path, records, options, summary)

    print(json.dumps(make_jsonable({
        "adapter_path": str(adapter_path),
        "summary_path": str(summary_path),
        "backend": "diffusers",
        "task": config["task"],
        "training_pairs": len(records),
        "steps": summary["steps"],
        "final_loss": summary["final_loss"],
        "device": options.device.type,
    }), indent=2))


def load_pair_records(config):
    data_config = config["data"]
    pairs_path = resolve_path(config, data_config["pairs_path"])

    if not pairs_path.exists():
        raise FileNotFoundError(f"Pairs CSV not found: {pairs_path}")

    frame = pd.read_csv(pairs_path)
    before_column = data_config.get("before_column", "before")
    after_column = data_config.get("after_column", "after")
    instruction_column = data_config.get("instruction_column", "instruction")
    required_columns = [before_column, after_column, instruction_column]
    missing_columns = [column for column in required_columns if column not in frame.columns]

    if missing_columns:
        raise ValueError(f"Pairs CSV is missing columns: {missing_columns}")

    records = [
        PairRecord(
            resolve_pair_path(config, row[before_column]),
            resolve_pair_path(config, row[after_column]),
            clean_instruction(row[instruction_column]),
        )
        for _, row in frame.iterrows()
    ]

    return validate_pair_records(records)


def resolve_pair_path(config, value):
    path = Path(str(value))

    if path.is_absolute():
        return path

    return resolve_path(config, path)


def clean_instruction(value):
    if pd.isna(value) or not str(value).strip():
        raise ValueError("Pairs CSV contains an empty instruction")

    return str(value).strip()


def validate_pair_records(records):
    if not records:
        raise ValueError("Pairs CSV has no training rows")

    missing_paths = []

    for record in records:
        if not record.before_path.exists():
            missing_paths.append(str(record.before_path))

        if not record.after_path.exists():
            missing_paths.append(str(record.after_path))

    if missing_paths:
        raise FileNotFoundError(f"Paired training images not found: {missing_paths}")

    return records


def load_pair_components(config, dependencies, options):
    model_config = config["model"]
    base_model = model_config["base_model"]
    base_kwargs, weight_kwargs = huggingface_load_kwargs(model_config)
    scheduler = dependencies["DDPMScheduler"].from_pretrained(base_model, subfolder="scheduler", **base_kwargs)
    tokenizer = dependencies["CLIPTokenizer"].from_pretrained(base_model, subfolder="tokenizer", **base_kwargs)
    text_encoder = dependencies["CLIPTextModel"].from_pretrained(base_model, subfolder="text_encoder", **base_kwargs)
    vae = dependencies["AutoencoderKL"].from_pretrained(base_model, subfolder="vae", **weight_kwargs)
    unet = dependencies["UNet2DConditionModel"].from_pretrained(base_model, subfolder="unet", **weight_kwargs)

    ensure_instruct_pix2pix_channels(unet)
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

    return PairComponents(
        scheduler=scheduler,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
    )


def ensure_instruct_pix2pix_channels(unet):
    if unet.config.in_channels == 8:
        return

    raise ValueError(
        "paired_image_translation saves LoRA weights only, so model.base_model must already be an "
        f"InstructPix2Pix-style 8-channel UNet. Got {unet.config.in_channels} channels."
    )


def build_pair_loader(records, tokenizer, image_module, options):
    dataset = PairedImageDataset(records, tokenizer, image_module, options.resolution)

    return DataLoader(
        dataset,
        batch_size=options.batch_size,
        shuffle=True,
        num_workers=options.num_workers,
        collate_fn=collate_pair_batch,
    )


def collate_pair_batch(batch):
    return {
        "before_pixel_values": torch.stack([item["before_pixel_values"] for item in batch]),
        "after_pixel_values": torch.stack([item["after_pixel_values"] for item in batch]),
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
    }


def train_pair_model(components, loader, options, config):
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
            final_loss = train_pair_step(components, batch, optimizer, options, config)
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


def train_pair_step(components, batch, optimizer, options, config):
    before_pixel_values = batch["before_pixel_values"].to(options.device, dtype=options.weight_dtype)
    after_pixel_values = batch["after_pixel_values"].to(options.device, dtype=options.weight_dtype)
    input_ids = batch["input_ids"].to(options.device)

    with torch.no_grad():
        latents = components.vae.encode(after_pixel_values).latent_dist.sample()
        latents = latents * components.vae.config.scaling_factor
        before_latents = components.vae.encode(before_pixel_values).latent_dist.mode()
        encoder_hidden_states = components.text_encoder(input_ids)[0]

    noise = torch.randn_like(latents)
    timesteps = random_timesteps(components.scheduler, latents, options.device)
    noisy_latents = components.scheduler.add_noise(latents, noise, timesteps)
    encoder_hidden_states, before_latents = apply_conditioning_dropout(
        components,
        encoder_hidden_states,
        before_latents,
        input_ids,
        config,
    )
    model_input = torch.cat([noisy_latents, before_latents], dim=1)

    with autocast_context(options):
        model_prediction = components.unet(model_input, timesteps, encoder_hidden_states, return_dict=False)[0]
        target = diffusion_target(components.scheduler, latents, noise, timesteps)
        loss = functional.mse_loss(model_prediction.float(), target.float(), reduction="mean")

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return float(loss.detach().cpu())


def apply_conditioning_dropout(components, encoder_hidden_states, before_latents, input_ids, config):
    dropout_probability = (config.get("training") or {}).get("conditioning_dropout_prob")

    if not dropout_probability:
        return encoder_hidden_states, before_latents

    batch_size = before_latents.shape[0]
    random_values = torch.rand(batch_size, device=before_latents.device)
    prompt_mask = (random_values < 2 * float(dropout_probability)).reshape(batch_size, 1, 1)
    null_conditioning = components.text_encoder(empty_input_ids(components, input_ids.shape[0], input_ids.device))[0]
    encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states)
    image_mask = 1 - (
        (random_values >= float(dropout_probability)).to(before_latents.dtype)
        * (random_values < 3 * float(dropout_probability)).to(before_latents.dtype)
    )
    image_mask = image_mask.reshape(batch_size, 1, 1, 1)
    return encoder_hidden_states, image_mask * before_latents


def empty_input_ids(components, batch_size, device):
    tokenized = components.tokenizer(
        [""] * batch_size,
        max_length=components.tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return tokenized.input_ids.to(device)


def save_pair_lora_adapter(unet, dependencies, adapter_path, weight_name):
    adapter_path.mkdir(parents=True, exist_ok=True)
    unet = unet.to(torch.float32)
    lora_state = dependencies["convert_state_dict_to_diffusers"](
        dependencies["get_peft_model_state_dict"](unet)
    )
    dependencies["StableDiffusionInstructPix2PixPipeline"].save_lora_weights(
        save_directory=adapter_path,
        unet_lora_layers=lora_state,
        weight_name=weight_name,
        safe_serialization=True,
    )


def save_pair_training_summary(config, adapter_path, records, options, training_summary):
    summary_path = output_metrics_path(config) or adapter_path / "training-summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "artifact_format": "ai-model-trainer-paired-image-lora-summary-v1",
        "project_name": project_name(config),
        "backend": "diffusers",
        "task": config["task"],
        "base_model": config["model"]["base_model"],
        "adapter_path": str(adapter_path),
        "training_pairs": len(records),
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


def image_tensor(image, resolution):
    image = resize_center_crop(image, resolution)
    image_array = np.asarray(image).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(image_array.transpose(2, 0, 1))
