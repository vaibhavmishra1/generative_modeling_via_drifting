"""
Training script for Drifting Models on ImageNet 256x256.

Implements the complete training pipeline from:
  "Generative Modeling via Drifting" (Deng et al., 2025)

Supports:
- Latent-space training (with SD-VAE)
- Pixel-space training
- Multi-GPU training via DDP
- Classifier-free guidance via label dropout
- Multi-scale feature-space drifting loss
- EMA model for generation
- Sample queues for positive/negative features
- WandB logging

Usage:
  # Single GPU
  python train.py --config configs/latent_dit_l2.yaml

  # Multi-GPU with torchrun
  torchrun --nproc_per_node=8 train.py --config configs/latent_dit_l2.yaml

  # Override config values
  python train.py --config configs/latent_dit_l2.yaml --data-path /path/to/imagenet
"""

import os
import sys
import math
import copy
import time
import argparse
import logging
from pathlib import Path
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast

try:
    from omegaconf import OmegaConf
    HAS_OMEGACONF = True
except ImportError:
    HAS_OMEGACONF = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from models.dit import DiT_models, DriftingDiT
from models.vae import AutoencoderKLWrapper
from models.feature_encoder import build_feature_encoder
from drifting.drift_field import compute_drift
from drifting.loss import DriftingLoss
from drifting.queue import SampleQueue
from data.imagenet import build_imagenet_dataset, build_dataloader


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────

def is_main_process():
    """Check if this is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank():
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Update EMA model parameters.

    EMA update rule: θ_ema = decay * θ_ema + (1 - decay) * θ
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """Set requires_grad for all parameters in a model."""
    for p in model.parameters():
        p.requires_grad = flag


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_logger(logging_dir):
    """Create a logger that writes to a file and stdout."""
    if is_main_process():
        os.makedirs(logging_dir, exist_ok=True)
        log_path = os.path.join(logging_dir, "train.log")
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
    return logger


# ─────────────────────────────────────────────────────────────
# Default Configuration
# ─────────────────────────────────────────────────────────────

def get_default_config():
    """
    Default training configuration based on the paper.

    These values are extracted from the paper's experimental setup
    and standard practices for DiT-based models.
    """
    config = {
        # Model
        "model": "DiT-L/2",                # DiT-L/2 for main results (~463M params)
        "input_size": 32,                    # Latent size: 256/8 = 32
        "in_channels": 4,                    # SD-VAE latent channels
        "num_classes": 1000,                 # ImageNet classes
        "class_dropout_prob": 0.1,           # 10% label dropout for CFG

        # Training mode
        "latent_space": True,                # True: latent-space, False: pixel-space
        "vae_path": "stabilityai/sd-vae-ft-mse",  # SD-VAE for latent space

        # Feature encoder (for drifting field computation)
        "feature_encoder": "mae",            # MAE ViT-L/16
        "feature_blocks": [7, 15, 23],       # Multi-scale block indices
        "feature_input_size": 256,           # Input size for feature encoder

        # Drifting field
        "drift_temps": [0.05, 0.05, 0.05],  # Temperature per scale
        "drift_weights": [1.0, 1.0, 1.0],   # Loss weight per scale
        "v_norm": True,                      # V-normalization

        # Queue
        "per_class_queue_size": 128,         # Per-class queue size
        "global_queue_size": 1000,           # Global unconditional queue
        "queue_push_size": 64,               # Samples pushed per step

        # Optimizer
        "optimizer": "adamw",
        "learning_rate": 1e-4,               # Peak learning rate
        "weight_decay": 0.0,                 # Weight decay (0 following DiT convention)
        "beta1": 0.9,
        "beta2": 0.999,
        "grad_clip": 1.0,                   # Gradient clipping norm

        # Training schedule
        "total_steps": 800000,               # Total training iterations
        "warmup_steps": 5000,                # Linear warmup steps
        "batch_size": 32,                    # Per-GPU batch size (256 global / 8 GPUs)

        # EMA
        "ema_decay": 0.9999,

        # Mixed precision
        "use_amp": True,                     # Use automatic mixed precision
        "amp_dtype": "bf16",                 # bf16 or fp16

        # Data
        "data_path": "/path/to/imagenet",    # ImageNet root
        "latent_dir": None,                  # Optional precomputed latents
        "image_size": 256,                   # Image resolution
        "num_workers": 8,

        # Logging & Checkpointing
        "output_dir": "./outputs",
        "log_every": 100,
        "save_every": 10000,
        "sample_every": 5000,
        "use_wandb": False,
        "wandb_project": "drifting-models",
        "experiment_name": "drifting-dit-l2-latent",

        # Reproducibility
        "seed": 42,
        "global_seed": 0,
    }
    return config


# ─────────────────────────────────────────────────────────────
# Learning Rate Schedule
# ─────────────────────────────────────────────────────────────

def get_lr(step, config):
    """
    Learning rate schedule: linear warmup + cosine decay.
    """
    warmup_steps = config["warmup_steps"]
    total_steps = config["total_steps"]
    base_lr = config["learning_rate"]

    if step < warmup_steps:
        return base_lr * step / warmup_steps
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ─────────────────────────────────────────────────────────────
# Training Step
# ─────────────────────────────────────────────────────────────

def train_step(
    model: nn.Module,
    noise: torch.Tensor,
    labels: torch.Tensor,
    real_images: torch.Tensor,
    feature_encoder: nn.Module,
    drift_loss_fn: DriftingLoss,
    sample_queue: SampleQueue,
    vae: Optional[nn.Module] = None,
    scaler: Optional[GradScaler] = None,
    amp_dtype: torch.dtype = torch.bfloat16,
    use_amp: bool = True,
):
    """
    Single training step for the drifting model.

    Flow:
    1. Generate samples: x_gen = f_θ(noise, labels)
    2. If latent mode: decode to pixels for feature extraction
    3. Extract multi-scale features for generated and real samples
    4. Compute drifting loss across scales
    5. Update queue with features

    Args:
        model: Generator network
        noise: [B, C, H, W] input noise
        labels: [B] class labels
        real_images: [B, 3, 256, 256] real images (always in pixel space)
        feature_encoder: Pretrained feature encoder
        drift_loss_fn: Multi-scale drifting loss
        sample_queue: Feature queue system
        vae: VAE for latent-space mode (None for pixel-space)
        scaler: GradScaler for mixed precision
        amp_dtype: AMP dtype

    Returns:
        loss: Scalar loss value
    """
    device = noise.device
    B = noise.shape[0]

    # 1. Generate samples
    with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        gen_output = model(noise, labels)  # [B, C, H, W]

    # 2. Decode to pixel space if in latent mode
    if vae is not None:
        with torch.no_grad():
            gen_pixels = vae.decode(gen_output.float())  # [B, 3, 256, 256]
    else:
        gen_pixels = gen_output  # Already in pixel space

    # 3. Extract features
    with torch.no_grad():
        # Real image features
        pos_features = feature_encoder(real_images)  # List of [B, D_i]

        # Push real features to queue
        sample_queue.push_data_features(pos_features, labels)

    # Generated image features (needs gradient for the generator)
    gen_pixels_for_feat = gen_pixels
    if not gen_pixels_for_feat.requires_grad:
        gen_pixels_for_feat = gen_pixels.detach().requires_grad_(False)

    # Extract features from generated images
    # Note: we need gradients through the feature encoder → generated pixels → generator
    # But the feature encoder is frozen, so gradients flow through its forward pass
    gen_features_for_loss = feature_encoder.extract_features(gen_pixels.detach())

    # Get positive features from queue (includes current batch + historical)
    pos_features_queue = sample_queue.get_pos_features(
        labels, batch_features=pos_features
    )

    # Get negative features (from generated queue)
    neg_features_queue = sample_queue.get_neg_features()

    # 4. Compute drifting loss
    # We need the loss to flow gradients into the generator.
    # Strategy: compute drift targets with detached features,
    # then compute features again with gradient for the loss.
    with torch.no_grad():
        # Compute drift for each scale
        drift_targets = []
        for s in range(drift_loss_fn.num_scales if hasattr(drift_loss_fn, 'num_scales') else len(drift_loss_fn.temps)):
            gen_feat_s = gen_features_for_loss[s]  # [B, D_s]
            pos_feat_s = pos_features_queue[s] if pos_features_queue[s].shape[0] > 0 else pos_features[s]
            neg_feat_s = neg_features_queue[s] if neg_features_queue is not None else None

            V = compute_drift(
                gen=gen_feat_s,
                pos=pos_feat_s,
                temp=drift_loss_fn.temps[s],
                v_norm=drift_loss_fn.v_norm,
                neg=neg_feat_s,
            )
            target = (gen_feat_s + V).detach()
            drift_targets.append(target)

    # Now compute features WITH gradient and apply loss
    # The gradient flows: loss → gen_features (with grad) → gen_pixels → gen_output → model
    with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        if vae is not None:
            gen_pixels_grad = vae.decode(gen_output)
        else:
            gen_pixels_grad = gen_output

        gen_features_grad = feature_encoder.extract_features(gen_pixels_grad)

        total_loss = 0.0
        for s in range(len(drift_loss_fn.temps)):
            scale_loss = F.mse_loss(gen_features_grad[s], drift_targets[s])
            total_loss = total_loss + drift_loss_fn.weights[s] * scale_loss

    # Push generated features to queue (for future negative sampling)
    with torch.no_grad():
        sample_queue.push_gen_features(gen_features_for_loss)

    return total_loss


# ─────────────────────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────────────────────

def main(config):
    """Main training function."""

    # ── Setup distributed ──
    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Seed ──
    seed = config["seed"] + rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ── Output directory ──
    output_dir = Path(config["output_dir"]) / config["experiment_name"]
    ckpt_dir = output_dir / "checkpoints"
    sample_dir = output_dir / "samples"
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(exist_ok=True)
        sample_dir.mkdir(exist_ok=True)

    log = create_logger(str(output_dir))
    if is_main_process():
        log.info(f"Config: {config}")
        log.info(f"Output: {output_dir}")
        log.info(f"World size: {world_size}")

    # ── WandB ──
    if HAS_WANDB and config["use_wandb"] and is_main_process():
        wandb.init(
            project=config["wandb_project"],
            name=config["experiment_name"],
            config=config,
        )

    # ── Build VAE (for latent-space mode) ──
    vae = None
    if config["latent_space"]:
        if is_main_process():
            log.info("Loading SD-VAE for latent-space training...")
        vae = AutoencoderKLWrapper(
            pretrained_path=config["vae_path"],
            scaling_factor=0.18215,
        ).to(device)
        vae.eval()
        requires_grad(vae, False)

    # ── Build feature encoder ──
    if is_main_process():
        log.info(f"Loading feature encoder: {config['feature_encoder']}...")
    feature_encoder = build_feature_encoder(
        encoder_type=config["feature_encoder"],
        extract_blocks=config["feature_blocks"],
        input_size=config["feature_input_size"],
    ).to(device)
    feature_encoder.eval()
    requires_grad(feature_encoder, False)

    feature_dims = feature_encoder.feature_dims
    if is_main_process():
        log.info(f"Feature dims per scale: {feature_dims}")

    # ── Build generator model ──
    if is_main_process():
        log.info(f"Building generator: {config['model']}...")
    model = DiT_models[config["model"]](
        input_size=config["input_size"],
        in_channels=config["in_channels"],
        num_classes=config["num_classes"],
        class_dropout_prob=config["class_dropout_prob"],
    ).to(device)

    if is_main_process():
        n_params = count_parameters(model)
        log.info(f"Generator parameters: {n_params / 1e6:.1f}M")

    # ── EMA model ──
    ema_model = copy.deepcopy(model)
    requires_grad(ema_model, False)
    ema_model.eval()

    # ── DDP ──
    if distributed:
        model = DDP(model, device_ids=[rank])
    raw_model = model.module if distributed else model

    # ── Drifting loss ──
    drift_loss_fn = DriftingLoss(
        temps=config["drift_temps"],
        weights=config["drift_weights"],
        v_norm=config["v_norm"],
    )

    # ── Sample queue ──
    sample_queue = SampleQueue(
        num_classes=config["num_classes"],
        per_class_size=config["per_class_queue_size"],
        global_size=config["global_queue_size"],
        feature_dims=feature_dims,
        device=device,
    )

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=(config["beta1"], config["beta2"]),
    )

    # ── GradScaler for mixed precision ──
    scaler = None
    amp_dtype = torch.bfloat16 if config["amp_dtype"] == "bf16" else torch.float16
    if config["use_amp"] and config["amp_dtype"] == "fp16":
        scaler = GradScaler()

    # ── Dataset ──
    if is_main_process():
        log.info("Loading ImageNet dataset...")
    dataset = build_imagenet_dataset(
        data_path=config["data_path"],
        image_size=config["image_size"],
        split="train",
        latent_dir=config.get("latent_dir"),
    )

    loader, sampler = build_dataloader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        distributed=distributed,
        seed=config["seed"],
    )

    if is_main_process():
        log.info(f"Dataset size: {len(dataset)}")
        log.info(f"Batch size per GPU: {config['batch_size']}")
        log.info(f"Global batch size: {config['batch_size'] * world_size}")
        log.info(f"Total steps: {config['total_steps']}")

    # ── Resume from checkpoint ──
    start_step = 0
    resume_path = config.get("resume", None)
    if resume_path and os.path.exists(resume_path):
        if is_main_process():
            log.info(f"Resuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        raw_model.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        if scaler and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        if is_main_process():
            log.info(f"Resumed at step {start_step}")

    # ── Training loop ──
    if is_main_process():
        log.info("Starting training...")

    model.train()
    data_iter = iter(loader)
    epoch = 0

    running_loss = 0.0
    log_steps = 0
    start_time = time.time()

    for step in range(start_step, config["total_steps"]):
        # Get data batch (handle epoch boundaries)
        try:
            real_images, labels = next(data_iter)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            data_iter = iter(loader)
            real_images, labels = next(data_iter)

        real_images = real_images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        B = real_images.shape[0]

        # Encode to latent space if needed
        if config["latent_space"] and vae is not None:
            with torch.no_grad():
                # Keep pixel images for feature extraction
                pixel_images = real_images.clone()
                # Encode to latent for noise shape reference
                latents = vae.encode(real_images)
        else:
            pixel_images = real_images
            latents = None

        # Generate noise matching the generator input shape
        if config["latent_space"]:
            noise = torch.randn(
                B, config["in_channels"],
                config["input_size"], config["input_size"],
                device=device,
            )
        else:
            noise = torch.randn_like(real_images)

        # Update learning rate
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Forward + backward
        optimizer.zero_grad()

        loss = train_step(
            model=model,
            noise=noise,
            labels=labels,
            real_images=pixel_images,
            feature_encoder=feature_encoder,
            drift_loss_fn=drift_loss_fn,
            sample_queue=sample_queue,
            vae=vae if config["latent_space"] else None,
            scaler=scaler,
            amp_dtype=amp_dtype,
            use_amp=config["use_amp"],
        )

        if scaler is not None:
            scaler.scale(loss).backward()
            if config["grad_clip"] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if config["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()

        # Update EMA
        update_ema(ema_model, raw_model, decay=config["ema_decay"])

        # Logging
        running_loss += loss.item()
        log_steps += 1

        if (step + 1) % config["log_every"] == 0 and is_main_process():
            avg_loss = running_loss / log_steps
            elapsed = time.time() - start_time
            steps_per_sec = log_steps / elapsed

            log.info(
                f"Step {step+1}/{config['total_steps']} | "
                f"Loss: {avg_loss:.4e} | "
                f"LR: {lr:.2e} | "
                f"Steps/s: {steps_per_sec:.2f}"
            )

            if HAS_WANDB and config["use_wandb"]:
                wandb.log({
                    "loss": avg_loss,
                    "lr": lr,
                    "step": step + 1,
                    "steps_per_sec": steps_per_sec,
                })

            running_loss = 0.0
            log_steps = 0
            start_time = time.time()

        # Save checkpoint
        if (step + 1) % config["save_every"] == 0 and is_main_process():
            ckpt = {
                "model": raw_model.state_dict(),
                "ema": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step + 1,
                "config": config,
            }
            if scaler is not None:
                ckpt["scaler"] = scaler.state_dict()

            ckpt_path = ckpt_dir / f"step_{step+1:07d}.pt"
            torch.save(ckpt, str(ckpt_path))
            log.info(f"Saved checkpoint: {ckpt_path}")

            # Also save latest
            latest_path = ckpt_dir / "latest.pt"
            torch.save(ckpt, str(latest_path))

        # Generate samples
        if (step + 1) % config["sample_every"] == 0 and is_main_process():
            generate_samples(
                ema_model, vae, device, config,
                save_dir=str(sample_dir),
                step=step + 1,
            )

    # Final save
    if is_main_process():
        ckpt = {
            "model": raw_model.state_dict(),
            "ema": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": config["total_steps"],
            "config": config,
        }
        torch.save(ckpt, str(ckpt_dir / "final.pt"))
        log.info("Training complete!")

    if distributed:
        dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────
# Sample Generation (during training)
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_samples(model, vae, device, config, save_dir, step, n_samples=64, cfg_scale=1.0):
    """Generate sample images for visualization during training."""
    from torchvision.utils import save_image

    model.eval()
    num_classes = config["num_classes"]

    # Generate class-conditional samples
    labels = torch.randint(0, num_classes, (n_samples,), device=device)

    if cfg_scale > 1.0:
        # CFG: need unconditional + conditional
        noise = torch.randn(
            n_samples, config["in_channels"],
            config["input_size"], config["input_size"],
            device=device,
        )
        # Unconditional labels
        uncond_labels = torch.full_like(labels, num_classes)
        all_labels = torch.cat([labels, uncond_labels])
        all_noise = torch.cat([noise, noise])

        output = model(all_noise, all_labels)
        cond_out, uncond_out = output.chunk(2, dim=0)
        output = uncond_out + cfg_scale * (cond_out - uncond_out)
    else:
        noise = torch.randn(
            n_samples, config["in_channels"],
            config["input_size"], config["input_size"],
            device=device,
        )
        output = model(noise, labels)

    # Decode from latent if needed
    if config["latent_space"] and vae is not None:
        images = vae.decode(output)
    else:
        images = output

    # Save grid
    images = (images + 1) / 2  # [-1, 1] → [0, 1]
    images = images.clamp(0, 1)
    save_image(
        images, os.path.join(save_dir, f"samples_step{step:07d}.png"),
        nrow=int(n_samples ** 0.5),
        padding=2,
    )

    model.train()


# ─────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train Drifting Model")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--data-path", type=str, default=None, help="ImageNet data path")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--model", type=str, default=None, help="Model name (e.g., DiT-L/2)")
    parser.add_argument("--batch-size", type=int, default=None, help="Per-GPU batch size")
    parser.add_argument("--total-steps", type=int, default=None, help="Total training steps")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--latent-space", action="store_true", default=None)
    parser.add_argument("--pixel-space", action="store_true", default=None)
    parser.add_argument("--use-wandb", action="store_true", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Start with defaults
    config = get_default_config()

    # Override with YAML config if provided
    if args.config and HAS_OMEGACONF:
        yaml_config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
        config.update(yaml_config)

    # Override with command-line args
    cli_overrides = {
        "data_path": args.data_path,
        "output_dir": args.output_dir,
        "model": args.model,
        "batch_size": args.batch_size,
        "total_steps": args.total_steps,
        "learning_rate": args.learning_rate,
        "resume": args.resume,
        "use_wandb": args.use_wandb,
        "seed": args.seed,
        "experiment_name": args.experiment_name,
    }

    if args.pixel_space:
        cli_overrides["latent_space"] = False
        cli_overrides["in_channels"] = 3
        cli_overrides["input_size"] = 256

    for k, v in cli_overrides.items():
        if v is not None:
            config[k] = v

    main(config)
