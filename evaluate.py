"""
Evaluation script for Drifting Models.

Computes:
- FID (Fréchet Inception Distance) using clean-fid
- IS (Inception Score)

Usage:
  # Full evaluation pipeline (generate + evaluate)
  python evaluate.py --ckpt checkpoints/final.pt --ref-path /path/to/imagenet/train \
                     --num-samples 50000 --cfg-scale 1.0

  # Evaluate existing generated samples
  python evaluate.py --gen-dir generated_samples --ref-path /path/to/imagenet/train

  # Quick evaluation with fewer samples
  python evaluate.py --ckpt checkpoints/final.pt --ref-path /path/to/imagenet/train \
                     --num-samples 10000 --cfg-scale 1.0
"""

import os
import argparse
import torch

try:
    from cleanfid import fid as cleanfid
    HAS_CLEANFID = True
except ImportError:
    HAS_CLEANFID = False

try:
    from pytorch_fid import fid_score
    HAS_PYTORCH_FID = True
except ImportError:
    HAS_PYTORCH_FID = False


def compute_fid(gen_dir: str, ref_path: str, mode: str = "clean", device=None):
    """
    Compute FID between generated images and reference dataset.

    Args:
        gen_dir: Directory of generated images (.png)
        ref_path: Path to reference images (ImageNet train)
        mode: FID computation mode ("clean" uses clean-fid)
        device: CUDA device

    Returns:
        FID score
    """
    if HAS_CLEANFID:
        print("Computing FID with clean-fid...")
        score = cleanfid.compute_fid(
            gen_dir,
            ref_path,
            mode=mode,
            num_workers=8,
            device=device,
        )
        return score
    elif HAS_PYTORCH_FID:
        print("Computing FID with pytorch-fid...")
        score = fid_score.calculate_fid_given_paths(
            [gen_dir, ref_path],
            batch_size=64,
            device=device or "cuda",
            dims=2048,
        )
        return score
    else:
        raise ImportError(
            "No FID computation library found. Install clean-fid or pytorch-fid:\n"
            "  pip install clean-fid\n"
            "  pip install pytorch-fid"
        )


def compute_inception_score(gen_dir: str, splits: int = 10):
    """
    Compute Inception Score for generated images.

    Args:
        gen_dir: Directory of generated images
        splits: Number of splits for IS calculation

    Returns:
        (mean IS, std IS)
    """
    try:
        from torchmetrics.image.inception import InceptionScore
        import torchvision.transforms as T
        from PIL import Image
        from tqdm import tqdm

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inception = InceptionScore(splits=splits).to(device)

        transform = T.Compose([
            T.Resize((299, 299)),
            T.ToTensor(),
        ])

        image_files = sorted([
            f for f in os.listdir(gen_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])

        batch_size = 64
        for i in tqdm(range(0, len(image_files), batch_size), desc="Computing IS"):
            batch_files = image_files[i:i + batch_size]
            images = []
            for f in batch_files:
                img = Image.open(os.path.join(gen_dir, f)).convert("RGB")
                img = transform(img)
                images.append(img)

            images = torch.stack(images).to(device)
            images = (images * 255).to(torch.uint8)
            inception.update(images)

        mean_is, std_is = inception.compute()
        return mean_is.item(), std_is.item()

    except ImportError:
        print("Warning: torchmetrics not installed. Skipping IS computation.")
        print("Install with: pip install torchmetrics[image]")
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Evaluate Drifting Model")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--gen-dir", type=str, default=None, help="Pre-generated images directory")
    parser.add_argument("--ref-path", type=str, required=True, help="Reference ImageNet path")
    parser.add_argument("--output-dir", type=str, default="eval_samples")
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compute-is", action="store_true", help="Also compute Inception Score")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate samples if needed
    gen_dir = args.gen_dir
    if gen_dir is None:
        assert args.ckpt is not None, "Either --ckpt or --gen-dir must be provided"

        from generate import load_model_from_checkpoint, generate

        model, config, vae = load_model_from_checkpoint(args.ckpt, device)

        gen_dir = args.output_dir
        generate(
            model=model,
            vae=vae,
            device=device,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            num_classes=config["num_classes"],
            cfg_scale=args.cfg_scale,
            latent_space=config.get("latent_space", True),
            in_channels=config["in_channels"],
            input_size=config["input_size"],
            output_dir=gen_dir,
            save_grid=True,
            save_individual=True,
            seed=args.seed,
        )

    # Compute FID
    print(f"\nComputing FID...")
    print(f"  Generated: {gen_dir}")
    print(f"  Reference: {args.ref_path}")

    fid_value = compute_fid(gen_dir, args.ref_path, device=device)
    print(f"\n  FID: {fid_value:.2f}")

    # Compute IS if requested
    if args.compute_is:
        print(f"\nComputing Inception Score...")
        is_mean, is_std = compute_inception_score(gen_dir)
        if is_mean is not None:
            print(f"  IS: {is_mean:.1f} ± {is_std:.1f}")

    # Save results
    results = {
        "fid": fid_value,
        "cfg_scale": args.cfg_scale,
        "num_samples": args.num_samples,
    }
    if args.compute_is:
        results["is_mean"] = is_mean
        results["is_std"] = is_std

    import json
    results_path = os.path.join(gen_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
