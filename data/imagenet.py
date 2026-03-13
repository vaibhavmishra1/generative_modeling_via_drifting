"""
ImageNet Dataset and DataLoader for Drifting Models.

Provides:
- ImageNet dataset with proper transforms for training
- Support for both raw pixels and VAE-encoded latents
- Distributed data loading with DistributedSampler
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from typing import Optional, Tuple


def center_crop_arr(pil_image, image_size):
    """
    Center-crop and resize a PIL image to the target size.
    Follows the standard crop used in DiT/ADM.
    """
    from PIL import Image

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = torch.from_numpy(
        __import__("numpy").array(pil_image)
    )

    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


class ImageNetDataset(Dataset):
    """
    ImageNet dataset for drifting model training.

    Returns images in [-1, 1] range with their class labels.

    Args:
        root: Path to ImageNet dataset (should contain 'train' subdirectory)
        image_size: Target image size (default 256)
        split: 'train' or 'val'
    """

    def __init__(
        self,
        root: str,
        image_size: int = 256,
        split: str = "train",
    ):
        self.image_size = image_size

        # Standard ImageNet transforms (following DiT)
        self.transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda arr: arr.float() / 127.5 - 1.0),  # [-1, 1]
            transforms.Lambda(lambda arr: arr.permute(2, 0, 1) if arr.dim() == 3 else arr),
        ])

        data_path = os.path.join(root, split) if split else root
        self.dataset = datasets.ImageFolder(data_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = self.transform(img)
        return img, label


class PrecomputedLatentDataset(Dataset):
    """
    Dataset of precomputed VAE latents for faster training.

    The latent-space training variant can precompute VAE encodings
    to avoid redundant encoding during training.

    Args:
        latent_dir: Directory containing .pt files with latents and labels
        The directory should contain:
            - latents.pt: [N, 4, 32, 32] tensor of latent codes
            - labels.pt: [N] tensor of class labels
    """

    def __init__(self, latent_dir: str):
        self.latents = torch.load(
            os.path.join(latent_dir, "latents.pt"), map_location="cpu"
        )
        self.labels = torch.load(
            os.path.join(latent_dir, "labels.pt"), map_location="cpu"
        )
        assert len(self.latents) == len(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.latents[idx], self.labels[idx]


def build_imagenet_dataset(
    data_path: str,
    image_size: int = 256,
    split: str = "train",
    latent_dir: Optional[str] = None,
) -> Dataset:
    """
    Build ImageNet dataset.

    Args:
        data_path: Path to ImageNet root
        image_size: Target image size
        split: Dataset split
        latent_dir: Optional path to precomputed latents

    Returns:
        Dataset instance
    """
    if latent_dir is not None and os.path.exists(latent_dir):
        return PrecomputedLatentDataset(latent_dir)
    return ImageNetDataset(data_path, image_size=image_size, split=split)


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 256,
    num_workers: int = 8,
    distributed: bool = False,
    seed: int = 0,
) -> Tuple[DataLoader, Optional[torch.utils.data.distributed.DistributedSampler]]:
    """
    Build DataLoader with optional distributed sampling.

    Args:
        dataset: Dataset to load from
        batch_size: Per-GPU batch size
        num_workers: Number of data loading workers
        distributed: Whether to use DistributedSampler
        seed: Random seed for sampler

    Returns:
        (DataLoader, sampler) tuple
    """
    sampler = None
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            shuffle=True,
            seed=seed,
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    return loader, sampler
