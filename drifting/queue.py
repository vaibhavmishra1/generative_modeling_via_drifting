"""
Sample Queue for Drifting Models.

Maintains per-class queues of data features and a global queue for
unconditional features. Used for:
1. Providing stable positive/negative samples for drift computation
2. Supporting classifier-free guidance during training
3. Reducing variance in the drift estimate

From the paper:
- Per-class queue size: 128 samples per class
- Global unconditional queue: 1000 samples
- Latest 64 real samples pushed per step
"""

import torch
from typing import Optional, Dict, List, Tuple


class SampleQueue:
    """
    Queue system for managing data and generated feature samples.

    Maintains:
    - Per-class queues of data features for conditional generation
    - A global queue for unconditional generation (CFG)
    - Optionally, queues of generated features for negative sampling

    The queues are FIFO: new samples push out the oldest ones.

    Args:
        num_classes: Number of classes (1000 for ImageNet)
        per_class_size: Maximum samples per class queue
        global_size: Maximum samples in the global (unconditional) queue
        feature_dims: List of feature dimensions for each scale
        device: Device to store queues on
    """

    def __init__(
        self,
        num_classes: int = 1000,
        per_class_size: int = 128,
        global_size: int = 1000,
        feature_dims: List[int] = [1024],
        device: torch.device = torch.device("cpu"),
    ):
        self.num_classes = num_classes
        self.per_class_size = per_class_size
        self.global_size = global_size
        self.feature_dims = feature_dims
        self.num_scales = len(feature_dims)
        self.device = device

        # Per-class data feature queues: {class_id: [List of [queue_tensor per scale]]}
        # Each queue_tensor: [current_size, D_i]
        self.class_queues: Dict[int, List[torch.Tensor]] = {}
        self.class_queue_ptrs: Dict[int, int] = {}  # Circular buffer pointers

        # Global unconditional queue
        self.global_queue: List[torch.Tensor] = [
            torch.zeros(0, d, device=device) for d in feature_dims
        ]
        self.global_ptr: int = 0

        # Generated feature queue (for negative sampling beyond current batch)
        self.gen_queue: List[torch.Tensor] = [
            torch.zeros(0, d, device=device) for d in feature_dims
        ]
        self.gen_ptr: int = 0
        self.gen_queue_size: int = per_class_size  # Same size as class queues

        self._initialized_classes = set()

    def _init_class_queue(self, class_id: int):
        """Initialize a class queue if it doesn't exist."""
        if class_id not in self._initialized_classes:
            self.class_queues[class_id] = [
                torch.zeros(0, d, device=self.device)
                for d in self.feature_dims
            ]
            self.class_queue_ptrs[class_id] = 0
            self._initialized_classes.add(class_id)

    @torch.no_grad()
    def push_data_features(
        self,
        features: List[torch.Tensor],
        labels: torch.Tensor,
    ):
        """
        Push data features into per-class queues and global queue.

        Args:
            features: List of [B, D_i] feature tensors (one per scale)
            labels: [B] class labels
        """
        B = labels.shape[0]

        # Push to global queue
        for s in range(self.num_scales):
            feat = features[s].detach()
            if self.global_queue[s].shape[0] < self.global_size:
                # Still filling up
                self.global_queue[s] = torch.cat(
                    [self.global_queue[s], feat], dim=0
                )[:self.global_size]
            else:
                # Circular buffer
                n = min(B, self.global_size)
                idx = torch.arange(n) + self.global_ptr
                idx = idx % self.global_size
                self.global_queue[s][idx] = feat[:n]

        self.global_ptr = (self.global_ptr + B) % self.global_size

        # Push to per-class queues
        unique_labels = labels.unique()
        for cls_id in unique_labels:
            cls_id_int = cls_id.item()
            self._init_class_queue(cls_id_int)
            mask = labels == cls_id
            cls_features = [features[s][mask].detach() for s in range(self.num_scales)]
            n_cls = mask.sum().item()

            for s in range(self.num_scales):
                if self.class_queues[cls_id_int][s].shape[0] < self.per_class_size:
                    self.class_queues[cls_id_int][s] = torch.cat(
                        [self.class_queues[cls_id_int][s], cls_features[s]], dim=0
                    )[:self.per_class_size]
                else:
                    n = min(n_cls, self.per_class_size)
                    ptr = self.class_queue_ptrs[cls_id_int]
                    idx = torch.arange(n) + ptr
                    idx = idx % self.per_class_size
                    self.class_queues[cls_id_int][s][idx] = cls_features[s][:n]

            self.class_queue_ptrs[cls_id_int] = (
                self.class_queue_ptrs[cls_id_int] + n_cls
            ) % self.per_class_size

    @torch.no_grad()
    def push_gen_features(self, features: List[torch.Tensor]):
        """
        Push generated features into the generated queue (for negatives).

        Args:
            features: List of [B, D_i] feature tensors
        """
        B = features[0].shape[0]

        for s in range(self.num_scales):
            feat = features[s].detach()
            if self.gen_queue[s].shape[0] < self.gen_queue_size:
                self.gen_queue[s] = torch.cat(
                    [self.gen_queue[s], feat], dim=0
                )[:self.gen_queue_size]
            else:
                n = min(B, self.gen_queue_size)
                idx = torch.arange(n) + self.gen_ptr
                idx = idx % self.gen_queue_size
                self.gen_queue[s][idx] = feat[:n]

        self.gen_ptr = (self.gen_ptr + B) % self.gen_queue_size

    @torch.no_grad()
    def get_pos_features(
        self,
        labels: torch.Tensor,
        batch_features: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        """
        Get positive (data) features for given class labels.

        Combines current batch features with queue features for each class.
        Samples are drawn without replacement from the class queue.

        Args:
            labels: [B] class labels for the current batch
            batch_features: Optional current batch data features to include

        Returns:
            List of [total_pos, D_i] positive feature tensors
        """
        all_pos = [[] for _ in range(self.num_scales)]

        # Include current batch features
        if batch_features is not None:
            for s in range(self.num_scales):
                all_pos[s].append(batch_features[s].detach())

        # Include queue features for classes in this batch
        unique_labels = labels.unique()
        for cls_id in unique_labels:
            cls_id_int = cls_id.item()
            if cls_id_int in self._initialized_classes:
                for s in range(self.num_scales):
                    q = self.class_queues[cls_id_int][s]
                    if q.shape[0] > 0:
                        all_pos[s].append(q)

        # Concatenate all positive features
        result = []
        for s in range(self.num_scales):
            if all_pos[s]:
                result.append(torch.cat(all_pos[s], dim=0))
            else:
                result.append(torch.zeros(0, self.feature_dims[s], device=self.device))

        return result

    @torch.no_grad()
    def get_neg_features(self) -> Optional[List[torch.Tensor]]:
        """
        Get negative (generated) features from the queue.

        Returns:
            List of [N, D_i] feature tensors, or None if queue is empty
        """
        if self.gen_queue[0].shape[0] == 0:
            return None
        return [q.clone() for q in self.gen_queue]

    @torch.no_grad()
    def get_global_features(self) -> Optional[List[torch.Tensor]]:
        """
        Get global (unconditional) features for CFG.

        Returns:
            List of [G, D_i] feature tensors, or None if queue is empty
        """
        if self.global_queue[0].shape[0] == 0:
            return None
        return [q.clone() for q in self.global_queue]

    def to(self, device: torch.device) -> "SampleQueue":
        """Move all queues to the specified device."""
        self.device = device
        for cls_id in self._initialized_classes:
            self.class_queues[cls_id] = [
                q.to(device) for q in self.class_queues[cls_id]
            ]
        self.global_queue = [q.to(device) for q in self.global_queue]
        self.gen_queue = [q.to(device) for q in self.gen_queue]
        return self
