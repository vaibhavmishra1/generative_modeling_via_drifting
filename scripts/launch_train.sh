#!/bin/bash
# =========================================================
# Launch distributed training for Drifting Models
# =========================================================
# Usage:
#   bash scripts/launch_train.sh [CONFIG] [NUM_GPUS] [DATA_PATH]
#
# Examples:
#   # 8-GPU latent space training
#   bash scripts/launch_train.sh configs/latent_dit_l2.yaml 8 /data/imagenet
#
#   # 4-GPU pixel space training
#   bash scripts/launch_train.sh configs/pixel_dit_l2.yaml 4 /data/imagenet
#
#   # Single GPU (debugging)
#   bash scripts/launch_train.sh configs/latent_dit_l2.yaml 1 /data/imagenet
# =========================================================

CONFIG=${1:-"configs/latent_dit_l2.yaml"}
NUM_GPUS=${2:-8}
DATA_PATH=${3:-"/path/to/imagenet"}

# Training parameters
MASTER_PORT=${MASTER_PORT:-29500}

echo "================================================"
echo "Drifting Model Training"
echo "================================================"
echo "Config:    ${CONFIG}"
echo "GPUs:      ${NUM_GPUS}"
echo "Data:      ${DATA_PATH}"
echo "Port:      ${MASTER_PORT}"
echo "================================================"

if [ ${NUM_GPUS} -eq 1 ]; then
    # Single GPU
    python train.py \
        --config ${CONFIG} \
        --data-path ${DATA_PATH}
else
    # Multi-GPU with torchrun
    torchrun \
        --nproc_per_node=${NUM_GPUS} \
        --master_port=${MASTER_PORT} \
        train.py \
        --config ${CONFIG} \
        --data-path ${DATA_PATH}
fi
