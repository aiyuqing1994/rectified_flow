import torch
from constants import (
    MODEL_UNET, MODEL_MOBILE_RES_UNET, DATASET_CIFAR10, DATASET_MNIST
)

config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dataset": DATASET_CIFAR10,  # Options: DATASET_MNIST, DATASET_CIFAR10
    "experiment_id": "exp_010",
    "image_size": 32,
    "model_name": MODEL_MOBILE_RES_UNET,  # Options: MODEL_MOBILE_RES_UNET, MODEL_UNET
    "channel_multiplier": 0.375,
    "batch_size": 64,
    "epochs_stage1": 75,
    "epochs_stage2": 50,
    "lr": 3e-3,
    "dropout_rate": 0.15,
    "ema_decay": 0.999999,
    "n_images_to_generate": 36,
    "generation_steps": 100,
    "checkpoint_interval": 5,
    "fid_num_images": 2000,
    "fid_stats_dir": "data/fid_stats",  # Directory to save/load FID statistics
    "calculate_fid": False,
    "load_model_path": None
}
