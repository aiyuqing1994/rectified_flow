import torch
from constants import (
    MODEL_UNET, MODEL_MOBILE_RES_UNET, DATASET_CIFAR10, DATASET_MNIST, LOSS_L2, LOSS_LPIPS_HUBER
)

config = {
    "dataset": DATASET_MNIST,  # Options: DATASET_MNIST, DATASET_CIFAR10
    "loss_function": LOSS_LPIPS_HUBER,  # Options: LOSS_L2, LOSS_LPIPS_HUBER
    "sampler": "U-shape",  # Options: "Uniform", "U-shape"
    "model_name": MODEL_MOBILE_RES_UNET,  # Options: MODEL_MOBILE_RES_UNET, MODEL_UNET

    "experiment_id": "Stage_2+3_Lpips_Huber_UShape",

    "device": "cuda" if torch.cuda.is_available() else "cpu",

    "image_size": 32,
    "channel_multiplier": 0.75,
    "batch_size": 64,
    "dropout_rate": 0.1,
    "ema_decay": 0.999999,
    "n_images_to_generate": 36,
    "generation_steps": 100,
    "checkpoint_interval": 5,

    "epochs_stage1": 0,
    "epochs_stage2": 50,
    "epochs_stage3": 50,
    "lr_stage1": 5e-3,
    "lr_stage2": 5e-4,
    "lr_stage3": 2e-4,

    "reflow_num_images": 100000,
    "distillation_num_images": 100000,

    "calculate_fid": False,
    "fid_num_images": 20000,
    "fid_stats_dir": "data/fid_stats",  # Directory to save/load FID statistics

    "load_model_path": "experiments/MNIST/Stage_1_MSE_Uniform/models/stage1_epoch_200.pth",
    #"reflow_data_path": "experiments/MNIST/exp_027/reflow_data",
    #"distillation_data_path": "experiments/MNIST/exp_028/distillation_data",
}
