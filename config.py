import torch

config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dataset": "CIFAR10", # Options: "MNIST", "CIFAR10"
    "experiment_id": "exp_009",
    "image_size": 32,
    "model_name": "Unet", # Options: "MobileResUNet", "Unet"
    "channel_multiplier": 1.0, 
    "batch_size": 64,
    "epochs_stage1": 75,
    "epochs_stage2": 50,
    "lr": 2e-4,
    "dropout_rate": 0.15,
    "ema_decay": 0.999999,
    "n_images_to_generate": 36,
    "generation_steps": 100, 
    "checkpoint_interval": 5,
    "fid_num_images": 2000,
    "fid_stats_dir": "data/fid_stats", # Directory to save/load FID statistics
    "calculate_fid": False,
    "load_model_path": None
}