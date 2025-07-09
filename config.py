import torch

config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dataset": "MNIST", # Options: "MNIST", "CIFAR10"
    "experiment_id": "exp_008",
    "image_size": 32,
    "channel_multiplier": 0.375, 
    "batch_size": 64,
    "epochs_stage1": 75,
    "epochs_stage2": 50,
    "lr": 3e-3,
    "n_images_to_generate": 36,
    "generation_steps": 100, 
    "checkpoint_interval": 5,
    "fid_num_images": 2000,
    "fid_stats_dir": "data/fid_stats", # Directory to save/load FID statistics
    "calculate_fid": False,
    "load_model_path": None #"/home/yuqing/rectified_flow/experiments/CIFAR10/exp_007/models/stage1_epoch_75.pth", # Path to a model checkpoint to load (e.g., "/path/to/model.pth")
}
