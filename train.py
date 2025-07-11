import torch
import torch.nn as nn
import os
import json
from ema_pytorch import EMA

from config import config
from data_utils.datasets import get_dataloaders
from models import Unet, MobileResUNet
from training.trainer import train_stage1, train_stage2
from evaluation.fid import InceptionV3, calculate_fid_stats

from constants import EXPERIMENTS_DIR, MODEL_UNET, MODEL_MOBILE_RES_UNET

if __name__ == "__main__":
    base_experiment_dir = f"{EXPERIMENTS_DIR}/{config['dataset']}/{config['experiment_id']}"
    models_save_dir = os.path.join(base_experiment_dir, "models")
    images_save_dir = os.path.join(base_experiment_dir, "images")
    configs_save_dir = os.path.join(base_experiment_dir, "configs")

    os.makedirs(models_save_dir, exist_ok=True)
    os.makedirs(images_save_dir, exist_ok=True)
    os.makedirs(configs_save_dir, exist_ok=True)

    config_save_path = os.path.join(configs_save_dir, "config.json")
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Config saved to {config_save_path}")

    dataloader, fid_dataloader = get_dataloaders(config)

    inception_model = InceptionV3().to(config['device'])
    inception_model.eval()

    mu_real, sigma_real = None, None
    if config["calculate_fid"]:
        fid_stats_path = os.path.join(config['fid_stats_dir'], f"{config['dataset']}_{config['image_size']}x{config['image_size']}_fid_stats.npz")
        os.makedirs(config['fid_stats_dir'], exist_ok=True)

        if os.path.exists(fid_stats_path):
            print(f"Loading real image statistics from {fid_stats_path}...")
            fid_stats = torch.load(fid_stats_path)
            mu_real, sigma_real = fid_stats['mu_real'], fid_stats['sigma_real']
            print("Done.")
        else:
            print("Pre-calculating real image statistics for FID...")
            mu_real, sigma_real = calculate_fid_stats(fid_dataloader, inception_model, config['device'], config['fid_num_images'], config['batch_size'])
            torch.save({'mu_real': mu_real, 'sigma_real': sigma_real}, fid_stats_path)
            print(f"Done. Statistics saved to {fid_stats_path}")

    from constants import UNET_DIM, UNET_DIM_MULTS

    if config["model_name"] == "Unet":
        model_1 = Unet(
            dim=UNET_DIM,
            dim_mults=UNET_DIM_MULTS,
            channels=config['in_channels'],
            out_dim=config['out_channels'],
            dropout=config['dropout_rate']
        ).to(config['device'])
    elif config["model_name"] == "MobileResUNet":
        model_1 = MobileResUNet(
            in_channels=config['in_channels'],
            out_channels=config['out_channels'],
            channel_multiplier=config['channel_multiplier']
        ).to(config['device'])
    else:
        raise ValueError(f"Unknown model name: {config['model_name']}")

    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=config['lr'])
    loss_fn = nn.MSELoss()
    ema = EMA(model_1, beta=config['ema_decay'], update_every=10)
    ema.to(config['device'])

    if config["load_model_path"]:
        print(f"Loading model from {config['load_model_path']}...")
        checkpoint = torch.load(config['load_model_path'], map_location=config['device'])
        model_1.load_state_dict(checkpoint['model_state_dict'])
        ema.load_state_dict(checkpoint['ema_state_dict'])
        print("Model loaded successfully.")

    train_stage1(config, model_1, ema, dataloader, optimizer_1, loss_fn, config['device'], models_save_dir, images_save_dir, mu_real, sigma_real, inception_model)

    if config["model_name"] == "Unet":
        model_2 = Unet(
            dim=64,
            dim_mults=(1, 2, 4, 8),
            channels=config['in_channels'],
            out_dim=config['out_channels'],
            dropout=config['dropout_rate']
        ).to(config['device'])
    elif config["model_name"] == "MobileResUNet":
        model_2 = MobileResUNet(
            in_channels=config['in_channels'],
            out_channels=config['out_channels'],
            channel_multiplier=config['channel_multiplier']
        ).to(config['device'])
    else:
        raise ValueError(f"Unknown model name: {config['model_name']}")

    optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=config['lr'])
    ema_2 = EMA(model_2, beta=config['ema_decay'], update_every=10)
    ema_2.to(config['device'])

    train_stage2(config, model_1, model_2, ema_2, dataloader, optimizer_2, loss_fn, config['device'], models_save_dir, images_save_dir, mu_real, sigma_real, inception_model)

    print("--- Training Finished ---")