import argparse
import json
import os

import numpy as np
import torch

from data_utils.datasets import get_dataloaders
from evaluation.fid import InceptionV3, calculate_fid_for_model, calculate_fid_stats
from models import Unet, MobileResUNet

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the FID score of a trained model.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint.')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the config.json file for the experiment.')
    parser.add_argument('--steps', type=int, required=True, help='Number of generation steps.')
    parser.add_argument('--stage', type=int, required=True, help='The training stage of the model (1, 2, or 3).')
    args = parser.parse_args()

    # --- Load Configuration ---
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    device = config["device"]
    config["generation_steps"] = args.steps

    # --- Load Data and FID Stats ---
    _, _, fid_dataloader = get_dataloaders(config)
    inception_model = InceptionV3().to(device)
    inception_model.eval()

    # --- Load Model ---
    if config["model_name"] == "Unet":
        model = Unet(
            dim=64,
            dim_mults=(1, 2, 4, 8),
            channels=config['in_channels'],
            out_dim=config['out_channels'],
            dropout=config['dropout_rate']
        ).to(device)
    elif config["model_name"] == "MobileResUNet":
        model = MobileResUNet(
            in_channels=config['in_channels'],
            out_channels=config['out_channels'],
            channel_multiplier=config['channel_multiplier'],
            dropout_rate=config['dropout_rate']
        ).to(device)
    else:
        raise ValueError(f"Unknown model name: {config['model_name']}")

    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    fid_stats_path = os.path.join(config['fid_stats_dir'], f"{config['dataset']}_{config['image_size']}x{config['image_size']}_fid_stats.npz")
    if not os.path.exists(fid_stats_path):
        print(f"FID stats not found at {fid_stats_path}. Generating them now...")
        os.makedirs(config['fid_stats_dir'], exist_ok=True)
        mu_real, sigma_real = calculate_fid_stats(fid_dataloader, inception_model, device, config['fid_num_images'], config['batch_size'])
        np.savez(fid_stats_path, mu_real=mu_real, sigma_real=sigma_real)
        print(f"Saved FID stats to {fid_stats_path}")

    stats = np.load(fid_stats_path)
    mu_real, sigma_real = stats['mu_real'], stats['sigma_real']

    # --- Calculate FID ---
    fid_score = calculate_fid_for_model(model, mu_real, sigma_real, inception_model, config, device, args.stage)
    print(f"FID Score ({args.steps} steps): {fid_score:.2f}")