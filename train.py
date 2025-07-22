import json
import os

import numpy as np
import torch
from ema_pytorch import EMA
from torch.utils.tensorboard import SummaryWriter

from config import config
from constants import EXPERIMENTS_DIR, LOSS_L2, LOSS_LPIPS_HUBER
from constants import UNET_DIM, UNET_DIM_MULTS
from data_utils.data_generator import generate_next_stage_data
from data_utils.datasets import get_dataloaders
from evaluation.fid import InceptionV3, calculate_fid_stats
from models import Unet, MobileResUNet
from training.loss import L2Loss, LPIPSHuberLoss, LPIPSLoss
from training.trainer import train_stage


def create_model(config):
    if config["model_name"] == "Unet":
        return Unet(
            dim=UNET_DIM,
            dim_mults=UNET_DIM_MULTS,
            channels=config['in_channels'],
            out_dim=config['out_channels'],
            dropout=config['dropout_rate']
        ).to(config['device'])
    elif config["model_name"] == "MobileResUNet":
        return MobileResUNet(
            in_channels=config['in_channels'],
            out_channels=config['out_channels'],
            channel_multiplier=config['channel_multiplier'],
            dropout_rate=config['dropout_rate']
        ).to(config['device'])
    else:
        raise ValueError(f"Unknown model name: {config['model_name']}")


def prepare_next_stage_data(config, base_experiment_dir, model, stage_name, num_images):
    data_path = config.get(f"{stage_name}_data_path")
    if data_path is None:
        data_path = os.path.join(base_experiment_dir, f"{stage_name}_data")

    if not (os.path.exists(os.path.join(data_path, "z0_seed0.npy")) and os.path.exists(
            os.path.join(data_path, "z1_seed0.npy"))):
        print(f"--- Generating data for {stage_name.capitalize()} ---")
        generate_next_stage_data(model, config, data_path, num_images)

    print(f"--- Loading offline {stage_name} data ---")
    z0_all = np.load(os.path.join(data_path, "z0_seed0.npy"))
    z1_all = np.load(os.path.join(data_path, "z1_seed0.npy"))

    # Split data into train and validation sets
    num_samples = len(z0_all)
    train_size = int(0.9 * num_samples)

    train_z0, val_z0 = torch.from_numpy(z0_all[:train_size]), torch.from_numpy(z0_all[train_size:])
    train_z1, val_z1 = torch.from_numpy(z1_all[:train_size]), torch.from_numpy(z1_all[train_size:])

    train_dataset = torch.utils.data.TensorDataset(train_z0, train_z1)
    val_dataset = torch.utils.data.TensorDataset(val_z0, val_z1)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    return train_loader, val_loader


if __name__ == "__main__":
    base_experiment_dir = f"{EXPERIMENTS_DIR}/{config['dataset']}/{config['experiment_id']}"
    models_save_dir = os.path.join(base_experiment_dir, "models")
    images_save_dir = os.path.join(base_experiment_dir, "images")
    configs_save_dir = os.path.join(base_experiment_dir, "configs")
    logs_save_dir = os.path.join(base_experiment_dir, "logs")

    os.makedirs(models_save_dir, exist_ok=True)
    os.makedirs(images_save_dir, exist_ok=True)
    os.makedirs(configs_save_dir, exist_ok=True)
    os.makedirs(logs_save_dir, exist_ok=True)

    writer = SummaryWriter(logs_save_dir)

    config_save_path = os.path.join(configs_save_dir, "config.json")
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Config saved to {config_save_path}")

    dataloader, val_loader, fid_dataloader = get_dataloaders(config)

    inception_model = InceptionV3().to(config['device'])
    inception_model.eval()

    mu_real, sigma_real = None, None
    if config["calculate_fid"]:
        fid_stats_path = os.path.join(config['fid_stats_dir'],
                                      f"{config['dataset']}_{config['image_size']}x{config['image_size']}_fid_stats.npz")
        os.makedirs(config['fid_stats_dir'], exist_ok=True)

        if os.path.exists(fid_stats_path):
            print(f"Loading real image statistics from {fid_stats_path}...")
            fid_stats = torch.load(fid_stats_path)
            mu_real, sigma_real = fid_stats['mu_real'], fid_stats['sigma_real']
            print("Done.")
        else:
            print("Pre-calculating real image statistics for FID...")
            mu_real, sigma_real = calculate_fid_stats(fid_dataloader, inception_model, config['device'],
                                                      config['fid_num_images'], config['batch_size'])
            torch.save({'mu_real': mu_real, 'sigma_real': sigma_real}, fid_stats_path)
            print(f"Done. Statistics saved to {fid_stats_path}")

    model_1 = create_model(config)

    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=config['lr_stage1'])

    if config["loss_function"] == LOSS_L2:
        loss_fn = L2Loss()
    elif config["loss_function"] == LOSS_LPIPS_HUBER:
        loss_fn = LPIPSHuberLoss(config['device'])
    else:
        raise ValueError(f"Unknown loss function: {config['loss_function']}")

    ema = EMA(model_1, beta=config['ema_decay'], update_every=10)
    ema.to(config['device'])

    if config["load_model_path"]:
        print(f"Loading model from {config['load_model_path']}...")
        checkpoint = torch.load(config['load_model_path'], map_location=config['device'])
        model_1.load_state_dict(checkpoint['model_state_dict'])
        ema.load_state_dict(checkpoint['ema_state_dict'])
        print("Model loaded successfully.")

    train_stage(config, "Stage 1", model_1, ema, dataloader, val_loader, optimizer_1, loss_fn, config['device'],
                models_save_dir, images_save_dir, mu_real, sigma_real, inception_model, writer)

    if config["epochs_stage2"] > 0:
        reflow_train_loader, reflow_val_loader = prepare_next_stage_data(config, base_experiment_dir, ema.ema_model, "reflow", config["reflow_num_images"])

        model_2 = create_model(config)
        model_2.load_state_dict(ema.ema_model.state_dict())

        optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=config['lr_stage2'])
        ema_2 = EMA(model_2, beta=config['ema_decay'], update_every=10)
        ema_2.to(config['device'])

        train_stage(config, "Stage 2", model_2, ema_2, reflow_train_loader, reflow_val_loader, optimizer_2, loss_fn,
                    config['device'], models_save_dir, images_save_dir, mu_real, sigma_real, inception_model, writer)
    else:
        ema_2 = ema

    if config["epochs_stage3"] > 0:
        distillation_train_loader, distillation_val_loader = prepare_next_stage_data(config, base_experiment_dir, ema_2.ema_model, "distillation", config["distillation_num_images"])

        model_3 = create_model(config)
        model_3.load_state_dict(ema_2.ema_model.state_dict())

        optimizer_3 = torch.optim.Adam(model_3.parameters(), lr=config['lr_stage3'])
        ema_3 = EMA(model_3, beta=config['ema_decay'], update_every=10)
        ema_3.to(config['device'])

        # Substitute loss_fn by LPIPSLoss if LPIPSHuberLoss is specified as there is no t involved in this stage
        if isinstance(loss_fn, LPIPSHuberLoss):
            loss_fn = LPIPSLoss(config['device'])

        train_stage(config, "Stage 3", model_3, ema_3, distillation_train_loader, distillation_val_loader, optimizer_3,
                    loss_fn, config['device'], models_save_dir, images_save_dir, mu_real, sigma_real, inception_model,
                    writer)

    writer.close()

    print("--- Training Finished ---")
