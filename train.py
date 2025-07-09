import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import os
from tqdm import tqdm
import numpy as np
import json
from ema_pytorch import EMA

from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import config
from models import Unet
from utils import InceptionV3, calculate_fid_stats, calculate_fid_for_model, generate_images, show_images


def train_one_epoch(model, ema, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc="Training Stage 1")
    beta_dist = torch.distributions.beta.Beta(0.5, 0.5)
    for z1, _ in pbar:
        z1 = z1.to(device)
        z0 = torch.randn_like(z1)
        t_scalar = beta_dist.sample((z0.shape[0],)).to(device)
        t = t_scalar.view(-1, 1, 1, 1)
        zt = (1 - t) * z0 + t * z1
        target = z1 - z0
        pred = model(zt, t_scalar)
        loss = loss_fn(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update()
        total_loss += loss.item()
        pbar.set_postfix({"Loss": loss.item()})
    return total_loss / len(dataloader)


if __name__ == "__main__":
    base_experiment_dir = f"experiments/{config['dataset']}/{config['experiment_id']}"
    models_save_dir = os.path.join(base_experiment_dir, "models")
    images_save_dir = os.path.join(base_experiment_dir, "images")
    configs_save_dir = os.path.join(base_experiment_dir, "configs")

    os.makedirs(models_save_dir, exist_ok=True)
    os.makedirs(images_save_dir, exist_ok=True)
    os.makedirs(configs_save_dir, exist_ok=True)

    # Save config to experiment folder
    config_save_path = os.path.join(configs_save_dir, "config.json")
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Config saved to {config_save_path}")

    if config["dataset"] == "MNIST":
        config.update({"in_channels": 1, "out_channels": 1, "norm_mean": (0.5,), "norm_std": (0.5,)})
        dataset_cls = torchvision.datasets.MNIST
    else:
        config.update({"in_channels": 3, "out_channels": 3, "norm_mean": (0.5, 0.5, 0.5), "norm_std": (0.5, 0.5, 0.5)})
        dataset_cls = torchvision.datasets.CIFAR10
    
    transform = transforms.Compose([transforms.Resize((config['image_size'], config['image_size'])), transforms.ToTensor(), transforms.Normalize(config["norm_mean"], config["norm_std"])])
    dataset = dataset_cls(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    fid_dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

    inception_model = InceptionV3().to(config['device'])
    inception_model.eval()

    # --- Load or Calculate Real Image FID Statistics ---
    mu_real, sigma_real = None, None # Initialize to None
    if config["calculate_fid"]:
        fid_stats_path = os.path.join(config['fid_stats_dir'], f"{config['dataset']}_{config['image_size']}x{config['image_size']}_fid_stats.npz")
        os.makedirs(config['fid_stats_dir'], exist_ok=True)

        if os.path.exists(fid_stats_path):
            print(f"Loading real image statistics from {fid_stats_path}...")
            fid_stats = np.load(fid_stats_path)
            mu_real, sigma_real = fid_stats['mu_real'], fid_stats['sigma_real']
            print("Done.")
        else:
            print("Pre-calculating real image statistics for FID...")
            mu_real, sigma_real = calculate_fid_stats(fid_dataloader, inception_model, config['device'])
            np.savez(fid_stats_path, mu_real=mu_real, sigma_real=sigma_real)
            print(f"Done. Statistics saved to {fid_stats_path}")

    if config["model_name"] == "Unet":
        model_1 = Unet(
            dim = 64,
            dim_mults = (1, 2, 4, 8),
            channels = config['in_channels'],
            out_dim = config['out_channels'],
            dropout = config['dropout_rate']
        ).to(config['device'])
    else:
        raise ValueError(f"Unknown model name: {config['model_name']}")

    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=config['lr'])
    scheduler_1 = ReduceLROnPlateau(optimizer_1, mode='min', factor=0.5, patience=5)
    loss_fn = nn.MSELoss()
    ema = EMA(model_1, beta = config['ema_decay'], update_every = 10)
    ema.to(config['device'])

    if config["load_model_path"]:
        print(f"Loading model from {config['load_model_path']}...")
        checkpoint = torch.load(config['load_model_path'], map_location=config['device'])
        model_1.load_state_dict(checkpoint['model_state_dict'])
        ema.load_state_dict(checkpoint['ema_state_dict'])
        print("Model loaded successfully.")

    print("--- Stage 1: Training 1-Rectified Flow Model ---")
    for epoch in range(config['epochs_stage1']):
        avg_loss = train_one_epoch(model_1, ema, dataloader, optimizer_1, loss_fn, config['device'])
        print(f"Stage 1 | Epoch {epoch+1}/{config['epochs_stage1']}, Loss: {avg_loss:.4f}")
        scheduler_1.step(avg_loss)
        vis_images = generate_images(ema.ema_model, config['n_images_to_generate'], config['generation_steps'], config['device'])
        img_title = f"Stage 1, Epoch {epoch+1}"
        if (epoch + 1) % config['checkpoint_interval'] == 0:
            torch.save({'model_state_dict': model_1.state_dict(), 'ema_state_dict': ema.state_dict()}, f"{models_save_dir}/stage1_epoch_{epoch+1}.pth")
            if config["calculate_fid"]:
                fid_score = calculate_fid_for_model(ema.ema_model, mu_real, sigma_real, inception_model, config['device'])
                print(f"FID Score after {epoch+1} epochs: {fid_score:.2f}")
                img_title += f"\nFID: {fid_score:.2f}"
        show_images(vis_images, img_title, f"{images_save_dir}/stage1_epoch_{epoch+1}.png")

    print("--- Stage 1 Finished ---")
    model_1.eval()

    print("\n--- Stage 2: Training 2-Rectified Flow Model (Reflow) ---")
    if config["model_name"] == "Unet":
        model_2 = Unet(
            dim = 64,
            dim_mults = (1, 2, 4, 8),
            channels = config['in_channels'],
            out_dim = config['out_channels'],
            dropout = config['dropout_rate']
        ).to(config['device'])
    else:
        raise ValueError(f"Unknown model name: {config['model_name']}")

    optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=config['lr'])
    scheduler_2 = ReduceLROnPlateau(optimizer_2, mode='min', factor=0.5, patience=5)
    ema_2 = EMA(model_2, beta = config['ema_decay'], update_every = 10)
    ema_2.to(config['device'])
    beta_dist = torch.distributions.beta.Beta(0.5, 0.5)
    
    for epoch in range(config['epochs_stage2']):
        model_2.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Training Stage 2, Epoch {epoch+1}")
        for z1, _ in pbar:
            optimizer_2.zero_grad()

            z1 = z1.to(config['device'])
            z0 = torch.randn_like(z1)

            t_scalar = beta_dist.sample((z0.shape[0],)).to(config['device'])
            t = t_scalar.view(-1, 1, 1, 1)

            zt = (1 - t) * z0 + t * z1
            target = z1 - z0

            pred = model_2(zt, t_scalar)
            loss = loss_fn(pred, target)

            loss.backward()
            optimizer_2.step()
            ema_2.update()

            total_loss += loss.item()
            pbar.set_postfix({"Loss": loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Stage 2 | Epoch {epoch+1}/{config['epochs_stage2']}, Loss: {avg_loss:.4f}")
        scheduler_2.step(avg_loss)
        vis_images = generate_images(ema_2.ema_model, config['n_images_to_generate'], config['generation_steps'], config['device'])
        img_title = f"Stage 2, Epoch {epoch+1}"
        if (epoch + 1) % config['checkpoint_interval'] == 0:
            torch.save({'model_state_dict': model_2.state_dict(), 'ema_state_dict': ema_2.state_dict()}, f"{models_save_dir}/stage2_epoch_{epoch+1}.pth")
            if config["calculate_fid"]:
                fid_score = calculate_fid_for_model(ema_2.ema_model, mu_real, sigma_real, inception_model, config['device'])
                print(f"FID Score after {epoch+1} epochs: {fid_score:.2f}")
                img_title += f"\nFID: {fid_score:.2f}"
        show_images(vis_images, img_title, f"{images_save_dir}/stage2_epoch_{epoch+1}.png")

    print("--- Training Finished ---")