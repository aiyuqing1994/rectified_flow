import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import os
from tqdm import tqdm
import numpy as np
import json

from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import config
from models import MobileResUNet
from utils import InceptionV3, calculate_fid_stats, calculate_fid_for_model, generate_images, show_images


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc="Training Stage 1")
    beta_dist = torch.distributions.beta.Beta(0.5, 0.5)
    for z1, _ in pbar:
        z1, z0 = z1.to(device), torch.randn_like(z1)
        t_scalar = beta_dist.sample((z0.shape[0],)).to(device)
        t = t_scalar.view(-1, 1, 1, 1)
        zt = (1 - t) * z0 + t * z1
        target = z1 - z0
        pred = model(zt, t_scalar)
        loss = loss_fn(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({"Loss": loss.item()})
    return total_loss / len(dataloader)


if __name__ == "__main__":
    base_experiment_dir = f"/home/yuqing/rectified_flow/experiments/{config['dataset']}/{config['experiment_id']}"
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

    model_1 = MobileResUNet(config['in_channels'], config['out_channels'], channel_multiplier=config['channel_multiplier'], num_groups=4).to(config['device'])
    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=config['lr'])
    scheduler_1 = ReduceLROnPlateau(optimizer_1, mode='min', factor=0.5, patience=5)
    loss_fn = nn.MSELoss()

    if config["load_model_path"]:
        print(f"Loading model from {config['load_model_path']}...")
        checkpoint = torch.load(config['load_model_path'], map_location=config['device'])
        model_1.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully.")

    print("--- Stage 1: Training 1-Rectified Flow Model ---")
    for epoch in range(config['epochs_stage1']):
        avg_loss = train_one_epoch(model_1, dataloader, optimizer_1, loss_fn, config['device'])
        print(f"Stage 1 | Epoch {epoch+1}/{config['epochs_stage1']}, Loss: {avg_loss:.4f}")
        scheduler_1.step(avg_loss)
        vis_images = generate_images(model_1, config['n_images_to_generate'], config['generation_steps'], config['device'])
        img_title = f"Stage 1, Epoch {epoch+1}"
        if (epoch + 1) % config['checkpoint_interval'] == 0:
            torch.save({'model_state_dict': model_1.state_dict()}, f"{models_save_dir}/stage1_epoch_{epoch+1}.pth")
            if config["calculate_fid"]:
                fid_score = calculate_fid_for_model(model_1, mu_real, sigma_real, inception_model, config['device'])
                print(f"FID Score after {epoch+1} epochs: {fid_score:.2f}")
                img_title += f"\nFID: {fid_score:.2f}"
        show_images(vis_images, img_title, f"{images_save_dir}/stage1_epoch_{epoch+1}.png")

    print("--- Stage 1 Finished ---")
    model_1.eval()

    print("\n--- Stage 2: Training 2-Rectified Flow Model (Reflow) ---")
    model_2 = MobileResUNet(config['in_channels'], config['out_channels'], channel_multiplier=config['channel_multiplier'], num_groups=4).to(config['device'])
    optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=config['lr'])
    scheduler_2 = ReduceLROnPlateau(optimizer_2, mode='min', factor=0.5, patience=5)
    beta_dist = torch.distributions.beta.Beta(0.5, 0.5)
    
    for epoch in range(config['epochs_stage2']):
        model_2.train()
        total_loss = 0
        pbar = tqdm(range(len(dataloader)), desc=f"Training Stage 2, Epoch {epoch+1}")
        for _ in pbar:
            z0 = torch.randn(config['batch_size'], config['in_channels'], config['image_size'], config['image_size'], device=config['device'])
            z1 = generate_images(model_1, config['batch_size'], config['generation_steps'], config['device'], start_noise=z0)
            t = beta_dist.sample((z0.shape[0],)).to(config['device']).view(-1, 1, 1, 1)
            zt = (1 - t) * z0 + t * z1
            target = z1 - z0
            pred = model_2(zt, t.squeeze())
            loss = loss_fn(pred, target)
            optimizer_2.zero_grad()
            loss.backward()
            optimizer_2.step()
            total_loss += loss.item()
            pbar.set_postfix({"Loss": loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Stage 2 | Epoch {epoch+1}/{config['epochs_stage2']}, Loss: {avg_loss:.4f}")
        scheduler_2.step(avg_loss)
        vis_images = generate_images(model_2, config['n_images_to_generate'], config['generation_steps'], config['device'])
        img_title = f"Stage 2, Epoch {epoch+1}"
        if (epoch + 1) % config['checkpoint_interval'] == 0:
            torch.save({'model_state_dict': model_2.state_dict()}, f"{models_save_dir}/stage2_epoch_{epoch+1}.pth")
            if config["calculate_fid"]:
                fid_score = calculate_fid_for_model(model_2, mu_real, sigma_real, inception_model, config['device'])
                print(f"FID Score after {epoch+1} epochs: {fid_score:.2f}")
                img_title += f"\nFID: {fid_score:.2f}"
        show_images(vis_images, img_title, f"{images_save_dir}/stage2_epoch_{epoch+1}.png")

    print("--- Training Finished ---")
