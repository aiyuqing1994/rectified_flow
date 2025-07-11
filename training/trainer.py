import torch
import torch.nn as nn
from tqdm import tqdm
from ema_pytorch import EMA
from torch.optim.lr_scheduler import ReduceLROnPlateau

from generation.generators import generate_images, show_images
from evaluation.fid import calculate_fid_for_model

from constants import BETA_DIST_ALPHA, BETA_DIST_BETA

def train_one_epoch(model, ema, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc="Training Stage 1")
    beta_dist = torch.distributions.beta.Beta(BETA_DIST_ALPHA, BETA_DIST_BETA)
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

def train_stage1(config, model_1, ema, dataloader, optimizer_1, loss_fn, device, models_save_dir, images_save_dir, mu_real, sigma_real, inception_model):
    print("--- Stage 1: Training 1-Rectified Flow Model ---")
    scheduler_1 = ReduceLROnPlateau(optimizer_1, mode='min', factor=0.5, patience=5)
    for epoch in range(config['epochs_stage1']):
        avg_loss = train_one_epoch(model_1, ema, dataloader, optimizer_1, loss_fn, device)
        print(f"Stage 1 | Epoch {epoch+1}/{config['epochs_stage1']}, Loss: {avg_loss:.4f}")
        scheduler_1.step(avg_loss)
        vis_images = generate_images(ema.ema_model, config, device)
        img_title = f"Stage 1, Epoch {epoch+1}"
        if (epoch + 1) % config['checkpoint_interval'] == 0:
            torch.save({'model_state_dict': model_1.state_dict(), 'ema_state_dict': ema.state_dict()}, f"{models_save_dir}/stage1_epoch_{epoch+1}.pth")
            if config["calculate_fid"]:
                fid_score = calculate_fid_for_model(ema.ema_model, mu_real, sigma_real, inception_model, config, device)
                print(f"FID Score after {epoch+1} epochs: {fid_score:.2f}")
                img_title += f"\nFID: {fid_score:.2f}"
        show_images(vis_images, img_title, f"{images_save_dir}/stage1_epoch_{epoch+1}.png")

    print("--- Stage 1 Finished ---")
    model_1.eval()

def train_stage2(config, model_1, model_2, ema_2, dataloader, optimizer_2, loss_fn, device, models_save_dir, images_save_dir, mu_real, sigma_real, inception_model):
    print("\n--- Stage 2: Training 2-Rectified Flow Model (Reflow) ---")
    scheduler_2 = ReduceLROnPlateau(optimizer_2, mode='min', factor=0.5, patience=5)
    beta_dist = torch.distributions.beta.Beta(0.5, 0.5)
    
    for epoch in range(config['epochs_stage2']):
        model_2.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Training Stage 2, Epoch {epoch+1}")
        for z1, _ in pbar:
            optimizer_2.zero_grad()

            z1 = z1.to(device)
            z0 = torch.randn_like(z1)

            t_scalar = beta_dist.sample((z0.shape[0],)).to(device)
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
        vis_images = generate_images(ema_2.ema_model, config, device)
        img_title = f"Stage 2, Epoch {epoch+1}"
        if (epoch + 1) % config['checkpoint_interval'] == 0:
            torch.save({'model_state_dict': model_2.state_dict(), 'ema_state_dict': ema_2.state_dict()}, f"{models_save_dir}/stage2_epoch_{epoch+1}.pth")
            if config["calculate_fid"]:
                fid_score = calculate_fid_for_model(ema_2.ema_model, mu_real, sigma_real, inception_model, config, device)
                print(f"FID Score after {epoch+1} epochs: {fid_score:.2f}")
                img_title += f"\nFID: {fid_score:.2f}"
        show_images(vis_images, img_title, f"{images_save_dir}/stage2_epoch_{epoch+1}.png")