import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
from config import config


# --- FID Calculation ---
class InceptionV3(nn.Module):
    """Pretrained InceptionV3 model for FID calculation."""
    def __init__(self):
        super().__init__()
        self.model = inception_v3(pretrained=True)
        self.model.fc = nn.Identity() # Use as a feature extractor

    def forward(self, x):
        if x.shape[1] == 1: x = x.repeat(1, 3, 1, 1)
        x = nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        return self.model(x)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance."""
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


@torch.no_grad()
def get_activations(images, model, device):
    """Get InceptionV3 activations for a set of images."""
    model.eval()
    activations = []
    pbar_desc = "Getting activations"
    for i in tqdm(range(0, len(images), config['batch_size']), desc=pbar_desc):
        batch = torch.from_numpy(images[i:i+config['batch_size']]).to(device)
        activations.append(model(batch).cpu().numpy())
    return np.concatenate(activations)


def calculate_fid_stats(dataloader, model, device):
    """Calculate mean and covariance of activations for a dataset."""
    images = np.concatenate([imgs.numpy() for imgs, _ in dataloader], axis=0)[:config['fid_num_images']]
    activations = get_activations(images, model, device)
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


@torch.no_grad()
def calculate_fid_for_model(gen_model, mu_real, sigma_real, inception_model, device):
    """Calculate FID score for a generator model given pre-calculated real stats."""
    gen_model.eval()
    fake_images = []
    pbar_desc = "Generating images for FID"
    for _ in tqdm(range(0, config['fid_num_images'], config['batch_size']), desc=pbar_desc):
        n = min(config['batch_size'], config['fid_num_images'] - len(fake_images))
        imgs = generate_images(gen_model, n, config['generation_steps'], device)
        fake_images.append(imgs.cpu().numpy())
    fake_images = np.concatenate(fake_images, axis=0)
    
    fake_activations = get_activations(fake_images, inception_model, device)
    mu_fake, sigma_fake = np.mean(fake_activations, axis=0), np.cov(fake_activations, rowvar=False)
    
    return calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)


# --- Generation & Visualization ---
@torch.no_grad()
def generate_images(model, n_images, steps, device, start_noise=None):
    model.eval()
    z = start_noise if start_noise is not None else torch.randn(n_images, config['in_channels'], config['image_size'], config['image_size'], device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((z.shape[0],), i / steps, device=device)
        v = model(z, t)
        z = z + v * dt
    return torch.clamp((z + 1) / 2, 0, 1)


def show_images(images, title, save_path):
    images = images.detach().cpu().numpy()
    n_cols = int(np.sqrt(images.shape[0]))
    n_rows = int(np.ceil(images.shape[0] / n_cols))
    plt.figure(figsize=(n_cols * 1.5, n_rows * 1.5))
    for i in range(images.shape[0]):
        plt.subplot(n_rows, n_cols, i + 1)
        img = images[i, 0] if images.shape[1] == 1 else np.transpose(images[i], (1, 2, 0))
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.suptitle(title); plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"Saved generated images to {save_path}")
