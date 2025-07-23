import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import sqrtm
from torchvision.models import inception_v3
from tqdm import tqdm

from constants import FID_EPS, INCEPTION_V3_INPUT_SIZE
from generation.generators import generate_images


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 model for FID calculation."""
    def __init__(self):
        super().__init__()
        self.model = inception_v3(pretrained=True)
        self.model.fc = nn.Identity() # Use as a feature extractor

    def forward(self, x):
        if x.shape[1] == 1: x = x.repeat(1, 3, 1, 1)
        x = nn.functional.interpolate(x, size=INCEPTION_V3_INPUT_SIZE, mode='bilinear', align_corners=False)
        return self.model(x)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=FID_EPS):
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
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


@torch.no_grad()
def get_activations(images, model, device, batch_size):
    """Get InceptionV3 activations for a set of images."""
    model.eval()
    activations = []
    pbar_desc = "Getting activations"
    for i in tqdm(range(0, len(images), batch_size), desc=pbar_desc):
        batch = torch.from_numpy(images[i:i+batch_size]).to(device)
        activations.append(model(batch).cpu().numpy())
    return np.concatenate(activations)


def calculate_fid_stats(dataloader, model, device, num_images, batch_size):
    """Calculate mean and covariance of activations for a dataset."""
    images = np.concatenate([imgs.numpy() for imgs, _ in dataloader], axis=0)[:num_images]
    activations = get_activations(images, model, device, batch_size)
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


@torch.no_grad()
def calculate_fid_for_model(gen_model, mu_real, sigma_real, inception_model, config, device, stage):
    """Calculate FID score for a generator model given pre-calculated real stats."""
    gen_model.eval()
    fake_images = []
    pbar_desc = "Generating images for FID"
    num_images_generated = 0
    for _ in tqdm(range(0, config['fid_num_images'], config['batch_size']), desc=pbar_desc):
        n = min(config['batch_size'], config['fid_num_images'] - num_images_generated)
        if n <= 0:
            break

        if stage == 3:
            imgs = gen_model(torch.randn(n, config['in_channels'], config['image_size'], config['image_size'], device=device))
        else:
            imgs = generate_images(gen_model, config, device, num_samples=n)
        
        fake_images.append(imgs.cpu().numpy())
        num_images_generated += imgs.shape[0]

    print(f"{num_images_generated} images generated for FID score calculation")
    fake_images = np.concatenate(fake_images, axis=0)
    
    fake_activations = get_activations(fake_images, inception_model, device, config['batch_size'])
    mu_fake, sigma_fake = np.mean(fake_activations, axis=0), np.cov(fake_activations, rowvar=False)
    
    return calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
