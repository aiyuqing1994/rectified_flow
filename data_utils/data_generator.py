import os

import numpy as np
import torch
from tqdm import tqdm

from generation.generators import generate_images


def generate_next_stage_data(model, config, save_dir, num_samples):
    batch_size = config["batch_size"]
    seed = 0

    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(seed)
    device = config["device"]

    model.eval()

    z0_list = []
    z1_list = []

    for _ in tqdm(range(0, num_samples, batch_size)):
        z0 = torch.randn(batch_size, config['in_channels'], config['image_size'], config['image_size'], device=device)
        z1 = generate_images(model, config, device, start_noise=z0)

        z0_list.append(z0.cpu().numpy())
        z1_list.append(z1.cpu().numpy())

    z0_all = np.concatenate(z0_list, axis=0)
    z1_all = np.concatenate(z1_list, axis=0)

    np.save(os.path.join(save_dir, f"z0_seed{seed}.npy"), z0_all)
    np.save(os.path.join(save_dir, f"z1_seed{seed}.npy"), z1_all)

    print(f"Successfully generated and saved {num_samples} data pairs to {save_dir}")
