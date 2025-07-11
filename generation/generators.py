import torch
import numpy as np
import matplotlib.pyplot as plt

@torch.no_grad()
def generate_images(model, config, device, start_noise=None):
    model.eval()
    z = start_noise if start_noise is not None else torch.randn(config['n_images_to_generate'], config['in_channels'], config['image_size'], config['image_size'], device=device)
    dt = 1.0 / config['generation_steps']
    for i in range(config['generation_steps']):
        t = torch.full((z.shape[0],), i / config['generation_steps'], device=device)
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