import matplotlib.pyplot as plt
import numpy as np
import torch


@torch.no_grad()
def generate_images(model, config, device, start_noise=None, steps=None):
    model.eval()
    if steps is None:
        steps = config['generation_steps']
    z = start_noise if start_noise is not None else torch.randn(config.get('n_images_to_generate', 36), config['in_channels'], config['image_size'], config['image_size'], device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((z.shape[0],), i / steps, device=device)
        v = model(z, t)
        z = z + v * dt
    return z


def show_images(images, title, save_path):
    images = torch.clamp((images + 1) / 2, 0, 1)
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


def generate_and_save_comparison_images(model, config, device, epoch, stage, images_save_dir, writer):
    if stage == 3:
        # For the distilled model, we only care about 1-step generation from noise
        model.eval()
        with torch.no_grad():
            z0 = torch.randn(config['n_images_to_generate'], config['in_channels'], config['image_size'], config['image_size'], device=device)
            one_step_images = model(z0)
        one_step_title = f"Stage {stage}, Epoch {epoch+1} (One-step)"
        one_step_save_path = f"{images_save_dir}/stage{stage}_epoch_{epoch+1}_onestep.png"
        show_images(one_step_images, one_step_title, one_step_save_path)
        writer.add_images(f"Generated Images/Stage {stage} (One-step)", one_step_images, epoch)
    else:
        # Multi-step generation
        multi_step_images = generate_images(model, config, device)
        multi_step_title = f"Stage {stage}, Epoch {epoch+1} (Multi-step)"
        multi_step_save_path = f"{images_save_dir}/stage{stage}_epoch_{epoch+1}_multistep.png"
        show_images(multi_step_images, multi_step_title, multi_step_save_path)
        writer.add_images(f"Generated Images/Stage {stage} (Multi-step)", multi_step_images, epoch)

        # One-step generation
        one_step_images = generate_images(model, config, device, steps=1)
        one_step_title = f"Stage {stage}, Epoch {epoch+1} (One-step)"
        one_step_save_path = f"{images_save_dir}/stage{stage}_epoch_{epoch+1}_onestep.png"
        show_images(one_step_images, one_step_title, one_step_save_path)
        writer.add_images(f"Generated Images/Stage {stage} (One-step)", one_step_images, epoch)