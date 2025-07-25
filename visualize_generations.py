import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from generation.generators import generate_images
from models import Unet, MobileResUNet


@torch.no_grad()
def visualize_generations(model, config, device, stage, num_images=36, num_plots=10):
    """
    Generates and visualizes images from a trained model.

    Args:
        model: The trained model.
        config: The experiment configuration.
        device: The device to run the model on.
        num_images: The number of images to generate per plot.
        num_plots: The maximum number of plots to generate.
    """
    for i in range(num_plots):
        print(f"Generating plot {i + 1}/{num_plots}...")
        
        if stage == 3:
            generated_images = model(torch.randn(num_images, config['in_channels'], config['image_size'], config['image_size'], device=device))
        else:
            config['n_images_to_generate'] = num_images
            generated_images = generate_images(model, config, device)
        
        # Create a grid of images
        fig, axes = plt.subplots(6, 6, figsize=(10, 10))
        for j, ax in enumerate(axes.flat):
            img = generated_images[j].permute(1, 2, 0).cpu().numpy()
            # Clamp and scale the image to [0, 1] for display
            img = np.clip((img + 1) / 2, 0, 1)
            ax.imshow(img)
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize images generated by a trained model.')
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

    # --- Set Channel Info Based on Dataset ---
    if config["dataset"] == "MNIST":
        config['in_channels'] = 1
        config['out_channels'] = 1
    elif config["dataset"] == "CIFAR10":
        config['in_channels'] = 3
        config['out_channels'] = 3
    else:
        raise f"Invalid Dataset {config['dataset']}"

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

    # --- Visualize Generations ---
    visualize_generations(model, config, device, args.stage)
