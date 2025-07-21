import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from constants import DATASET_MNIST, DATA_DIR


def get_dataloaders(config):
    if config["dataset"] == DATASET_MNIST:
        config.update({"in_channels": 1, "out_channels": 1, "norm_mean": (0.5,), "norm_std": (0.5,)})
        dataset_cls = torchvision.datasets.MNIST
        train_dataset = dataset_cls(root=DATA_DIR, train=True, download=True, transform=transforms.Compose([transforms.Resize((config['image_size'], config['image_size'])), transforms.ToTensor(), transforms.Normalize(config["norm_mean"], config["norm_std"]) ]))
        val_dataset = dataset_cls(root=DATA_DIR, train=False, download=True, transform=transforms.Compose([transforms.Resize((config['image_size'], config['image_size'])), transforms.ToTensor(), transforms.Normalize(config["norm_mean"], config["norm_std"]) ]))
    else:
        config.update({"in_channels": 3, "out_channels": 3, "norm_mean": (0.5, 0.5, 0.5), "norm_std": (0.5, 0.5, 0.5)})
        dataset_cls = torchvision.datasets.CIFAR10
        train_dataset = dataset_cls(root=DATA_DIR, train=True, download=True, transform=transforms.Compose([transforms.Resize((config['image_size'], config['image_size'])), transforms.ToTensor(), transforms.Normalize(config["norm_mean"], config["norm_std"]) ]))
        val_dataset = dataset_cls(root=DATA_DIR, train=False, download=True, transform=transforms.Compose([transforms.Resize((config['image_size'], config['image_size'])), transforms.ToTensor(), transforms.Normalize(config["norm_mean"], config["norm_std"]) ]))

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    fid_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False)
    return train_loader, val_loader, fid_dataloader