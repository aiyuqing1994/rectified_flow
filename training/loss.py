import lpips
import torch
import torch.nn as nn


def pseudo_huber_loss(pred, target, c=0.00054, d=3072):
    diff = pred - target
    c_scaled = c * d
    return torch.sqrt((diff ** 2) + c_scaled ** 2) - c_scaled


class LPIPSHuberLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)

    def forward(self, v_pred, v_target, x_original, x_predicted, t):
        # Ensure t is a tensor with the same shape as the batch size for broadcasting
        t = t.view(-1, 1, 1, 1)

        # Pseudo-Huber loss component
        phub_loss = pseudo_huber_loss(v_pred, v_target).mean()

        # LPIPS component
        lpips_dist = self.lpips_loss_fn(x_original, x_predicted).mean()

        # Combined loss according to the paper's formula
        # loss = (1 - t) * phub_loss + lpips_dist
        # The paper mentions mlp-hub and mlp-hub-1/t. Let's implement the better one: mlp-hub-1/t
        # mlp-hub-1/t = (1-t)*mhub + (1/t)*LPIPS
        # The paper mentions mlp-hub and mlp-hub-t. Let's implement the better one: mlp-hub-t
        # mlp-hub-t = (1-t)*mhub + t*LPIPS
        loss = (1 - t) * phub_loss + t * lpips_dist
        
        return loss.mean()


class L2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred, target):
        return self.loss(pred, target)