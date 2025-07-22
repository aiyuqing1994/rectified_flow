import lpips
import torch
import torch.nn as nn


class LPIPSHuberLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)

    @staticmethod
    def pseudo_huber_loss(pred, target, c=0.00054):
        d = target[0].numel()
        diff = pred - target
        c_scaled = c * d
        return torch.sqrt((diff ** 2) + c_scaled ** 2) - c_scaled

    def forward(self, v_pred, v_target, x_original, x_predicted, t):
        # Ensure t is a tensor with the same shape as the batch size for broadcasting
        t = t.view(-1, 1, 1, 1)

        # Pseudo-Huber loss component
        phub_loss = self.pseudo_huber_loss(v_pred, v_target).mean()

        # LPIPS component
        lpips_dist = self.lpips_loss_fn(x_original, x_predicted).mean()

        loss = (1 - t) * phub_loss + lpips_dist

        return loss.mean()


class LPIPSLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)

    def forward(self, x_original, x_predicted):
        return self.lpips_loss_fn(x_original, x_predicted).mean()


class L2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred, target):
        return self.loss(pred, target)
