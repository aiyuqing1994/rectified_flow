import torch


class UShapedSampler:
    def __init__(self, a=4):
        self.a = a

    def sample(self, shape, device):
        # This is a simplified implementation using inverse transform sampling.
        # A more robust implementation might be needed for production use.
        u = torch.rand(shape, device=device)
        # The CDF of the U-shaped distribution is proportional to sinh(a*u)/sinh(a)
        # To get the inverse CDF, we need to solve for u in y = sinh(a*u)/sinh(a)
        # u = (1/a) * asinh(y * sinh(a))
        return (1.0 / self.a) * torch.asinh(u * torch.sinh(torch.tensor(self.a)))


class UniformSampler:
    def sample(self, shape, device):
        return torch.rand(shape, device=device)