"""Final MLP architecture (adopted in the depth experiment): (64, 32)."""

from torch import nn

HIDDEN = (64, 32)


class MLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in HIDDEN:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
