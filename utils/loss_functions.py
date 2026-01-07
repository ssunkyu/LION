"""
This module handles task-dependent operations (A) and noises (n) 
for Point Cloud Data to simulate a measurement y = Ax + n.
"""

from abc import ABC, abstractmethod
import torch
from third_party.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist, SoftChamferDistance
from third_party.PyTorchEMD.emd import earth_mover_distance


# =================
# Base Classes & Registration
# =================

__LOSS_FUNCTION__ = {}

def register_loss_function(name: str):
    """A decorator to register a new loss function class."""
    def wrapper(cls):
        if __LOSS_FUNCTION__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __LOSS_FUNCTION__[name] = cls
        return cls
    return wrapper

def get_loss_function(name: str, **kwargs):
    """Fetches a loss function class by its registered name."""
    if __LOSS_FUNCTION__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __LOSS_FUNCTION__[name](**kwargs)

class LossFunction(ABC):
    """Abstract base class for loss functions."""
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data, **kwargs):
        # Calculate mean loss.
        pass

# =================
# Point Cloud Linear Operators
# =================

@register_loss_function(name='cd')
class ChamferLossFunction(LossFunction):
    """
    The Chamfer loss function measures the distance between two point clouds.
    """
    def __init__(self, device):
        self.cd = chamfer_3DDist()
        self.device = device

    def forward(self, p1, p2, **kwargs):
        return self.cd(p1, p2)

@register_loss_function(name='l2')
class EuclideanLossFunction(LossFunction):
    """
    The Euclidean loss function measures the distance between two point clouds.
    """
    def __init__(self, device):
        self.device = device

    def forward(self, p1, p2, **kwargs):
        return torch.norm(p1 - p2, dim=-1).mean()

@register_loss_function(name='emd')
class EarthMoverDistanceLossFunction(LossFunction):
    """
    The Earth Mover's Distance (EMD) loss function measures the distance between two point clouds.
    """
    def __init__(self, device):
        self.emd = earth_mover_distance()
        self.device = device

    def forward(self, p1, p2, **kwargs):
        return self.emd(p1, p2)