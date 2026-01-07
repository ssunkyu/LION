"""
This module handles task-dependent operations (A) and noises (n) 
for Point Cloud Data to simulate a measurement y = Ax + n.
"""

from abc import ABC, abstractmethod
import torch
from pointnet2_ops import pointnet2_utils

# ============================
# Point Cloud Linear Operators
# ============================

__OPERATOR__ = {}

def register_operator(name: str):
    """A decorator to register a new operator class."""
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper

def get_operator(name: str, **kwargs):
    """Fetches an operator class by its registered name."""
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)

class LinearOperator(ABC):
    """Abstract base class for linear operators."""
    @abstractmethod
    def forward(self, data, **kwargs):
        # A * x
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # A^T * x
        pass


@register_operator(name='denoise_pcd')
class DenoiseOperatorPCD(LinearOperator):
    """
    The denoising operator A is the identity operator.
    For a problem y = x + n, the forward operation A(x) is simply x.
    """
    def __init__(self, device):
        self.device = device
    
    def forward(self, data, **kwargs):
        return data

    def transpose(self, data, **kwargs):
        return data

@register_operator(name='inpainting_pcd')
class InpaintingOperatorPCD(LinearOperator):
    """
    The inpainting operator selects a subset of points using a mask.
    A is a multiplication op with the mask, making A self-adjoint (A^T = A).
    """
    def __init__(self, device):
        self.device = device
    
    def forward(self, data, **kwargs):
        mask = kwargs.get('mask', None)
        if mask is None:
            raise ValueError("InpaintingOperatorPCD requires a 'mask'")
        # Ensure mask is broadcastable: (B, N) -> (B, N, 1)
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        return data * mask
    
    def transpose(self, data, **kwargs):
        # The masking operation is self-adjoint, so A^T is the same as A.
        return self.forward(data, **kwargs)

@register_operator(name='super_resolution_pcd')
class SuperResolutionOperatorPCD(LinearOperator):
    """
    The super-resolution operator A corresponds to downsampling.
    For point clouds, Farthest Point Sampling (FPS) is a standard method.
    """
    def __init__(self, num_points_out, device):
        self.num_points_out = num_points_out
        self.device = device

    def _farthest_point_sampling_optimized(self, pcd, n_samples):
        """
        Performs Farthest Point Sampling using a highly optimized CUDA kernel.

        Args:
            pcd (torch.Tensor): The input point cloud with shape (B, N, C).
            n_samples (int): The number of points to sample.

        Returns:
            torch.Tensor: The sampled point cloud with shape (B, n_samples, C).
        """
        # pointnet2_utils.furthest_point_sample expects (B, N, C) tensor.
        # It returns the indices of the sampled points.
        # shape: (B, n_samples)
        sampled_indices = pointnet2_utils.furthest_point_sample(pcd, n_samples)
        
        # Use the indices to gather the points from the original point cloud.
        # This is equivalent to torch.gather but often more intuitive.
        # sampled_indices must be expanded to match the coordinate dimension (C).
        # shape: (B, n_samples, C)
        sampled_pcd = pointnet2_utils.gather_operation(pcd.transpose(1, 2).contiguous(), sampled_indices)
        
        return sampled_pcd.transpose(1, 2).contiguous()

    def forward(self, data, **kwargs):
        """ Downsample using the optimized FPS. """
        # The input point cloud 'data' is expected to be in (B, N, C) format.
        return self._farthest_point_sampling_optimized(data, self.num_points_out)

    def transpose(self, data, **kwargs):
        """
        The exact transpose of FPS is complex and ill-defined.
        Upsampling is typically learned by the generative model itself.
        This function is rarely used in practice.
        """
        print("Warning: Transpose of Farthest Point Sampling is ill-defined. Returning identity.")
        return data

@register_operator(name='blur_pcd')
class BlurOperatorPCD_Sparse(LinearOperator):
    """
    Point cloud blurring, memory-efficiently implemented using a sparse weight matrix.
    """
    def __init__(self, device, k=20, sigma=0.05):
        super().__init__(device)
        self.k = k
        self.sigma = sigma
        self.device = device
        self.W = None # Weight matrix

    def _build_weight_matrix(self, pcd):
        B, N, _ = pcd.shape
        dist_matrix = torch.cdist(pcd, pcd, p=2.0)
        
        nn_dists, nn_indices = torch.topk(dist_matrix, self.k, dim=-1, largest=False)

        dists_sq = nn_dists.pow(2)
        weights = torch.exp(-dists_sq / (2 * self.sigma**2))
        normalized_weights = weights / weights.sum(dim=-1, keepdim=True)

        batch_indices = torch.arange(B, device=self.device).view(B, 1, 1).expand(B, N, self.k)
        row_indices = torch.arange(N, device=self.device).view(1, N, 1).expand(B, N, self.k)
        
        i = torch.stack([batch_indices.flatten(), row_indices.flatten(), nn_indices.flatten()])
        v = normalized_weights.flatten()
        
        self.W = torch.sparse_coo_tensor(i, v, (B, N, N))

    def forward(self, data, **kwargs):
        if self.W is None:
            self._build_weight_matrix(data)
        
        # CORRECTED: No transpose needed for the input data.
        # Shape: (B, N, N) @ (B, N, 3) -> (B, N, 3)
        return self.W.bmm(data)

    def transpose(self, data, **kwargs):
        if self.W is None:
            raise RuntimeError("Weight matrix W is not built. Call forward first.")
        
        # For transpose operation, we use the transpose of the weight matrix.
        W_T = self.W.transpose(1, 2)
        
        # CORRECTED: No transpose needed for the input data.
        # Shape: (B, N, N) @ (B, N, 3) -> (B, N, 3)
        return W_T.bmm(data)


# ==========================
# Point Cloud Noise classes
# ==========================

__NOISE__ = {}

def register_noise(name: str):
    """A decorator to register a new noise class."""
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __NOISE__[name] = cls
        return cls
    return wrapper

def get_noise(name: str, **kwargs):
    """Fetches a noise class by its registered name."""
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser


class Noise(ABC):
    """Abstract base class for noise models."""
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

@register_noise(name='clean')
class CleanPCD(Noise):
    """Applies no noise."""
    def forward(self, data):
        return data

@register_noise(name='gaussian')
class GaussianNoisePCD(Noise):
    """Adds signal-independent Gaussian noise."""
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma
    
@register_noise(name='chamfer')
class ChamferNoisePCD(Noise):
    """Adds signal-independent Chamfer noise."""
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma

@register_noise(name='earth_mover')
class EarthMoverNoisePCD(Noise):
    """Adds signal-independent Earth Mover's distance noise."""
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma

@register_noise(name='song_chamfer')
class SongChamferNoisePCD(Noise):
    """Adds signal-independent song chamfer noise."""
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma

# @register_noise(name='poisson_pcd_like')
# class CoordinateDependentGaussianNoise(Noise):
#     """
#     A signal-dependent noise model for point clouds.
#     Strict Poisson noise is ill-defined for continuous coordinates.
#     Instead, this adds Gaussian noise with a standard deviation
#     proportional to each point's distance from the origin.
#     """
#     def __init__(self, scale):
#         self.scale = scale

#     def forward(self, data):
#         # Calculate the L2 norm (distance from origin) for each point.
#         norm = torch.norm(data, p=2, dim=-1, keepdim=True)
#         # Calculate a per-point sigma proportional to the norm.
#         sigma_per_point = norm * self.scale
#         # Add noise using the per-point sigma.
#         return data + torch.randn_like(data, device=data.device) * sigma_per_point