"""
This module handles task-dependent operations (A) and noises (n) 
for Point Cloud Data to simulate a measurement y = Ax + n.
"""

from abc import ABC, abstractmethod
import torch

# =================
# Base Classes & Registration
# =================

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

# =================
# Point Cloud Linear Operators
# =================

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

    def _farthest_point_sampling(self, pcd, n_samples):
        B, N, C = pcd.shape
        centroids = torch.zeros(B, n_samples, dtype=torch.long, device=self.device)
        distance = torch.ones(B, N, device=self.device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=self.device)
        batch_indices = torch.arange(B, dtype=torch.long, device=self.device)
        
        for i in range(n_samples):
            centroids[:, i] = farthest
            centroid = pcd[batch_indices, farthest, :].view(B, 1, C)
            dist = torch.sum((pcd - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
            
        # Index the points using the FPS result.
        # Expand centroids for torch.gather to index all coordinates.
        centroids_expanded = centroids.unsqueeze(-1).expand(B, n_samples, C)
        return torch.gather(pcd, 1, centroids_expanded)

    def forward(self, data, **kwargs):
        # Downsample using FPS.
        return self._farthest_point_sampling(data, self.num_points_out)

    def transpose(self, data, **kwargs):
        # The exact transpose of FPS is complex and ill-defined.
        # Upsampling is typically learned by the generative model itself.
        # This function is rarely used in practice.
        print("Warning: Transpose of Farthest Point Sampling is ill-defined. Returning identity.")
        return data

@register_operator(name='blur_pcd')
class BlurOperatorPCD(LinearOperator):
    """
    Point cloud blurring, approximated as a linear operator.
    This is achieved by pre-calculating a weight matrix W based on k-NN,
    linearizing an otherwise non-linear operation.
    """
    def __init__(self, k, sigma, device):
        self.k = k
        self.sigma = sigma
        self.device = device
        self.W = None # Weight matrix

    def _build_weight_matrix(self, pcd):
        B, N, _ = pcd.shape
        dist_matrix = torch.cdist(pcd, pcd, p=2.0)
        
        # Find the k-nearest neighbors for each point.
        _, nn_indices = torch.topk(dist_matrix, self.k, dim=-1, largest=False) # (B, N, k)

        # Calculate Gaussian weights.
        W = torch.zeros(B, N, N, device=self.device)
        batch_indices = torch.arange(B, device=self.device).view(B, 1)
        
        # Vectorized implementation for building the weight matrix
        k_indices = nn_indices.view(B, -1) # (B, N*k)
        row_indices = torch.arange(N, device=self.device).view(1, N, 1).expand(B, N, self.k).reshape(B, -1) # (B, N*k)
        
        # Gather distances for all neighbors
        dists_sq = dist_matrix[batch_indices, row_indices, k_indices].view(B, N, self.k).pow(2)
        
        # Calculate and normalize weights
        weights = torch.exp(-dists_sq / (2 * self.sigma**2))
        normalized_weights = weights / weights.sum(dim=-1, keepdim=True)
        
        # Populate the sparse weight matrix W
        W[batch_indices, row_indices, k_indices] = normalized_weights.view(B, -1)
        self.W = W

    def forward(self, data, **kwargs):
        # The weight matrix is built on the first forward call,
        # assuming the geometry of the data is fixed for this operator instance.
        if self.W is None:
            self._build_weight_matrix(data)
        # Linear operation: y = Wx
        return torch.bmm(self.W, data)

    def transpose(self, data, **kwargs):
        if self.W is None:
            raise RuntimeError("Weight matrix W is not built. Call forward first.")
        # Transpose operation: y = W^T x
        return torch.bmm(self.W.transpose(1, 2), data)

# =============
# Point Cloud Noise classes
# =============

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

@register_noise(name='clean_pcd')
class CleanPCD(Noise):
    """Applies no noise."""
    def forward(self, data):
        return data

@register_noise(name='gaussian_pcd')
class GaussianNoisePCD(Noise):
    """Adds signal-independent Gaussian noise."""
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