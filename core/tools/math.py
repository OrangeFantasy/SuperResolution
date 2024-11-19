import cv2
import math
import numpy as np
import torch

from torch.nn import functional as F
from typing import Any, Tuple, Optional


def backward_warping(frame: torch.Tensor, motion: torch.Tensor, mode: str = "nearest", align_corners: bool = True) -> torch.Tensor:
    h, w = frame.shape[-2:]
    grid_x = torch.arange(w, device=frame.device).view(1, w).expand(h, -1).float()
    grid_y = torch.arange(h, device=frame.device).view(h, 1).expand(-1, w).float()
    grid_x = ((grid_x[None, ...] + motion[:, 0]) / (w - 1)) * 2.0 - 1.0
    grid_y = ((grid_y[None, ...] + motion[:, 1]) / (h - 1)) * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return F.grid_sample(frame, grid.permute(0, 2, 3, 1), mode, "zeros", align_corners)


# def backward_warping(frame: torch.Tensor, motion: torch.Tensor, 
#         dim_xy: int = 1, mode: str = "nearest", align_corners: bool = True) -> torch.Tensor:
#     assert dim_xy == 1 or dim_xy == 3 or dim_xy == -1, "dim_xy should be 1, 3 or -1."
    
#     h, w = frame.shape[-2:]
#     grid_x = torch.arange(w, device=frame.device).view(1, w).expand(h, -1).float()
#     grid_y = torch.arange(h, device=frame.device).view(h, 1).expand(-1, w).float()

#     if dim_xy == 1:
#         grid_x = ((grid_x[None, ...] + motion[:, 0]) / (w - 1)) * 2.0 - 1.0
#         grid_y = ((grid_y[None, ...] + motion[:, 1]) / (h - 1)) * 2.0 - 1.0
#     else:
#         grid_x = ((grid_x[None, ...] + motion[..., 0]) / (w - 1)) * 2.0 - 1.0
#         grid_y = ((grid_y[None, ...] + motion[..., 1]) / (h - 1)) * 2.0 - 1.0

#     grid = torch.stack([grid_x, grid_y], dim=-1)
#     return F.grid_sample(frame, grid.permute(0, 2, 3, 1), mode, "zeros", align_corners)


def backward_warping_np(frame: np.ndarray, motion: np.ndarray, mode: str = "nearest") -> np.ndarray: # img[i-1], mv[i] -> img[i]
    """
    :param frame: (H, W, C)
    :param motion: (2, H, W). The first dimension is the x-axis and the second dimension is the y-axis.
    :param mode: "nearest" or "bilinear"
    :return: (H, W, C)
    """
    if mode == "nearest":
        interpolation = cv2.INTER_NEAREST
    elif mode == "bilinear":
        interpolation = cv2.INTER_LINEAR
    else:
        raise ValueError(f"Invalid interpolation mode: {mode}")
    
    h, w, _ = frame.shape    
    grid_x = np.arange(w).reshape(1, w).repeat(h, axis=0).astype(np.float32)
    grid_y = np.arange(h).reshape(h, 1).repeat(w, axis=1).astype(np.float32)
    grid_x = (grid_x + motion[0]) / (w - 1)
    grid_y = (grid_y + motion[1]) / (h - 1)

    return cv2.remap(frame, grid_x, grid_y, interpolation)


def tone_map(hdr_image: torch.Tensor, mu: int = 8) -> torch.Tensor:
    log_1_mu = torch.log(torch.tensor(1 + mu, dtype=hdr_image.dtype, device=hdr_image.device))
    return torch.log(1 + mu * hdr_image) / log_1_mu


def tone_map_np(hdr_image: np.ndarray, mu: int = 8) -> np.ndarray:
    return np.log(1 + mu * hdr_image) / np.log(1 + mu)


def inverse_tone_map(ldr_image: torch.Tensor, mu: int = 8) -> torch.Tensor:
    log_1_mu = torch.log(torch.tensor(1 + mu, dtype=hdr_image.dtype, device=hdr_image.device))
    return (torch.exp(log_1_mu * ldr_image) - 1) / mu


def inverse_tone_map_np(ldr_image: np.ndarray, mu: int = 8) -> np.ndarray:
    return (np.exp(np.log(1 + mu) * ldr_image) - 1) / mu


def dilate_velocity(motion: torch.Tensor, depth: torch.Tensor, ksize: int = 3):
    """
    param motion: (B, H, W, 2)
    param depth: (B, 1, H, W)
    return: (B, 2, H, W)
    """
    b, h, w, _ = motion.shape

    depth = F.unfold(depth, ksize, padding=ksize // 2).view(b, ksize * ksize, h, w)
    min_pos = torch.argmin(depth, dim=1, keepdim=True)

    x_offset = min_pos %  ksize - ksize // 2
    y_offset = min_pos // ksize - ksize // 2

    grid_x = torch.arange(w, dtype=torch.int32, device=motion.device).view(1, 1, 1, w).expand(-1, -1, h, -1)
    grid_y = torch.arange(h, dtype=torch.int32, device=motion.device).view(1, 1, h, 1).expand(-1, -1, -1, w)

    x_idx = (grid_x + x_offset).clamp_(0, w - 1)
    y_idx = (grid_y + y_offset).clamp_(0, h - 1)

    return motion[:, y_idx, x_idx, :]


def dilate_velocity_np(motion: np.ndarray, depth: np.ndarray, ksize: int = 3):
    raise NotImplementedError


def make_coord(shape: Tuple[int, int], ranges: Optional[Any] = None, acquire_batch_dim: bool = True, 
               dtype: torch.dtype = torch.float32, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    param shape: Shape of the grid - (H, W).
    param device: torch.device or str, e.g., "cpu" or "cuda:0"
    param ranges: ((x_min, x_max), (y_min, y_max))
    return: (H, W, 2). The layout of the last dimension is (x, y).
    """
    if ranges is None:
        ranges = ((-1, 1), (-1, 1))
    x_min, x_max = ranges[0]
    y_min, y_max = ranges[1]

    h, w = shape
    rx = (x_max - x_min) / (2 * w)
    ry = (y_max - y_min) / (2 * h)

    coord_x = x_min + rx + (2 * rx) * torch.arange(w, dtype=dtype, device=device).view(1, w).expand(h, -1).float()
    coord_y = y_min + ry + (2 * ry) * torch.arange(h, dtype=dtype, device=device).view(h, 1).expand(-1, w).float()

    coord = torch.stack([coord_x, coord_y], dim=0)
    if acquire_batch_dim:
        coord = coord.unsqueeze(0)
    return coord


if __name__ == "__main__":
    # x = math.log(2)
    hdr_image = torch.rand(3, 12, 12)
    tone_map(hdr_image, 8)