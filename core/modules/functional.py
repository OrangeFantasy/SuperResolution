import torch
from torch import Tensor
from torch.nn import functional as F


def gaussian(window_size: int, sigma: float) -> Tensor:
    coord = torch.arange(window_size, dtype=torch.float32)
    gauss = torch.exp_(-(coord - window_size // 2)**2 / (2 * sigma**2))
    return gauss / gauss.sum()


# ******************************
#       Contextual Loss
# ******************************

def _compute_meshgrid(shape, device) -> Tensor:
    b, _, h, w = shape
    rows = torch.arange(0, h, dtype=torch.float32, device=device) / (h + 1)
    cols = torch.arange(0, w, dtype=torch.float32, device=device) / (w + 1)

    feature_grid = torch.meshgrid(rows, cols, indexing="ij")
    feature_grid = torch.stack(feature_grid).unsqueeze(0)
    feature_grid = torch.cat([feature_grid for _ in range(b)], dim=0)
    return feature_grid


def _compute_mae_distance(x: torch.Tensor, y: torch.Tensor) -> Tensor:
    b, c, h, w = x.size()
    x_vec = x.view(b, c, -1)
    y_vec = y.view(b, c, -1)

    distance = x_vec.unsqueeze(2) - y_vec.unsqueeze(3)
    distance = distance.sum(dim=1).abs()
    distance = distance.transpose(1, 2).reshape(b, h * w, h * w)
    distance = distance.clamp(min=0.)
    return distance


def _compute_mse_distance(x: Tensor, y: Tensor) -> Tensor:
    b, c, h, w = x.shape
    x_vec = x.view(b, c, -1)
    y_vec = y.view(b, c, -1)
    x_s = torch.sum(x_vec ** 2, dim=1)
    y_s = torch.sum(y_vec ** 2, dim=1)

    A = y_vec.transpose(1, 2) @ x_vec  # N x (HW) x (HW)
    distance = y_s.unsqueeze(2) - 2 * A + x_s.unsqueeze(1)
    distance = distance.transpose(1, 2).reshape(b, h * w, h * w)
    distance = distance.clamp(min=0.)
    return distance


def _compute_cosine_distance(x: Tensor, y: Tensor) -> Tensor:
    # mean shifting by channel-wise mean of `y`.
    y_mean = y.mean(dim=(0, 2, 3), keepdim=True)
    x_centered = x - y_mean
    y_centered = y - y_mean

    # L2 normalization
    x_normalized = F.normalize(x_centered, p=2, dim=1)
    y_normalized = F.normalize(y_centered, p=2, dim=1)

    # Channel-wise vectorization
    b, c, _, _ = x.size()
    x_normalized = x_normalized.reshape(b, c, -1)  # (B, C, H*W)
    y_normalized = y_normalized.reshape(b, c, -1)  # (B, C, H*W)

    # cosine similarity
    cosine_sim = torch.bmm(x_normalized.transpose(1, 2), y_normalized)  # (B, H*W, H*W)

    # convert to distance
    distance = 1. - cosine_sim
    return distance


def _compute_relative_distance(distance: Tensor) -> Tensor:
    dist_min, _ = torch.min(distance, dim=2, keepdim=True)
    relative_distance = distance / (dist_min + 1e-5)
    return relative_distance


def _compute_contextual(relative_distance: Tensor, bandwidth: float) -> Tensor:
    # w = torch.exp((1 - relative_distance) / bandwidth)  # Eq(3)
    # contextual = w / torch.sum(w, dim=2, keepdim=True)  # Eq(4)

    contextual = torch.softmax((1 - relative_distance) / bandwidth, dim=2)
    return contextual


def contextual_loss(x: Tensor, y: Tensor, band_width: float = 0.5, loss_type: str = "cosine") -> Tensor:
    """Computes contextual loss between x and y.

    Code:
        <https://gist.github.com/yunjey/3105146c736f9c1055463c33b4c989da>

    Args:
        x (Tensor): Features of shape (B, C, H, W).
        y (Tensor): Features of shape (B, C, H, W).
        band_width (float): Parameter used to convert distance to similarity.
        loss_type (str): Type to measure the distance between features.

    Returns:
        Tensor: Contextual bilateral loss between x and y.
    """
    if loss_type == "l1":
        distance = _compute_mae_distance(x, y)
    elif loss_type == "l2":
        distance = _compute_mse_distance(x, y)
    elif loss_type == "cosine":
        distance = _compute_cosine_distance(x, y)
    else:
        raise TypeError

    relative_distance = _compute_relative_distance(distance)
    contextual = _compute_contextual(relative_distance, band_width)
    contextual = torch.mean(torch.max(contextual, dim=1)[0], dim=1)  # Eq(1)
    loss = torch.mean(-torch.log(contextual + 1e-5))  # Eq(5)

    return loss


def contextual_bilateral_loss(x: Tensor, y: Tensor, weight_spatial: float, bandwidth: float, loss_type: str = "cosine") -> Tensor:
    """Computes Contextual Bilateral (CoBi) Loss between x and y.

    Ref:
        <https://arxiv.org/pdf/1905.05169.pdf>

    Args:
        x (Tensor): Features of shape (B, C, H, W).
        y (Tensor): Features of shape (B, C, H, W).
        weight_spatial (float): The weight of the spatial feature.
        band_width (float): Parameter used to convert distance to similarity.
        loss_type (str): Type to measure the distance between features.

    Returns:
        Tensor: Contextual bilateral loss between x and y.
    """
    # Calculate two image spatial loss
    grid = _compute_meshgrid(x.shape, x.device)
    distance = _compute_mse_distance(grid, grid)
    relative_distance = _compute_relative_distance(distance)
    contextual_spatial = _compute_contextual(relative_distance, bandwidth)

    # Calculate feature loss
    if loss_type == "l1":
        distance = _compute_mae_distance(x, y)
    elif loss_type == "l2":
        distance = _compute_mse_distance(x, y)
    elif loss_type == "cosine":
        distance = _compute_cosine_distance(x, y)
    else:
        raise TypeError
    relative_distance = _compute_relative_distance(distance)
    contextual_feature = _compute_contextual(relative_distance, bandwidth)

    # Combine loss
    contextual_combine = (1. - weight_spatial) * contextual_feature + weight_spatial * contextual_spatial
    k_max_NC, _ = torch.max(contextual_combine, dim=2, keepdim=True)
    contextual = k_max_NC.mean(dim=1)
    loss = torch.mean(-torch.log(contextual + 1e-5))
    
    return loss


# ******************************
#           SSIM Loss
# ******************************

def _ssim(x: Tensor, y: Tensor, window: Tensor, window_size: int, channel: int, size_average: bool) -> Tensor:
    mu_x = F.conv2d(x, window, padding=window_size // 2, groups=channel)
    mu_y = F.conv2d(y, window, padding=window_size // 2, groups=channel)

    mu_x2 = torch.square(mu_x)
    mu_y2 = torch.square(mu_y)
    mu_x_mul_mu_y = torch.multiply(mu_x, mu_y)

    sigma_x2 = F.conv2d(x * x, window, padding=window_size // 2, groups=channel) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=window_size // 2, groups=channel) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=window_size // 2, groups=channel) - mu_x_mul_mu_y

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu_x_mul_mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map


def ssim_loss(x: Tensor, y: Tensor, window_size: int = 11, size_average: bool = True) -> Tensor:
    assert x.shape == y.shape, "Two tensor must have the same shape."
    channel = x.shape[1]

    sigma = 1.5
    _1d_window = gaussian(window_size, sigma).view(window_size, 1)
    _2d_window = torch.mm(_1d_window, _1d_window.t())
    window = _2d_window.expand(channel, 1, window_size, window_size)

    return 1. - _ssim(x, y, window, window_size, channel, size_average)


def gradient_map(x: Tensor, padding_mode: str = "replicate", return_xy: bool = False, eps = 1e-6) -> Tensor:
    x = F.pad(x, (1, 0, 1, 0), mode=padding_mode)
    grad_x = x[:, :, 1:, 1:] - x[:, :, 1:, :-1]
    grad_y = x[:, :, 1:, 1:] - x[:, :, :-1, 1:]
    
    if return_xy:
        return grad_x, grad_y
    else:
        return torch.sqrt(grad_x**2 + grad_y**2 + eps)
    