import torch
from torch import Tensor
from torch.autograd import Function
from typing import List, Optional, Tuple

import gs_renderer


class GaussianRenderFunction(Function):
    @staticmethod
    def forward(ctx, ksize: Tensor, covariances: Tensor, colors: Tensor, opacities: Tensor) -> Tensor:
        height, width = colors.shape[-2:]
        render_image = gs_renderer.forward(width, height, 
            ksize.permute(0, 2, 3, 1), covariances.permute(0, 2, 3, 1), colors.permute(0, 2, 3, 1), opacities.permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2)

        ctx.save_for_backward(ksize, covariances, colors, opacities, render_image)
        ctx.width, ctx.height = width, height

        return render_image

    @staticmethod
    def backward(ctx, grad) -> Tuple[None, Tensor, Tensor, Tensor]:
        ksize, covariances, colors, opacities, render_image = ctx.saved_tensors
        width, height = ctx.width, ctx.height

        grad_covariances, grad_colors, grad_opacities = gs_renderer.backward(width, height, 
            ksize.permute(0, 2, 3, 1), covariances.permute(0, 2, 3, 1), colors.permute(0, 2, 3, 1), opacities.permute(0, 2, 3, 1), 
            render_image.permute(0, 2, 3, 1), grad.permute(0, 2, 3, 1)
        )
        
        return None, grad_covariances.permute(0, 3, 1, 2), grad_colors.permute(0, 3, 1, 2), grad_opacities.permute(0, 3, 1, 2)


def compute_covariance(scales: Tensor, rotation: Tensor) -> Tensor:
    scale_x, scale_y = scales[:, 0:1, :, :], scales[:, 1:2, :, :]
    scale_x2 = scale_x**2
    scale_y2 = scale_y**2

    cosθ = torch.cos(rotation)
    sinθ = torch.sin(rotation)
    cos2_θ = cosθ**2
    sin2_θ = sinθ**2

    sigma_x2 = scale_x2 * cos2_θ + scale_y2 * sin2_θ
    sigma_y2 = scale_x2 * sin2_θ + scale_y2 * cos2_θ
    sigma_xy = (scale_y2 - scale_x2) * sinθ * cosθ

    return torch.concat((sigma_x2, sigma_xy, sigma_y2), dim=1)


def gaussian_render(scales: Tensor, rotations: Tensor, colors: Tensor, opacities: Optional[Tensor] = None, max_ksize: int = 5) -> Tensor:
    covariances = compute_covariance(scales, rotations)
    ksize = torch.clamp(scales, 0, max_ksize).detach()
    if opacities is None:
        opacities = torch.ones_like(rotations)
    return GaussianRenderFunction.apply(ksize, covariances, colors, opacities)


def gaussian_render_py(scales: Tensor, rotations: Tensor, colors: Tensor, opacities: Optional[Tensor] = None, max_ksize: int = 5) -> Tensor:
    def _2d_gaussian(kernel: List[torch.Tensor], covariance: List[torch.Tensor], device: torch.device = torch.device('cpu')):
        kx, ky = kernel
        sigma_x2, sigma_xy, sigma_y2 = covariance

        rx = 1 / kx
        ry = 1 / ky
        x_coord = (-1 + rx + (2 * rx) * torch.arange(kx.item(), device=device).view(1, kx).expand(ky, -1).float()) * (kx / 2)
        y_coord = (-1 + ry + (2 * ry) * torch.arange(ky.item(), device=device).view(ky, 1).expand(-1, kx).float()) * (ky / 2)

        kernel = torch.exp(-0.5 * (x_coord**2 / sigma_x2 + y_coord**2 / sigma_y2 - 2 * sigma_xy * x_coord * y_coord / (sigma_x2 * sigma_y2)))
        return kernel

    def _covariance(scale_x: torch.Tensor, scale_y: torch.Tensor, rotation: torch.Tensor):
        cosθ = torch.cos(rotation)
        sinθ = torch.sin(rotation)

        cos2_θ = cosθ**2
        sin2_θ = sinθ**2

        scale_x2 = scale_x**2
        scale_y2 = scale_y**2

        sigma_x2 = scale_x2 * cos2_θ + scale_y2 * sin2_θ
        sigma_y2 = scale_x2 * sin2_θ + scale_y2 * cos2_θ
        sigma_xy = (scale_y2 - scale_x2) * sinθ * cosθ

        return sigma_x2, sigma_xy, sigma_y2


    batch_size = scales.shape[0]
    height, width = scales.shape[2], scales.shape[3]
    image = torch.zeros((batch_size, 3, height, width), dtype=torch.float32, device=scales.device)

    scale_x, scale_y = scales[:, 0, :, :], scales[:, 1, :, :]
    sigma_x2, sigma_xy, sigma_y2 = _covariance(scale_x, scale_y, rotations.squeeze(1))
    scale_x, scale_y = scale_x.int(), scale_y.int()

    for batch_idx in range(batch_size):
        for i in range(height):
            for j in range(width):
                _scale_x, _scale_y = scale_x[batch_idx, i, j], scale_y[batch_idx, i, j]
                _sigma_x2, _sigma_xy, _sigma_y2 = \
                    sigma_x2[batch_idx, i, j], sigma_xy[batch_idx, i, j], sigma_y2[batch_idx, i, j]

                if _scale_x == 0 or _scale_y == 0 or _sigma_x2 == 0 or _sigma_y2 == 0:
                    _rgb = colors[batch_idx, :, i, j][:, None, None] * opacities[batch_idx, 0, i, j]
                    image[batch_idx, :, i:i+1, j:j+1] += _rgb
                else:
                    _kx, _ky = _scale_x * 2, _scale_y * 2
                    _gaussian = _2d_gaussian((_kx, _ky), (_sigma_x2, _sigma_xy, _sigma_y2), device=scales.device)

                    _l = max(i - _scale_x, 0)
                    _r = min(i + _scale_x, height)
                    _t = max(j - _scale_y, 0)
                    _b = min(j + _scale_y, width)

                    _l_gs = _l - (i - _scale_x)
                    _r_gs = _kx - ((i + _scale_x) - _r)
                    _t_gs = _t - (j - _scale_y)
                    _b_gs = _ky - ((j + _scale_y) - _b)

                    _rgb = _gaussian[None, :, :] * colors[batch_idx, :, i, j][:, None, None] * opacities[batch_idx, 0, i, j]
                    image[batch_idx, :, _t:_b, _l:_r] += _rgb[:, _t_gs:_b_gs, _l_gs:_r_gs]

    return image


# if __name__ == "__main__":
#     from torchvision.utils import save_image

#     batch_size = 2
#     width = 1920
#     height = 1080

#     colors = torch.randn((batch_size, 3, height, width), requires_grad=True).cuda()
#     scales = torch.randn((batch_size, 2, height, width), requires_grad=True).cuda()
#     rotations = torch.zeros((batch_size, 1, height, width), requires_grad=True).cuda()
#     opacity = torch.ones((batch_size, 1, height, width), requires_grad=True).cuda()

#     save_image(colors, "colors.png", padding=0)
#     output_image = gaussian_render(scales, rotations, colors, opacity)
#     save_image(output_image, "output_image.png", padding=0, normalize=False)

#     loss = output_image.sum()
#     loss.backward()

#     pass