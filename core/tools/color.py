import torch
from torch import Tensor

# assert torch.cuda.is_available(), "CUDA is not available"
# if torch.get_float32_matmul_precision() != "high":
#     torch.set_float32_matmul_precision("high")


_25_7: float = 6103515625

_MATRIX_RGB_TO_XYZ: Tensor = torch.tensor([
    [0.412453, 0.357580, 0.180423],
    [0.212671, 0.715160, 0.072169],
    [0.019334, 0.119193, 0.950227]
], dtype=torch.float32).to(torch.cuda.current_device())
_MATRIX_XYZ_TO_RGB: Tensor = torch.linalg.inv(_MATRIX_RGB_TO_XYZ)
_XYZ_REF_WHITE: Tensor = torch.tensor([
    0.95047, 1.00000, 1.08883
], dtype=torch.float32).view(1, 3, 1, 1).to(torch.cuda.current_device())


def rgb_colour_distance(rgb1: Tensor, rgb2: Tensor):
    r1, b1, g1 = rgb1[:, 0:1], rgb1[:, 1:2], rgb1[:, 2:3]
    r2, b2, g2 = rgb2[:, 0:1], rgb2[:, 1:2], rgb2[:, 2:3]

    r = (r1 + r2) / 2
    Δr = r1 - r2
    Δg = g1 - g2
    Δb = b1 - b2

    return torch.sqrt((2 + r) * Δr**2 + 4 * Δg**2 + (2 + (1 - r)) * Δb**2)
   

@torch.compile
def _rgb_to_xyz_(rgb: Tensor):
    # gamma correction.
    mask = rgb > 0.04045
    rgb = torch.where(mask, torch.pow((rgb + 0.055) / 1.055, 2.4), rgb / 12.92)

    # linear transform.
    b, _, h, w = rgb.shape
    rgb = rgb.permute(0, 2, 3, 1).view(b, -1, 3)
    xyz = rgb @ _MATRIX_RGB_TO_XYZ.T
    xyz = xyz.permute(0, 2, 1).view(b, 3, h, w)
    return xyz


@torch.compile
def _xyz_to_rgb_(xyz: Tensor):
    # linear transform.
    b, _, h, w = xyz.shape
    xyz = xyz.permute(0, 2, 3, 1).view(b, -1, 3)
    rgb = xyz @ _MATRIX_XYZ_TO_RGB.T
    rgb = rgb.permute(0, 2, 1).view(b, 3, h, w)

    # gamma correction.
    mask = rgb > 0.0031308
    rgb = torch.where(mask, 1.055 * torch.pow(rgb, 1 / 2.4) - 0.055, 12.92 * rgb)

    # clip to [0, 1].
    rgb = torch.clip(rgb, 0, 1)
    return rgb


@torch.compile
def _xyz_to_lab_(xyz: Tensor):
    # normalization
    xyz = xyz / _XYZ_REF_WHITE

    # nonlinear transform
    mask = xyz > 0.008856
    xyz = torch.where(mask, torch.pow(xyz, 1 / 3), (7.787 * xyz) + (16 / 116))

    # linear transform
    lab = torch.zeros_like(xyz)
    lab[:, 0] = 116.0 * xyz[:, 1] - 16.0
    lab[:, 1] = 500.0 * (xyz[:, 0] - xyz[:, 1])
    lab[:, 2] = 200.0 * (xyz[:, 1] - xyz[:, 2])
    return lab


@torch.compile
def _lab_to_xyz_(lab: Tensor):
    # linear transform
    xyz = torch.zeros_like(lab)
    xyz[:, 1] = (lab[:, 0] + 16.0) / 116.0
    xyz[:, 0] = (lab[:, 1] / 500.0) + xyz[:, 1]
    xyz[:, 2] = xyz[:, 1] - (lab[:, 2] / 200.0)

    # nonlinear transform
    mask = xyz > 0.2068966
    xyz = torch.where(mask, torch.pow(xyz, 3.0), (xyz - 16.0 / 116.0) / 7.787)

    # de-normalization
    xyz *= _XYZ_REF_WHITE
    return xyz


@torch.compile
def _CIEDE2000_(lab1: Tensor, lab2: Tensor, k_L: float = 1, k_C: float = 1, k_H: float = 1):
    L1, a1, b1 = lab1[:, 0:1], lab1[:, 1:2], lab1[:, 2:3]
    L2, a2, b2 = lab2[:, 0:1], lab2[:, 1:2], lab2[:, 2:3]

    # Compute `C_i^\prime` and `h_i^\prime`
    C1 = torch.sqrt(a1**2 + b1**2)
    C2 = torch.sqrt(a2**2 + b2**2)
    C_bar = (C1 + C2) / 2
    G = 0.5 * (1 - torch.sqrt(C_bar**7 / (C_bar**7 + _25_7)))

    a1_p = (1 + G) * a1
    a2_p = (1 + G) * a2

    C1_p = torch.sqrt(a1_p**2 + b1**2)
    C2_p = torch.sqrt(a2_p**2 + b2**2)

    h1_p = torch.where((b1 == 0) & (a1_p == 0), 0, 
                       torch.where(a1_p >= 0, torch.atan2(b1, a1_p), torch.atan2(b1, a1_p) + 2 * torch.pi))
    h2_p = torch.where((b2 == 0) & (a2_p == 0), 0,
                       torch.where(a2_p >= 0, torch.atan2(b2, a2_p), torch.atan2(b2, a2_p) + 2 * torch.pi))
    
    # Compute `\delta L^\prime`, `\delta C^\prime`, and `\delta H^\prime`
    ΔL_p = L2 - L1
    ΔC_p = C2_p - C1_p
   
    Δh_p = torch.where(C1_p * C2_p == 0, 0,
                       torch.where(h2_p - h1_p > torch.pi, h2_p - h1_p - 2 * torch.pi,
                                   torch.where(h2_p - h1_p < -torch.pi, h2_p - h1_p + 2 * torch.pi, h2_p - h1_p)))
    ΔH_p = 2 * torch.sqrt(C1_p * C2_p) * torch.sin(Δh_p / 2)

    # Compute `delta E_00`
    L_bar_p = (L1 + L2) / 2
    C_bar_p = (C1_p + C2_p) / 2

    h_bar_p = torch.where(C1_p * C2_p == 0, 0,
                          torch.where((h1_p - h2_p).abs() <= torch.pi, (h1_p + h2_p) / 2,
                                      torch.where(h1_p + h2_p < 2 * torch.pi, (h1_p + h2_p) / 2 + torch.pi, (h1_p + h2_p) / 2 - torch.pi)))
    
    T = 1 - 0.17 * torch.cos(h_bar_p - torch.pi / 6) + 0.24 * torch.cos(2 * h_bar_p) \
        + 0.32 * torch.cos(3 * h_bar_p + torch.pi / 30) - 0.2 * torch.cos(4 * h_bar_p - 63 * torch.pi / 180)
    
    h_bar_p_degree = (h_bar_p * 180 / torch.pi) % 360
    Δθ = 30 * torch.exp(-(((h_bar_p_degree - 275) / 25)**2))

    R_C = 2 * torch.sqrt(C_bar_p**7 / (C_bar_p**7 + _25_7))

    _temp = (L_bar_p - 50)**2
    S_L = 1 + ((0.015 * _temp) / torch.sqrt(20 + _temp))

    S_C = 1 + 0.045 * C_bar_p
    S_H = 1 + 0.015 * C_bar_p * T
    R_T = -torch.sin(2 * Δθ) * R_C

    f_L = ΔL_p / (k_L * S_L)
    f_C = ΔC_p / (k_C * S_C)
    f_H = ΔH_p / (k_H * S_H)
    ΔE_00 = torch.sqrt(f_L**2 + f_C**2 + f_H**2 + R_T * f_C * f_H)
    
    return ΔE_00


def ciede2000(color1: Tensor, color2: Tensor, is_rgb: bool = True, pixel_range: float = 1.0, k: list[float] = [1.0, 1.0, 1.0]) -> Tensor:
    if is_rgb:
        color1 = _xyz_to_lab_(_rgb_to_xyz_(color1 / pixel_range))
        color2 = _xyz_to_lab_(_rgb_to_xyz_(color2 / pixel_range))

    return _CIEDE2000_(color1, color2, *k)


# if __name__ == "__main__":
#     # lab1 = torch.tensor([17.7900,  7.9800, 11.1100]).view(1, 3, 1, 1).cuda()
#     # lab3 = torch.tensor([48.4500,  9.5700, 13.0700]).view(1, 3, 1, 1).cuda()
#     # lab2 = torch.tensor([37.5420, 12.0180, 13.3300]).view(1, 3, 1, 1).cuda()
#     # lab4 = torch.tensor([65.2000, 14.8210, 17.5450]).view(1, 3, 1, 1).cuda()

#     # lab1 = torch.concat([lab1, lab3], dim=-1)
#     # lab2 = torch.concat([lab2, lab4], dim=-1)
#     # delta = CIEDE2000(lab1, lab2)
#     # print(delta)
#     # print(delta.shape)

#     # rgb1 = torch.tensor([0.5, 0.2, 0.7]).view(1, 3, 1, 1)

#     # lab1 = _MATRIX_RGB_TO_XYZ.view(3, 3, 1, 1) @ rgb1
#     # print(lab1)

#     rgb1 = torch.tensor([255, 150, 150]).view(1, 3, 1, 1).cuda()
#     rgb1 = rgb1 / 255.0

#     lab = _xyz_to_lab_(_rgb_to_xyz_(rgb1))
#     rgb = _xyz_to_rgb_(_lab_to_xyz_(lab))

#     print(lab)
#     print(rgb)

#     ciede2000(rgb1, rgb)
