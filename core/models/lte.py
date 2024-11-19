import torch
from functools import partial
from torch import nn, Tensor
from torch.nn import functional as F
from typing import List, Tuple, Union, Optional
from torchvision.utils import save_image

from ..tools.math import make_coord


def save_coord(coord: torch.Tensor, path: str):
    x, y = coord.chunk(2, dim=-1)
    coord = torch.cat((x.squeeze(-1).unsqueeze(1), y.squeeze(-1).unsqueeze(1)), dim=-2)
    save_image(coord, path, normalize=True)


def ConvActivation(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(inplace=True)
    )


def MLP(in_channels: int, out_channels: Optional[int] = None, n_layers: int = 3, linear_mlp: bool = False, bias: bool = False):
    if out_channels is None: out_channels = in_channels
    __LayerBlock = nn.Linear if linear_mlp else partial(nn.Conv2d, kernel_size=1)

    blks = []
    for _ in range(n_layers - 1):
        blks += [
            __LayerBlock(in_channels, in_channels, bias=bias),
            nn.ReLU(inplace=True)
        ]
    blks += [__LayerBlock(in_channels, out_channels, bias=bias)]
    return nn.Sequential(*blks)


class Residual(nn.Sequential):
    def __init__(self, *modules: nn.Module):
        super().__init__(*modules)
    
    def forward(self, x: Tensor) -> Tensor:
        return x + super().forward(x)


class FrameEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                 n_layers: int = 3, inner_channels: List[int] = [32, 32], acquire_shallow: bool = True):
        super().__init__()
        inner_channels = inner_channels + [out_channels]
        assert len(inner_channels) == n_layers, "inner_channels must have same length as num_layers - 1"

        curr_ch = in_channels
        if acquire_shallow:
            self.first = ConvActivation(curr_ch, out_channels)
            curr_ch = out_channels

        blks = []
        for i in range(n_layers):
            blks += [
                nn.Conv2d(curr_ch, inner_channels[i], 3, padding=1),
                nn.ReLU(inplace=True),
            ]
            curr_ch = inner_channels[i]
        self.blks = nn.Sequential(*blks)

        self.acquire_shallow = acquire_shallow
    
    def forward(self, x: Tensor) -> Union[Tuple[Tensor, Tensor], Tensor]:
        if self.acquire_shallow:
            f_head = self.first.forward(x)
            f_main = self.blks.forward(f_head)
            return f_head, f_main
        return self.blks.forward(x)


class GBuffersEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor: int = 4):
        super().__init__()
        inner_channels = in_channels # * factor**2

        # self.fold = nn.PixelUnshuffle(factor)
        self.conv_main = nn.Sequential(
            ConvActivation(inner_channels, inner_channels),
            ConvActivation(inner_channels, inner_channels, kernel_size=1, padding=0),
            ConvActivation(inner_channels, inner_channels, kernel_size=1, padding=0),
            ConvActivation(inner_channels, out_channels, kernel_size=1, padding=0),
        )
        self.conv_enhance = nn.Sequential(
            ConvActivation(7, out_channels),
            ConvActivation(out_channels, out_channels, kernel_size=1, padding=0),
            ConvActivation(out_channels, out_channels, kernel_size=1, padding=0),
            ConvActivation(out_channels, out_channels, kernel_size=1, padding=0),        
        ) 

        self.enhance_param = nn.Parameter(torch.tensor(0.45), requires_grad=True)

    def forward(self, buffer: Tensor) -> Tensor:
        # fold_normal = self.fold.forward(buffer[:, 0:3])
        # fold_albedo = self.fold.forward(buffer[:, 6:9])  # 3 * f**2
        # fold_depth = self.fold.forward(buffer[:, 9:10])  # 1 * f**2
        fold_enhance = torch.concat((buffer[:, 0:3], buffer[:, 6:9], buffer[:, 9:10]), dim=1)

        fmap = self.conv_main.forward(buffer)
        fmap = fmap + self.enhance_param * self.conv_enhance.forward(fold_enhance)
        return fmap


class MainNetwork(nn.Module):
    def __init__(self, in_channels: int, basic_channels: int, n_layers):
        super().__init__()
        self.conv_in = ConvActivation(in_channels, basic_channels)

        __Block = Residual(
            nn.Conv2d(basic_channels, basic_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(basic_channels, basic_channels, 3, padding=1)
        )
        self.main = nn.Sequential(*[__Block for _ in range(n_layers)])

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_in.forward(x)
        x = self.main.forward(x)
        return x


class SuperSampler(nn.Module):
    def __init__(self, frame_channels: int = 3, buffer_channels: int = 10):
        super().__init__()

        _SR_CH = 64

        self.frame_encoder = FrameEncoder(frame_channels, _SR_CH, acquire_shallow=False)
        self.gbuffer_encoder = GBuffersEncoder(buffer_channels, _SR_CH, factor=4)
   
        self.conv_coef = nn.Conv2d(_SR_CH, _SR_CH, 3, padding=1)
        self.conv_freq = nn.Conv2d(_SR_CH, _SR_CH, 3, padding=1)
        self.phase = MLP(1, _SR_CH // 2, linear_mlp=True)

        self.mlp = MLP(_SR_CH, 3, n_layers=4)

        # self.main = MainNetwork(2 * _SR_CH, _SR_CH, n_layers=4)
        # self.after_main = ConvActivation(_SR_CH, 3, 3, padding=1)

        # self.conv_final = nn.Conv2d(3, 3, 3, padding=1)

        # self.gs_params = nn.Conv2d(_SR_CH, 8, 3, padding=0, bias=False)
        
        # self.local_ensemble = False
        # self.factor_embedding = nn.Embedding(_SR_CH)
        # self.blending = bool(True)

    def forward(self, lr_frame: Tensor, hr_buffer: Tensor, factor: Tensor = 4):
        # NOTE: G-buffers layout: WorldNormal(3), Metallic(1), Specular(1), Roughness(1), BaseColor(3), Depth(1)
        B, _, IH, IW = lr_frame.shape; OH, OW = hr_buffer.shape[-2:]

        f_current = self.frame_encoder.forward(lr_frame)  # Extract Features
        f_gbuffer = self.gbuffer_encoder.forward(hr_buffer)

        # Local texture estimation
        inv_factor = 1 / factor
        inv_factor = torch.tensor([1 / factor], dtype=torch.float32, device=lr_frame.device).unsqueeze(0)

        coord = make_coord((OH, OW), device=lr_frame.device).expand(B, OH, OW, 2)  # Reconstruction image coordinates

        lr_coord = make_coord((IH, IW), device=lr_frame.device).expand(B, IH, IW, 2)  # Feature coordinates
        q_coord = F.grid_sample(lr_coord.permute(0, 3, 1, 2), coord, mode="nearest", align_corners=False).permute(0, 2, 3, 1)
        relative_coord = coord - q_coord
        relative_coord[..., 0] *= IW
        relative_coord[..., 1] *= IH
        relative_coord = relative_coord.permute(0, 3, 1, 2)

        coef = self.conv_coef.forward(f_current)
        q_coef = F.grid_sample(coef, coord, mode="nearest", align_corners=False)

        freq = self.conv_freq.forward(f_gbuffer)
        q_freq = freq
        q_freq = torch.stack(torch.split(q_freq, 2, dim=1), dim=1)
        q_freq = torch.mul(q_freq, relative_coord.unsqueeze(1)).sum(dim=2)
        q_freq += self.phase.forward(inv_factor)[:, :, None, None]
        q_freq = torch.cat((torch.cos(torch.pi * q_freq), torch.sin(torch.pi * q_freq)), dim=1)

        f_lte = torch.mul(q_coef, q_freq)
        f_lte = self.mlp.forward(f_lte)
        sr_frame = f_lte
        # f_current_up = F.interpolate(f_current, size=(OH, OW), mode="bilinear", align_corners=False)
        # f_main = torch.cat((f_current_up, f_gbuffer), dim=1)
        # f_main = self.main.forward(f_main)
        # f_main = self.after_main.forward(f_main)
        
        # sr_frame = self.conv_final.forward(f_lte + f_main)
        return sr_frame

        print(relative_coord)
        pass


class LossGroup(nn.Module):
    def __init__(self, lambda_l1: float, lambda_ssim: float, lambda_perc: float = 0.05):
        super().__init__()
        from ..modules.loss import SSIMLoss, PerceptualLoss
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.perc = PerceptualLoss(net="vgg16", loss_type="l2")

        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_perc = lambda_perc

    def forward(self, target: Tensor, output: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        l1_loss = self.l1.forward(output, target)
        ssim_loss = self.ssim.forward(output, target)
        perc_loss = self.perc.forward(output, target)

        loss = self.lambda_l1 * l1_loss + self.lambda_ssim * ssim_loss + self.lambda_perc * perc_loss
        loss_dict = {
            "l1": l1_loss.item(),
            "ssim": ssim_loss.item(),
            "perc": perc_loss.item(),
            "total": loss.item()
        }
        return loss, loss_dict
        


if __name__ == '__main__':
    module = FrameEncoder(3, 32, acquire_shallow=False)
    print(module)

    x = torch.randn(1, 3, 256, 256)
    out = module.forward(x)

    if isinstance(out, tuple):
        print(out[0].shape)
        print(out[1].shape)
    else:
        print(out.shape)