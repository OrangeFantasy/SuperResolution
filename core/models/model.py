import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.utils import save_image
from typing import Dict, List, Tuple, Union

from ..tools.math import backward_warping, make_coord


def ConvActivation(in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(inplace=True)
    )


class Residual(nn.Sequential):
    def forward(self, x: Tensor) -> Tensor:
        return x + super().forward(x)


class SRResidualBlock(nn.Module):
    def __init__(self, channels: int, ratio: float = 0.2) -> None:
        super(SRResidualBlock, self).__init__()
        self.blks = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )
        self._ratio = ratio

    def forward(self, x: Tensor) -> Tensor:
        return 0.2 * self.blks.forward(x) + x


class FrameExtractor(nn.Module):
    def __init__(self, in_channels: int, inner_channels: List[int] = [32, 32, 32], n_layers: int = 3, require_first: bool = True):
        super(FrameExtractor, self).__init__()
        assert len(inner_channels) == n_layers, "inner_channels must have same length as num_layers - 1"
        curr_ch = in_channels

        blks = []
        for i in range(n_layers):
            blks += [ConvActivation(curr_ch, inner_channels[i], 3, 1, 1)]
            curr_ch = inner_channels[i]
        self.blks = nn.ModuleList(blks)

        self._require_first = require_first
    
    def forward(self, x: Tensor) -> Union[Tuple[Tensor, Tensor], Tensor]:
        f_first = self.blks[0].forward(x)
        f_frame = f_first
        for blk in self.blks[1:]:
            f_frame = blk.forward(f_frame)
    
        if self._require_first:
            return f_first, f_frame
        return f_frame
    

class GBuffersExtractor(nn.Module):
    def __init__(self, in_channels: int, inner_channels: List[int] = [64, 64, 64], n_layers: int = 3):
        super(GBuffersExtractor, self).__init__()
        curr_ch = in_channels

        blks = []
        for i in range(n_layers):
            blks += [
                ConvActivation(curr_ch, inner_channels[i], 3, 1, 1),
                Residual(
                    ConvActivation(inner_channels[i], inner_channels[i], 3, 1, 1),
                    ConvActivation(inner_channels[i], inner_channels[i], 3, 1, 1)
                )
            ]
            curr_ch = inner_channels[i]
        self.blks = nn.Sequential(*blks)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.blks.forward(x)


class MainNetwork(nn.Module):
    def __init__(self, in_channels: int, basic_channels: int, n_layers: int, res_raatio: float = 0.2) -> None:
        super(MainNetwork, self).__init__()
        self.conv_in = ConvActivation(in_channels, basic_channels, 3, 1, 1)
        self.conv_main = nn.ModuleList([
            SRResidualBlock(basic_channels, res_raatio)
            for _ in range(n_layers)
        ])
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_in(x)
        for blk in self.conv_main:
            x = blk(x)
        return x
    

class SuperSampler(nn.Module):
    def __init__(self, frame_channels: int = 3, buffer_channels: int = 10, n_history: int = 1) -> None:
        super(SuperSampler, self).__init__()
        _fr_ch = 32
        _bf_ch = 64
        _sr_ch = 64

        self.frame_extractor = FrameExtractor(frame_channels, [_fr_ch, _fr_ch, _fr_ch], 3, require_first=False)
        self.gbuffers_extractor = GBuffersExtractor(buffer_channels, [_bf_ch, _bf_ch, _bf_ch], 3)

        self.main = MainNetwork(_fr_ch + _bf_ch, _sr_ch, 6)
        self.after_main = ConvActivation(_sr_ch + _fr_ch, _sr_ch, 3, 1, 1)

        self.final_conv = nn.Sequential(
            nn.Conv2d(_sr_ch, _sr_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(_sr_ch, frame_channels, 3, 1, 1)
        )

        self.n_history = n_history
        self._init_parameters()

    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _embed_scale(scale: float, dtype: torch.dtype, device: torch.device) -> Tensor:
        pass

    @staticmethod
    def _embed_relative_coord(coord: Tensor) -> Tensor:
        pass

    @staticmethod
    def _backward_warping_sequence(frames: Tensor, velocity: Tensor, mode: str = "nearest", align_corners: bool = False) -> Tensor:
        b, l, _, h, w = frames.shape
        return backward_warping(frames.view(b * l, -1, h, w), velocity.view(b * l, -1, h, w), mode, align_corners).view(b, l, -1, h, w)

    def forward(self, batch: Dict[str, Union[Tensor, int]]):
        lr_frames = batch.pop("lr_frames")
        hr_buffers = batch.pop("hr_buffers")

        b, _, _, sh, sw = lr_frames.shape; th, tw = hr_buffers.shape[-2:]

        hr_coord = make_coord((th, tw), dtype=lr_frames.dtype, device=lr_frames.device).expand(b, 2, th, tw)  # Reconstruction image coordinates
        # lr_coord = make_coord((sh, sw), dtype=lr_frames.dtype, device=lr_frames.device).expand(b, 2, sh, sw)  # Feature coordinates

        # coord_sample = F.grid_sample(lr_coord, hr_coord.permute(0, 2, 3, 1), mode="nearest", align_corners=False)
        # relative_coord = hr_coord - coord_sample
        # relative_coord[:, 0] *= sw
        # relative_coord[:, 1] *= sh

        fmap_curr = self.frame_extractor.forward(lr_frames[:, -1])
        fmap_curr_sample = F.grid_sample(fmap_curr, hr_coord.permute(0, 2, 3, 1), mode="bilinear", align_corners=False)

        # curr_sample = F.grid_sample(lr_frames[:, -1], hr_coord.permute(0, 2, 3, 1), mode="nearest", align_corners=False)
        # save_image(curr_sample, "curr_sample.png")

        fmap_buff = self.gbuffers_extractor.forward(hr_buffers[:, -1])

        fmap = self.main.forward(torch.concat((fmap_curr_sample, fmap_buff), dim=1))
        fmap = self.after_main.forward(torch.concat((fmap_curr_sample, fmap), dim=1))

        sr_frames = self.final_conv.forward(fmap)
        batch["sr_frames"] = sr_frames
        
        return batch

        # velocity = batch["velocity"]
        # for idx in range(self.n_history):
        #     if idx > 0:
        #         velocity[:, idx] = backward_warping(velocity[:, idx], velocity[:, idx - 1]) + velocity[:, idx - 1]


class LossGroup(nn.Module):
    def __init__(self, lambda_l1: float, lambda_ssim: float):
        super().__init__()
        from ..modules.loss import SSIMLoss
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()

        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim

    def forward(self, batch) -> Tensor:
        target = batch["hr_frames"]
        output = batch["sr_frames"]

        l1_loss = self.l1.forward(output, target)
        ssim_loss = self.ssim.forward(output, target)

        loss = self.lambda_l1 * l1_loss + self.lambda_ssim * ssim_loss
        loss_dict = {
            "l1": l1_loss.item(),
            "ssim": ssim_loss.item(),
            "total": loss.item()
        }

        return loss, loss_dict


if __name__ == "__main__":
    module = FrameExtractor(3, [32, 32, 32], 3, True)
    x = torch.randn(1, 3, 64, 64)
    print(module.forward(x)[0].shape, module.forward(x)[1].shape)