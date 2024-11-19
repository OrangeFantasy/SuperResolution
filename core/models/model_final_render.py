import torch
from torch import nn, Tensor
from torch.nn import functional as F

import math
import torch
from torch import Tensor
from torch.nn import functional as F
from typing import Any, Dict, List, Optional, Tuple

from core.modules.loss import SSIMLoss
from core import constant
from core.tools import color, parse_loss_group
from core.tools import math as mutil
from core.tools.renderer import gaussian_render
ENABLE_DEBUG = False
EXPORT_ONNX = False


def backward_warping(motion_vectors: Tensor, history_frames: Tensor, history_buffer: Tensor, history_features: Optional[Tensor] = None,
                     mode: str = "bilinear", align_corners: bool = True) -> tuple[Tensor, ...]:
    # Resize and reshape inputs to low resolution.
    b, f, _, h, w = history_frames.shape

    for idx in range(f):
        if idx > 0:
            motion_vectors[:, idx] = mutil.backward_warping(motion_vectors[:, idx], motion_vectors[:, idx - 1]) + motion_vectors[:, idx - 1]
    motion_vectors = motion_vectors.view(b * f, -1, h, w)

    history_frames = history_frames.view(b * f, -1, h, w)
    history_buffer = history_buffer.view(b * f, -1, h, w)

    warped_frames = mutil.backward_warping(history_frames, motion_vectors, mode, align_corners)
    warped_buffer = mutil.backward_warping(history_buffer, motion_vectors, mode, align_corners)

    if history_features is not None:
        history_features = history_features.view(b * f, -1, h, w)
        return warped_frames.view(b, f, -1, h, w), warped_buffer.view(b, f, -1, h, w), motion_vectors.view(b, f, -1, h, w), history_features.view(b, f, -1, h, w)
    else:
        return warped_frames.view(b, f, -1, h, w), warped_buffer.view(b, f, -1, h, w), motion_vectors.view(b, f, -1, h, w)


def clamp_depth(depths: tuple[Tensor, ...], thres1: float = 3., thres2: float = 7):
    def _clamp(depth: Tensor) -> Tensor:
        if depth.max() > thres1:
            depth = torch.where(depth >= thres2, thres1 + math.log2(thres2 - thres1) + torch.log10((depth - thres2) + 1),
                torch.where(depth >= thres1, thres1 + torch.log2((depth - thres1) + 1), depth))
        return depth
    return [_clamp(d) for d in depths]


def dot_product(x: Tensor, y: Tensor, dim: int) -> Tensor:
    return torch.sum(x * y, dim=dim, keepdim=True)


def compute_mask_normal(current_normal: Tensor, history_normal: Tensor, thres: float = 0.707) -> Tensor:
    current_normal = (current_normal - 0.5) * 2.0
    history_normal = (history_normal - 0.5) * 2.0
    _dot = dot_product(current_normal, history_normal, dim=2)
    mask = torch.logical_not(_dot.abs() < 1e-3) & (_dot < thres)  # cosine != 0 and cosine < 0.707

    if ENABLE_DEBUG:
        _mask = torch.where(mask, 1.0, 0.0)
        _mask = _mask.view(-1, *_mask.shape[-3:])
        _save_buffer(".log/.temp/mask_normal.png", _mask, n_frames=2)
    
    return mask


def compute_mask_metallic(current_metallic: Tensor, history_metallic: Tensor, mark_max_value: float = 1000., compare_diff: bool = True) -> Tensor:
    mask = current_metallic > mark_max_value
    if compare_diff:
        mask |= torch.abs(current_metallic - history_metallic) > 0.25
    if ENABLE_DEBUG:
        _mask = torch.where(mask, 1.0, 0.0)
        _mask = _mask.view(-1, *_mask.shape[-3:])
        _save_buffer(".log/.temp/mask_metallic.png", _mask, n_frames=2)
    return mask


def compute_mask_roughness(current_roughness: Tensor, history_roughness: Tensor, mark_min_value: float = -1000., compare_diff: bool = True) -> Tensor:
    mask = current_roughness < mark_min_value
    if compare_diff:
        mask |= torch.abs(current_roughness - history_roughness) > 0.25
    if ENABLE_DEBUG:
        _mask = torch.where(mask, 1.0, 0.0)
        if mask.dim() == 5:
            _mask = _mask.view(-1, *_mask.shape[2:])
        _save_buffer(".log/.temp/mask_roughness.png", _mask, n_frames=2)
    return mask


def compute_mask_basecolor(current_color: Tensor, history_color: Tensor, ksize: int = 3) -> Tensor:
    b, f, c, h, w = history_color.shape
    current_color = current_color.reshape(b * f, c, h, w)

    pad_size = ksize // 2
    current_color_pad = F.pad(current_color, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
    current_color_unfold = current_color_pad.unfold(2, ksize, 1).unfold(3, ksize, 1).reshape(*current_color.shape, ksize * ksize)

    history_color = history_color.reshape(b * f, c, h, w)[..., None]
    distance = color.rgb_colour_distance(current_color_unfold, history_color).view(b, f, 1, h, w, ksize * ksize)
    mask = (distance > 0.5).float().mean(dim=-1) > 0.7

    if ENABLE_DEBUG:
        _mask = torch.where(mask, 1.0, 0.0)
        _mask = _mask.view(-1, *_mask.shape[2:])
        _save_buffer(".log/.temp/mask_color_bf.png", _mask, n_frames=2)

    return mask


def compute_mask_depth(current_depth: Tensor, history_depth: Tensor) -> Tensor:
    mask = (current_depth - history_depth) > (0.1 * current_depth)

    if ENABLE_DEBUG:
        _mask = torch.where(mask, 1.0, 0.0)
        _mask = _mask.view(-1, *_mask.shape[2:])
        _save_buffer(".log/.temp/mask_depth.png", _mask, n_frames=2)

    return mask


def compute_mask_frame(current_frame: Tensor, history_frame: Tensor, ksize: int = 3) -> Tensor:
    pad_size = ksize // 2
    if EXPORT_ONNX:
        raise NotImplementedError
    else:
        current_frame_pad = F.pad(current_frame, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
    current_frame_unfold = current_frame_pad.unfold(2, ksize, 1).unfold(3, ksize, 1).reshape(*current_frame.shape, ksize * ksize)

    current_frame_unfold = current_frame_unfold[:, None, ...]
    mask = (history_frame < current_frame_unfold.min(dim=-1)[0]) | (history_frame > current_frame_unfold.max(dim=-1)[0])
    mask = torch.all(mask, dim=2, keepdim=True)

    if ENABLE_DEBUG:
        _mask = torch.where(mask, 1.0, 0.0)
        if _mask.dim() == 5:
            _mask = _mask.view(-1, *_mask.shape[2:])
        _save_buffer(".log/.temp/mask_frame_frame.png", _mask, n_frames=2)

    return mask

# ******************************
# For debug

from torchvision.utils import save_image

def _save_buffer(path: str, buffer: Tensor, n_frames: int = -1):
    if buffer.dim() == 4:
        bf, c, h, w = buffer.shape
        b, f = bf // n_frames, n_frames
    else:
        b, f, c, h, w = buffer.shape

    buffer = buffer.view(b, f, c, h, w)
    buffer = [buffer[:, idx] for idx in range(f)]

    to_save_buffer = torch.concat([buffer[0], *buffer[1:]], dim=-2)
    save_image(to_save_buffer, path)


class GradientMap(nn.Module):
    def __init__(self, channels: int | None = None, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

        _kernel_v = torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        _kernel_h = torch.tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        if channels is not None:
            _kernel_v = _kernel_v.repeat(channels, 1, 1, 1)
            _kernel_h = _kernel_h.repeat(channels, 1, 1, 1)
        
        self.register_buffer("kernel_v", _kernel_v)
        self.register_buffer("kernel_h", _kernel_h)

        self.pre_expand = channels is not None
    
    def forward(self, x: Tensor) -> Tensor:
        channels = x.shape[1]
        if self.pre_expand:
            kernel_v, kernel_h = self.kernel_v, self.kernel_h
        else:
            kernel_v = self.kernel_v.expand(channels, 1, 3, 3)
            kernel_h = self.kernel_h.expand(channels, 1, 3, 3)

        grad_v = F.conv2d(x, kernel_v, padding=1, groups=channels)
        grad_h = F.conv2d(x, kernel_h, padding=1, groups=channels)

        return torch.sqrt(grad_v ** 2 + grad_h ** 2 + self.eps)
    

class ConvActivation(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True)
        )


class TransConvActivation(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 2, stride: int = 2, padding: int = 0):
        super().__init__(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True)
        )


class FrameEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, inner_channels: int = 24, return_head_feature: bool = False):
        super().__init__()
        self.conv_head = ConvActivation(in_channels, out_channels)
        self.conv_main = nn.Sequential(
            ConvActivation(out_channels, out_channels),
            ConvActivation(out_channels, inner_channels),
            ConvActivation(inner_channels, out_channels),
            ConvActivation(out_channels, out_channels)
        )

        self.return_head_feature = return_head_feature
    
    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        head = self.conv_head.forward(x)
        main = self.conv_main.forward(head)

        if self.return_head_feature:
            return main, head
        else:
            return main


class GBuffersEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor: int = 4):
        super().__init__()
        inner_channels = in_channels * factor**2

        self.fold = nn.PixelUnshuffle(factor)
        self.conv_main = nn.Sequential(
            ConvActivation(inner_channels, inner_channels),
            ConvActivation(inner_channels, inner_channels, kernel_size=1, padding=0),
            ConvActivation(inner_channels, inner_channels, kernel_size=1, padding=0),
            ConvActivation(inner_channels, out_channels, kernel_size=1, padding=0),
        )
        self.conv_enhance = nn.Sequential(
            ConvActivation(7 * factor**2, 7 * factor**2),
            ConvActivation(7 * factor**2, out_channels, kernel_size=1, padding=0),
            ConvActivation(out_channels, out_channels, kernel_size=1, padding=0),
            ConvActivation(out_channels, out_channels, kernel_size=1, padding=0),        
        ) 

        self.enhance_param = nn.Parameter(torch.tensor(0.45), requires_grad=True)

    def forward(self, buffer: Tensor) -> Tensor:
        fold_normal = self.fold.forward(buffer[:, 0:3])
        fold_albedo = self.fold.forward(buffer[:, 6:9])  # 3 * f**2
        fold_depth = self.fold.forward(buffer[:, 9:10])  # 1 * f**2
        fold_enhance = torch.concat([fold_normal, fold_albedo, fold_depth], dim=1)

        fmap = self.conv_main.forward(self.fold.forward(buffer))
        fmap = fmap + self.enhance_param * self.conv_enhance.forward(fold_enhance)
        return fmap


class HistoryRefiner(nn.Module):
    def __init__(self, in_channels: int = 64, out_channels: int = 32, aux_channels: int = 16,
                 down_chs = (32, 48, 64, 64), up_chs = (32, 32, 48, 64)):
        super().__init__()
        assert down_chs[0] == up_chs[0], "Output channels must be equal to the sum of the last two channels in the encoder."

        # Adjust the input channels.
        self.refine_in = ConvActivation(in_channels, down_chs[0])  # 64 -> 32

        # Main network.
        self.conv0 = ConvActivation(down_chs[0] + aux_channels, down_chs[0])
        self.down0 = ConvActivation(down_chs[0], down_chs[1], kernel_size=3, stride=2)
        self.conv1 = ConvActivation(down_chs[1], down_chs[1])
        self.down1 = ConvActivation(down_chs[1], down_chs[2], kernel_size=3, stride=2)
        self.conv2 = ConvActivation(down_chs[2], down_chs[2])
        self.down2 = ConvActivation(down_chs[2], down_chs[3], kernel_size=3, stride=2)
        self.conv3 = ConvActivation(down_chs[3], down_chs[3])

        self.up3 = nn.Sequential(
            ConvActivation(down_chs[3], up_chs[3]),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.up2 = nn.Sequential(
            ConvActivation(up_chs[3], up_chs[2]),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.up1 = nn.Sequential(
            ConvActivation(up_chs[2], up_chs[1]),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.up0 = ConvActivation(up_chs[1], up_chs[0])
        
        self.refine_out = ConvActivation(up_chs[0], out_channels, kernel_size=1, padding=0)  # 32 -> 32

    def forward(self, x: Tensor, aux: Tensor, motion: Tensor) -> Tensor:  # mask: valid regions.
        x_in = self.refine_in.forward(x)  # 64 -> 32

        x0 = torch.concat((x_in, aux, torch.abs(motion)), dim=1)
        x0 = self.conv0.forward(x0)
        x1 = self.down0.forward(x0)
        x1 = self.conv1.forward(x1)
        x2 = self.down1.forward(x1)
        x2 = self.conv2.forward(x2)
        x3 = self.down2.forward(x2)
        x3 = self.conv3.forward(x3)

        u3 = self.up3.forward(x3)
        u2 = self.up2.forward(u3 + x2)
        u1 = self.up1.forward(u2 + x1)
        u0 = self.up0.forward(u1 + x0)

        out = u0 + x_in
        out = self.refine_out.forward(out)

        return out


class FusionNetwork(nn.Module):
    def __init__(self,  in_channels: int, first_channel: int | None = None,
                 down_chs: tuple[int, int, int, int] = (64, 128, 192, 256), up_chs: tuple[int, int, int, int] = (64, 64, 128, 192)):
        super().__init__()
        if first_channel is None:
            first_channel = down_chs[0]
        
        self.conv0 = nn.Sequential(
            ConvActivation(in_channels, first_channel),
            ConvActivation(first_channel, down_chs[0])
        )# - fmap0
        self.down1 = ConvActivation(down_chs[0], down_chs[1], kernel_size=3, stride=2)
        self.conv1 = nn.Sequential(
            ConvActivation(down_chs[1], down_chs[1]),
            ConvActivation(down_chs[1], down_chs[1])   
        ) # - fmap1
        self.down2 = ConvActivation(down_chs[1], down_chs[2], kernel_size=3, stride=2)
        self.conv2 = nn.Sequential(
            ConvActivation(down_chs[2], down_chs[2]),
            ConvActivation(down_chs[2], down_chs[2])
        ) # - fmap2
        self.down3 = ConvActivation(down_chs[2], down_chs[3], kernel_size=3, stride=2)
        self.conv3 = nn.Sequential(
            ConvActivation(down_chs[3], down_chs[3]),
            ConvActivation(down_chs[3], down_chs[3])
        ) # - fmap3

        self.up3 = nn.Sequential(
            ConvActivation(down_chs[3], up_chs[3]),
            ConvActivation(up_chs[3], up_chs[3]),
            TransConvActivation(up_chs[3], up_chs[3], kernel_size=2, stride=2)
        )
        self.up2 = nn.Sequential(
            ConvActivation(up_chs[3] + down_chs[2], 2 * up_chs[2]),
            ConvActivation(2 * up_chs[2], up_chs[2]),
            TransConvActivation(up_chs[2], up_chs[2], kernel_size=2, stride=2)
        )
        self.up1 = nn.Sequential(
            ConvActivation(up_chs[2] + down_chs[1], 2 * up_chs[1]),
            ConvActivation(2 * up_chs[1], up_chs[1]),
            TransConvActivation(up_chs[1], up_chs[1], kernel_size=2, stride=2)
        )
        self.up0 = nn.Sequential(
            ConvActivation(up_chs[1] + down_chs[0], up_chs[0]),
            ConvActivation(up_chs[0], up_chs[0])
        )
        
    def forward(self, x: Tensor) -> Tensor:
        fmap0 = self.conv0.forward(x)
        fmap1 = self.down1.forward(fmap0)
        fmap1 = self.conv1.forward(fmap1)
        fmap2 = self.down2.forward(fmap1)
        fmap2 = self.conv2.forward(fmap2)
        fmap3 = self.down3.forward(fmap2)
        fmap3 = self.conv3.forward(fmap3)

        fmap3 = self.up3.forward(fmap3)
        fmap2 = torch.concat([fmap3, fmap2], dim=1)
        fmap2 = self.up2.forward(fmap2)       
        fmap1 = torch.concat([fmap2, fmap1], dim=1)
        fmap1 = self.up1.forward(fmap1)    
        fmap0 = torch.concat([fmap1, fmap0], dim=1)
        fmap0 = self.up0.forward(fmap0)

        return fmap0


class Attention(nn.Module):
    def __init__(self, in_channels, out_channels = None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        self.conv = ConvActivation(in_channels, out_channels)
        self.w_self = ConvActivation(in_channels, 1)
        self.w_cross = ConvActivation(in_channels, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x_attn = self.conv.forward(x)
        w_self = self.w_self.forward(x)
        w_cross = self.w_cross.forward(x)
        return x_attn, w_self, w_cross


class MainNetwork(nn.Module):
    def __init__(self, basic_channels: int = 64, curr_channels: int = 64, hist_channels: int = 32, gbuffer_channels: int = 64, sr_scale: int = 4):
        super().__init__()
        assert curr_channels == hist_channels, "curr_channels and hist_channels must be the same"

        grad_channels = 6 * sr_scale**2
        self.grad_map = nn.Sequential(GradientMap(6), nn.PixelUnshuffle(sr_scale))

        self.attn_grad = Attention(grad_channels, basic_channels)
        self.attn_fmap = Attention(basic_channels, basic_channels)

        self.main_network = FusionNetwork(basic_channels + hist_channels + gbuffer_channels, basic_channels + gbuffer_channels)

    def forward(self, fmap_current, fmap_history, fmap_gbuffer, *other_inputs):
        grad_input = torch.concat((other_inputs[0][:, 0:3], other_inputs[0][:, 6:9]), dim=1)
        grad_map = self.grad_map.forward(grad_input)

        grad, g_w_self, g_w_cross = self.attn_grad.forward(grad_map)
        fmap, f_w_self, f_w_cross = self.attn_fmap.forward(fmap_current)
        
        fmap_current = fmap_current + torch.sigmoid(f_w_self * g_w_cross) * fmap + torch.sigmoid(g_w_self * f_w_cross) * grad
        
        fmap_main = torch.concat((fmap_current, fmap_history, fmap_gbuffer), dim=1)
        fmap_main = self.main_network.forward(fmap_main)

        return fmap_main


class SuperSampler(nn.Module):
    def __init__(self, frame_ch: int, buffer_ch: int, sr_scale: int, n_history: int):
        super().__init__()
        self.n_history = n_history

        _FR_CHS = 64
        _HS_CHS = _FR_CHS // self.n_history
        _BF_CHS = 64
        _DW_CHS = [64, 128, 192, 256]
        _UP_CHS = [64, 64, 128, 192]

        self.frame_encoder = FrameEncoder(frame_ch, _FR_CHS, return_head_feature=True)
        self.buffers_encoder = GBuffersEncoder(buffer_ch, _BF_CHS, factor=sr_scale)
        self.history_refiner = HistoryRefiner(_FR_CHS, _HS_CHS, aux_channels=(3 + 1 + 3) * 2 + 2)

        self.main_network = MainNetwork(hist_channels=_HS_CHS * n_history, sr_scale=sr_scale)#, _DW_CHS, _UP_CHS)

        self.conv_out = nn.Sequential(
            nn.Conv2d(_UP_CHS[0] + _DW_CHS[0], _FR_CHS * sr_scale**2, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(sr_scale),
            nn.Conv2d(_FR_CHS, 6, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True)
        )
        # self.conv_final = ConvActivation(frame_ch, frame_ch)

        self.sr_scale = sr_scale
        self.reset_parameters()
        print("Build model: SuperSampler")

    def forward(self, lr_frame: Tensor, hr_buffer: Tensor, lr_buffer: Tensor, 
                motion_vectors: Tensor, history_frames: Tensor, history_buffer: Tensor) -> Tensor:
        constant.GLOBAL_FORWARD_COUNT += 1

        # NOTE: G-buffers layout: WorldNormal(3), Metallic(1), Specular(1), Roughness(1), BaseColor(3), Depth(1)
          
        hr_buffer[:, -1], lr_buffer[:, -1], history_buffer[:, :, -1] = clamp_depth([hr_buffer[:, -1], lr_buffer[:, -1], history_buffer[:, :, -1]])
        assert hr_buffer.max() < 10, "Depth value is too large"

        # Extract current frame and G-buffers features.
        fmap_bf = self.buffers_encoder.forward(hr_buffer)
        fmap_current, fmap_shadow = self.frame_encoder.forward(lr_frame)  # Extract features

        # Backward warping.
        b, f, _, h, w = history_frames.shape
        warped_frames, warped_buffer, motion_vectors = backward_warping(motion_vectors, history_frames, history_buffer)

        # Mark invalid regions and extract history frames features.
        mask_frame = compute_mask_frame(lr_frame, warped_frames, ksize=5)

        current_buffer = lr_buffer.view(b, 1, -1, h, w).expand_as(warped_buffer)
        mask_basecolor = compute_mask_basecolor( current_buffer[:, :, 6:9], warped_buffer[:, :, 6:9] )
        mask_depth     = compute_mask_depth(     current_buffer[:, :, -1:], warped_buffer[:, :, -1:] )

        mask = mask_frame | mask_basecolor | mask_depth
        marked_frames = torch.where(mask, 0, warped_frames).view(b * f, -1, h, w)
        marked_buffer = torch.where(mask, 0, warped_buffer)

        aux_buffer = torch.concat(
            (current_buffer[:, :, 0:3], current_buffer[:, :, 6:10], marked_buffer[:, :, 0:3], marked_buffer[:, :, 6:10]), dim=2
            ).view(b * f, -1, h, w)
        aux_motion = motion_vectors.view(b * f, -1, h, w)

        fmap_history = self.frame_encoder.forward(marked_frames)[0].detach()
        fmap_history = self.history_refiner.forward(fmap_history, aux_buffer, aux_motion).view(b, -1, h, w)

        # Fusion neetwork.
        fmap = self.main_network.forward(fmap_current, fmap_history, fmap_bf, hr_buffer)

        # Upsample to high-resolution.
        fmap_out = torch.concat([fmap, fmap_shadow], dim=1)
        sr_frame = self.conv_out.forward(fmap_out)

        scales, rotations, colors = sr_frame[:, :2], sr_frame[:, 2:3], sr_frame[:, 3:6] #, sr_frame[:, 6:7]
        scales = F.relu(scales) + 1e-6
        # opacitices = torch.clamp(opacitices, 0, 1)
        colors = torch.clamp(colors, constant.CLIP_MIN_PIXEL, constant.CLIP_MAX_PIXEL)

        render_image = gaussian_render(scales, rotations, colors, max_ksize=7)
        sr_frame = 0.5 * colors + 0.5 * render_image
        if constant.GLOBAL_FORWARD_COUNT % 250 == 0:
            save_image(colors, ".log/.temp/.colors.png")
            save_image(render_image, ".log/.temp/.render_image.png")
        # save_image(render_image, 'render_image.png')
        # sr_frame = sr_frame + gaussian_render(scales, rotations, colors, opacitices, )

        # Pixel-wise loss weight.
        # mask_normal    = compute_mask_normal(    current_buffer[:, :, 0:3], warped_buffer[:, :, 0:3] )
        # mask_roughness = compute_mask_roughness( current_buffer[:, :, 5:6], warped_buffer[:, :, 5:6], mark_min_value=0.02 )
        # mask_metalloc  = compute_mask_metallic(  current_buffer[:, :, 3:4], warped_buffer[:, :, 3:4], mark_max_value=0.95 )
        # enhance_weight = torch.concat([mask_normal, mask_roughness, mask_metalloc], dim=1).float().mean(dim=2, keepdim=True).mean(dim=1)

        return sr_frame #, enhance_weight

    def reset_parameters(self):
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
        

class LossGroup(nn.Module):
    def __init__(self, lambda_l1: float, lambda_ssim: float):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()

        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim

    def forward(self, target: Tensor, output: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        l1_loss = self.l1.forward(output, target)
        ssim_loss = self.ssim.forward(output, target)

        loss = self.lambda_l1 * l1_loss + self.lambda_ssim * ssim_loss
        loss_dict = {
            "l1": l1_loss.item(),
            "ssim": ssim_loss.item(),
            "total": loss.item()
        }

        return loss, loss_dict
        