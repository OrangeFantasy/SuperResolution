from sympy import im
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models import feature_extraction

from .filter import CannyFilter
from ..modules import functional as lf


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11, channel: int = 3, size_average: bool = True) -> None:
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.size_average = size_average

        sigma = 1.5
        _1d_window = lf.gaussian(window_size, sigma).view(window_size, 1)
        _2d_window = torch.mm(_1d_window, _1d_window.t()).view(1, 1, window_size, window_size)

        self.register_buffer("window", _2d_window.expand(channel, 1, window_size, window_size))

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        assert x.shape == y.shape, "Two tensor must have the same shape."
        assert x.shape[1] == self.channel, "Input and gaussian window must have the same channels."

        return 1 - lf._ssim(x, y, self.window, self.window_size, self.channel, self.size_average)


class PerceptualLoss(nn.Module):
    __supported_nets = ["vgg16", "vgg19"]
    __feature_layers = {
        "vgg16": ["features.3", "features.8", "features.15", "features.22", "features.29"],
        "vgg19": ["features.3", "features.8", "features.17", "features.26", "features.35"],
    }

    def __init__(self, net: str = "vgg16", layers = None , loss_type: str = "l1", normalize_input: bool = True) -> None:
        super().__init__()
        from omegaconf import OmegaConf

        assert net in self.__supported_nets, f"Unsupported network [{net}]"
        self.layers = self.__feature_layers[net] if layers is None else OmegaConf.to_container(layers)
        
        if net == "vgg16":
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        elif net == "vgg19":
            model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
            
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        # print(model)

        self.feature_extractor = feature_extraction.create_feature_extractor(model, self.layers)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()

        self.normalize_input = normalize_input
        if normalize_input:
            from torchvision import transforms
            self.normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        loss_type = loss_type.lower()
        if loss_type == "l1":
            self.loss_fn = F.l1_loss
        elif loss_type == "mse" or loss_type == "l2":
            self.loss_fn = F.mse_loss
        else:
            raise NotImplementedError("Unsupported loss type [%s]" % (loss_type))
        
        # print(self.feature_extractor)
        print(f"[PerceptualLoss] Using {net} with {self.layers} layers. Loss type: {loss_type}")

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert input.shape == target.shape, "Two tensor must have the same shape."

        # NOTE: input and target value range are [0, 1]
        if self.normalize_input:
            input = self.normalize_transform(input)
            target = self.normalize_transform(target.detach())

        input_features = self.feature_extractor(input)
        target_features = self.feature_extractor(target)

        losses = 0
        for layer in self.layers:
            losses += self.loss_fn(input_features[layer], target_features[layer], reduction="mean")

        return losses


class CharbonnierLoss(torch.nn.Module):
    """L1 Charbonnierloss. Reference: <<Fast and Accurate Image Super-Resolution with Deep Laplacian Pyramid Networks>>"""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, input, target):
        diff = torch.add(input, -target)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


class EnhanceL1Loss(nn.Module):
    def __init__(self, enhance_factor: float = 0.5) -> None:
        super().__init__()
        self.factor = enhance_factor

    def forward(self, input: Tensor, target: Tensor, ext_weight: Tensor):
        assert input.shape[-1] % ext_weight.shape[-1] == 0, "The size of input and mask must be divisible."
        weight = 1 + self.factor * F.interpolate(ext_weight, size=input.shape[-2:], mode="bilinear", align_corners=True)
        loss = weight * F.l1_loss(input, target, reduction="none")
        return loss.mean()


class EnhancePerceptualLoss(PerceptualLoss):
    def __init__(self, net: str = "vgg16", layers: list[str] | None = None, loss_type: str = "l1", 
                 layer_weights: list[int] | None = None, enhance_factor: float = 0.) -> None:
        super().__init__(net, layers, loss_type)
        self.factor = enhance_factor

        if layer_weights is not None:
            assert len(layer_weights) == len(self.layers), "The number of layer weights must be the same as the number of layers."
            self.layer_weights = layer_weights
        else:
            self.layer_weights = [1. for _ in range(len(self.layers))]

    def forward(self, input: Tensor, target: Tensor, ext_weight: Tensor) -> Tensor:
        assert input.shape == target.shape, "Two tensor must have the same shape."
        assert input.shape[-1] % ext_weight.shape[-1] == 0, "The size of input and mask must be divisible."

        if self.normalize_input:
            input = self.normalize_transform(input)
            target = self.normalize_transform(target.detach())

        input_features = self.feature_extractor(input)
        target_features = self.feature_extractor(target)

        losses = 0
        for idx, layer in enumerate(self.layers):
            weight = 1 + self.factor * F.interpolate(ext_weight, size=input_features[layer].shape[-2:], mode="bilinear", align_corners=True)
            losses += self.layer_weights[idx] * (weight * self.loss_fn(input_features[layer], target_features[layer], reduction="none")).mean()
        
        return losses


class EnhanceCharbonnierLoss(torch.nn.Module):
    def __init__(self, enhance_factor: float = 0.5, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.factor = enhance_factor

    def forward(self, input: Tensor, target: Tensor, ext_weight: Tensor) -> Tensor:
        assert input.shape[-1] % ext_weight.shape[-1] == 0, "The size of input and mask must be divisible."
        weight = 1 + self.factor * F.interpolate(ext_weight, size=input.shape[-2:], mode="bilinear", align_corners=True)

        diff = torch.add(input, -target)
        error = torch.sqrt(diff * diff + self.eps)
        loss = (weight * error).mean() #  torch.mean(error)
        return loss


class CannyL1Loss(nn.Module):
    def __init__(self, enhance_factor: float = 1., filter_size: int = 5, std: float = 1.0, 
                 threshold1: float = 10.0, threshold2: float = 100.0) -> None:
        super().__init__()
        self.canny = CannyFilter(filter_size, std)
        self.threshold1 = threshold1
        self.threshold2 = threshold2

        self.factor = enhance_factor

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            input (Tensor): Tensor of arbitrary shape as probabilities.
            target (Tensor): Tensor of the same shape as input.

        Returns:
            Tensor: `L1` loss (mean).
        """
        target_edge = self.canny.forward(self.normalize(target), self.threshold1, self.threshold2)
        l1_weight = 1. + self.factor * target_edge
        l1_loss = l1_weight * F.l1_loss(input, target, reduction="none")
        l1_loss = torch.sum(l1_loss) / torch.sum(l1_weight)

        return l1_loss
    
    @staticmethod
    def normalize(x: Tensor) -> Tensor:
        return (x + 1) / 2
    

class FocalFrequencyLoss(nn.Module):
    def __init__(self, loss_weight: float = 1.0, alpha: float = 1.0, patch_factor: int = 1, 
                 ave_spectrum: bool = False, log_matrix: bool = False, batch_matrix: bool = False):
        """ The torch.nn.Module class that implements focal frequency loss - a frequency domain loss function for optimizing generative models.

        Ref:
        Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021. <https://arxiv.org/pdf/2012.12821.pdf>

        Code:
        <https://github.com/EndlessSora/focal-frequency-loss/blob/master/focal_frequency_loss/focal_frequency_loss.py>

        Args:
            loss_weight (float): weight for focal frequency loss. Default: 1.0
            alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
            patch_factor (int): the factor to crop image patches for patch-based focal frequency loss. Default: 1
            ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
            log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
            batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
        """
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x: Tensor) -> Tensor:
        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, ("Patch factor should be divisible by image height and width")
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h: (i + 1) * patch_h, j * patch_w: (j + 1) * patch_w])

        # stack to patch tensor
        y = torch.stack(patch_list, 1)

        # perform 2D DFT (real-to-complex, orthonormalization)
        freq = torch.fft.fft2(y, norm="ortho")
        freq = torch.stack([freq.real, freq.imag], -1)

        return freq

    def loss_formulation(self, recon_freq: Tensor, real_freq: Tensor, matrix: Tensor = None) -> Tensor:
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            "The values of spectrum weight matrix should be in the range [0, 1], "
            "but got Min: %.10f Max: %.10f" % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(self, input: Tensor, target: Tensor, matrix: Tensor | None = None, **kwargs):
        """Forward function to calculate focal frequency loss.

        Args:
            input (torch.Tensor): of shape `(N, C, H, W)`. Predicted tensor.
            target (torch.Tensor): of shape `(N, C, H, W)`. Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        input_freq = self.tensor2freq(input)
        target_freq = self.tensor2freq(target)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            input_freq = torch.mean(input_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        # calculate focal frequency loss
        return self.loss_formulation(input_freq, target_freq, matrix) * self.loss_weight


class ContextualLoss(nn.Module):
    def __init__(self, bandwidth: float = 1.0, vgg19_layers: list[str] = ["features.7", "features.12"], loss_type: str = "cosine") -> None:
        super().__init__()
        self.bandwidth = bandwidth
        self.vgg_layers = vgg19_layers
        self.loss_type = loss_type

        from torchvision import models
        from torchvision.models.feature_extraction import create_feature_extractor
        vgg_model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self.feature_extractor = create_feature_extractor(vgg_model, vgg19_layers)

        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()

        from torchvision import transforms
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        assert x.shape == y.shape, "Two tensor must have the same shape."

        x = self.normalize(x)
        y = self.normalize(y)

        x_features = self.feature_extractor(x)
        y_features = self.feature_extractor(y)
        
        losses = []
        for layer in self.vgg_layers:
            losses.append(lf.contextual_loss(x_features[layer], y_features[layer], self.bandwidth, self.loss_type))

        if len(losses) == 1:
            return losses[0]
        return torch.stack(losses)


class ContextualBilateralLoss(ContextualLoss):
    def __init__(self, weight_spatial: float = 0.1, bandwidth: float = 1.0, 
                 vgg19_layers: list[str] = ["features.7", "features.12"], loss_type: str = "cosine") -> None:
        super().__init__(bandwidth, vgg19_layers, loss_type)
        self.weight_spatial = weight_spatial
   
    def forward(self, x: Tensor, y: Tensor):
        assert x.shape == y.shape, "Two tensor must have the same shape."

        x = self.normalize(x)
        y = self.normalize(y)

        x_features = self.feature_extractor(x)
        y_features = self.feature_extractor(y)
        
        losses = []
        for layer in self.vgg_layers:
            losses.append(
                lf.contextual_bilateral_loss(x_features[layer], y_features[layer], self.weight_spatial, self.bandwidth, self.loss_type))

        if len(losses) == 1:
            return losses[0]
        return torch.stack(losses)


class GANLoss(nn.Module):
    def __init__(self, gan_type: str, real_label_val: float = 1.0, fake_label_val: float = 0.0):
        super().__init__()
        if gan_type == "gan" or gan_type == "ragan":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_type == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_type == "wgan-gp":
            def wgan_loss(input: Tensor, target: Tensor) -> Tensor:
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError("GAN type [%s] is not found" % (self.gan_type))
        
        self.gan_type = gan_type
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

    def get_target_label(self, input: Tensor, target_is_real: bool):
        if self.gan_type == "wgan-gp":
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input: Tensor, target_is_real: bool) -> Tensor:
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class VGG19Loss(nn.Module):
    def __init__(self, vgg19_layers: list[str] = ["features.35"], loss_type: str = "l1") -> None:
        super().__init__()
        self.vgg_layers = vgg19_layers
        
        from torchvision import models
        from torchvision.models.feature_extraction import create_feature_extractor
        vgg_model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self.feature_extractor = create_feature_extractor(vgg_model, vgg19_layers)

        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()

        from torchvision import transforms
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if loss_type == "l1":
            self.cri_loss = F.l1_loss
        elif loss_type == "mse":
            self.cri_loss = F.mse_loss
        
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        assert x.shape == y.shape, "Two tensor must have the same shape."

        x = self.normalize(x)
        y = self.normalize(y)

        x_features = self.feature_extractor(x)
        y_features = self.feature_extractor(y)

        losses = []
        for layer in self.vgg_layers:
            losses.append(self.cri_loss(x_features[layer], y_features[layer]))

        if len(losses) == 1:
            return losses[0]
        return torch.stack(losses)
    

class LPIPSLoss(nn.Module):
    __supported_nets__ = ["alex", "squeeze", "vgg"]

    def __init__(self, net: str = "alex"):
        super().__init__()
        assert net in self.__supported_nets__, "Unsupported net [%s]" % (net)

        from lpips import LPIPS
        self.net = LPIPS(pretrained=True, net=net)

        for param in self.net.parameters():
            param.requires_grad = False
        self.net.eval()
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        lpips_output = self.net.forward(input, target).view(input.shape[0])
        loss = lpips_output.mean()  # sum_(layers)
        return loss


class RepRFNLoss(torch.nn.Module):
    """pixle +fft L1 Charbonnierloss."""

    def __init__(self):
        super(RepRFNLoss, self).__init__()
        self.charbonnierloss = CharbonnierLoss()
        self.l1loss = torch.nn.L1Loss()

    def forward(self, input: Tensor, target: Tensor):
        pixel_loss = self.charbonnierloss.forward(input, target)
        fft_loss = self.l1loss.forward(torch.fft.fft2(input, dim=(-2, -1)), torch.fft.fft2(target, dim=(-2, -1)))
        return 0.9 * pixel_loss + 0.1 * fft_loss


class FrequencyLoss(nn.Module):
    """from https://github.com/yanzq95/SGNet/blob/main/train.py#L51
    """

    def __init__(self, loss_type: str = "l1") -> None:
        super().__init__()
        if loss_type == "l1":
            self.criterion = F.l1_loss
        elif loss_type == "mse":
            self.criterion = F.mse_loss
        else:
            raise NotImplementedError(f"loss type [{loss_type}] is not found")
        
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input_amp, input_pha = self._rfft(input)
        target_amp, target_pha = self._rfft(target)

        loss_fre_amp = self.criterion(input_amp, target_amp)
        loss_fre_pha = self.criterion(input_pha, target_pha)

        loss_fre = 0.5 * loss_fre_amp + 0.5 * loss_fre_pha
        return loss_fre

    @staticmethod
    def _rfft(dp: Tensor) -> Tensor:
        dp = torch.fft.rfft2(dp, norm="backward")
        dp_amp = torch.abs(dp)
        dp_pha = torch.angle(dp)
        return dp_amp, dp_pha


class GradientLoss(nn.Module):
    def __init__(self, channels: int = 3, loss_type: str = "l1", eps: float = 1e-6):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.register_buffer("kernel_v", 
            torch.tensor([[0, -1, 0], [0, 0, 0], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3).expand(channels, 1, 3, 3))
        self.register_buffer("kernel_h", 
            torch.tensor([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3).expand(channels, 1, 3, 3))
    
        if loss_type == "l1":
            self.cri_loss = F.l1_loss
        else:
            raise NotImplementedError(f"loss type [{loss_type}] is not found")

    def _gradient(self, x: Tensor) -> Tensor:
        grad_v = F.conv2d(x, self.kernel_v, padding=1, groups=self.channels)
        grad_h = F.conv2d(x, self.kernel_h, padding=1, groups=self.channels)

        return torch.sqrt(grad_v**2 + grad_h**2 + self.eps)
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        grad_input = self._gradient(input)
        grad_target = self._gradient(target)

        loss = self.cri_loss(grad_input, grad_target)
        return loss


class EnhanceSSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11, channel: int = 3, enhance_factor: float = 0.5) -> None:
        super().__init__()
        self.window_size = window_size
        self.channel = channel

        sigma = 1.5
        _1d_window = lf.gaussian(window_size, sigma).view(window_size, 1)
        _2d_window = torch.mm(_1d_window, _1d_window.t()).view(1, 1, window_size, window_size)

        self.register_buffer("window", _2d_window.expand(channel, 1, window_size, window_size))

        self.factor = enhance_factor

    def forward(self, x: Tensor, y: Tensor, ext_weight: Tensor) -> Tensor:
        assert x.shape == y.shape, "Two tensor must have the same shape."
        assert x.shape[1] == self.channel, "Input and gaussian window must have the same channels."

        assert x.shape[-1] % ext_weight.shape[-1] == 0, "The size of input and mask must be divisible."
        if x.shape[-1] != ext_weight.shape[-1]:
            ext_weight = F.interpolate(ext_weight, size=x.shape[-2:], mode="bilinear", align_corners=True)
        weight = 1 + self.factor * ext_weight
        ssim_loss = 1 - lf._ssim(x, y, self.window, self.window_size, self.channel, size_average=False)
        return (ssim_loss * weight).sum() / weight.sum()


class EnhanceL1Loss_2(nn.Module):
    def __init__(self, enhance_factor: float = 0.5) -> None:
        super().__init__()
        self.factor = enhance_factor

    def forward(self, input: Tensor, target: Tensor, ext_weight: Tensor):
        assert input.shape[-1] % ext_weight.shape[-1] == 0, "The size of input and mask must be divisible."
        if input.shape[-1] != ext_weight.shape[-1]:
            ext_weight = F.interpolate(ext_weight, size=input.shape[-2:], mode="bilinear", align_corners=True)
        weight = 1 + self.factor * ext_weight
        loss = weight * F.l1_loss(input, target, reduction="none")
        return loss.sum() / weight.sum()
    

class GradientLoss_2(nn.Module):
    def __init__(self, channels: int = 3, loss_type: str = "l1", eps: float = 1e-6):
        super().__init__()
        self.channels = channels
        self.eps = eps

        # self.register_buffer("kernel_v", 
        #     torch.tensor([[0, -1, 0], [0, 0, 0], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3).expand(channels, 1, 3, 3))
        # self.register_buffer("kernel_h", 
        #     torch.tensor([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3).expand(channels, 1, 3, 3))
    
        self.pad = nn.ReflectionPad2d((1, 0, 1, 0))

        if loss_type == "l1":
            self.cri_loss = F.l1_loss
        else:
            raise NotImplementedError(f"loss type [{loss_type}] is not found")

    def _gradient(self, x: Tensor) -> Tensor:
        x = self.pad.forward(x)

        grad_x = x[:, :, 1:, 1:] - x[:, :, 1:, :-1]
        grad_y = x[:, :, 1:, 1:] - x[:, :, :-1, 1:]
        return torch.sqrt(grad_x**2 + grad_y**2 + self.eps)

    # def _gradient_2(self, x: Tensor) -> Tensor:
    #     grad_v = F.conv2d(x, self.kernel_v, padding=1, groups=self.channels)
    #     grad_h = F.conv2d(x, self.kernel_h, padding=1, groups=self.channels)

    #     return torch.sqrt(grad_v**2 + grad_h**2 + self.eps)
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        grad_input = lf.gradient_map(input, eps=self.eps)
        grad_target = lf.gradient_map(target, eps=self.eps)
        # grad_input = self._gradient(input)
        # grad_target = self._gradient(target)

        # from torchvision.utils import save_image
        # save_image(grad_input, ".log/.temp/grad_input.png")
        # save_image(grad_target, ".log/.temp/grad_target.png")

        # grad_input = self._gradient_2(input)
        # grad_target = self._gradient_2(target)

        # from torchvision.utils import save_image
        # save_image(grad_input, ".log/.temp/grad_input_2.png")
        # save_image(grad_target, ".log/.temp/grad_target_2.png")

        loss = self.cri_loss(grad_input, grad_target)
        return loss