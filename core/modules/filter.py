import torch
from torch import Tensor, nn
from torch.nn import functional as F
from einops import rearrange
from scipy.signal.windows import gaussian


class CannyFilter(nn.Module):
    def __init__(self, filter_size: int = 5, std: float = 1.0):
        super().__init__()
        self.filter_size = filter_size

        gaussian_filter = torch.from_numpy(gaussian(filter_size, std)).float()
        self.register_buffer("gaussian_filter_x", rearrange(gaussian_filter, "k -> 1 1 1 k").repeat(3, 1, 1, 1))
        self.register_buffer("gaussian_filter_y", rearrange(gaussian_filter, "k -> 1 1 k 1").repeat(3, 1, 1, 1))

        sobel_filter = torch.tensor(
            [[1, 0, -1],
             [2, 0, -2], 
             [1, 0, -1]], dtype=torch.float32)
        self.register_buffer("sobel_filter_x", rearrange(sobel_filter, "k1 k2 -> 1 1 k1 k2").repeat(3, 1, 1, 1))
        self.register_buffer("sobel_filter_y", rearrange(sobel_filter, "k1 k2 -> 1 1 k2 k1").repeat(3, 1, 1, 1))

        directional_filter = torch.tensor(
            [[[[ 0,  0,  0],
               [ 0,  1, -1],
               [ 0,  0,  0]]],

             [[[ 0,  0,  0],
               [ 0,  1,  0],
               [ 0,  0, -1]]],

             [[[ 0,  0,  0],
               [ 0,  1,  0],
               [ 0, -1,  0]]],

             [[[ 0,  0,  0],
               [ 0,  1,  0],
               [-1,  0,  0]]],

             [[[ 0,  0,  0],
               [-1,  1,  0],
               [ 0,  0,  0]]],

             [[[-1,  0,  0],
               [ 0,  1,  0],
               [ 0,  0,  0]]],

             [[[ 0, -1,  0],
               [ 0,  1,  0],
               [ 0,  0,  0]]],

             [[[ 0,  0, -1],
               [ 0,  1,  0],
               [ 0,  0,  0]]]], dtype=torch.float32)
        self.register_buffer("directional_filter", directional_filter)
        
        connect_filter = torch.tensor(
            [[1, 1, 1], 
             [1, 0, 1], 
             [1, 1, 1]], dtype=torch.float32)
        self.register_buffer("connect_filter", rearrange(connect_filter, "k1 k2 -> 1 1 k1 k2"))

    @torch.no_grad()
    def forward(self, img: Tensor, threshold1: float = 10.0, threshold2: float = 100.0) -> Tensor:
        assert img.shape[1] == 3, "Image channels must be 3."
        assert threshold1 < threshold2, "Threshold 1 must lower threshold 2."

        # Gaussian filter.
        blurred_img = F.conv2d(img, self.gaussian_filter_x, padding=(0, self.filter_size // 2), groups=3)
        blurred_img = F.conv2d(blurred_img, self.gaussian_filter_y, padding=(self.filter_size // 2, 0), groups=3)

        # Compute grad.
        grad_x = F.conv2d(blurred_img, self.sobel_filter_x, padding=1, groups=3)
        grad_y = F.conv2d(blurred_img, self.sobel_filter_y, padding=1, groups=3)
        
        grad_mag = torch.sum(torch.sqrt(grad_x**2 + grad_y**2), dim=1, keepdim=True)
        grad_orientation = torch.sum(torch.atan2(grad_y, grad_x), dim=1, keepdim=True) * (180. / torch.pi) + 180.
        grad_orientation = torch.round(grad_orientation / 45.) * 45.

        # Non-Maximum Suppression.
        all_filtered = F.conv2d(grad_mag, self.directional_filter, padding=1)

        inidices_positive = (grad_orientation / 45) % 8
        inidices_negative = ((grad_orientation / 45) + 4) % 8

        batch, _, height, width = inidices_positive.shape
        pixel_count = height * width * batch
        pixel_range = torch.arange(pixel_count, device=img.device)

        indices = (inidices_positive.long().reshape(-1) * pixel_count + pixel_range)
        positive = all_filtered.view(-1)[indices].view((batch, 1, height, width))

        indices = (inidices_negative.long().reshape(-1) * pixel_count + pixel_range)
        negative = all_filtered.view(-1)[indices].view((batch, 1, height, width))

        is_max = torch.stack([positive, negative]).min(dim=0)[0] > 0.0
        thin_edges = grad_mag
        thin_edges[is_max == 0] = 0.0

        # Double threshold.
        edges = thin_edges.clone()
        lower = thin_edges < threshold1
        edges[lower] = 0.0
        higher = thin_edges > threshold2
        edges[higher] = 1.0

        connect_map = F.conv2d(higher.float(), self.connect_filter, padding=1)
        middle = torch.logical_and(thin_edges >= threshold1, thin_edges <= threshold2)
        edges[middle] = 0.0
        connect_map[torch.logical_not(middle)] = 0
        edges[connect_map > 0] = 1.0
        
        edges[...,  0, :] = 0.0
        edges[..., -1, :] = 0.0
        edges[..., :,  0] = 0.0
        edges[..., :, -1] = 0.0
        edges = (edges > 0.0).float()

        return edges
    

if __name__ == '__main__':
    import cv2
    import numpy as np
    img_path = "/media/orae/HDD/Dataset/SuperResolution/DIV2K/DIV2K_train_LR_bicubic/X2/0008x2.png"
    res_path = "canny_torch_our.png"
    img = cv2.imread(img_path) / 255.0 # height, width, channel
    img = np.transpose(img, [2, 1, 0]) # channel width height
    canny_operator = CannyFilter(5).cuda()
    result = canny_operator.forward(torch.from_numpy(np.expand_dims(img, axis=0)).float().cuda(), threshold1=2, threshold2=4) # batch channel width height
    res = np.squeeze(result.cpu().numpy())
    res = np.transpose(res, [1, 0])
    res = (res*255).astype(np.uint8)
    cv2.imwrite(res_path, res)
    