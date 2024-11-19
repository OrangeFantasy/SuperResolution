# import torch
# from torch.nn import functional as F

# from core.utils.math import make_coord, backward_warping, dilate_velocity


# def swap_xy(coord):
#     return coord.flip(-1)

# def swap_hw(coord):
#     return coord.permute(0, 2, 1, 3)

# def swap_xy_1(coord):
#     return coord.flip(1)

# def swap_hw_1(coord):
#     return coord.permute(0, 1, 3, 2)

# from core.utils.math import make_coord

# from torch.utils.data import DataLoader, Dataset


# class TestDataset(Dataset):
#     def __init__(self):
#         super().__init__()
    
#     def __len__(self):
#         return 100

#     def __getitem__(self, idx):
#         return torch.randn(1, 3, 2, 4)



# # if __name__ == '__main__':
# #     lr_frame = torch.randn(2, 3, 2, 4)
# #     motion = torch.zeros(2, 2, 4, 2)
# #     depth = torch.ones(2, 1, 2, 4)
# #     hr_buffer = torch.randn(2, 10, 4, 8)
# #     factor = torch.tensor([[2], [2]], dtype=torch.float32)

# #     from core.models.lte import Sampler

# #     model = Sampler()
# #     out = model.forward(lr_frame, hr_buffer, lr_frame, lr_frame, lr_frame, lr_frame, factor)
# #     print(out.shape)

# #     x = torch.tensor(
# #         [[1, 0, 0, 2],
# #          [0, 1, 2, 0],
# #          [0, 3, 4, 0],
# #          [3, 0, 0, 4]],
# #     dtype=torch.float32).view(1, 1, 4, 4)
# #     coord = make_coord((4, 4), device=x.device, flatten=False)  # [B, h, w, 2] => 2 for x, y

# #     rotate = torch.tensor(
# #         [[0, 1],
# #          [-1, 0]],
# #     dtype=torch.float32).view(1, 2, 2)  # rotate 90 degree
# #     coord = coord @ rotate
    
# #     x_ = F.grid_sample(x, coord, mode='nearest', align_corners=False)

# #     print(x)
# #     print(x_)

#     # print(coord)
#     # print(coord.shape)
    

#     # motion = dilate_velocity(motion, depth)

#     # backward_warping(inputs, motion)



#     # device = "cuda:0" if torch.cuda.is_available() else "cpu"
#     # B = 2
#     # IH, IW = 2, 4
#     # OH, OW = 4, 8

#     # coord = make_coord((OH, OW), device=device).expand(B, OH * OW, 2)  # Reconstruction image coordinates
#     # feat_coord = make_coord((IH, IW), device=device, flatten=False).permute(0, 3, 1, 2).expand(B, 2, IH, IW)  # Feature coordinates
#     # q_coord = F.grid_sample(feat_coord, coord.unsqueeze(1), mode="bilinear", align_corners=False).permute(0, 2, 3, 1).squeeze(1) # Query coordinates

#     # print(q_coord.shape)



#     # ih, iw = 2, 3
#     # th, tw = 4, 6

#     # lr = torch.randn(1, 3, ih, iw, device=device)
#     # hr = torch.randn(1, 3, th, tw, device=device)

#     # coord = make_coord((th, tw), device=device).unsqueeze(0)
#     # coord_ = make_coord_2((th, tw), device=device).unsqueeze(0)

#     # feat_coord = make_coord((ih, iw), device=device, flatten=False) \
#     #     .permute(2, 0, 1) \
#     #     .unsqueeze(0).expand(1, 2, ih, iw)
#     # feat_coord_ = make_coord_2((ih, iw), device=device, flatten=False) \
#     #     .permute(2, 0, 1) \
#     #     .unsqueeze(0).expand(1, 2, ih, iw)

#     # q_coord = F.grid_sample(
#     #     feat_coord, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :] \
#     #     .permute(0, 2, 1)
#     # q_coord_ = F.grid_sample(
#     #     feat_coord_, coord_.unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :] \
#     #     .permute(0, 2, 1)


#     # rel_coord = coord - q_coord
#     # rel_coord[:, :, 0] *= ih
#     # rel_coord[:, :, 1] *= iw
#     # my_rel_coord = rel_coord.permute(0, 2, 1).view(rel_coord.shape[0], rel_coord.shape[2], th, tw)

#     # rel_coord_ = coord_ - q_coord_
#     # rel_coord_[:, :, 0] *= iw
#     # rel_coord_[:, :, 1] *= ih
#     # my_rel_coord_ = rel_coord_.permute(0, 2, 1).view(rel_coord_.shape[0], rel_coord_.shape[2], th, tw)



#     # print(swap_xy_1(my_rel_coord))
#     # print(my_rel_coord_)
#     # print(swap_xy_1(my_rel_coord) == my_rel_coord_)


#     # rel_coord = coord - q_coord
#     # rel_coord[:, :, 0] *= ih
#     # rel_coord[:, :, 1] *= iw
#     # rel_coord = rel_coord.permute(0, 2, 1).view(rel_coord.shape[0], rel_coord.shape[2], th, tw)
#     # print(q_coord.view(1, th, tw, 2))
#     # print(q_coord)



#     # print(swap_hw(coord.view(1, 2, tw, th)).view(1, 2, tw * th, 2))
#     # print(coord_)





#     # feat_coord_ = make_coord_2((ih, iw), device=device, flatten=False) \
#     #     .permute(2, 0, 1) \
#     #     .unsqueeze(0).expand(lr.shape[0], 2, ih, iw)
#     # q_coord_ = F.grid_sample(
#     #     feat_coord_, coord_.unsqueeze(1), mode='nearest', align_corners=False)
#     # q_coord_ = q_coord_[:, :, 0, :].permute(0, 2, 1)

#     # rel_coord_ = coord_ - q_coord_
#     # rel_coord_[:, :, 0] *= iw
#     # rel_coord_[:, :, 1] *= ih
#     # rel_coord_ = rel_coord_.permute(0, 2, 1).view(rel_coord_.shape[0], rel_coord_.shape[2], th, tw)
#     # q_coord_ = q_coord_.view(1, th, tw, 2)

#     # q_coord__ = q_coord_.clone()
#     # q_coord__[..., 0] = q_coord_[..., 1]
#     # q_coord__[..., 1] = q_coord_[..., 0]
#     # tmp = q_coord_[..., 0].contiguous()
#     # q_coord_[..., 0] = q_coord_[..., 1]
#     # q_coord_[..., 1] = tmp
#     # print(q_coord__)
#     # print(q_coord_)
#     # print((coord == coord_).all())
#     # print((q_coord == q_coord_).all())
#     # print(q_coord)


# for i in range(10, 80):
#     print(i / 10, 480 / (i / 10))

# scales = [1.2, 1.6, 2.0, 2.4, 3.2, 4.0, 4.8, 6.0]
# for i in range(0, 14000, 10):
#     flag = True
#     for scale in scales:
#         if i % int(scale * 10) != 0:
#             flag = False
#     if flag:
#         print(i)

# l = [1, 1, 1]
# print(l[1:1])
import torch
from torch.nn import functional as F
from torchvision.utils import save_image
from core.tools.math import make_coord

th, tw = 48, 64
sh, sw = 3, 4

hr_coord = make_coord((th, tw)).expand(1, 2, th, tw)  # Reconstruction image coordinates
lr_coord = make_coord((sh, sw)).expand(1, 2, sh, sw)  # Feature coordinates

coord_sample = F.grid_sample(lr_coord, hr_coord.permute(0, 2, 3, 1), mode="nearest", align_corners=False)
relative_coord = hr_coord - coord_sample
relative_coord[:, 0] *= sw
relative_coord[:, 1] *= sh

relative_coord[:, 0] = torch.sin(relative_coord[:, 0])
relative_coord[:, 1] = torch.sin(relative_coord[:, 1])

save_image(relative_coord[:, 0], "x.png")
save_image(relative_coord[:, 1], "y.png")

print(relative_coord)