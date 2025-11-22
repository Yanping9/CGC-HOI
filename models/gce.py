import torch
from torch import nn, Tensor
from torchvision.ops import FeaturePyramidNetwork
from typing import List
from transformers import SwinTransformer
from collections import OrderedDict
import torch.nn.functional as F

class FeatureHead(nn.Module):
    def __init__(self, dim, dim_backbone, return_layer, num_layers):
        super().__init__()
        self.dim = dim
        self.dim_backbone = dim_backbone
        self.return_layer = return_layer

        in_channel_list = [
            int(dim_backbone * 2 ** i)
            for i in range(return_layer + 1, 1)
        ]
        self.fpn_src = FeaturePyramidNetwork(in_channel_list, dim)
        self.fpn_line = FeaturePyramidNetwork(in_channel_list, dim)
        self.layers_src = nn.Sequential(
            Permute([0, 2, 3, 1]),
            SwinTransformer(dim, num_layers)
        )
        self.layers_line = nn.Sequential(
            Permute([0, 2, 3, 1]),
            SwinTransformer(dim, num_layers)
        )
        self.pose_linear = nn.Linear(768, 256)
        self.src_linear = nn.Linear(768, 256)
    def transfer_pose_features(self, feat):
        feat = feat.permute(0, 2, 3, 1)
        feat = self.pose_linear(feat)
        return feat
    def forward(self, src, line, pose):
        pyramid_src = OrderedDict(  # 只拿“x[-1]”这个特征，并没有用到多层
            (f"{i}", src[i].tensors)
            for i in range(self.return_layer, 0)
        )
        pyramid_line = OrderedDict(  # 只拿“x[-1]”这个特征，并没有用到多层
            (f"{i}", line)
            for i in range(self.return_layer, 0)
        )
        mask = src[self.return_layer].mask
        src = self.fpn_src(pyramid_src)[f"{self.return_layer}"]
        line = self.fpn_line(pyramid_line)[f"{self.return_layer}"]
        src = self.layers_src(src)
        line = self.layers_line(line)
        line = bilinear_fea(line, src.shape[1], src.shape[2])
        pose = self.transfer_pose_features(pose)
        pose = bilinear_fea(pose, src.shape[1], src.shape[2])
        if src.shape == pose.shape == line.shape:
            src = torch.cat([src, line, pose], dim=3)
            src = self.src_linear(src)
        else:
            error_msg = f"shape mismatch. src:{src.shape}; line:{line.shape}, pose:{pose.shape}\n"
            with open("output/logs/gce_cat_error_hico.txt", 'a') as f:
                f.write(error_msg)
        return src, mask

class Permute(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims
    def forward(self, x: Tensor) -> Tensor:
        return x.permute(self.dims)

def bilinear_fea(feat, h, w):
    feat = feat.permute(0, 3, 1, 2)
    feat = F.interpolate(
        feat,
        size=(h, w),
        mode='bilinear',  # 双线性插值
        align_corners=False  # 是否对齐角点，根据任务需求设置
    )
    return feat.permute(0, 2, 3, 1)