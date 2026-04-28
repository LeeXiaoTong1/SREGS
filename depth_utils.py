

import cv2
import torch
import numpy as np
import os
from depth_anything_v2.dpt import DepthAnythingV2

def estimate_depth(img, model):
    if model is None:
        return None
    # 确保图像是RGB格式
    if len(img.shape) == 2:  # 如果是灰度图，转换为RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # 使用传入的模型进行深度推理
    depth = model.infer_image(img)  # HxW 深度图
    return depth


def affine_align_1d(p, t, m, eps=1e-6):
    pv = p[m].reshape(-1)
    tv = t[m].reshape(-1)
    if pv.numel() < 256:
        return p.new_tensor(1.0), p.new_tensor(0.0)
    mp, mt = pv.mean(), tv.mean()
    vp = (pv - mp).pow(2).mean().clamp_min(eps)
    cov = ((pv - mp) * (tv - mt)).mean()
    a = cov / vp
    b = mt - a * mp
    return a, b

def silog_from_logdiff(d, eps=1e-6):
    return torch.sqrt(d.pow(2).mean() - d.mean().pow(2) + eps)

