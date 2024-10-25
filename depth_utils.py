

import cv2
import torch
import numpy as np
import os
from depth_anything_v2.dpt import DepthAnythingV2

def estimate_depth(img, model):
    # 确保图像是RGB格式
    if len(img.shape) == 2:  # 如果是灰度图，转换为RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # 使用传入的模型进行深度推理
    depth = model.infer_image(img)  # HxW 深度图
    return depth