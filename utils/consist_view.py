import math
import torch
import sys
import os
import uuid
from argparse import ArgumentParser, Namespace
from random import randint

import torch
import torch.nn.functional as F



@torch.no_grad()
def quick_inb_ratio(cam_i, pkg_i, cam_j, n=512, alpha_th=0.2, z_eps=1e-6):
    """
    用 view-i 的 depth/alpha 抽样像素 -> 反投影到世界 -> 投影到 view-j
    返回落在 view-j 视野内且 z>0 的比例（inb_ratio）
    """
    depth_i = pkg_i["depth"][0]  # [H,W]
    H, W = depth_i.shape
    Hj, Wj = _get_hw(cam_j)
    device = depth_i.device

    alpha = pkg_i.get("rend_alpha", None)
    if alpha is not None:
        a2d = alpha[0] if alpha.dim() == 3 else alpha
        valid = (a2d > alpha_th) & torch.isfinite(depth_i) & (depth_i > z_eps)
    else:
        valid = torch.isfinite(depth_i) & (depth_i > z_eps)

    idx = torch.nonzero(valid.flatten(), as_tuple=False).flatten()
    if idx.numel() < 64:
        return 0.0

    # 抽样 n 个像素（不足就全用）
    if idx.numel() > n:
        pick = idx[torch.randint(0, idx.numel(), (n,), device=device)]
    else:
        pick = idx

    pick = pick.to(torch.int64)

    ys_i = torch.div(pick, W, rounding_mode="trunc")      
    xs_i = torch.remainder(pick, W)           

    ys = ys_i.to(torch.float32)
    xs = xs_i.to(torch.float32)


    K_i, K_i_inv = get_K_Kinv(cam_i, device=device, dtype=torch.float32)
    K_j, _       = get_K_Kinv(cam_j, device=device, dtype=torch.float32)

    ones = torch.ones_like(xs)
    pix = torch.stack([xs, ys, ones], dim=0)  # [3,N]
    z = depth_i[ys.long(), xs.long()].unsqueeze(0)  # [1,N]
    X_ci = (K_i_inv @ pix) * z  # [3,N]

    
    W2C_i, C2W_i = get_W2C_C2W(cam_i, device=device, dtype=torch.float32)
    Xw = (C2W_i[:3,:3] @ X_ci) + C2W_i[:3,3:4]

    W2C_j, _     = get_W2C_C2W(cam_j, device=device, dtype=torch.float32)
    X_cj = (W2C_j[:3,:3] @ Xw) + W2C_j[:3,3:4]
    z_j = X_cj[2, :]

    uv = (K_j @ X_cj)
    u = uv[0, :] / (uv[2, :] + 1e-12)
    v = uv[1, :] / (uv[2, :] + 1e-12)

    # inb = (z_j > z_eps) & (u >= 0) & (u <= (W - 1)) & (v >= 0) & (v <= (H - 1))
    inb = (z_j > z_eps) & (u >= 0) & (u <= (Wj - 1)) & (v >= 0) & (v <= (Hj - 1))
    return float(inb.float().mean().item())


# =========================
# Camera matrix cache
# =========================

# NOTE: K depends on (FoVx, FoVy, H, W). W2C depends on camera extrinsics.
# In training they are constant per camera, so caching is safe.

_K_cache = {}       # key -> K (3x3)
_Kinv_cache = {}    # key -> inv(K) (3x3)
_W2C_cache = {}     # key -> W2C (4x4)
_C2W_cache = {}     # key -> inv(W2C) (4x4)

def _cam_uid(cam):
    # Prefer stable uid; fallback to python object id
    if hasattr(cam, "uid"):
        try:
            return int(cam.uid)
        except Exception:
            pass
    return int(id(cam))

def _cache_key(cam, device, dtype):
    # include resolution because K uses H/W
    H, W = _get_hw(cam)
    dev = torch.device(device)
    # include device index explicitly for cuda
    dev_key = (dev.type, dev.index) if dev.type == "cuda" else (dev.type, None)
    return (_cam_uid(cam), dev_key, str(dtype), int(H), int(W))

@torch.no_grad()
def get_K_Kinv(cam, device="cuda", dtype=torch.float32):
    key = _cache_key(cam, device, dtype)
    K = _K_cache.get(key, None)
    Ki = _Kinv_cache.get(key, None)
    if K is None or Ki is None:
        K = _get_K_from_fov(cam, device=device).to(dtype)
        Ki = torch.inverse(K)
        _K_cache[key] = K
        _Kinv_cache[key] = Ki
    return K, Ki

@torch.no_grad()
def get_W2C_C2W(cam, device="cuda", dtype=torch.float32):
    key = _cache_key(cam, device, dtype)
    W2C = _W2C_cache.get(key, None)
    C2W = _C2W_cache.get(key, None)
    if W2C is None or C2W is None:
        W2C = _get_W2C(cam, device=device).to(dtype)
        C2W = torch.inverse(W2C)
        _W2C_cache[key] = W2C
        _C2W_cache[key] = C2W
    return W2C, C2W

def clear_consist_view_cache():
    """Call once if you switch scenes / reload cameras to avoid stale entries."""
    _K_cache.clear()
    _Kinv_cache.clear()
    _W2C_cache.clear()
    _C2W_cache.clear()


def _get_hw(cam):
    # cam.original_image: [3,H,W] 或 cam.image: [3,H,W]
    if hasattr(cam, "original_image") and cam.original_image is not None:
        H, W = cam.original_image.shape[1], cam.original_image.shape[2]
    else:
        # 兜底：用 depth_image 的尺寸
        H, W = cam.depth_image.shape[:2]
    return H, W

def _get_K_from_fov(cam, device="cuda"):
    H, W = _get_hw(cam)
    # GraphDECO/3DGS 通常 FoVx/FoVy 是弧度
    fx = 0.5 * W / math.tan(float(cam.FoVx) * 0.5)
    fy = 0.5 * H / math.tan(float(cam.FoVy) * 0.5)
    cx = 0.5 * (W - 1)
    cy = 0.5 * (H - 1)
    K = torch.tensor([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
    return K

def _get_W2C(cam, device="cuda"):
    # 优先用 Camera 里现成的 world_view_transform（GraphDECO 通常存的是 transpose 后的）
    if hasattr(cam, "world_view_transform"):
        W2C = cam.world_view_transform
        if W2C.shape == (4, 4):
            # GraphDECO 常见：world_view_transform = getWorld2View2(...).T
            W2C = W2C.transpose(0, 1).contiguous()
        return W2C.to(device).float()

    # 兜底：用 COLMAP 的 R,T (x_cam = R x_world + T)
    R = torch.tensor(cam.R, device=device, dtype=torch.float32)
    T = torch.tensor(cam.T, device=device, dtype=torch.float32).view(3, 1)
    W2C = torch.eye(4, device=device, dtype=torch.float32)
    W2C[:3, :3] = R
    W2C[:3, 3:4] = T
    return W2C

def _charbonnier(x, eps=1e-3):
    return torch.sqrt(x * x + eps * eps)

def xview_reproj_depth_loss(
    cam_i, pkg_i,
    cam_j, pkg_j,
    tau_rel=0.02,       
    alpha_th=1e-3,
    n_samples=8192,
    detach_j=False,
    debug=False,
    debug_prefix="xview",
):

    device = pkg_i["depth"].device
    depth_i = pkg_i["depth"][0] if pkg_i["depth"].dim() == 3 else pkg_i["depth"]
    depth_j = pkg_j["depth"][0] if pkg_j["depth"].dim() == 3 else pkg_j["depth"] 
    if detach_j:
        depth_j = depth_j.detach()

    H, W = depth_i.shape
    alpha_i = pkg_i.get("rend_alpha", pkg_i.get("alpha", None))
    alpha_j = pkg_j.get("rend_alpha", pkg_j.get("alpha", None))

    if alpha_i is not None:
        alpha_i = alpha_i[0] if alpha_i.dim() == 3 else alpha_i
    if alpha_j is not None:
        alpha_j = alpha_j[0] if alpha_j.dim() == 3 else alpha_j
        if detach_j:
            alpha_j = alpha_j.detach()


    valid = torch.isfinite(depth_i) & (depth_i > 1e-6)
    if alpha_i is not None:
        valid = valid & (alpha_i > alpha_th)

    # 采样像素
    idx = torch.nonzero(valid, as_tuple=False)
    if idx.numel() < 128:
        return depth_i.new_tensor(0.0)

    if idx.shape[0] > n_samples:
        perm = torch.randperm(idx.shape[0], device=device)[:n_samples]
        idx = idx[perm]
    
    N = int(idx.shape[0])

    v = idx[:, 0].float()  # row
    u = idx[:, 1].float()  # col
    z = depth_i[idx[:, 0], idx[:, 1]]  # camera-z depth

    # 相机矩阵
    # cached intrinsics/extrinsics (no repeated inverse)
    K_i, K_i_inv = get_K_Kinv(cam_i, device=device, dtype=torch.float32)
    K_j, _       = get_K_Kinv(cam_j, device=device, dtype=torch.float32)

    W2C_i, C2W_i = get_W2C_C2W(cam_i, device=device, dtype=torch.float32)
    W2C_j, _     = get_W2C_C2W(cam_j, device=device, dtype=torch.float32)


    # 反投影到 camera i 坐标
    ones = torch.ones_like(u)
    pix = torch.stack([u, v, ones], dim=1)  # [N,3]
    xyz_i = (pix @ K_i_inv.T) * z[:, None]  # [N,3]

    # camera i -> world
    xyz_i_h = torch.cat([xyz_i, torch.ones((xyz_i.shape[0], 1), device=device)], dim=1)  # [N,4]
    xyz_w = (xyz_i_h @ C2W_i.T)  # [N,4]

    # world -> camera j
    xyz_j = (xyz_w @ W2C_j.T)[:, :3]  # [N,3]
    z_ij = xyz_j[:, 2].clamp_min(1e-6)

    # 投影到 j 像素
    uvw = (xyz_j @ K_j.T)  # [N,3]
    u2 = uvw[:, 0] / z_ij
    v2 = uvw[:, 1] / z_ij

    inb = (u2 >= 0) & (u2 <= (W - 1)) & (v2 >= 0) & (v2 <= (H - 1))
    N_inb = int(inb.sum().item())

    if debug:
        print(f"[{debug_prefix}] N={N} | inb={N_inb}/{N} ({N_inb/max(N,1):.3f})")

    if N_inb < 128:
        return depth_i.new_tensor(0.0)


    Hi, Wi = depth_i.shape
    Hj, Wj = depth_j.shape

    # ... u2,v2 是投影到 j 的像素坐标
    inb = (u2 >= 0) & (u2 <= (Wj - 1)) & (v2 >= 0) & (v2 <= (Hj - 1))

    u2 = u2[inb]; v2 = v2[inb]; z_ij = z_ij[inb]

    u2i = torch.round(u2).long().clamp(0, Wj - 1)
    v2i = torch.round(v2).long().clamp(0, Hj - 1)

    dj = depth_j[v2i, u2i]

    if alpha_j is not None:
        aj = alpha_j[v2i, u2i]

    dist_j = pkg_j.get("rend_dist", None)
    if dist_j is not None:
        dist_j = dist_j[0] if dist_j.dim()==3 else dist_j
        dj_dist = dist_j[v2i, u2i]


    # ---- sample alpha -> vis weight ----
    aj = None
    if alpha_j is not None:
        aj = alpha_j[v2i, u2i]  # [N_inb]
        if debug:
            print(f"[{debug_prefix}] aj min/med/max = "
                f"{aj.min().item():.4g}, {aj.median().item():.4g}, {aj.max().item():.4g}")

    # ---- base valid ----
    base = torch.isfinite(dj) & (dj > 1e-6) & torch.isfinite(z_ij) & (z_ij > 1e-6)

    # ---- occlusion soft weight (TSDF-friendly) ----
    # rel = (dj - z_ij) / z_ij
    rel = (dj - z_ij) / z_ij.clamp_min(1e-6)

    tau_occ = 0.02      # 0.02~0.05 可调
    temp_occ = 0.01     # 越小越接近 hard gate
    w_occ = torch.sigmoid((rel + tau_occ) / temp_occ).detach()  # [N_inb]

    # ---- visibility: 建议别再用 0.01 了，你 vis_ratio 永远 1.0 ----
    if aj is not None:
        alpha_vis_th = 0.3   # 先试 0.3/0.5（你之前 0.01 基本等于没 gate）
        temp_vis = 0.05
        w_vis = torch.sigmoid((aj - alpha_vis_th) / temp_vis).detach()
    else:
        w_vis = torch.ones_like(dj)

    # ---- dist soft weight (替代 hard dist_ok) ----
    if dj_dist is not None:
        dj_dist = dj_dist.clamp_min(0.0)
        if debug:
            print(f"[{debug_prefix}] dist min/med/max = "
                f"{dj_dist.min().item():.4g}, {dj_dist.median().item():.4g}, {dj_dist.max().item():.4g}")

        # sigma_dist 控制“多严格”：越小越像 hard gate
        sigma_dist = 5e-4    # 经验起点：5e-4；你看到 max 到 0.05 时也不至于全砍
        w_dist = torch.exp(-dj_dist / sigma_dist).detach()
    else:
        w_dist = torch.ones_like(dj)

    # ---- final mask + weights ----
    mask = base
    w_extra = (w_vis * w_occ * w_dist)


    if mask.sum() < 64:
        return depth_i.new_tensor(0.0)

    # ---- apply mask ----
    rel = rel[mask]
    w_extra = w_extra[mask]

    # ---- your original robust weight (error-based) ----
    err = rel.abs()
    w_err = torch.exp(-err / tau_rel).detach()

    w = (w_err * w_extra)
    loss = (w * _charbonnier(rel, eps=1e-3)).sum() / (w.sum() + 1e-6)

    return loss

