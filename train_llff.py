#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import sys
import os
import uuid
import numpy as np
import matplotlib as plt
from argparse import ArgumentParser, Namespace
from random import randint
import math
import torch
import torch.nn.functional as F
from torchmetrics.functional.regression import pearson_corrcoef
from tqdm import tqdm

from utils.loss_utils import (
    l1_loss,
    l1_loss_mask,
    patch_norm_mse_loss,
    ssim,
)
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.normal_utils import stable_normal_prior_term
from utils.consist_view import xview_reproj_depth_loss, quick_inb_ratio
from arguments import ModelParams, PipelineParams, OptimizationParams
from lpipsPyTorch import lpips
from scene.gaussian_model import build_scaling_rotation
from depth_utils import affine_align_1d, silog_from_logdiff
from depth_anything_v2.dpt import DepthAnythingV2


DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
def load_depth_model(mode='vitl'):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    model = DepthAnythingV2(**model_configs[mode])
    model.load_state_dict(torch.load(f'checkp/depth_anything_v2_{mode}.pth', map_location='cpu'))
    model = model.to(DEVICE).eval()
    return model
depth_model = load_depth_model('vitl')

def training(dataset, opt, pipe, args, depth_model):
    testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from = args.test_iterations, \
            args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, shuffle=False, depth_model=depth_model)
    gaussians.training_setup(opt)

    train_cams = scene.getTrainCameras()
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

    viewpoint_stack, pseudo_stack = None, None
    ema_loss_for_log = 0.0
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None
        
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))


        train_cams = scene.getTrainCameras()
        if len(train_cams) >= 2:
            centers_cpu = torch.stack([c.camera_center.detach().cpu() for c in train_cams], dim=0)
            uid2idx = {c.uid: i for i, c in enumerate(train_cams)}

            def sample_neighbor_cam(cam_i, k=8, min_rank=1):
                ci = cam_i.camera_center.detach().cpu()
                d2 = ((centers_cpu - ci[None, :]) ** 2).sum(dim=1)

                i_idx = uid2idx.get(cam_i.uid, None)
                if i_idx is not None:
                    d2[i_idx] = 1e9

                k_eff = min(k, len(train_cams) - 1)
                nn = torch.topk(d2, k=k_eff, largest=False).indices

                start = min(min_rank - 1, k_eff - 1)
                pool = nn[start:]
                j_idx = pool[torch.randint(0, pool.numel(), (1,)).item()].item()
                return train_cams[j_idx]
        else:
            def sample_neighbor_cam(cam_i, k=8, min_rank=1):
                return cam_i

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()

        Ll1 =  l1_loss_mask(image, gt_image)
        loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)))
        
        # regularization
        lambda_normal = opt.lambda_normal if iteration > 2000 else 0.0
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()

        loss = loss + normal_loss
        # if iteration < 2000:
        loss = loss + args.opacity_reg * torch.abs(gaussians.get_opacity).mean()

        rendered_depth = render_pkg["depth"][0]
        midas_depth = torch.tensor(viewpoint_cam.depth_image).cuda().squeeze()

        # Ensure both depths have the same dimensions
        midas_depth_resized = torch.nn.functional.interpolate(
            midas_depth.unsqueeze(0).unsqueeze(0),
            size=rendered_depth.shape,
            mode='bicubic',
            align_corners=False
        ).squeeze()

        rendered_depth = rendered_depth.reshape(-1, 1)
        midas_depth_t = midas_depth_resized.reshape(-1, 1)

        depth_loss = min(
            (1 - pearson_corrcoef(-midas_depth_t, rendered_depth)),
            (1 - pearson_corrcoef(1 / (midas_depth_t + 200.), rendered_depth))
        )
        loss += args.depth_weight * depth_loss

        if iteration > args.end_sample_pseudo:
            args.depth_weight = 0.001

        patch_range = (5, 17)
        depth_n = render_pkg["depth"][0].unsqueeze(0)
        anyth_n = midas_depth.unsqueeze(0)
        anyth_n = 255.0 - anyth_n
        loss_l2_dpt = patch_norm_mse_loss(depth_n[None,...], anyth_n[None,...], randint(patch_range[0], patch_range[1]), opt.error_tolerance)
        loss += 0.03 * loss_l2_dpt   


        # ===== Module-1: cross-view reprojection consistency =====
        train_cams = scene.getTrainCameras()
        if len(train_cams) >= 2:
            centers_cpu = torch.stack([c.camera_center.detach().cpu() for c in train_cams], dim=0)  # [N,3]
            uid2idx = {c.uid: i for i, c in enumerate(train_cams)}

            def sample_neighbor_cam(cam_i, k=8, min_rank=2):
                ci = cam_i.camera_center.detach().cpu()
                d2 = ((centers_cpu - ci[None, :]) ** 2).sum(dim=1)

                # 排除自身
                i_idx = uid2idx.get(cam_i.uid, None)
                if i_idx is not None:
                    d2[i_idx] = 1e9

                k_eff = min(k, len(train_cams) - 1)
                nn = torch.topk(d2, k=k_eff, largest=False).indices  # 距离从近到远

                # 跳过最接近的 (min_rank-1) 个，避免“几乎同视角”
                start = min(min_rank - 1, k_eff - 1)
                pool = nn[start:]
                j_idx = pool[torch.randint(0, pool.numel(), (1,)).item()].item()
                return train_cams[j_idx]

        else:
            def sample_neighbor_cam(cam_i, k=10):
                return cam_i
        
        if iteration > 2000:
            # 建议：min_rank=1，让最近邻也可选（你已经排除了自身 uid）
            k_nn = 6
            tries = 5
            accept_th = 0.2   # 达标就用
            skip_th   = 0.10   # 低于这个直接跳过 xview（避免噪声梯度）

            best_cam = None
            best_r = -1.0

            # 先用 pkg_i 做 quick overlap 预检，不渲染 cand
            for t in range(tries):
                cand = sample_neighbor_cam(viewpoint_cam, k=k_nn, min_rank=1)
                r = quick_inb_ratio(viewpoint_cam, render_pkg, cand, n=512, alpha_th=0.2)

                if r > best_r:
                    best_r = r
                    best_cam = cand

                if r >= accept_th:
                    best_cam = cand
                    break

            if iteration % 2000 == 0:
                print(f"[xview_pick@{iteration}] best_inb_pre={best_r:.3f} accept_th={accept_th:.2f}")

            # 不达标：跳过本轮 xview（比“硬算一个很差的 xview”更稳）
            if best_cam is None or best_r < skip_th:
                loss_xview = None
            else:
                cam_j = best_cam
                # 真正需要时再 render cam_j
                pkg_j = render(cam_j, gaussians, pipe, background)
                loss_xview = xview_reproj_depth_loss(
                    viewpoint_cam, render_pkg,
                    cam_j, pkg_j,
                    tau_rel=0.04,
                    alpha_th=1e-3,
                    n_samples=8192,
                    detach_j=False,
                    debug=(iteration % 2000 == 0),
                    debug_prefix=f"xview@{iteration}",
                )

            if loss_xview is not None:
                q_inb = (best_r - skip_th) / (accept_th - skip_th + 1e-6)
                q_inb = float(max(0.0, min(1.0, q_inb)))
                xw = min(1.0, (iteration - 2000) / 2000.0)
                photo_ref = Ll1.detach()
                w_dyn = (0.1 * photo_ref / (loss_xview.detach() + 1e-6)).clamp(0.0, 2.0)
                w_eff = xw * w_dyn * (q_inb ** 2)
                loss = loss + w_eff * loss_xview

        loss.backward()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                            testing_iterations, scene, render, (pipe, background))

            if iteration > first_iter and (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration > first_iter and (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration),
                           scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # Densification
            if  iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_threshold, scene.cameras_extent, size_threshold, iteration)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            gaussians.update_learning_rate(iteration)
            if (iteration - args.start_sample_pseudo - 1) % opt.opacity_reset_interval == 0 and \
                    iteration > args.start_sample_pseudo:
                gaussians.reset_opacity()


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer



def training_report(tb_writer, iteration, Ll1, loss, l1_loss, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        # tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test, psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 8):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()

                    _mask = None
                    _psnr = psnr(image, gt_image, _mask).mean().double()
                    _ssim = ssim(image, gt_image, _mask).mean().double()
                    _lpips = lpips(image, gt_image, _mask, net_type='vgg')
                    psnr_test += _psnr
                    ssim_test += _ssim
                    lpips_test += _lpips
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {} ".format(
                    iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5000, 9000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[5000, 9000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[5000, 9000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--train_bg", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print(args.test_iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args, depth_model)

    # All done
    print("\nTraining complete.")
