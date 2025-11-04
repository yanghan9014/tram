import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/../..')

import cv2
import torch
import argparse
import numpy as np
import pickle as pkl
from glob import glob
from tqdm import tqdm
import time
import math

from lib.camera import run_metric_slam, align_cam_to_world
from lib.pipeline.tools import arrange_boxes
from lib.utils.utils_detectron2 import DefaultPredictor_Lazy

from torch.amp import autocast
from segment_anything import SamPredictor, sam_model_registry
from detectron2.config import LazyConfig

from write_tum import to_tum_lines, convert_npz_to_tum
from render_scene_and_cam import render_cameras_gt_traj
import pdb

def format_fps(value):
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        if math.isinf(value):
            return "inf"
        return f"{value:.2f}"
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isinf(numeric_value):
        return "inf"
    return f"{numeric_value:.2f}"

def average(values):
    return sum(values) / len(values) if values else 0.0

def main(args):
    roots = []
    for p in range(10):
        folder = os.path.join(args.data_dir, f"P{p}")
        root = sorted(glob(f'{folder}/*'))
        roots.extend(root)

    emdb = []
    spl = args.split
    for root in roots:
        annfile = f'{root}/{root.split("/")[-2]}_{root.split("/")[-1]}_data.pkl'
        ann = pkl.load(open(annfile, 'rb'))
        if ann[f'emdb{spl}']:
            emdb.append(root)

    # ViTDet
    device = 'cuda'
    cfg_path = 'data/pretrain/cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)

    # SAM
    sam = sam_model_registry["vit_h"](checkpoint="data/pretrain/sam_vit_h_4b8939.pth")
    _ = sam.to(device)
    predictor = SamPredictor(sam)

    suffix_parts = []
    if args.no_mask:
        suffix_parts.append("no_mask")
    if args.no_ba:
        suffix_parts.append("no_ba")
    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""

    profile_dir = os.path.join(args.result_dir, f"profiler{suffix}")
    camera_dir = os.path.join(args.result_dir, f"camera{suffix}")
    pose_dir = os.path.join(args.result_dir, f"poses{suffix}")

    for directory in (profile_dir, camera_dir, pose_dir):
        os.makedirs(directory, exist_ok=True)
    profile_records = []

    # Estimate camera motion on EMDB (subset: spl)
    for root in emdb:
        try:
            print(f'Running on {root}...')

            seq = root.split('/')[-1]
            img_folder = f'{root}/images'
            imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
            seq_start_time = time.perf_counter()
            seq_folder = os.path.join(args.result_dir, f"{seq}")

            if not args.no_mask:
                mask_compute_time = 0.0
                masks_ = []
                for t, imgpath in enumerate(tqdm(imgfiles)):
                    mask_step_start = time.perf_counter()
                    img_cv2 = cv2.imread(imgpath)

                    ### --- Detection ---
                    with torch.no_grad():
                        with autocast('cuda'):
                            det_out = detector(img_cv2)
                            det_instances = det_out['instances']
                            valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
                            boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
                            confs = det_instances.scores[valid_idx].cpu().numpy()

                            boxes = np.hstack([boxes, confs[:, None]])
                            boxes = arrange_boxes(boxes, mode='size', min_size=100)

                    ### --- SAM --- 
                    if len(boxes)>0:
                        with autocast('cuda'):
                            predictor.set_image(img_cv2, image_format='BGR')

                            # multiple boxes
                            bb = torch.tensor(boxes[:, :4]).to('cuda')
                            bb = predictor.transform.apply_boxes_torch(bb, img_cv2.shape[:2])  
                            masks, scores, _ = predictor.predict_torch(
                                point_coords=None,
                                point_labels=None,
                                boxes=bb,
                                multimask_output=False
                            )
                            scores = scores.cpu()
                            masks = masks.cpu().squeeze(1)
                            mask = masks.sum(dim=0)
                    else:
                        mask = torch.zeros(img_cv2.shape[:2], dtype=torch.float32)
                    mask_compute_time += time.perf_counter() - mask_step_start
                    masks_.append(mask.byte())
                masks = torch.stack(masks_)

            ### --- Camera Motion ---
            annfile = f'{root}/{root.split("/")[-2]}_{root.split("/")[-1]}_data.pkl'
            ann = pkl.load(open(annfile, 'rb'))
            intr = ann['camera']['intrinsics']

            cam_int = [intr[0,0], intr[1,1], intr[0,2], intr[1,2]]
            slam_start_time = time.perf_counter()
            if args.no_mask:
                cam_R, cam_T = run_metric_slam(img_folder, masks=None, calib=cam_int)
            else:
                cam_R, cam_T = run_metric_slam(img_folder, masks=masks, calib=cam_int)
            slam_elapsed = time.perf_counter() - slam_start_time
            wd_cam_R, wd_cam_T, spec_f = align_cam_to_world(imgfiles[0], cam_R, cam_T)

            camera = {'pred_cam_R': cam_R.numpy(), 'pred_cam_T': cam_T.numpy(), 
                    'world_cam_R': wd_cam_R.numpy(), 'world_cam_T': wd_cam_T.numpy(),
                    'img_focal': cam_int[0], 'img_center': cam_int[2:], 'spec_focal': spec_f}
            
            ### --- Save results ---
            camera_path = os.path.join(camera_dir, f"{seq}.npz")
            pose_path = os.path.join(pose_dir, f"{seq}_poses.txt")
                
            np.savez(camera_path, **camera)
            convert_npz_to_tum(camera_path, pose_path)
            seq_total_time = time.perf_counter() - seq_start_time

            frames = len(imgfiles)
            slam_fps = frames / slam_elapsed if slam_elapsed > 0 else float('inf')
            total_fps = frames / seq_total_time if seq_total_time > 0 else float('inf')

            if args.no_mask:
                slam_percent = (slam_elapsed / seq_total_time * 100) if seq_total_time > 0 else 0.0
                other_time = max(seq_total_time - slam_elapsed, 0.0)
                other_percent = (other_time / seq_total_time * 100) if seq_total_time > 0 else 0.0
                print(
                    f"[Profiler] {seq}: SLAM {slam_elapsed:.2f}s ({slam_percent:.1f}%), "
                    f"total {seq_total_time:.2f}s (100%), "
                    f"SLAM FPS {slam_fps:.2f}, total FPS {total_fps:.2f}"
                )
                record = {
                    'seq': seq,
                    'frames': frames,
                    'mask_time': None,
                    'mask_percent': None,
                    'mask_fps': None,
                    'slam_time': slam_elapsed,
                    'slam_percent': slam_percent,
                    'slam_fps': slam_fps,
                    'total_time': seq_total_time,
                    'total_fps': total_fps,
                    'other_time': other_time,
                    'other_percent': other_percent,
                    'other_fps': (frames / other_time) if other_time > 0 else None,
                }
            else:
                mask_fps = frames / mask_compute_time if mask_compute_time > 0 else float('inf')
                mask_percent = (mask_compute_time / seq_total_time * 100) if seq_total_time > 0 else 0.0
                slam_percent = (slam_elapsed / seq_total_time * 100) if seq_total_time > 0 else 0.0
                other_time = max(seq_total_time - (mask_compute_time + slam_elapsed), 0.0)
                other_percent = (other_time / seq_total_time * 100) if seq_total_time > 0 else 0.0
                other_fps = (frames / other_time) if other_time > 0 else None

                print(
                    f"[Profiler] {seq}: mask {mask_compute_time:.2f}s ({mask_percent:.1f}%), "
                    f"masked DROID-SLAM {slam_elapsed:.2f}s ({slam_percent:.1f}%), "
                    f"total {seq_total_time:.2f}s (100%), "
                    f"mask FPS {mask_fps:.2f}, SLAM FPS {slam_fps:.2f}, total FPS {total_fps:.2f}"
                )
                record = {
                    'seq': seq,
                    'frames': frames,
                    'mask_time': mask_compute_time,
                    'mask_percent': mask_percent,
                    'mask_fps': mask_fps,
                    'slam_time': slam_elapsed,
                    'slam_percent': slam_percent,
                    'slam_fps': slam_fps,
                    'total_time': seq_total_time,
                    'total_fps': total_fps,
                    'other_time': other_time,
                    'other_percent': other_percent,
                    'other_fps': other_fps,
                }

            profile_records.append(record)

            seq_log_path = os.path.join(profile_dir, f"{seq}.txt")
            with open(seq_log_path, 'w') as log_f:
                log_f.write(f"Sequence: {seq}\n")
                log_f.write(f"Frames: {frames}\n")
                log_f.write(f"Total time: {seq_total_time:.4f}s (100.0%), FPS: {format_fps(total_fps)}\n")
                if record['mask_time'] is not None:
                    log_f.write(
                        f"Mask time: {record['mask_time']:.4f}s ({record['mask_percent']:.2f}%), "
                        f"FPS: {format_fps(record['mask_fps'])}\n"
                    )
                log_f.write(
                    f"SLAM time: {record['slam_time']:.4f}s ({record['slam_percent']:.2f}%), "
                    f"FPS: {format_fps(record['slam_fps'])}\n"
                )
                if record['other_time'] > 0:
                    log_f.write(
                        f"Other time: {record['other_time']:.4f}s ({record['other_percent']:.2f}%)"
                    )
                    if record['other_fps'] is not None:
                        log_f.write(f", FPS: {format_fps(record['other_fps'])}")
                    log_f.write("\n")

            # official_cam_path = os.path.join(args.result_dir, "camera", f"{seq}.npz")
            # official_pose_path = os.path.join(args.result_dir, "poses", f"{seq}_poses.txt")
            # convert_npz_to_tum(official_cam_path, official_pose_path)

            # out_gif = os.path.join(args.result_dir, "vis", f"{seq}.gif")
            # render_cameras(official_pose_path, pose_path, out_gif=out_gif)
        except:
            print(f"{seq} failed")
    summary_path = os.path.join(profile_dir, "summary.txt")
    if profile_records:
        total_frames = sum(r['frames'] for r in profile_records)
        avg_total_time = sum(r['total_time'] for r in profile_records) / len(profile_records)
        avg_total_fps = sum(r['total_fps'] for r in profile_records) / len(profile_records)

        slam_times = [r['slam_time'] for r in profile_records]
        slam_percents = [r['slam_percent'] for r in profile_records]
        slam_fps_values = [r['slam_fps'] for r in profile_records]

        mask_times = [r['mask_time'] for r in profile_records if r['mask_time'] is not None]
        mask_percents = [r['mask_percent'] for r in profile_records if r['mask_percent'] is not None]
        mask_fps_values = [r['mask_fps'] for r in profile_records if r['mask_fps'] is not None]

        other_times = [r['other_time'] for r in profile_records if r['other_time'] > 0]
        other_percents = [r['other_percent'] for r in profile_records if r['other_percent'] > 0]
        other_fps_values = [r['other_fps'] for r in profile_records if r['other_fps'] is not None]

        with open(summary_path, 'w') as summary_f:
            summary_f.write(f"Sequences processed: {len(profile_records)}\n")
            summary_f.write(f"Total frames: {total_frames}\n")
            summary_f.write(f"Avg total time: {avg_total_time:.4f}s, Avg total FPS: {avg_total_fps:.2f}\n")
            summary_f.write(
                f"Avg SLAM time: {average(slam_times):.4f}s, "
                f"Avg SLAM percent: {average(slam_percents):.2f}%, "
                f"Avg SLAM FPS: {average(slam_fps_values):.2f}\n"
            )
            if mask_times:
                summary_f.write(
                    f"Avg mask time: {average(mask_times):.4f}s, "
                    f"Avg mask percent: {average(mask_percents):.2f}%, "
                    f"Avg mask FPS: {average(mask_fps_values):.2f}\n"
                )
            if other_times:
                summary_f.write(
                    f"Avg other time: {average(other_times):.4f}s, "
                    f"Avg other percent: {average(other_percents):.2f}%, "
                    f"Avg other FPS: {average(other_fps_values):.2f}\n"
                )
    else:
        with open(summary_path, 'w') as summary_f:
            summary_f.write("No sequences processed.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=int, default=2)
    parser.add_argument('--no_mask', action='store_true', default=False)
    parser.add_argument('--no_ba', action='store_true', default=False)
    parser.add_argument('--data_dir', type=str, default='emdb_data/')
    parser.add_argument('--result_dir', type=str, default='results/emdb/')
    parser.add_argument("--visualize_mask", action='store_true', help='save deva vos for visualization')
    args = parser.parse_args()

    # seqs = []
    # roots = []
    # for p in range(3,4):
    #     folder = os.path.join(args.data_dir, f"P{p}")
    #     root = sorted(glob(f'{folder}/*'))
    #     roots.extend(root)
    # for root in roots:
    #     seq = root.split('/')[-1]
    #     seqs.append(seq)
    # seqs = seqs[2:3]

    # for seq in seqs:
    #     pose_path = os.path.join(args.result_dir, "poses_no_mask", f"{seq}_poses.txt")
    #     official_pose_path = os.path.join(args.result_dir, "poses", f"{seq}_poses.txt")
    #     out_gif = os.path.join(args.result_dir, "vis", f"{seq}.gif")

    #     out_mp4 = os.path.join(args.result_dir, "vis", f"{seq}.mp4")
    #     orig_video_path = os.path.join(args.data_dir, "P3", seq, f"P3_{seq}_video.mp4")
    #     # render_cameras_traj(official_pose_path, pose_path, out_gif=out_gif)
    #     render_cameras_traj(official_pose_path, pose_path, orig_video=orig_video_path, out_mp4=out_mp4)
    #     print(f"✅ Saved mp4 to {out_mp4}")

    main(args)
