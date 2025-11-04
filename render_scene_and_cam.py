import os
import open3d as o3d
from glob import glob
import numpy as np
import imageio
import argparse
import pickle as pkl
from tqdm import tqdm
import cv2
import time
from lib.utils.rotation_conversions import *

import evo
from evo.core import trajectory
from evo.core.trajectory import PoseTrajectory3D
from evo.core import sync
import evo.main_ape as main_ape
from evo.core.metrics import PoseRelation
from torchvision.transforms import Resize

import pdb

def make_y0_plane(xmin=-500, xmax=500, zmin=-500, zmax=500, y=0.0):
    verts = np.array([
        [xmin, y, zmin],
        [xmax, y, zmin],
        [xmax, y, zmax],
        [xmin, y, zmax],
    ], dtype=np.float64)
    tris = np.array([[0,1,2], [0,2,3]], dtype=np.int32)

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(tris),
    )
    mesh.compute_vertex_normals()
    return mesh

def read_camera_poses(txt_file):
    """
    Read camera poses from text file.
    Each line: <timestep> tx ty tz qx qy qz qw

    Returns:
        cam_t: (N,3) translations
        cam_r: (N,3,3) rotation matrices
        cam_q: (N,4) quaternions in (qx,qy,qz,qw) order
    """
    cam_r, cam_t, cam_q = [], [], []
    with open(txt_file, "r") as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            _, tx, ty, tz, qx, qy, qz, qw = vals

            # quaternion → rotation matrix (expects w,x,y,z)
            R = o3d.geometry.get_rotation_matrix_from_quaternion([qw, qx, qy, qz])
            t = np.array([tx, ty, tz], dtype=float)
            q = np.array([qx, qy, qz, qw], dtype=float)

            cam_r.append(R)
            cam_t.append(t)
            cam_q.append(q)

    return np.asarray(cam_t, dtype=float), np.asarray(cam_r, dtype=float), np.asarray(cam_q, dtype=float)

def make_camera_frustums(cam_r, cam_t, scale=0.2):
    """
    Build LineSet pyramids (camera frustums) from cam_r and cam_t.
    Returns:
        poses: list of LineSet pyramids.
    """
    poses = []
    for R, t in zip(cam_r, cam_t):
        # pyramid points (camera frustum)
        cam_points = np.array([
            [0, 0, 0],
            [scale, scale, scale * 1.5],
            [scale, -scale, scale * 1.5],
            [-scale, -scale, scale * 1.5],
            [-scale, scale, scale * 1.5],
        ])
        cam_lines = [
            [0, 1], [0, 2], [0, 3], [0, 4],
            [1, 2], [2, 3], [3, 4], [4, 1]
        ]

        cam = o3d.geometry.LineSet()
        cam.points = o3d.utility.Vector3dVector(cam_points @ R.T + t)
        cam.lines = o3d.utility.Vector2iVector(cam_lines)
        cam.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(cam_lines))  # red lines
        poses.append(cam)

    return poses

def align_to_reference_at_t0(cam_R, cam_t, ref_R, ref_t):
    """
    R, t here are the SAME convention you feed to make_camera_frustums:
      world point of frustum = cam_points @ R.T + t

    Align the whole (cam_R, cam_t) sequence so that frame 0 coincides with the
    reference's frame 0 (ref_R[0], ref_t[0]) up to a single global SE(3) transform.
    Scale is untouched.

    Args:
        cam_R: (T,3,3) sequence to be aligned
        cam_t: (T,3)   sequence to be aligned
        ref_R: (T,3,3) reference sequence (only index 0 is used)
        ref_t: (T,3)   reference sequence (only index 0 is used)

    Returns:
        cam_R_aligned: (T,3,3)
        cam_t_aligned: (T,3)
    """
    R0, t0 = cam_R[0], cam_t[0]
    Rr0, tr0 = ref_R[0], ref_t[0]

    # Find world transform G = (R_g, t_g) such that:
    #   R' = R * R_g^T,      t' = R_g * t + t_g,
    # and at t=0 we have R'(0)=Rr0, t'(0)=tr0.
    R_g = Rr0.T @ R0
    t_g = tr0 - R_g @ t0

    # Apply to the whole sequence
    cam_R_aligned = cam_R @ (R0.T @ Rr0)   # since R_g^T = R0^T @ Rr0
    cam_t_aligned = (cam_t @ R_g.T) + t_g  # (R_g @ t) + t_g

    return cam_R_aligned, cam_t_aligned

def align_traj_evo(cam_t_ref, cam_q_ref, cam_t_est, cam_q_est, correct_scale=True):
    """
    Align an estimated trajectory to a reference using evo's Umeyama alignment.
    Args:
        cam_t_ref: (N,3) reference translations
        cam_q_ref: (N,4) reference quaternions (qx,qy,qz,qw)
        cam_t_est: (N,3) estimated translations
        cam_q_est: (N,4) estimated quaternions (qx,qy,qz,qw)
        correct_scale: whether to also scale the estimated trajectory
    Returns:
        cam_t_aligned: (N,3) aligned translations
        cam_R_aligned: (N,3,3) aligned rotation matrices
    """
    traj_ref = trajectory.PosePath3D(
        positions_xyz=cam_t_ref,
        orientations_quat_wxyz=cam_q_ref[:, [3,0,1,2]],  # evo expects w,x,y,z
    )
    traj_est = trajectory.PosePath3D(
        positions_xyz=cam_t_est,
        orientations_quat_wxyz=cam_q_est[:, [3,0,1,2]],
    )

    traj_est.align(traj_ref, correct_scale=correct_scale)

    cam_t_aligned = np.array([p[:3, 3] for p in traj_est.poses_se3])
    cam_R_aligned = np.array([p[:3, :3] for p in traj_est.poses_se3])
    return cam_t_aligned, cam_R_aligned

def render_cameras_gt_traj(pose_masked, pose_no_mask=None, gt_anno=None, stride=5, orig_video=None, out_mp4="scene.mp4", viser=False):
    # Load cameras
    if gt_anno is not None:
        ann = pkl.load(open(gt_anno, 'rb'))
        ext = ann['camera']['extrinsics']
        cam_R_gt = ext[:,:3,:3].transpose(0,2,1)
        cam_t_gt = np.einsum('bij, bj->bi', cam_R_gt, -ext[:, :3, -1])
        
        S = np.diag([-1., 1., 1.])
        cam_R_gt = np.einsum('ij,bjk->bik', S, cam_R_gt)                  # R' = S @ R_{w->c}
        cam_t_gt = np.einsum('bi,ij->bj', cam_t_gt, S)                    # t' = t_{w->c} @ S
        cam_q_gt = matrix_to_quaternion(torch.from_numpy(cam_R_gt)).numpy()

        # set gt starting point to origin
        cam_t_gt = cam_t_gt - cam_t_gt[0,:]

    cam_t_masked, cam_R_masked, cam_q_masked = read_camera_poses(pose_masked)
    if gt_anno is not None:
        cam_t_masked, cam_R_masked = align_traj_evo(cam_t_gt, cam_q_gt, cam_t_masked, cam_q_masked)

    if pose_no_mask is not None:
        cam_t_no_mask, cam_R_no_mask, cam_q_no_mask = read_camera_poses(pose_no_mask)
        # cam_R_no_mask, cam_t_no_mask = align_to_reference_at_t0(cam_R_no_mask, cam_t_no_mask, cam_R_masked, cam_t_masked)
        if gt_anno is not None:
            if len(cam_t_gt) < len(cam_t_no_mask):
                cam_t_no_mask = cam_t_no_mask[:len(cam_t_gt)]
                cam_R_no_mask = cam_R_no_mask[:len(cam_t_gt)]
                cam_q_no_mask = cam_q_no_mask[:len(cam_t_gt)]
            elif len(cam_t_gt) > len(cam_t_no_mask):
                pad_len = len(cam_t_gt) - len(cam_t_no_mask)
                cam_t_no_mask = np.pad(cam_t_no_mask, ((0, pad_len), (0, 0)), mode='edge')
                cam_R_no_mask = np.pad(cam_R_no_mask, ((0, pad_len), (0, 0), (0, 0)), mode='edge')
                cam_q_no_mask = np.pad(cam_q_no_mask, ((0, pad_len), (0, 0)), mode='edge')
            cam_t_no_mask, cam_R_no_mask = align_traj_evo(cam_t_gt, cam_q_gt, cam_t_no_mask, cam_q_no_mask)
    if viser:
        import viser as viser_mod

        def _make_line_segments(points_xyz):
            if points_xyz.shape[0] < 2:
                return np.empty((0, 2, 3), dtype=np.float32)
            return np.stack([points_xyz[:-1], points_xyz[1:]], axis=1).astype(np.float32)

        def _make_frustum_segments(cam_r, cam_t):
            cam_points = np.array(
                [
                    [0, 0, 0],
                    [0.2, 0.2, 0.3],
                    [0.2, -0.2, 0.3],
                    [-0.2, -0.2, 0.3],
                    [-0.2, 0.2, 0.3],
                ],
                dtype=np.float32,
            )
            cam_lines = np.array(
                [
                    [0, 1],
                    [0, 2],
                    [0, 3],
                    [0, 4],
                    [1, 2],
                    [2, 3],
                    [3, 4],
                    [4, 1],
                ],
                dtype=np.int32,
            )
            segments = []
            for R, t in zip(cam_r, cam_t):
                frustum_points = cam_points @ R.T + t
                segments.append(frustum_points[cam_lines])
            if not segments:
                return np.empty((0, 2, 3), dtype=np.float32)
            return np.concatenate(segments, axis=0)

        server = viser_mod.ViserServer()
        server.scene.set_up_direction("+y")

        stride = max(stride, 1)
        pose_fps = 30.0 / stride
        step_dt = 1.0 / pose_fps if pose_fps > 0 else 0.0

        traj_sequences = [
            ("masked", cam_t_masked, cam_R_masked, (255, 0, 0)),
        ]
        if pose_no_mask is not None:
            traj_sequences.append(("no_mask", cam_t_no_mask, cam_R_no_mask, (0, 255, 0)))
        if gt_anno is not None:
            traj_sequences.append(("gt", cam_t_gt, cam_R_gt, (64, 64, 64)))

        max_steps = max((positions.shape[0] for _, positions, _, _ in traj_sequences if positions is not None), default=0)

        def _add_segment(label, step_idx, positions, color):
            if step_idx == 0 or step_idx >= positions.shape[0]:
                return
            segment = np.array(
                [[positions[step_idx - 1], positions[step_idx]]],
                dtype=np.float32,
            )
            server.scene.add_line_segments(
                name=f"/trajectory/{label}/segments/{step_idx:05d}",
                points=segment,
                colors=color,
                line_width=2.0,
            )

        def _add_frustum(label, step_idx, rotations, positions, color):
            if step_idx >= positions.shape[0] or step_idx >= rotations.shape[0]:
                return
            frustum_segments = _make_frustum_segments(
                rotations[step_idx : step_idx + 1], positions[step_idx : step_idx + 1]
            )
            if frustum_segments.size:
                server.scene.add_line_segments(
                    name=f"/trajectory/{label}/frustums/{step_idx:05d}",
                    points=frustum_segments,
                    colors=color,
                    line_width=1.0,
                )

        # Incrementally stream poses into Viser to mimic the offline rendering loop.
        for step_idx in range(max_steps):
            for label, positions, rotations, color in traj_sequences:
                if positions is None or step_idx >= positions.shape[0]:
                    continue
                if step_idx == 0:
                    _add_frustum(label, step_idx, rotations, positions, color)
                    continue
                _add_segment(label, step_idx, positions, color)
                if step_idx % stride == 0:
                    _add_frustum(label, step_idx, rotations, positions, color)
            if step_dt > 0:
                time.sleep(step_dt)

        print("Viser viewer running. Press Enter in this terminal when you want to continue.")
        try:
            input("→ Press Enter to close the viewer and move to the next sequence...")
        except KeyboardInterrupt:
            print("Keyboard interrupt received, closing viewer.")

        for attr in ("stop", "close", "shutdown"):
            if hasattr(server, attr):
                try:
                    getattr(server, attr)()
                except Exception:
                    pass
                break
        return

    cams_masked = make_camera_frustums(cam_R_masked[::stride], cam_t_masked[::stride])
    if pose_no_mask is not None:
        cams_no_mask = make_camera_frustums(cam_R_no_mask[::stride], cam_t_no_mask[::stride])
    if gt_anno is not None:
        cams_gt = make_camera_frustums(cam_R_gt[::stride], cam_t_gt[::stride])

    num_traj = 1 + (pose_no_mask is not None) + (gt_anno is not None)
    cam_centers = [np.empty((0,3), float),
               np.empty((0,3), float),
               np.empty((0,3), float)]

    # Offscreen renderer
    w, h = 540, 720
    render = o3d.visualization.rendering.OffscreenRenderer(w, h)
    render.scene.set_background([1, 1, 1, 1])

    orig_fps = 30.0
    pose_fps = orig_fps / stride

    if orig_video is not None:
        cap = cv2.VideoCapture(orig_video)
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_orig = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1e9)
        n_steps = min(len(cams_masked), n_orig // stride)

        writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*'mp4v'), pose_fps, (2*w, h))  # side-by-side
    else:
        n_steps = len(cams_masked)
        writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*'mp4v'), pose_fps, (w, h))

    plane = make_y0_plane()
    # plane = make_y0_plane(y=cam_t_gt[0,1])
    mat_plane = o3d.visualization.rendering.MaterialRecord()
    mat_plane.shader = "defaultLitTransparency" 
    mat_plane.base_color = (0.85, 0.85, 0.85, 0.4)  # RGBA
    render.scene.add_geometry("z0_plane", plane, mat_plane)

    # Camera line material
    cam_mat = o3d.visualization.rendering.MaterialRecord()
    cam_mat.shader = "unlitLine"

    k = 4.0
    # r_xy  = max(k * np.linalg.norm(std[[0, 2]]), 1e-3)
    up = np.array([0, -1, 0])
    spin_deg_per_frame = 0

    imgs = []
    for i in range(n_steps):  # or range(0, len(cams_masked), stride)
        if orig_video is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * stride)
            ret, orig_bgr = cap.read()
            orig_bgr = orig_bgr[:, w:, :]
            if not ret: break

        render.scene.add_geometry(f"cam_masked_{i}", cams_masked[i], cam_mat)
        # cam_centers[0] = np.concatenate((cam_centers[0], np.asarray(cams_masked[i].points)[None, 0]), axis=0)
        cam_centers[0] = np.vstack([cam_centers[0], np.asarray(cams_masked[i].points)[0]])

        if pose_no_mask is not None and i < len(cams_no_mask):
            cams_no_mask[i].colors = o3d.utility.Vector3dVector([[0, 1, 0]] * len(cams_no_mask[i].lines))
            render.scene.add_geometry(f"cam_no_mask_{i}", cams_no_mask[i], cam_mat)
            # cam_centers[1] = np.concatenate((cam_centers[1], np.asarray(cams_no_mask[i].points)[None, 0]), axis=0)
            cam_centers[1] = np.vstack([cam_centers[1], np.asarray(cams_no_mask[i].points)[0]])
        if gt_anno is not None and i < len(cams_gt):
            cams_gt[i].colors = o3d.utility.Vector3dVector([[0.2, 0.2, 0.2]] * len(cams_gt[i].lines))
            render.scene.add_geometry(f"cam_gt_{i}", cams_gt[i], cam_mat)
            # cam_centers[2] = np.concatenate((cam_centers[2], np.asarray(cams_gt[i].points)[None, 0]), axis=0)
            cam_centers[2] = np.vstack([cam_centers[2], np.asarray(cams_gt[i].points)[0]])

        theta = i * np.deg2rad(spin_deg_per_frame)
        window = 80

        win = np.vstack([c[-window:] for c in cam_centers])
        center = win.mean(axis=0)
        std    = win.std(axis=0)

        # mesh = o3d.geometry.create_mesh_coordinate_frame()
        if render.scene.has_geometry("axes"):
            render.scene.remove_geometry("axes")        
        mesh = o3d.geometry.TriangleMesh().create_coordinate_frame(size=std[::2].mean())
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        render.scene.add_geometry("axes", mesh, mat)    

        r_xy  = max(k * np.linalg.norm(std[[0, 2]]), 5.0)

        eye = center + np.array([r_xy * np.sin(theta), 0, r_xy * np.cos(theta)])
        render.setup_camera(60, center, eye, up)
        
        img_o3d = render.render_to_image()
        img_np = np.flipud(np.asarray(img_o3d))
        imgs.append(img_np)

        if orig_video is not None:
            viz_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            combined = np.hstack([viz_bgr, orig_bgr])  # (h, 2*w, 3)
            writer.write(combined)
        else:
            viz_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            writer.write(viz_bgr)

    if orig_video is not None:
        cap.release()
    writer.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a scene with camera trajectories and save as a GIF.")
    # parser.add_argument("--cam_traj_path", type=str, default=None, help="Path to the camera trajectory file (txt).")
    # parser.add_argument("--cam_traj_path_2", type=str, default=None, help="Path to the second camera trajectory file (txt).")
    # parser.add_argument("--gt_anno", type=str, default=None, help="Path to the ground truth annotation file (optional).")
    # parser.add_argument("--output_mp4", type=str, default="output.mp4", help="Output MP4 file name (default: output.mp4).")
    # parser.add_argument("--stride", type=int, default=5, help="Stride for sampling camera poses (default: 5).")
    # parser.add_argument("--orig_video", type=str, default=None, help="Path to the original video file (optional).")

    # args = parser.parse_args()
    # render_cameras_gt_traj(args.cam_traj_path, args.cam_traj_path_2, gt_anno=args.gt_anno, stride=args.stride, orig_video=args.orig_video, out_mp4=args.output_mp4)

    parser.add_argument('--pred_dir', type=str, default='results/emdb')
    parser.add_argument('--gt_dir', type=str, default='emdb_data')
    parser.add_argument("--stride", type=int, default=5, help="Stride for sampling camera poses (default: 5).")
    parser.add_argument('--split', type=int, default=2)
    parser.add_argument('--viser', action='store_true', help='Use viser instead of Open3D for visualization.')
    args = parser.parse_args()

    pred_dir = args.pred_dir
    gt_dir = args.gt_dir
    roots = []
    for p in range(10):
        folder = os.path.join(gt_dir, f"P{p}")
        root = sorted(glob(f'{folder}/*'))
        roots.extend(root)

    emdb = []
    spl = args.split
    for root in roots:
        annfile = f'{root}/{root.split("/")[-2]}_{root.split("/")[-1]}_data.pkl'
        ann = pkl.load(open(annfile, 'rb'))
        if ann[f'emdb{spl}']:
            emdb.append(root)
    
    for root in emdb:
        p = root.split('/')[-2]
        seq = root.split('/')[-1]
        pose_masked = os.path.join(pred_dir, "poses", f"{seq}_poses.txt")
        pose_no_mask = os.path.join(pred_dir, "poses_no_mask", f"{seq}_poses.txt")
        # pose_no_mask_no_ba = os.path.join(pred_dir, "poses_no_mask_no_ba", f"{seq}_poses.txt")
        # pose_no_ba = os.path.join(pred_dir, "poses_no_ba", f"{seq}_poses.txt")
        # pose_mast3rSLAM_inc = os.path.join(pred_dir, "poses_mast3rSLAM_inc", f"{p}_{seq[:2]}_incremental_all.txt")
        # pose_mast3rSLAM_inc_kf = os.path.join(pred_dir, "poses_mast3rSLAM_inc_kf", f"{p}_{seq[:2]}_incremental_kf.txt")
        # pose_mast3rSLAM_global_kf = os.path.join(pred_dir, "poses_mast3rSLAM_global_kf", f"{p}_{seq[:2]}_globalOptimized_kf.txt")

        gt_anno = os.path.join(root, f"{p}_{seq}_data.pkl")
        orig_video = os.path.join(root, f"{p}_{seq}_video.mp4")
        output_mp4 = os.path.join(pred_dir, "vis_no_mask", f"{seq}.mp4")
        # output_mp4 = os.path.join(pred_dir, "vis_no_mask_no_ba", f"{seq}.mp4")
        # output_mp4 = os.path.join(pred_dir, "vis_no_mask_to_no_mask_no_ba", f"{seq}.mp4")
        # output_mp4 = os.path.join(pred_dir, "vis_no_ba", f"{seq}.mp4")
        # output_mp4 = os.path.join(pred_dir, "vis_mast3rSLAM_inc", f"{seq}.mp4")
        # output_mp4 = os.path.join(pred_dir, "vis_mast3rSLAM_global_kf", f"{seq}.mp4")
        # output_mp4 = os.path.join(pred_dir, "vis_inc_z0", f"{seq}.mp4")
        
        if not os.path.exists(pose_masked):
            print(f"Masked pose file not found: {pose_masked}")
            continue
        if not os.path.exists(pose_no_mask):
            print(f"No-mask pose file not found: {pose_no_mask}")
            continue
        # if not os.path.exists(pose_no_mask_no_ba):
        #     print(f"No-mask pose file not found: {pose_no_mask_no_ba}")
        #     continue
        # if not os.path.exists(pose_no_ba):
        #     print(f"No-mask pose file not found: {pose_no_ba}")
        #     continue
        # if not os.path.exists(pose_mast3rSLAM_inc):
        #     print(f"No-mask pose file not found: {pose_mast3rSLAM_inc}")
        #     continue
        if not os.path.exists(gt_anno):
            print(f"Ground truth annotation file not found: {gt_anno}")
            continue
        if not os.path.exists(orig_video):
            print(f"Original video file not found: {orig_video}")
            continue

        render_cameras_gt_traj(pose_masked, pose_no_mask, gt_anno=gt_anno, orig_video=orig_video, out_mp4=output_mp4, viser=args.viser)
        print(f"Rendered sequence: {output_mp4}")
