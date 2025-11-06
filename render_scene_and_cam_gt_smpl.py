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
from typing import Any, Optional
import torch
from lib.utils.rotation_conversions import *
from lib.models.smpl import SMPL


_SMPL_MODEL_CACHE: Optional[SMPL] = None
_SMPL_FACES_CACHE: Optional[np.ndarray] = None
_FLIP_Y_ARRAY = np.array([1.0, -1.0, 1.0], dtype=np.float32)
_FLIP_Y_MATRIX = np.diag(_FLIP_Y_ARRAY)


def _compute_similarity_from_points(
    src: np.ndarray,
    dst: np.ndarray,
    allow_scale: bool = True,
) -> tuple[float, np.ndarray, np.ndarray]:
    if src.shape[0] == 0 or dst.shape[0] == 0:
        return 1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    n = min(src.shape[0], dst.shape[0])
    src = src[:n]
    dst = dst[:n]

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    src_centered = src - mu_src
    dst_centered = dst - mu_dst

    cov = dst_centered.T @ src_centered / float(n)
    U, S, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt

    if allow_scale:
        var_src = np.sum(src_centered ** 2) / float(n)
        if var_src > 1e-8:
            scale = float(np.sum(S) / var_src)
        else:
            scale = 1.0
    else:
        scale = 1.0

    translation = mu_dst - scale * (R @ mu_src)
    return scale, R.astype(np.float32), translation.astype(np.float32)


def _get_smpl_model() -> SMPL:
    global _SMPL_MODEL_CACHE, _SMPL_FACES_CACHE
    if _SMPL_MODEL_CACHE is None:
        model = SMPL()
        model.eval()
        _SMPL_MODEL_CACHE = model
        faces = getattr(model, "faces", None)
        if faces is not None:
            _SMPL_FACES_CACHE = np.asarray(faces, dtype=np.int32)
    elif _SMPL_FACES_CACHE is None:
        faces = getattr(_SMPL_MODEL_CACHE, "faces", None)
        if faces is not None:
            _SMPL_FACES_CACHE = np.asarray(faces, dtype=np.int32)
    return _SMPL_MODEL_CACHE


def _get_smpl_faces() -> Optional[np.ndarray]:
    if _SMPL_FACES_CACHE is None:
        _get_smpl_model()
    return _SMPL_FACES_CACHE

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

def make_camera_frustums(cam_r, cam_t, scale=0.2, color=(1.0, 0.0, 0.0)):
    """
    Build LineSet pyramids (camera frustums) from cam_r and cam_t.
    Returns:
        poses: list of LineSet pyramids.
    """
    color_arr = np.asarray(color, dtype=np.float64)
    if color_arr.shape != (3,):
        raise ValueError(f"color must be length-3, got shape {color_arr.shape}")
    color_arr = np.clip(color_arr, 0.0, 1.0)

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
        cam.colors = o3d.utility.Vector3dVector(np.broadcast_to(color_arr, (len(cam_lines), 3)))
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

    cam_R_est = quaternion_to_matrix(torch.from_numpy(cam_q_est).float()).numpy()
    scale, rotation, translation = _derive_alignment_from_cameras(
        cam_t_est,
        cam_R_est,
        cam_t_aligned,
        cam_R_aligned,
    )

    return cam_t_aligned, cam_R_aligned, (scale, rotation, translation)


def _load_emdb_camera_transforms(camera_txt_path: str) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not camera_txt_path or not os.path.exists(camera_txt_path):
        print(f"Camera trajectory file not found: {camera_txt_path}")
        return None, None

    scales: list[float] = []
    translations: list[list[float]] = []
    quaternions: list[list[float]] = []
    with open(camera_txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            vals = [float(v) for v in parts[:8]]
            scales.append(vals[0])
            translations.append(vals[1:4])
            # Store quaternion as w, x, y, z for quaternion_to_matrix
            quaternions.append([vals[7], vals[4], vals[5], vals[6]])

    if not translations:
        print(f"No valid camera entries found in {camera_txt_path}")
        return None, None

    scale_arr = np.asarray(scales, dtype=np.float32)
    if scale_arr.size == 0:
        naive_scale = 1.0
    else:
        window = min(scale_arr.size, 2)
        naive_scale = float(scale_arr[:window].mean())

    cam_t = torch.tensor(translations, dtype=torch.float32) * naive_scale
    cam_q = torch.tensor(quaternions, dtype=torch.float32)
    cam_R = quaternion_to_matrix(cam_q)

    return cam_t, cam_R


def _derive_alignment_from_cameras(
    cam_t_orig: np.ndarray,
    cam_R_orig: np.ndarray,
    cam_t_aligned: np.ndarray,
    cam_R_aligned: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    if cam_t_orig.shape[0] == 0 or cam_t_aligned.shape[0] == 0:
        return 1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    R_align = cam_R_aligned[0] @ cam_R_orig[0].T
    if np.linalg.det(R_align) < 0:
        R_align = -R_align

    disp_orig = cam_t_orig[1:] - cam_t_orig[:-1]
    disp_aligned = cam_t_aligned[1:] - cam_t_aligned[:-1]
    denom = np.linalg.norm(disp_orig, axis=1)
    numer = np.linalg.norm(disp_aligned, axis=1)
    valid = (denom > 1e-6) & (numer > 1e-6)
    if valid.any():
        scale = float(np.median(numer[valid] / denom[valid]))
    else:
        scale = 1.0

    translation = cam_t_aligned[0] - scale * (R_align @ cam_t_orig[0])
    return scale, R_align.astype(np.float32), translation.astype(np.float32)


def load_emdb_human_world_vertices(
    human_npz_path: str,
    camera_txt_path: str,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not human_npz_path or not os.path.exists(human_npz_path):
        print(f"Human prediction file not found or invalid: {human_npz_path}")
        return None, None

    cam_t, cam_R = _load_emdb_camera_transforms(camera_txt_path)
    if cam_t is None or cam_R is None:
        return None, None

    faces = _get_smpl_faces()
    smpl_model = _get_smpl_model()

    try:
        with np.load(human_npz_path) as data:
            pred_rotmat = torch.from_numpy(data["pred_rotmat"]).float()
            pred_shape = torch.from_numpy(data["pred_shape"]).float()
            pred_trans = torch.from_numpy(data["pred_trans"]).float()
    except Exception as exc:
        print(f"Failed to load human predictions from {human_npz_path}: {exc}")
        return None, None

    if pred_trans.ndim == 3:
        pred_trans = pred_trans.squeeze(1)
    if pred_trans.ndim == 1:
        pred_trans = pred_trans.unsqueeze(0)

    mean_shape = pred_shape.mean(dim=0, keepdim=True)
    pred_shape = mean_shape.expand(pred_rotmat.shape[0], -1).contiguous()

    num_frames = min(pred_rotmat.shape[0], pred_shape.shape[0], pred_trans.shape[0], cam_t.shape[0], cam_R.shape[0])
    if num_frames <= 0:
        print(f"No overlapping frames between human and camera data for {human_npz_path}")
        return None, faces

    pred_rotmat = pred_rotmat[:num_frames]
    pred_shape = pred_shape[:num_frames]
    pred_trans = pred_trans[:num_frames]
    cam_t = cam_t[:num_frames]
    cam_R = cam_R[:num_frames]

    with torch.no_grad():
        smpl_out = smpl_model(
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, [0]],
            betas=pred_shape,
            transl=pred_trans,
            pose2rot=False,
            default_smpl=True,
        )
        vertices_local = smpl_out.vertices  # (T, V, 3)

    vertices_world = torch.einsum("bij,bnj->bni", cam_R, vertices_local) + cam_t.unsqueeze(1)
    vertices_world_np = vertices_world.cpu().numpy().astype(np.float32, copy=False)

    # if alignment is not None:
    #     scale, rot, trans = alignment
    #     rot = np.asarray(rot, dtype=np.float32)
    #     trans = np.asarray(trans, dtype=np.float32)
    #     vertices_world_np = vertices_world_np @ rot.T
    #     if abs(scale - 1.0) > 1e-6:
    #         vertices_world_np *= scale
    #     vertices_world_np += trans
    return vertices_world_np, faces


def _load_gt_human_vertices_from_ann(ann: dict) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    smpl_data = ann.get("smpl")

    if not smpl_data:
        return None, None

    body_pose = smpl_data.get("poses_body")
    global_orient = smpl_data.get("poses_root")
    transl = smpl_data.get("trans")
    betas = smpl_data.get("betas")

    if body_pose is None or global_orient is None or transl is None:
        return None, None

    body_pose_t = torch.from_numpy(np.asarray(body_pose)).float()
    global_orient_t = torch.from_numpy(np.asarray(global_orient)).float()
    transl_t = torch.from_numpy(np.asarray(transl)).float()

    if body_pose_t.ndim == 3 and body_pose_t.shape[-1] == 3:
        body_pose_t = body_pose_t.reshape(body_pose_t.shape[0], -1)
    elif body_pose_t.ndim == 2 and body_pose_t.shape[1] == 69:
        pass
    else:
        body_pose_t = body_pose_t.reshape(body_pose_t.shape[0], -1)

    if global_orient_t.ndim == 3 and global_orient_t.shape[-1] == 3:
        global_orient_t = global_orient_t.reshape(global_orient_t.shape[0], -1)

    if betas is None:
        betas_t = torch.zeros((1, 10), dtype=torch.float32)
    else:
        betas_t = torch.from_numpy(np.asarray(betas)).float()
        if betas_t.ndim == 1:
            betas_t = betas_t.unsqueeze(0)
    if betas_t.shape[0] == 1:
        betas_t = betas_t.expand(body_pose_t.shape[0], -1).contiguous()

    transl_t = transl_t.reshape(body_pose_t.shape[0], 3)

    faces = _get_smpl_faces()
    smpl_model = _get_smpl_model()

    with torch.no_grad():
        smpl_out = smpl_model(
            body_pose=body_pose_t,
            global_orient=global_orient_t,
            betas=betas_t,
            transl=transl_t,
            pose2rot=True,
            default_smpl=True,
        )
    vertices = smpl_out.vertices.cpu().numpy().astype(np.float32, copy=False)
    return vertices, faces

def render_cameras_gt_traj(
    pose_masked,
    pose_no_mask=None,
    gt_anno=None,
    stride=5,
    orig_video=None,
    out_mp4="scene.mp4",
    viser=False,
    human_npz_path=None,
    human_camera_path=None,
    align_to_gt=False,
):
    # Load cameras
    traj_sequences: list[tuple[str, np.ndarray, np.ndarray, tuple[int, int, int]]] = []
    alignment_transform = (1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
    human_alignment_transform = (1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
    human_seq_len = 0
    human_cam_t_for_mesh: Optional[np.ndarray] = None
    human_cam_R_for_mesh: Optional[np.ndarray] = None
    gt_human_vertices_full: Optional[np.ndarray] = None
    gt_human_faces: Optional[np.ndarray] = None
    gt_human_seq_len = 0

    cam_t_masked_orig = cam_R_masked_orig = cam_q_masked = None
    cam_t_masked = cam_R_masked = None
    if pose_masked:
        if os.path.exists(pose_masked):
            cam_t_masked_orig, cam_R_masked_orig, cam_q_masked = read_camera_poses(pose_masked)
            cam_t_masked = cam_t_masked_orig.copy()
            cam_R_masked = cam_R_masked_orig.copy()
        else:
            print(f"Camera trajectory file not found: {pose_masked}")
            pose_masked = None

    cam_t_no_mask = cam_R_no_mask = cam_q_no_mask = None
    if pose_no_mask:
        if os.path.exists(pose_no_mask):
            cam_t_no_mask, cam_R_no_mask, cam_q_no_mask = read_camera_poses(pose_no_mask)
        else:
            print(f"Camera trajectory file not found: {pose_no_mask}")
            pose_no_mask = None

    cam_t_gt = cam_R_gt = cam_q_gt = None
    if gt_anno is not None:
        if os.path.exists(gt_anno):
            ann = pkl.load(open(gt_anno, 'rb'))
            ext = ann['camera']['extrinsics']
            cam_R_gt = ext[:,:3,:3].transpose(0,2,1)
            cam_t_gt = np.einsum('bij, bj->bi', cam_R_gt, -ext[:, :3, -1])

            gt_human_vertices_full, gt_human_faces = _load_gt_human_vertices_from_ann(ann)
            if gt_human_vertices_full is not None:
                # gt_human_vertices_full = np.einsum('bij, bwj->bwi', cam_R_gt, gt_human_vertices_full - cam_t_gt[:, None, :])
                gt_human_vertices_full = gt_human_vertices_full - cam_t_gt[0, None, :]
                gt_human_vertices_full = gt_human_vertices_full * np.array([-1.0, -1.0, 1.0], dtype=np.float32)
                gt_human_seq_len = gt_human_vertices_full.shape[0]


            S = np.diag([-1., 1., 1.])
            cam_R_gt = np.einsum('ij,bjk->bik', S, cam_R_gt)                  # R' = S @ R_{w->c}
            cam_t_gt = np.einsum('bi,ij->bj', cam_t_gt, S)                    # t' = t_{w->c} @ S
            cam_q_gt = matrix_to_quaternion(torch.from_numpy(cam_R_gt)).numpy()

            # set gt starting point to origin
            cam_t_gt = cam_t_gt - cam_t_gt[0,:]

            traj_sequences.append(("gt", cam_t_gt, cam_R_gt, (64, 64, 64)))

            if align_to_gt and cam_t_masked is not None and cam_q_masked is not None and cam_q_masked.size > 0:
                cam_t_aligned, cam_R_aligned, sim3 = align_traj_evo(cam_t_gt, cam_q_gt, cam_t_masked, cam_q_masked)
                if cam_t_masked_orig is not None and cam_R_masked_orig is not None and cam_t_masked_orig.shape[0] >= 1:
                    alignment_transform = sim3
                cam_t_masked = cam_t_aligned
                cam_R_masked = cam_R_aligned

            if align_to_gt and cam_t_no_mask is not None and cam_q_no_mask is not None:
                if len(cam_t_gt) < len(cam_t_no_mask):
                    cam_t_no_mask = cam_t_no_mask[:len(cam_t_gt)]
                    cam_R_no_mask = cam_R_no_mask[:len(cam_t_gt)]
                    cam_q_no_mask = cam_q_no_mask[:len(cam_t_gt)]
                elif len(cam_t_gt) > len(cam_t_no_mask):
                    pad_len = len(cam_t_gt) - len(cam_t_no_mask)
                    cam_t_no_mask = np.pad(cam_t_no_mask, ((0, pad_len), (0, 0)), mode='edge')
                    cam_R_no_mask = np.pad(cam_R_no_mask, ((0, pad_len), (0, 0), (0, 0)), mode='edge')
                    cam_q_no_mask = np.pad(cam_q_no_mask, ((0, pad_len), (0, 0)), mode='edge')
                cam_t_no_mask, cam_R_no_mask, _ = align_traj_evo(cam_t_gt, cam_q_gt, cam_t_no_mask, cam_q_no_mask)
        else:
            print(f"Ground truth annotation file not found: {gt_anno}")
            gt_anno = None

    if human_camera_path:
        human_cam_t_tensor, human_cam_R_tensor = _load_emdb_camera_transforms(human_camera_path)
        if human_cam_t_tensor is not None and human_cam_R_tensor is not None:
            human_cam_t_raw = human_cam_t_tensor.cpu().numpy() * _FLIP_Y_ARRAY
            human_cam_R_raw = human_cam_R_tensor.cpu().numpy()
            human_cam_R_raw = _FLIP_Y_MATRIX @ human_cam_R_raw @ _FLIP_Y_MATRIX
            human_cam_q = matrix_to_quaternion(torch.from_numpy(human_cam_R_raw.astype(np.float32))).numpy()

            human_cam_t_vis = human_cam_t_raw
            human_cam_R_vis = human_cam_R_raw
            human_alignment_transform = (
                1.0,
                np.eye(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
            )

            if align_to_gt and cam_t_gt is not None and human_cam_t_raw.shape[0] > 0:
                target_len = min(human_cam_t_raw.shape[0], cam_t_gt.shape[0])
                if target_len > 0:
                    src_points = human_cam_t_raw[:target_len]
                    dst_points = cam_t_gt[:target_len]
                    scale_align, rot_align, trans_align = _compute_similarity_from_points(
                        src_points,
                        dst_points,
                        allow_scale=True,
                    )

                    human_cam_t_vis = (human_cam_t_raw @ rot_align.T) * scale_align + trans_align
                    human_cam_R_vis = np.einsum("ij,tjk->tik", rot_align, human_cam_R_raw)

                    human_alignment_transform = (
                        float(scale_align),
                        rot_align.astype(np.float32),
                        trans_align.astype(np.float32),
                    )

            traj_sequences.append(
                (
                    "human_camera",
                    human_cam_t_vis,
                    human_cam_R_vis,
                    (0, 102, 204),
                )
            )
        else:
            print(f"Failed to load human camera trajectory: {human_camera_path}")
    if viser:
        import viser as viser_mod

        def _make_line_segments(points_xyz: np.ndarray) -> np.ndarray:
            if points_xyz.shape[0] < 2:
                return np.empty((0, 2, 3), dtype=np.float32)
            return np.stack([points_xyz[:-1], points_xyz[1:]], axis=1).astype(np.float32)

        def _make_frustum_segments(cam_r: np.ndarray, cam_t: np.ndarray) -> np.ndarray:
            cam_points = np.array(
                [
                    [0.0, 0.0, 0.0],
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

        processed_sequences = []
        max_steps = 0
        world_origin: Optional[np.ndarray] = None
        for label, positions, rotations, color in traj_sequences:
            if positions is None or rotations is None or positions.size == 0:
                continue
            positions_ds = positions[::stride]
            rotations_ds = rotations[::stride]
            max_steps = max(max_steps, positions_ds.shape[0])
            processed_sequences.append((label, positions_ds, rotations_ds, color))
            if label == "gt" and positions_ds.shape[0] > 0 and world_origin is None:
                world_origin = positions_ds[0]
        if world_origin is None and processed_sequences:
            world_origin = processed_sequences[0][1][0]

        human_vertices_seq: Optional[np.ndarray] = None
        human_faces: Optional[np.ndarray] = None
        gt_human_vertices_seq: Optional[np.ndarray] = None
        gt_human_faces_seq: Optional[np.ndarray] = None
        if human_npz_path and human_camera_path:
            human_vertices_full, human_faces = load_emdb_human_world_vertices(
                human_npz_path,
                human_camera_path,
            )
            if human_vertices_full is not None and human_vertices_full.size > 0:
                if align_to_gt and human_seq_len:
                    human_vertices_full = human_vertices_full[:human_seq_len]
                human_vertices_full = human_vertices_full @ _FLIP_Y_MATRIX
                if align_to_gt:
                    scale, rot, trans = human_alignment_transform
                    rot = np.asarray(rot, dtype=np.float32)
                    trans = np.asarray(trans, dtype=np.float32)
                    human_vertices_full = human_vertices_full @ rot.T
                    if abs(scale - 1.0) > 1e-6:
                        human_vertices_full *= scale
                    human_vertices_full += trans
                human_vertices_seq = human_vertices_full[::stride]
                if human_vertices_seq.size > 0:
                    max_steps = max(max_steps, human_vertices_seq.shape[0])
                    if world_origin is None:
                        world_origin = human_vertices_seq[0].mean(axis=0)
        if gt_human_vertices_full is not None and gt_human_faces is not None:
            gt_human_vertices_local = gt_human_vertices_full
            if align_to_gt and gt_human_seq_len:
                gt_human_vertices_local = gt_human_vertices_local[:gt_human_seq_len]
            gt_human_vertices_local = gt_human_vertices_local @ _FLIP_Y_MATRIX
            gt_human_vertices_seq = gt_human_vertices_local[::stride]
            gt_human_faces_seq = gt_human_faces
            if gt_human_vertices_seq.size > 0:
                max_steps = max(max_steps, gt_human_vertices_seq.shape[0])
                if world_origin is None:
                    world_origin = gt_human_vertices_seq[0].mean(axis=0)

        color_cache: dict[tuple[tuple[int, int, int], int], np.ndarray] = {}
        trajectory_handles: dict[str, Any] = {}
        frustum_handles: dict[str, Any] = {}
        world_frame_handle: Optional[Any] = None
        world_plane_handle: Optional[Any] = None
        has_humans = bool(human_vertices_seq is not None and human_vertices_seq.size > 0 and human_faces is not None)
        has_gt_humans = bool(gt_human_vertices_seq is not None and gt_human_vertices_seq.size > 0 and gt_human_faces_seq is not None)

        def _compute_world_scale() -> float:
            if world_origin is None:
                return 1.0
            max_radius = 0.0
            for _, positions_ds, _, _ in processed_sequences:
                if positions_ds.size == 0:
                    continue
                offsets = positions_ds - world_origin
                radii = np.linalg.norm(offsets, axis=1)
                if radii.size:
                    max_radius = max(max_radius, float(radii.max()))
            return max(max_radius, 1.0)

        if world_origin is not None:
            world_frame_handle = server.scene.add_frame(
                "/world_frame",
                position=world_origin.astype(np.float32),
                show_axes=True,
                axes_length=0.2,
                axes_radius=0.01,
            )

        def _color_array(color_rgb: tuple[int, int, int], n: int) -> np.ndarray:
            key = (color_rgb, n)
            if key in color_cache:
                return color_cache[key]
            base = np.asarray(color_rgb, dtype=np.float32) / 255.0
            arr = np.broadcast_to(base, (n, 2, 3)).copy()
            color_cache[key] = arr
            return arr

        with server.gui.add_folder("Playback"):
            gui_show_traj = server.gui.add_checkbox("Show Trajectory", True)
            gui_show_frustum = server.gui.add_checkbox("Show Frustum", True)
            gui_show_world = server.gui.add_checkbox(
                "Show World Frame",
                world_frame_handle is not None,
                disabled=world_frame_handle is None,
            )
            gui_show_plane = server.gui.add_checkbox(
                "Show Ground Plane",
                world_origin is not None,
                disabled=world_origin is None,
            )
            gui_show_humans = server.gui.add_checkbox(
                "Show Humans",
                has_humans,
                disabled=not has_humans,
            )
            gui_show_gt_humans = server.gui.add_checkbox(
                "Show GT Humans",
                has_gt_humans,
                disabled=not has_gt_humans,
            )
            gui_timestep = server.gui.add_slider(
                "Timestep",
                min=0,
                max=max(0, max_steps - 1),
                step=1,
                initial_value=0,
                disabled=max_steps == 0,
            )
            gui_next_frame = server.gui.add_button("Next Frame", disabled=max_steps == 0)
            gui_prev_frame = server.gui.add_button("Prev Frame", disabled=max_steps == 0)
            gui_playing = server.gui.add_checkbox("Playing", True)
            gui_framerate = server.gui.add_slider(
                "FPS",
                min=1,
                max=60,
                step=0.5,
                initial_value=pose_fps if pose_fps > 0 else 30.0,
            )
            gui_framerate_options = server.gui.add_button_group(
                "FPS options",
                tuple(str(v) for v in (10, 20, 30, 60)),
            )

        for label, positions, rotations, color in processed_sequences:
            server.scene.add_frame(f"/trajectory/{label}")
            trajectory_handles[label] = server.scene.add_line_segments(
                name=f"/trajectory/{label}/segments",
                points=np.empty((0, 2, 3), dtype=np.float32),
                colors=np.empty((0, 2, 3), dtype=np.float32),
                line_width=2.0,
            )
            frustum_handles[label] = server.scene.add_line_segments(
                name=f"/trajectory/{label}/frustum",
                points=np.empty((0, 2, 3), dtype=np.float32),
                colors=np.empty((0, 2, 3), dtype=np.float32),
                line_width=1.2,
            )

        human_mesh_handle: Optional[Any] = None
        if has_humans and human_vertices_seq is not None and human_faces is not None:
            human_mesh_handle = server.scene.add_mesh_simple(
                name="/humans/predicted",
                vertices=human_vertices_seq[0],
                faces=human_faces.astype(np.int32, copy=False),
                flat_shading=False,
                wireframe=False,
                opacity=0.8,
                color=(0.0, 1.0, 1.0),
                # color=(0.78, 0.78, 0.85),
            )
            human_mesh_handle.visible = gui_show_humans.value

        gt_human_mesh_handle: Optional[Any] = None
        if has_gt_humans and gt_human_vertices_seq is not None and gt_human_faces_seq is not None:
            gt_human_mesh_handle = server.scene.add_mesh_simple(
                name="/humans/gt",
                vertices=gt_human_vertices_seq[0],
                faces=gt_human_faces_seq.astype(np.int32, copy=False),
                flat_shading=False,
                wireframe=False,
                opacity=0.8,
                color=(0.55, 0.55, 0.55),
            )
            gt_human_mesh_handle.visible = gui_show_gt_humans.value

        if world_origin is not None:
            extent = max(_compute_world_scale(), 1.0)
            half = extent
            vertices = np.array(
                [
                    [-half, 0.0, -half],
                    [-half, 0.0, half],
                    [half, 0.0, half],
                    [half, 0.0, -half],
                ],
                dtype=np.float32,
            )
            faces = np.array(
                [
                    [0, 1, 2],
                    [0, 2, 3],
                    [2, 1, 0],
                    [3, 2, 0],
                ],
                dtype=np.int32,
            )
            world_plane_handle = server.scene.add_mesh_simple(
                name="/world_plane",
                vertices=vertices,
                faces=faces,
                flat_shading=False,
                wireframe=False,
                opacity=0.55,
                color=(0.55, 0.55, 0.55),
            )

        def _update_step(step_idx: int) -> None:
            for label, positions, rotations, color in processed_sequences:
                if positions.shape[0] == 0:
                    continue
                capped_idx = min(step_idx, positions.shape[0] - 1)
                segments = _make_line_segments(positions[: capped_idx + 1])
                traj_handle = trajectory_handles[label]
                if gui_show_traj.value and segments.size:
                    traj_handle.visible = True
                    traj_handle.points = segments
                    traj_handle.colors = _color_array(color, segments.shape[0])
                else:
                    traj_handle.visible = gui_show_traj.value and segments.size > 0

                frustum_segments = _make_frustum_segments(
                    rotations[capped_idx : capped_idx + 1],
                    positions[capped_idx : capped_idx + 1],
                )
                frustum_handle = frustum_handles[label]
                if gui_show_frustum.value and frustum_segments.size:
                    frustum_handle.visible = True
                    frustum_handle.points = frustum_segments
                    frustum_handle.colors = _color_array(color, frustum_segments.shape[0])
                else:
                    frustum_handle.visible = gui_show_frustum.value and frustum_segments.size > 0

            if human_mesh_handle is not None and human_vertices_seq is not None and human_vertices_seq.size:
                human_idx = min(step_idx, human_vertices_seq.shape[0] - 1)
                human_mesh_handle.vertices = human_vertices_seq[human_idx]
                human_mesh_handle.visible = gui_show_humans.value
            if gt_human_mesh_handle is not None and gt_human_vertices_seq is not None and gt_human_vertices_seq.size:
                gt_idx = min(step_idx, gt_human_vertices_seq.shape[0] - 1)
                gt_human_mesh_handle.vertices = gt_human_vertices_seq[gt_idx]
                gt_human_mesh_handle.visible = gui_show_gt_humans.value

        _update_step(0)

        @gui_show_traj.on_update
        def _(_) -> None:
            _update_step(gui_timestep.value)
            server.flush()

        @gui_show_frustum.on_update
        def _(_) -> None:
            _update_step(gui_timestep.value)
            server.flush()

        if world_frame_handle is not None:
            world_frame_handle.visible = gui_show_world.value

            @gui_show_world.on_update
            def _(_) -> None:
                world_frame_handle.visible = gui_show_world.value
                server.flush()

        if world_plane_handle is not None:
            world_plane_handle.visible = gui_show_plane.value

            @gui_show_plane.on_update
            def _(_) -> None:
                world_plane_handle.visible = gui_show_plane.value
                server.flush()

        if human_mesh_handle is not None:
            human_mesh_handle.visible = gui_show_humans.value

            @gui_show_humans.on_update
            def _(_) -> None:
                human_mesh_handle.visible = gui_show_humans.value
                server.flush()

        if gt_human_mesh_handle is not None:
            gt_human_mesh_handle.visible = gui_show_gt_humans.value

            @gui_show_gt_humans.on_update
            def _(_) -> None:
                gt_human_mesh_handle.visible = gui_show_gt_humans.value
                server.flush()

        @gui_timestep.on_update
        def _(_) -> None:
            _update_step(gui_timestep.value)
            server.flush()

        @gui_next_frame.on_click
        def _(_) -> None:
            if max_steps == 0:
                return
            gui_timestep.value = (gui_timestep.value + 1) % max_steps

        @gui_prev_frame.on_click
        def _(_) -> None:
            if max_steps == 0:
                return
            gui_timestep.value = (gui_timestep.value - 1) % max_steps

        @gui_playing.on_update
        def _(_) -> None:
            gui_timestep.disabled = gui_playing.value or max_steps == 0
            gui_next_frame.disabled = gui_playing.value or max_steps == 0
            gui_prev_frame.disabled = gui_playing.value or max_steps == 0

        @gui_framerate_options.on_click
        def _(_) -> None:
            try:
                gui_framerate.value = float(gui_framerate_options.value)
            except (TypeError, ValueError):
                pass

        try:
            while True:
                if gui_playing.value and max_steps > 0:
                    gui_timestep.value = (gui_timestep.value + 1) % max_steps
                time.sleep(1.0 / max(gui_framerate.value, 1e-3))
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

    if not processed_sequences:
        print("No camera trajectories available for rendering; skipping.")
        return

    frustum_sequences = []
    for label, positions_ds, rotations_ds, color in processed_sequences:
        if positions_ds is None or rotations_ds is None or positions_ds.size == 0:
            continue
        color_norm = tuple((np.asarray(color, dtype=np.float64) / 255.0).tolist())
        frustums = make_camera_frustums(rotations_ds, positions_ds, color=color_norm)
        frustum_sequences.append((label, frustums))

    if not frustum_sequences:
        print("No valid frustums constructed; skipping rendering.")
        return

    # Offscreen renderer
    w, h = 540, 720
    render = o3d.visualization.rendering.OffscreenRenderer(w, h)
    render.scene.set_background([1, 1, 1, 1])

    orig_fps = 30.0
    pose_fps = orig_fps / stride
    max_len = max((len(frustums) for _, frustums in frustum_sequences), default=0)
    if max_len == 0:
        print("Frustum sequences are empty after downsampling; skipping rendering.")
        return
    cam_centers = [np.empty((0, 3), dtype=float) for _ in frustum_sequences]

    if orig_video is not None:
        cap = cv2.VideoCapture(orig_video)
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_orig = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1e9)
        n_steps = min(max_len, n_orig // stride)

        writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*'mp4v'), pose_fps, (2*w, h))  # side-by-side
    else:
        n_steps = max_len
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
    up = np.array([0, 1, 0])
    spin_deg_per_frame = 0

    imgs = []
    for i in range(n_steps):  # or range(0, len(cams_masked), stride)
        if orig_video is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * stride)
            ret, orig_bgr = cap.read()
            orig_bgr = orig_bgr[:, w:, :]
            if not ret: break

        for seq_idx, (label, frustums) in enumerate(frustum_sequences):
            if i >= len(frustums):
                continue
            frustum = frustums[i]
            render.scene.add_geometry(f"{label}_{i}", frustum, cam_mat)
            cam_point = np.asarray(frustum.points)[0]
            if cam_centers[seq_idx].size == 0:
                cam_centers[seq_idx] = cam_point[None, :]
            else:
                cam_centers[seq_idx] = np.vstack([cam_centers[seq_idx], cam_point])

        theta = i * np.deg2rad(spin_deg_per_frame)
        window = 80

        recent_centers = [c[-window:] for c in cam_centers if c.size > 0]
        if recent_centers:
            win = np.vstack(recent_centers)
            center = win.mean(axis=0)
            std = win.std(axis=0)
        else:
            center = np.zeros(3, dtype=float)
            std = np.ones(3, dtype=float) * 0.5

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
    parser.add_argument('--human_dir', type=str, default='emdb2_0-9_to_daniel/human', help='Directory with human prediction npz files.')
    parser.add_argument('--camera_dir', type=str, default='emdb2_0-9_to_daniel/camera', help='Directory with camera trajectory txt files for human predictions.')
    parser.add_argument('--sequence', type=str, default=None, help='Optional sequence name (e.g., P0_09_outdoor_walk) to visualize a single human/camera pair.')
    parser.add_argument('--human_npz_path', type=str, default=None, help='Direct path to a human npz file for single-pair visualization.')
    parser.add_argument('--human_camera_path', type=str, default=None, help='Direct path to a human camera trajectory txt file for single-pair visualization.')
    parser.add_argument('--gt_anno_path', type=str, default=None, help='Direct path to the ground-truth annotation pickle for single-pair visualization.')
    parser.add_argument('--output_mp4', type=str, default=None, help='Output MP4 path when rendering without viser.')
    parser.add_argument('--align_to_gt', action='store_true', help='Align predicted trajectories and humans to ground truth when available.')
    args = parser.parse_args()

    if args.sequence or args.human_npz_path or args.human_camera_path:
        seq_name = args.sequence
        human_npz = args.human_npz_path
        human_camera_txt = args.human_camera_path
        gt_anno = args.gt_anno_path
        orig_video = None

        if seq_name:
            if human_npz is None:
                human_npz = os.path.join(args.human_dir, f"{seq_name}.npz")
            if human_camera_txt is None:
                human_camera_txt = os.path.join(args.camera_dir, f"{seq_name}_images_incremental_all.txt")
            if gt_anno is None:
                if "_" in seq_name:
                    person_id, seq_rest = seq_name.split("_", 1)
                    seq_root = os.path.join(args.gt_dir, person_id, seq_rest)
                    candidate_gt = os.path.join(seq_root, f"{person_id}_{seq_rest}_data.pkl")
                    if os.path.exists(candidate_gt):
                        gt_anno = candidate_gt
                    else:
                        print(f"Warning: ground truth annotation not found at {candidate_gt}")
                    candidate_video = os.path.join(seq_root, f"{person_id}_{seq_rest}_video.mp4")
                    if os.path.exists(candidate_video):
                        orig_video = candidate_video
                else:
                    print(f"Warning: sequence format '{seq_name}' does not match expected 'P*_...'")

        if human_camera_txt is None:
            raise SystemExit("A human camera trajectory path is required to visualize the pair.")

        if not os.path.exists(human_camera_txt):
            raise SystemExit(f"Human camera trajectory file not found: {human_camera_txt}")
        if human_npz is not None and not os.path.exists(human_npz):
            print(f"Warning: human npz file not found at {human_npz}, continuing without humans.")
            human_npz = None
        if gt_anno is not None and not os.path.exists(gt_anno):
            print(f"Warning: ground truth annotation file not found at {gt_anno}, proceeding without GT alignment.")
            gt_anno = None

        out_mp4 = args.output_mp4 or f"{seq_name or 'human_camera'}_vis.mp4"
        render_cameras_gt_traj(
            pose_masked=None,
            pose_no_mask=None,
            gt_anno=gt_anno,
            stride=args.stride,
            orig_video=orig_video,
            out_mp4=out_mp4,
            viser=args.viser,
            human_npz_path=human_npz,
            human_camera_path=human_camera_txt,
            align_to_gt=args.align_to_gt,
        )
        raise SystemExit(0)

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

        human_npz = None
        if args.human_dir:
            candidate_human = os.path.join(args.human_dir, f"{p}_{seq}.npz")
            if os.path.exists(candidate_human):
                human_npz = candidate_human
            elif args.viser:
                print(f"Human prediction file not found: {candidate_human}")

        human_camera_txt = None
        if args.camera_dir:
            candidate_camera = os.path.join(args.camera_dir, f"{p}_{seq}_images_incremental_all.txt")
            if os.path.exists(candidate_camera):
                human_camera_txt = candidate_camera
            elif args.viser:
                print(f"Camera trajectory file for humans not found: {candidate_camera}")

        render_cameras_gt_traj(
            pose_masked,
            pose_no_mask,
            gt_anno=gt_anno,
            stride=args.stride,
            orig_video=orig_video,
            out_mp4=output_mp4,
            viser=args.viser,
            human_npz_path=human_npz,
            human_camera_path=human_camera_txt,
            align_to_gt=args.align_to_gt,
        )
        print(f"Rendered sequence: {output_mp4}")
