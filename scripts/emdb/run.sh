#!/bin/bash

split=2
# cam_output_dir="results/emdb/camera"
# smpl_output_dir="results/emdb/smpl"
result_dir="results/emdb/"
# smpl_output_dir="results/emdb/smpl_no_mask"
eval_input_dir="emdb_data/"

CUDA_VISIBLE_DEVICES=2 python scripts/emdb/run_cam.py --split $split --result_dir "$result_dir"
# CUDA_VISIBLE_DEVICES=2 python scripts/emdb/run_cam.py --split $split --result_dir "$result_dir" --no_mask
# CUDA_VISIBLE_DEVICES=2 python scripts/emdb/run_cam.py --split $split --result_dir "$result_dir" --no_ba
# CUDA_VISIBLE_DEVICES=2 python scripts/emdb/run_cam.py --split $split --result_dir "$result_dir" --no_mask --no_ba
# CUDA_VISIBLE_DEVICES=2 python render_scene_and_cam.py 
# python scripts/emdb/run_smpl.py --split $split --output_dir "$smpl_output_dir"
# python scripts/emdb/run_eval.py --split $split --pred_dir  "$result_dir" --gt_dir "$eval_input_dir"
