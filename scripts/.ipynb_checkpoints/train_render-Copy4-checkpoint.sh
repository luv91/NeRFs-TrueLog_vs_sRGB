#!/bin/bash

# CONFIG=configs/360.gin  # For 360 scenes.
CONFIG=configs/llff.gin  # For forward-facing scenes.
# SCENE=pondbike
# SCENE=transfer_images_50only_llff_GX010290_log_Stars
# SCENE=transfer_images_72_throughcolmap_GX010374
# SCENE=GX010401_stars_llff_50Im
# SCENE=GX010401_stars_llff_150Images
# GX010401_processed_output
# SCENE=strat
# SCENE=GX010423_data
# SCENE=GX010430_data
# SCENE=lionpavilion
# SCENE=GX010557_LLFF_Processed
# SCENE=GX010431_LLFF_Processed
# SCENE=GX010549_LLFF_Processed
# SCENE=GX010555_LLFF_Processed
# SCENE=GX010590_LLFF_Processed
# DATA_ROOT=/work/SuperResolutionData/LuvNeRF/videos/LogNeRFStudioProcessed
# SCENE=GX010476_LLFF_Processed
EXPERIMENT=llff/"$SCENE"  # Checkpoints, results, logs will be saved to exp/${EXPERIMENT}.

# DATA_ROOT=/home/verma.lu/NeRFs/nerf_synthetic_data_all_types/nerf_synthetic

# DATA_ROOT=/work/SuperResolutionData/LuvNeRF/videos/sRGBNeRFStudioProcessed


SCENE=orchids
DATA_ROOT=/home/verma.lu/NeRFs/nerf_synthetic_data_all_types/nerf_synthetic

DATA_DIR="$DATA_ROOT"/"$SCENE"


# Training
# You can also run this with `accelerate launch`.
python train.py --gin_configs=${CONFIG} \
    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
    --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
    --gin_bindings="Model.bilateral_grid = True"


# Render testing views
python render.py --gin_configs=${CONFIG} \
    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
    --gin_bindings="Config.exp_name = '${EXPERIMENT}'"


# Render path
python render.py --gin_configs=${CONFIG} \
    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
    --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
    --gin_bindings="Config.render_path = True" \
    --gin_bindings="Config.render_path_frames = 120" \
    --gin_bindings="Config.render_video_fps = 60"


# # Render training views
# # Comment the last line to render training views without
# # per-view bilateral grids applied.
# python render.py --gin_configs=${CONFIG} \
#     --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
#     --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
#     --gin_bindings="Config.render_train = True" \
#     --gin_bindings="Model.bilateral_grid = True"