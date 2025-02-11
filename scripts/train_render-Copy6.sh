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
SCENE=GX010472_LLFF_Processed
# SCENE=GX010555_LLFF_Processed
# DATA_ROOT=/work/SuperResolutionData/LuvNeRF/videos/LogNeRFStudioProcessed
DATA_ROOT=/work/SuperResolutionData/LuvNeRF/videos/sRGBNeRFStudioProcessed

EXPERIMENT=llff/"$SCENE"  # Checkpoints, results, logs will be saved to exp/${EXPERIMENT}.
DATA_DIR="$DATA_ROOT"/"$SCENE"

# Create the results directory if it doesn't exist
RESULTS_DIR=/work/SuperResolutionData/LuvNeRF/results
# mkdir -p "$RESULTS_DIR"

# Define the directory for saving checkpoints and logs inside the results folder
CHECKPOINT_DIR="$RESULTS_DIR/$EXPERIMENT"

# Create the checkpoint directory if it doesn't exist
mkdir -p "$CHECKPOINT_DIR"

# Training
# You can also run this with `accelerate launch`.
python train.py --gin_configs=${CONFIG} \
    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
    --gin_bindings="Config.exp_name = '${CHECKPOINT_DIR}'" \
    --gin_bindings="Model.bilateral_grid = True"

# Render testing views
python render.py --gin_configs=${CONFIG} \
    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
    --gin_bindings="Config.exp_name = '${CHECKPOINT_DIR}'"

# Render path
python render.py --gin_configs=${CONFIG} \
    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
    --gin_bindings="Config.exp_name = '${CHECKPOINT_DIR}'" \
    --gin_bindings="Config.render_path = True" \
    --gin_bindings="Config.render_path_frames = 120" \
    --gin_bindings="Config.render_video_fps = 60"
