#!/bin/bash

CONFIG=configs/360.gin  # For 360 scenes.
# CONFIG=configs/llff.gin  # For forward-facing scenes.
# SCENE=pondbike
# SCENE=transfer_images_50only_llff_GX010290_log_Stars
# SCENE=transfer_images_72_throughcolmap_GX010374
# SCENE=GX010401_stars_llff_50Im

# SCENE=GX010401_stars_llff_150Images
# SCENE=GX010423_data
# SCENE=GX010430_data
# SCENE=chinesearch
# SCENE=lionpavilion
# SCENE=GX010557_LLFF_Processed
# SCENE=GX010431_LLFF_Processed
# DATA_ROOT=/work/SuperResolutionData/LuvNeRF/videos/sRGBNeRFStudioProcessed

# SCENE=GX010454_LLFF_Processed
# DATA_ROOT=/work/SuperResolutionData/LuvNeRF/videos/sRGBNeRFStudioProcessed

SCENE=GX010416_LLFF_Processed
DATA_ROOT=/work/SuperResolutionData/LuvNeRF/videos/LogNeRFStudioProcessed

# SCENE=GX010571_LLFF_Processed
# DATA_ROOT=/work/SuperResolutionData/LuvNeRF/videos/LogNeRFStudioProcessed
# SCENE=GX010401_processed_output
# SCENE=strat
EXPERIMENT=llff/"$SCENE"  # Checkpoints, results, and logs will be saved to exp/${EXPERIMENT}.
# DATA_ROOT=/home/verma.lu/NeRFs/nerf_synthetic_data_all_types/nerf_synthetic
DATA_DIR="$DATA_ROOT"/"$SCENE"


# Evaluation
python eval.py --gin_configs=${CONFIG} \
    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
    --gin_bindings="Config.exp_name = '${EXPERIMENT}'"
