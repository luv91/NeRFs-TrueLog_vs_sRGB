#!/bin/bash

# CONFIG=configs/360.gin  # For 360 scenes.
CONFIG=configs/llff.gin  # For forward-facing scenes.
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
# SCENE=GX010401_processed_output

# SCENE=GX010555_LLFF_Processed
# DATA_ROOT=/work/SuperResolutionData/LuvNeRF/videos/LogNeRFStudioProcessed

SCENE=GX010569_LLFF_Processed
DATA_ROOT=/work/SuperResolutionData/LuvNeRF/videos/sRGBNeRFStudioProcessed

# SCENE=strat
EXPERIMENT=llff/"$SCENE"  # Checkpoints, results, logs will be saved to exp/${EXPERIMENT}.

# DATA_ROOT=/home/verma.lu/NeRFs/nerf_synthetic_data_all_types/nerf_synthetic
DATA_DIR="$DATA_ROOT"/"$SCENE"


FT_NAME=edit1  # Finishing checkpoints and results will be saved to exp/${EXPERIMENT}/${FT_NAME}.
FT_TGT_IMAGE=/pathto/datasets/bilarf_data/editscenes/scibldg/edits/edit1_color_path_088.jpg  # Edited single view.
FT_TGT_POSE=path:88  # Camera pose of the edited view.


# Train a low-rank 4D bilateral grid for radiance-finishing.
python train_bilagrid4d.py --gin_configs=${CONFIG} \
    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
    --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
    --gin_bindings="Config.ft_name = '${FT_NAME}'" \
    --gin_bindings="Config.ft_tgt_image = '${FT_TGT_IMAGE}'" \
    --gin_bindings="Config.ft_tgt_pose = '${FT_TGT_POSE}'" \
    --gin_bindings="Model.bilateral_grid4d = True"


# Render path
python render.py --gin_configs=${CONFIG} \
    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
    --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
    --gin_bindings="Config.ft_name = '${FT_NAME}'" \
    --gin_bindings='Config.render_ft = True' \
    --gin_bindings="Config.render_path = True" \
    --gin_bindings="Config.render_path_frames = 120" \
    --gin_bindings="Config.render_video_fps = 60" \
    --gin_bindings='Model.bilateral_grid4d = True'


# Render testing views
python render.py --gin_configs=${CONFIG} \
    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
    --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
    --gin_bindings="Config.ft_name = '${FT_NAME}'" \
    --gin_bindings='Config.render_ft = True' \
    --gin_bindings='Model.bilateral_grid4d = True'
