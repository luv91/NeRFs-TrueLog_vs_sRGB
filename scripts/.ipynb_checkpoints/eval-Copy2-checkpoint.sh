#!/bin/bash
# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

export CUDA_VISIBLE_DEVICES=0

# SCENE=flower
# SCENE=GX010430_LLFF_Processed
# DATA_DIR=/work/SuperResolutionData/LuvNeRF/videos/LogNeRFStudioProcessed

# SCENE=GX010540_LLFF_Processed
# DATA_DIR=/work/SuperResolutionData/LuvNeRF/videos/sRGBNeRFStudioProcessed
# SCENE=GX010590_LLFF_Processed
# DATA_DIR=/work/SuperResolutionData/LuvNeRF/videos/LogNeRFStudioProcessed

# SCENE=GX010570_LLFF_Processed
# DATA_DIR=/work/SuperResolutionData/LuvNeRF/videos/LogNeRFStudioProcessed

SCENE=GX010580_LLFF_Processed
DATA_DIR=/work/SuperResolutionData/LuvNeRF/videos/LogNeRFStudioProcessed

EXPERIMENT=llff/"$SCENE"
CHECKPOINT_DIR=/work/SuperResolutionData/LuvNeRF/results/llff/"$SCENE"/checkpoints/015000

# Evaluation
python eval.py --gin_configs=${CONFIG} \
    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
    --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
    --gin_bindings="Config.resume_checkpoint_path = '${CHECKPOINT_DIR}'" \
    --logtostderr
