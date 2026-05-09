#!/bin/bash
################################################################################
#                     Luxia Digits Testing Script (Fldzhyan)              #
################################################################################
# File: olivetti_fldzhyan.sh
# Description: 
# This script test the trained fldzhyan model on the olivetti dataset.
# Usage:
#   bash olivetti_fldzhyan.sh
#
# Environment Variables Required:
#   GIT_PROJ: Base GIT directory path 
#
# Author: Tzamn Melendez Carmona
################################################################################

# Set the Git root directory path
if [ -z "$GIT_PROJ" ]; then
    export GIT_PROJ=$(git rev-parse --show-toplevel)
    echo "setting GIT Current Project to $GIT_PROJ"
fi

# Exit on error
set -e

#------------------------------------------------------------------------------#
#                              Global Configuration                            #
#------------------------------------------------------------------------------#

# Python executable path
PY=python3

# Dataset configuration
DS=olivetti

# list of the paths to the models

MOD_LIST=(
    ${GIT_PROJ}/results/olivetti/train/lx_fldzhyan_4096x8196_ep400_bs16_lr0.0000015_seed1/200525_132251/model_epoch079.pth
    ${GIT_PROJ}/results/olivetti/train/lx_fldzhyan_4096x8196_ep400_bs16_lr0.0000015_seed2/210525_195326/model_epoch178.pth
    ${GIT_PROJ}/results/olivetti/train/lx_fldzhyan_4096x8196_ep400_bs16_lr0.0000015_seed3/220525_205208/model_epoch048.pth
    ${GIT_PROJ}/results/olivetti/train/lx_fldzhyan_4096x8196_ep400_bs16_lr0.0000015_seed4/230525_144554/model_epoch331.pth
    ${GIT_PROJ}/results/olivetti/train/lx_fldzhyan_4096x8196_ep400_bs16_lr0.0000015_seed5/240525_085637/model_epoch186.pth
)
      
      
      
      
      

#------------------------------------------------------------------------------#
#                           Testing Configuration                             #
#------------------------------------------------------------------------------#

DEVICE=cpu   # Hardware device selection (cuda/cpu)
BS=16   # Batch size
VF=0.1       # Validation fraction

#------------------------------------------------------------------------------#
#                           Output Directory Setup                             #
#------------------------------------------------------------------------------#

# Generate timestamp for unique directory naming
DATE=$(date +%d%m%y_%H%M%S)

MODEL=fldzhyan
RESULTS_DIR="${GIT_PROJ}/results/${DS}/test"
# Ensure base results directory exists
mkdir -p "$RESULTS_DIR"

# Enable WandB logging
USE_WANDB="--use_wandb"
WANDB_KEY=""


# USE_WANDB=""

#------------------------------------------------------------------------------#
#                           Mesh Size Configuration                            #
#------------------------------------------------------------------------------#

NI=4096     # Number of inputs

#------------------------------------------------------------------------------#
#                        Virtual Environment Activation                         #
#------------------------------------------------------------------------------#
# Activate the LuxIA virtual environment
#
if ! source ${GIT_PROJ}/bash_scripts/activate_luxia.sh; then
    echo "Failed to activate virtual environment. Exiting."
    exit 1
fi

#------------------------------------------------------------------------------#
#                            TEST Execution                                #
#------------------------------------------------------------------------------#



# Iterate over the list of models and seeds
SEED=0
for PTH_MODEL in ${MOD_LIST[@]}; do
    SEED=$((SEED+1))
    NAME="${MODEL}"
    DIR="${RESULTS_DIR}/${NAME}/${DATE}"
    LOG_DIR="${DIR}/logs"

# Create necessary directories
mkdir -p "$LOG_DIR"
        # Construct the test command with all parameters
        EXEC="${PY} ${GIT_PROJ}/test.py --model=$MODEL --n_inputs=$NI --dataset=$DS \
        --batch_size=$BS --val_frac=$VF --seed=$SEED\
            --exp_dir=$DIR  --device=$DEVICE  --path_model=$PTH_MODEL \
             $USE_WANDB --wandb_key=$WANDB_KEY"

        # Execute test and log output
        echo " *  - - - - TESTING $MODEL of Size $NI on $DS dataset with - - - * *"
        luxia_exec $EXEC | tee $LOG_DIR/bs${BS}_${DATE}.log
done

