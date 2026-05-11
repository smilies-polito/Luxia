#!/bin/bash
################################################################################
#                     Luxia Digits Testing Script (Clements)              #
################################################################################
# File: digits_clements.sh
# Description: 
# This script test the trained clements model on the digits dataset.
# Usage:
#   bash digits_clements.sh
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
SRC_FOLDER="${GIT_PROJ}/examples/pnn-benchmarks"

# Exit on error
set -e

#------------------------------------------------------------------------------#
#                              Global Configuration                            #
#------------------------------------------------------------------------------#

# Python executable path
PY=python3

# Dataset configuration
DS=digits

# list of the paths to the models

MOD_LIST=(
    ${GIT_PROJ}/results/digits/train/pt_clements_64x64_ep150_bs512_lr0.0005_seed1/020525_160745/model_epoch124.pth
    ${GIT_PROJ}/results/digits/train/pt_clements_64x64_ep150_bs512_lr0.0005_seed2/030525_171429/model_epoch144.pth
    ${GIT_PROJ}/results/digits/train/pt_clements_64x64_ep150_bs512_lr0.0005_seed3/040525_181747/model_epoch130.pth
    ${GIT_PROJ}/results/digits/train/pt_clements_64x64_ep150_bs512_lr0.0005_seed4/050525_192002/model_epoch137.pth
    ${GIT_PROJ}/results/digits/train/pt_clements_64x64_ep150_bs512_lr0.0005_seed5/060525_203146/model_epoch140.pth
)

#------------------------------------------------------------------------------#
#                           Testing Configuration                             #
#------------------------------------------------------------------------------#

DEVICE=cpu   # Hardware device selection (cuda/cpu)
BS=512      # Batch size
VF=0.1       # Validation fraction

#------------------------------------------------------------------------------#
#                           Output Directory Setup                             #
#------------------------------------------------------------------------------#

# Generate timestamp for unique directory naming
DATE=$(date +%d%m%y_%H%M%S)

MODEL=clements
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

NI=64  # Number of inputs

#------------------------------------------------------------------------------#
#                        Virtual Environment Activation                         #
#------------------------------------------------------------------------------#
# Activate the LuxIA virtual environment
#
if ! source ${SRC_FOLDER}/bash_scripts/activate_photontorch.sh; then
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
        photontorch_exec $EXEC | tee $LOG_DIR/bs${BS}_${DATE}.log
done

