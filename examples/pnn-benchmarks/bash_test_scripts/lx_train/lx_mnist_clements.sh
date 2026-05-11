#!/bin/bash
################################################################################
#                     Luxia Digits Testing Script (Clements)              #
################################################################################
# File: mnist_clements.sh
# Description: 
# This script test the trained clements model on the mnist dataset.
# Usage:
#   bash mnist_clements.sh
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
DS=mnist

# list of the paths to the models

MOD_LIST=(
    ${GIT_PROJ}/results/mnist/train/lx_clements_pdb_784x784_ep150_bs512_lr0.000038_seed1/190525_170827/model_epoch120.pth
    ${GIT_PROJ}/results/mnist/train/lx_clements_pdb_784x784_ep150_bs512_lr0.000038_seed2/200525_010129/model_epoch114.pth
    ${GIT_PROJ}/results/mnist/train/lx_clements_pdb_784x784_ep150_bs512_lr0.000038_seed3/200525_092433/model_epoch061.pth
    ${GIT_PROJ}/results/mnist/train/lx_clements_pdb_784x784_ep150_bs512_lr0.000038_seed4/200525_180203/model_epoch108.pth
    ${GIT_PROJ}/results/mnist/train/lx_clements_pdb_784x784_ep150_bs512_lr0.000038_seed5/210525_022925/model_epoch062.pth
)
      
      
      
      
      

#------------------------------------------------------------------------------#
#                           Testing Configuration                             #
#------------------------------------------------------------------------------#

DEVICE=cpu   # Hardware device selection (cuda/cpu)
BS=512        # Batch size
VF=0.1       # Validation fraction

#------------------------------------------------------------------------------#
#                           Output Directory Setup                             #
#------------------------------------------------------------------------------#

# Generate timestamp for unique directory naming
DATE=$(date +%d%m%y_%H%M%S)

MODEL=clements_pdb
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

NI=784 # Number of inputs

#------------------------------------------------------------------------------#
#                        Virtual Environment Activation                         #
#------------------------------------------------------------------------------#
# Activate the LuxIA virtual environment
#
if ! source ${SRC_FOLDER}/bash_scripts/activate_luxia.sh; then
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

