#!/bin/bash
################################################################################
#                     Luxia Digits Training Script (Clements)              #
################################################################################
# File: train_digits_sota_nx.sh
# Description: 
# Runs a full Clements accelerator training on the digits dataset with the
# specified number of parallel networks and mesh size.
#
# Usage:
#   bash train_digits_sota_nx.sh
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

#------------------------------------------------------------------------------#
#                           Model Architecture Setup                           #
#------------------------------------------------------------------------------#

NN=2  # Number of parallel networks

#------------------------------------------------------------------------------#
#                           Training Configuration                             #
#------------------------------------------------------------------------------#

DEVICE=cuda     # Hardware device selection (cuda/cpu)
EPOCH=150   # Number of training epochs
BS=512      # Batch size
VF=0.1       # Validation fraction

#------------------------------------------------------------------------------#
#                           Output Directory Setup                             #
#------------------------------------------------------------------------------#

MODEL=nx_clements
RESULTS_DIR="${GIT_PROJ}/results/${DS}/train"
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
NL=64  # Number of layers

#------------------------------------------------------------------------------#
#                            Training Execution                                #
#------------------------------------------------------------------------------#
SAVE="--save_model"

LR=0.0005
#------------------------------------------------------------------------------#
#                        Virtual Environment Activation                         #
#------------------------------------------------------------------------------#
# Activate the LuxIA virtual environment
#
if ! source ${SRC_FOLDER}/bash_scripts/activate_neurophox.sh; then
    echo "Failed to activate virtual environment. Exiting."
    exit 1
fi

# Generate learning rates properly formatted for Bash
# lr=$($PY -c "import numpy as np; print(' '.join(map(str, np.logspace(-0.9, -7, num=40))))")

# Iterate through diverse seed values
seed_values=(1 2 3 4 5)
for SEED in ${seed_values[@]}; do
    DATE=$(date +%d%m%y_%H%M%S)
    NAME="${MODEL}_${NI}x${NL}_ep${EPOCH}_bs${BS}_lr${LR}_seed${SEED}"
    DIR="${RESULTS_DIR}/${NAME}/${DATE}"
    LOG_DIR="${DIR}/logs"
    # Create necessary directories
    mkdir -p "$LOG_DIR"

    # Construct the training command with all parameters
    EXEC="${PY} ${GIT_PROJ}/train.py --model=$MODEL --n_inputs=$NI --n_layers=$NL --n_networks=$NN \
        --dataset=$DS --epoch=$EPOCH --batch_size=$BS --val_frac=$VF --lr=$LR \
        --exp_dir=$DIR --seed=$SEED --device=$DEVICE\
        $SAVE $USE_WANDB --wandb_key=$WANDB_KEY"

    # Execute training and log output
    echo " *  - - - - Training $MODEL of Size $NI on $DS dataset with LR=$LR - - - - * *"
    neurophox_exec $EXEC | tee $LOG_DIR/lr${LR}_bs${BS}_${DATE}.log
done

