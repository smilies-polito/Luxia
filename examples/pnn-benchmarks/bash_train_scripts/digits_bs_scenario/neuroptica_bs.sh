#!/bin/bash
################################################################################
#                     Luxia Digits Training Script (Clements)              #
################################################################################
# File: neuroptica_bs.sh
# Description: 
# Runs a full Clements accelerator training on the digits dataset with the
# specified number of parallel networks and mesh size.
#
# Usage:
#   bash neuroptica_bs.sh
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

DEVICE=cpu   # Hardware device selection only cpu support for neuroptica
EPOCH=1     # Number of training epochs
BS=512      # Batch size
VF=0.1       # Validation fraction

#------------------------------------------------------------------------------#
#                           Output Directory Setup                             #
#------------------------------------------------------------------------------#

MODEL=neu_clements
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

NI=800  # Number of inputs
NL=800  # Number of layers

#------------------------------------------------------------------------------#
#                            Training Execution                                #
#------------------------------------------------------------------------------#
# SAVE="--save_model"

LR=0.0005
#------------------------------------------------------------------------------#
#                        Virtual Environment Activation                         #
#------------------------------------------------------------------------------#
# Activate the LuxIA virtual environment
#
if ! source ${SRC_FOLDER}/bash_scripts/activate_neuroptica.sh; then
    echo "Failed to activate virtual environment. Exiting."
    exit 1
fi

# run for different Batch Sizes starting from 128 to 1024
bs=$($PY -c "print(' '.join(map(str, [2**i for i in range(6, 11)])))")

for BS in $(echo $bs); do
    DATE=$(date +%d%m%y_%H%M%S)
    NAME="${MODEL}_${NI}x${NL}_ep${EPOCH}_bs${BS}_lr${LR}"
    DIR="${RESULTS_DIR}/${NAME}/${DATE}"
    LOG_DIR="${DIR}/logs"
    # Create necessary directories
    mkdir -p $LOG_DIR


    EXEC="${PY} ${GIT_PROJ}/train.py --model=$MODEL --n_inputs=$NI --n_layers=$NL --n_networks=$NN \
        --dataset=$DS --epoch=$EPOCH --batch_size=$BS --val_frac=$VF --lr=$LR \
        --exp_dir=$DIR --device=$DEVICE\
        $USE_WANDB --wandb_key=$WANDB_KEY"

    # Execute training and log output
    echo " *  - - - - Training $MODEL of Size $NI on $DS dataset with LR=$LR - - - - * *"
    neuroptica_exec $EXEC | tee $LOG_DIR/lr${LR}_bs${BS}_${DATE}.log
done

