#!/bin/bash
################################################################################
#                     Luxia Iris Training Script (Clements Bell)              #
################################################################################
# File: lx_iris_clements_bell.sh
# Description:
#  This script trains the lx_clements model on the Iris dataset
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
DS=iris

#------------------------------------------------------------------------------#
#                           Model Architecture Setup                           #
#------------------------------------------------------------------------------#

NN=2  # Number of parallel networks

#------------------------------------------------------------------------------#
#                           Training Configuration                             #
#------------------------------------------------------------------------------#

DEVICE=cuda # Hardware device selection (cuda/cpu)
EPOCH=200   # Number of training epochs
BS=16       # Batch size
VF=0.1      # Validation fraction

#------------------------------------------------------------------------------#
#                           Output Directory Setup                             #
#------------------------------------------------------------------------------#

MODEL=lx_clements_bell
RESULTS_DIR="${GIT_PROJ}/results/${DS}/train"
# Ensure base results directory exists
mkdir -p "$RESULTS_DIR"

# Enable WandB logging
USE_WANDB="--use_wandb"
WANDB_KEY=""
WANDB_PROJ="${MODEL}_${DS}"

# USE_WANDB=""
# WANDB_KEY=""

#------------------------------------------------------------------------------#
#                           Mesh Size Configuration                            #
#------------------------------------------------------------------------------#

NI=4  # Number of inputs
NL=4  # Number of layers

#------------------------------------------------------------------------------#
#                            Training Execution                                #
#------------------------------------------------------------------------------#
SAVE="--save_model"

LR=0.001

#------------------------------------------------------------------------------#
#                        Virtual Environment Activation                         #
#------------------------------------------------------------------------------#
# Activate the LuxIA virtual environment
#
if ! source ${SRC_FOLDER}/bash_scripts/activate_luxia.sh; then
    echo "Failed to activate virtual environment. Exiting."
    exit 1
fi

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
    luxia_exec $EXEC | tee $LOG_DIR/lr${LR}_bs${BS}_${DATE}.log
done

