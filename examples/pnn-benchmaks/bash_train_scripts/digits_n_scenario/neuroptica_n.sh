#!/bin/bash
################################################################################
#                     Neuroptica Digits Training Script (Clements)              #
################################################################################
# File: neuroptica_n.sh
# Description:
#   Runs a Clements model with different mesh sizes on the Digits dataset using
#   the Neuroptica simulator. Trains the model for a single epoch and logs output.
#
# Usage:
#   bash neuroptica_n.sh
#
# Environment Variables Required:
#   PATH_PROJ: Base GitHub directory path (source system_setup.sh to set)
#
# Author: Tzamn Melendez Carmona
################################################################################
################################################################################
#------------------------------------------------------------------------------#
#                        Project Path Configuration                            #
#------------------------------------------------------------------------------#
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
DS=digits

#------------------------------------------------------------------------------#
#                           Model Architecture Setup                           #
#------------------------------------------------------------------------------#

NN=2  # Number of parallel networks

#------------------------------------------------------------------------------#
#                           Training Configuration                             #
#------------------------------------------------------------------------------#

DEVICE=cpu   # Hardware device selection (cuda/cpu)
EPOCH=1      # Number of training epochs
BS=128       # Batch size
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
if ! source ${GIT_PROJ}/bash_scripts/activate_neuroptica.sh; then
    echo "Failed to activate virtual environment. Exiting."
    exit 1
fi

# Generate NI values from 100 to 900 (5 evenly spaced values)
NI_VALUES=$($PY -c "import numpy; print(' '.join(map(str, map(int, numpy.linspace(100, 900, 5)))))")

for NI in $NI_VALUES; do
    NL=$NI  # Number of layers matches the number of inputs
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
