#!/bin/bash
################################################################################
#      LuxIA Numerical Error Comparison Script (ref vs float32 run variants)
#
# For each input size, this runs:
#   - explicit windowed cases from WINDOW_SIZES
#   - one additional transfer-matrix / full-matrix comparison case
#     encoded as: is_matmul=true and window_size=n_layers
#
# The Python script always uses the 32-bit transfer-matrix model as reference.
# Each launched run is the float32 implementation being compared against that
# reference.
################################################################################

if [ -z "$GIT_PROJ" ]; then
    export GIT_PROJ=$(git rev-parse --show-toplevel)
    echo "Setting GIT_PROJ to $GIT_PROJ"
fi

set -e

PY=python3

DS=digits
VF=0.1
DEVICE=cpu

NN=1

MODELS=(
    "lx_clements"
)

INPUT_SIZES=(
    64
    128
    256
    512
)

BATCH_SIZES=(
    512
)

WINDOW_SIZES=(
    1
    2
    4
    8
    16
    32
)

SEEDS=(
    0
    1
    2
    3
    4
)

RESULTS_DIR="${GIT_PROJ}/results/${DS}/numerical_error"
mkdir -p "$RESULTS_DIR"

if ! source "${GIT_PROJ}/bash_scripts/activate_luxia.sh"; then
    echo "Failed to activate virtual environment"
    exit 1
fi

for MODEL in "${MODELS[@]}"; do
    for NI in "${INPUT_SIZES[@]}"; do
        NL=$((NI))

        for BS in "${BATCH_SIZES[@]}"; do
            for SEED in "${SEEDS[@]}"; do

                # -------------------------------------------------------------
                # 1) Windowed / sliced float32 runs
                # -------------------------------------------------------------
                for NW in "${WINDOW_SIZES[@]}"; do
                    if (( NW > NL )); then
                        echo "Skipping invalid config: window_size=${NW} > n_layers=${NL}"
                        continue
                    fi

                    DATE=$(date +%d%m%y_%H%M%S)

                    NAME="${MODEL}_${NI}x${NL}_bs${BS}_w${NW}_seed${SEED}"
                    DIR="${RESULTS_DIR}/${NAME}/${DATE}"
                    LOG_DIR="${DIR}/logs"

                    mkdir -p "$LOG_DIR"

                    EXEC="${PY} ${GIT_PROJ}/numerical_window_error.py \
                        --model=${MODEL} \
                        --dataset=${DS} \
                        --n_inputs=${NI} \
                        --n_layers=${NL} \
                        --n_networks=${NN} \
                        --batch_size=${BS} \
                        --window_size=${NW} \
                        --seed=${SEED} \
                        --val_frac=${VF} \
                        --device=${DEVICE} \
                        --exp_dir=${DIR}"

                    echo ""
                    echo "===================================================="
                    echo "Running numerical error (windowed float32 vs 32-bit TM ref):"
                    echo "MODEL:       ${MODEL}"
                    echo "INPUTS:      ${NI}  (Layers: ${NL})"
                    echo "BATCH:       ${BS}"
                    echo "WINDOW:      ${NW}"
                    echo "IS_MATMUL:   False"
                    echo "SEED:        ${SEED}"
                    echo "N_WINDOWS:   $(( (NL + NW - 1) / NW ))"
                    echo "RUN_LABEL:   windowed"
                    echo "===================================================="

                    luxia_exec $EXEC | tee "${LOG_DIR}/numerical_error_${MODEL}_${NI}x${NL}_bs${BS}_w${NW}_seed${SEED}_${DATE}.log"
                done

            done
        done
    done
done
