#!/bin/bash
################################################################################
#         LuxIA Benchmark Script (isolated per mode, CPU + GPU)
################################################################################

if [ -z "$GIT_PROJ" ]; then
    export GIT_PROJ=$(git rev-parse --show-toplevel)
    echo "Setting GIT_PROJ to $GIT_PROJ"
fi

set -u
set -o pipefail

# -------------------------------------------------------------------
# Simple logger
# -------------------------------------------------------------------
log() {
    local level="$1"
    shift
    echo "[$level] $*"
}

# -------------------------------------------------------------------
# Failure tracking
# -------------------------------------------------------------------
FAILED_RUNS=()
SUCCESS_COUNT=0
FAIL_COUNT=0

record_failure() {
    FAILED_RUNS+=("$1")
    FAIL_COUNT=$((FAIL_COUNT + 1))
}

record_success() {
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
}

# -------------------------------------------------------------------
# Function to detect GPU availability
# Returns:
#   0 -> GPU available
#   1 -> GPU not available
# -------------------------------------------------------------------
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        log "INFO" "GPU detected (via nvidia-smi)"
        nvidia-smi || log "WARN" "nvidia-smi command failed, but command exists"
        pip install pynvml || log "WARN" "Failed to install pynvml"
        return 0
    else
        if python3 -c "import torch; exit(0) if torch.cuda.is_available() else exit(1)"; then
            log "INFO" "GPU detected (via torch)"
            pip install pynvml || log "WARN" "Failed to install pynvml"
            return 0
        else
            log "INFO" "No GPU detected"
            return 1
        fi
    fi
}

# -------------------------------------------------------------------
# Safe command runner
# Runs a benchmark, logs output, continues on failure
# -------------------------------------------------------------------
run_benchmark() {
    local device="$1"
    local exec_cmd="$2"
    local log_file="$3"
    local label="$4"

    log "INFO" "Executing: ${label}"

    set +e
    if [ "${device}" = "cuda" ]; then
        CUDA_VISIBLE_DEVICES=${GPU_ID} luxia_exec ${exec_cmd} 2>&1 | tee "${log_file}"
        cmd_status=${PIPESTATUS[0]}
    else
        luxia_exec ${exec_cmd} 2>&1 | tee "${log_file}"
        cmd_status=${PIPESTATUS[0]}
    fi
    set -e

    if [ ${cmd_status} -ne 0 ]; then
        log "ERROR" "Failed: ${label}"
        record_failure "${label}"
    else
        log "INFO" "Execution successful: ${label}"
        record_success
    fi

    return 0
}

# -------------------------------------------------------------------
# Python
# -------------------------------------------------------------------
PY=python3

# -------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------
DS=digits
VF=0.1

# -------------------------------------------------------------------
# Devices to evaluate
# -------------------------------------------------------------------
DEVICES=("cpu")

# -------------------------------------------------------------------
# Parallel photonic networks
# -------------------------------------------------------------------
NN=1

# -------------------------------------------------------------------
# Models to evaluate
# -------------------------------------------------------------------
MODELS=(
    "lx_clements"
)

# -------------------------------------------------------------------
# Input sizes
# -------------------------------------------------------------------
INPUT_SIZES=(
    100
    300
    500
    700
    900
)

# -------------------------------------------------------------------
# Batch sizes
# -------------------------------------------------------------------
BATCH_SIZES=(
    64
    128
    256
    512
)

# -------------------------------------------------------------------
# Window sizes
# -------------------------------------------------------------------
WINDOW_SIZES=(
    1
    2
    4
    8
    16
    32
    64
)

# -------------------------------------------------------------------
# Output directory
# -------------------------------------------------------------------
RESULTS_DIR="${GIT_PROJ}/results/${DS}/benchmark"
mkdir -p "$RESULTS_DIR"

# -------------------------------------------------------------------
# Activate environment
# -------------------------------------------------------------------
if ! source "${GIT_PROJ}/bash_scripts/activate_luxia.sh"; then
    echo "Failed to activate virtual environment"
    exit 1
fi

# from here on, allow failures inside benchmark runs only through wrapper
set -e

# -------------------------------------------------------------------
# Detect GPU and extend devices list
# -------------------------------------------------------------------
if check_gpu; then
    DEVICES+=("cuda")
fi

log "INFO" "Devices to benchmark: ${DEVICES[*]}"

# -------------------------------------------------------------------
# Optional GPU selection
# -------------------------------------------------------------------
GPU_ID=${GPU_ID:-0}

# -------------------------------------------------------------------
# Run experiments
# -------------------------------------------------------------------
for DEVICE in "${DEVICES[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for NI in "${INPUT_SIZES[@]}"; do
            NL=$((NI))

            for BS in "${BATCH_SIZES[@]}"; do
                DATE=$(date +%d%m%y_%H%M%S)

                BASE_NAME="${MODEL}_${NI}x${NL}_bs${BS}_${DEVICE}"
                BASE_DIR="${RESULTS_DIR}/${BASE_NAME}/${DATE}"
                LOG_DIR="${BASE_DIR}/logs"
                mkdir -p "$LOG_DIR"

                echo ""
                echo "===================================================="
                echo "Running benchmark group:"
                echo "DEVICE:    ${DEVICE}"
                echo "MODEL:     ${MODEL}"
                echo "INPUTS:    ${NI}  (Layers: ${NL})"
                echo "BATCH:     ${BS}"
                echo "DIR:       ${BASE_DIR}"
                echo "===================================================="

                # ------------------------------------------------
                # Run FULL once
                # ------------------------------------------------
                MODE="full"
                FULL_DIR="${BASE_DIR}/full"
                mkdir -p "$FULL_DIR"

                EXEC="${PY} ${GIT_PROJ}/benchmark_window.py \
                    --mode=${MODE} \
                    --model=${MODEL} \
                    --dataset=${DS} \
                    --n_inputs=${NI} \
                    --n_layers=${NL} \
                    --n_networks=${NN} \
                    --batch_size=${BS} \
                    --window_size=1 \
                    --val_frac=${VF} \
                    --device=${DEVICE} \
                    --exp_dir=${FULL_DIR}"

                echo ""
                echo "----------------------------------------------------"
                echo "Device: ${DEVICE} | Mode: ${MODE}"
                echo "----------------------------------------------------"

                run_benchmark \
                    "${DEVICE}" \
                    "${EXEC}" \
                    "${LOG_DIR}/benchmark_${MODE}_${MODEL}_${NI}x${NL}_bs${BS}_${DEVICE}_${DATE}.log" \
                    "${PY} ${GIT_PROJ}/benchmark_window.py --mode=${MODE} --model=${MODEL} --dataset=${DS} --n_inputs=${NI} --n_layers=${NL} --n_networks=${NN} --batch_size=${BS} --window_size=1 --val_frac=${VF} --device=${DEVICE} --exp_dir=${FULL_DIR}"

                # ------------------------------------------------
                # Run SLICE for each window size
                # ------------------------------------------------
                for NW in "${WINDOW_SIZES[@]}"; do
                    if (( NW > NL )); then
                        log "WARN" "Skipping invalid config: window_size=${NW} > n_layers=${NL}"
                        continue
                    fi

                    MODE="slice"
                    SLICE_DIR="${BASE_DIR}/slice_w${NW}"
                    mkdir -p "$SLICE_DIR"

                    EXEC="${PY} ${GIT_PROJ}/benchmark_window.py \
                        --mode=${MODE} \
                        --model=${MODEL} \
                        --dataset=${DS} \
                        --n_inputs=${NI} \
                        --n_layers=${NL} \
                        --n_networks=${NN} \
                        --batch_size=${BS} \
                        --window_size=${NW} \
                        --val_frac=${VF} \
                        --device=${DEVICE} \
                        --exp_dir=${SLICE_DIR}"

                    echo ""
                    echo "----------------------------------------------------"
                    echo "Device: ${DEVICE} | Mode: ${MODE} | Window: ${NW}"
                    echo "----------------------------------------------------"

                    run_benchmark \
                        "${DEVICE}" \
                        "${EXEC}" \
                        "${LOG_DIR}/benchmark_${MODE}_${MODEL}_${NI}x${NL}_bs${BS}_w${NW}_${DEVICE}_${DATE}.log" \
                        "${PY} ${GIT_PROJ}/benchmark_window.py --mode=${MODE} --model=${MODEL} --dataset=${DS} --n_inputs=${NI} --n_layers=${NL} --n_networks=${NN} --batch_size=${BS} --window_size=${NW} --val_frac=${VF} --device=${DEVICE} --exp_dir=${SLICE_DIR}"
                done
            done
        done
    done
done

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------
echo ""
echo "===================================================="
echo "Benchmark summary"
echo "===================================================="
echo "Successful runs: ${SUCCESS_COUNT}"
echo "Failed runs:     ${FAIL_COUNT}"

if [ ${FAIL_COUNT} -gt 0 ]; then
    echo ""
    echo "Failed configurations:"
    for fail in "${FAILED_RUNS[@]}"; do
        echo " - ${fail}"
    done
fi
