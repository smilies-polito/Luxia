#!/bin/bash
################################################################################
#                        Neurophox Environment Activation Script                #
################################################################################
# Description:
#   This script creates and activates a virtual environment for Neurophox 
#   It handles environment validation, creation,
#   package installation, and GPU detection.
#
# Dependencies:
#   - Python 3
#   - Python-venv
#   - Pip3
#   - Launch system_setup.sh to install and check dependencies
#
# Author: Tzamn Melendez Carmona
################################################################################

#------------------------------------------------------------------------------#
#                        Project Path Configuration                            #
#------------------------------------------------------------------------------#
# Set project directory paths as environment variables
if [ -z "$GIT_PROJ" ]; then
    export GIT_PROJ=$(git rev-parse --show-toplevel)
    echo "GIT_CURRENT_PROJ is not set, please check if you are in a git repository."
fi

# Verify GIT_PROJ_VENV is set
if [ -z "$GIT_PROJ_VENV" ]; then
    export GIT_PROJ_VENV=$GIT_PROJ/.venv
    echo "GIT_PROJ_VENV is not set. Setting it to $GIT_PROJ_VENV"
fi

SIM_ENV=neurophox

#------------------------------------------------------------------------------#
#                      Virtual Environment Management                           #
#------------------------------------------------------------------------------#
# Create virtual environment if it doesn't exist
if [ ! -d $GIT_PROJ_VENV/$SIM_ENV ]; then
    echo "Creating virtual environment for $SIM_ENV on $GIT_PROJ_VENV"
    python3 -m venv $GIT_PROJ_VENV/$SIM_ENV
    source $GIT_PROJ_VENV/$SIM_ENV/bin/activate

    pip3 cache purge
    # Installing neurophox using pip and 
    pip3 install neurophox -c $GIT_PROJ/neurophox_constrains.txt
    pip3 install -r $GIT_PROJ/requirements.txt
else
    echo "Virtual environment for $SIM_ENV already exists"
    source $GIT_PROJ_VENV/${SIM_ENV}/bin/activate
fi

#------------------------------------------------------------------------------#
#                            GPU Detection                                      #
#------------------------------------------------------------------------------#
# Check for GPU availability and install GPU monitoring tools if available
if command -v nvidia-smi &> /dev/null; then
    log "INFO" "GPU detected (via nvidia-smi)"
    nvidia-smi || log "WARN" "nvidia-smi command failed, but command exists"
    pip install pynvml || log "WARN" "Failed to install pynvml"
else
    # Fallback using Python (e.g., torch)
    if python3 -c "import torch; exit(0) if torch.cuda.is_available() else exit(1)"; then
        log "INFO" "GPU detected (via torch)"
        pip install pynvml || log "WARN" "Failed to install pynvml"
    else
        log "INFO" "No GPU detected"
    fi
fi

#------------------------------------------------------------------------------#
#                        Environment Verification                               #
#------------------------------------------------------------------------------#
# Display current virtual environment
echo " [***] Current virtual environment: $VIRTUAL_ENV"

#------------------------------------------------------------------------------#
#                        Training Function Definition                           #
#------------------------------------------------------------------------------#
# Function to execute training with error handling
neurophox_exec() {
    EXEC="$*"
    echo "Executing { "$EXEC" }"
    if ! $EXEC; then
        echo "ERROR: FAILED WHEN TRYING TO EXECUTE: \"${EXEC} \""
        echo "Training failed"
    else
        echo "Training successful"
    fi
}

