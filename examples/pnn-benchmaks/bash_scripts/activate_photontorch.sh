#!/bin/bash
################################################################################
#                        Photontorch Environment Activation Script                     #
################################################################################
# Description:
#   This script creates and activates a virtual environment for Photontorch (Photonic
#   Neural Network framework). It handles environment validation, creation,
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

SIM_ENV=photontorch

#------------------------------------------------------------------------------#
#                      Virtual Environment Management                           #
#------------------------------------------------------------------------------#
# Create virtual environment if it doesn't exist
if [ ! -d $GIT_PROJ_VENV/$SIM_ENV ]; then
    echo "Creating virtual environment for $SIM_ENV on $GIT_PROJ_VENV"
    python3 -m venv $GIT_PROJ_VENV/$SIM_ENV
    source $GIT_PROJ_VENV/$SIM_ENV/bin/activate
    
    # Update pip and install required packages
    pip3 install --upgrade pip
    pip3 cache purge
    pip3 install photontorch
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

check_and_modify_photontorch() {
    echo "Calling modify_photontorch function."
    # Retrieve the installed version of PyTorch using Python
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
    # Check if torch was successfully imported
    if [[ -z "$TORCH_VERSION" ]]; then 
        echo "PyTorch is not installed or cannot be found." exit 1 
    fi

    # Define the minimum required version
    MAX_VERSION="1.9"

    # Compare the versions
    if [[ $(printf '%s\n' "$MAX_VERSION" "$TORCH_VERSION" | sort -V | head -n1) == "$MAX_VERSION" ]] && [[ "$TORCH_VERSION" != "$MAX_VERSION" ]]; then 
        echo "PyTorch version $TORCH_VERSION is newer than the max version required $MAX_VERSION." 
        echo "photontorch has some deprecated functions and may not work as expected." 
        read -p "Do you want to modify the code to use the newer version of PyTorch? [y/n]: " response

        if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            # Get the Python major version to locate the correct site-packages path
            PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)

            # Define the file path
            FILE="$GIT_PROJ/.venv/photontorch/lib/python${PYTHON_VERSION}/site-packages/photontorch/networks/network.py"
            # Check if the file exists
            if [[ ! -f "$FILE" ]]; then 
                echo "File $FILE not found." 
                exit 1 
            fi

            # Define the line number and new content
            LINE_NUMBER=522 NEW_CONTENT="            x= torch.linalg.solve(P, Cmlmc)"

            # Display 5 lines before and after the line to be modified
            echo "Current context around line $LINE_NUMBER:"
            sed -n "$((LINE_NUMBER - 5)),$((LINE_NUMBER + 5))p" "$FILE"

            # Show the line that will be modified
            echo "Modifying line $LINE_NUMBER in $FILE." 
            echo "Replacing: $(sed -n "${LINE_NUMBER}p" "$FILE")" 
            echo "With: $NEW_CONTENT"

            # Use sed to replace the specific line
            sed -i "${LINE_NUMBER}s/.*/$NEW_CONTENT/" "$FILE"

            # Display the file after the modification
            echo "Updated context around line $LINE_NUMBER:" 
            sed -n "$((LINE_NUMBER - 5)),$((LINE_NUMBER + 5))p" "$FILE"
            echo "Modification complete." 
        else 
            echo "No modifications were made." 
        fi 
    else 
        echo "PyTorch version $TORCH_VERSION is newer than $MAX_VERSION." 
    fi 
}

#------------------------------------------------------------------------------#
#                        Training Function Definition                           #
#------------------------------------------------------------------------------#
# Function to execute training with error handling
photontorch_exec() {
    EXEC="$*"
    echo "Executing { "$EXEC" }"
    if ! $EXEC; then
        echo "ERROR: FAILED WHEN TRYING TO EXECUTE: \"${EXEC}\""
        # Ask the user if the error is due to deprecated functions
        read -p "Are you experince errors due to torch functions that have been deprecated [y/n]: " response
        if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            echo "Checking and modifying photontorch ..."
            check_and_modify_photontorch
            echo "Retrying the training ..."
            photontorch_train $EXEC
        else
            echo "An unknown error occurred ... =( please check the error message."
        fi
    else
        echo "Training successful"
    fi
}
