#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status
set -o pipefail  # Catch errors in pipelines
set -u  # Treat unset variables as an error

#------------------------------------------------------------------------------#
#                        Helper Functions                                      #
#------------------------------------------------------------------------------#

# Function to print color-coded logs
log() {
    local level="$1"
    local message="$2"
    local color=""
    local prefix=""
    
    case "$level" in
        "INFO")  color="\033[0;32m"; prefix="[*]" ;;  # Green
        "WARN")  color="\033[0;33m"; prefix="[!]" ;;  # Yellow
        "ERROR") color="\033[0;31m"; prefix="[!]" ;;  # Red
        *)       color="\033[0m";    prefix="[*]" ;;  # Default
    esac
    
    echo -e "${color}${prefix} ${message}\033[0m"
}

# Function to check if we're inside the ParentGit repository
is_in_parent_git_repo() {
    local current_dir
    current_dir=$(pwd)
    while [[ "$current_dir" != "/" ]]; do
        if [[ -d "$current_dir/.git" && -d "$current_dir/luxia" ]]; then
            echo "$current_dir"
            return 0
        fi
        current_dir=$(dirname "$current_dir")
    done
    return 1
}

# Function to check if luxia is installed
is_luxia_installed() {
    log "INFO" "Checking if luxia is installed in the system..."
    if pip show luxia &> /dev/null; then
        log "INFO" "luxia is installed."
        return 0
    else
        log "WARN" "luxia is NOT installed."
        return 1
    fi
}

# Function to create a virtual environment if it doesn't exist
create_virtualenv() {
    local env_path="$1"
    if [[ ! -d "$env_path" ]]; then
        log "INFO" "Creating virtual environment at $env_path"
        python3 -m venv "$env_path" || {
            log "ERROR" "Failed to create virtual environment."
            return 1
        }
    else
        log "INFO" "Virtual environment already exists at $env_path"
    fi
}

# Function to install luxia package
install_luxia() {
    local install_path="$1"
    log "INFO" "Installing luxia from $install_path"
    pip install -e "$install_path" || {
        log "ERROR" "Failed to install luxia. Attempting to continue..."
    }
}

# Function to install additional requirements
install_requirements() {
    local req_file="$1"
    if [[ -f "$req_file" ]]; then
        log "INFO" "Installing additional requirements from $req_file"
        pip install -r "$req_file" || {
            log "ERROR" "Failed to install requirements from $req_file"
            return 1
        }
    else
        log "WARN" "No requirements.txt found at $req_file"
    fi
}

# Function to detect GPU availability
check_gpu() {
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
}

# Function to execute a training command with error handling
luxia_exec() {
    local exec_command="$*"
    log "INFO" "Executing: $exec_command"
    if ! eval "$exec_command"; then
        log "ERROR" "Failed to execute: \"$exec_command\""
        return 1
    else
        log "INFO" "Execution successful"
    fi
}

#------------------------------------------------------------------------------#
#                        Main Script Logic                                     #
#------------------------------------------------------------------------------#

log "INFO" "Setting up environment for your project..."

# Get repository paths
GIT_ROOT=$(is_in_parent_git_repo || true)
GIT_PROJ=$(git rev-parse --show-toplevel || {
    log "ERROR" "Failed to detect git project directory. Are you in a git repository?"
    exit 1
})

if [[ -n "${GIT_ROOT:-}" ]]; then
    log "INFO" "Project is inside ParentGit repository at: $GIT_ROOT"
    
    if is_luxia_installed; then
        log "INFO" "luxia is already installed in the system."
        log "INFO" "Ensuring it's installed from the ParentGit repo version..."
        install_luxia "$GIT_ROOT"
        log "INFO" "Using system's main virtual environment."
    else
        log "INFO" "luxia is not installed in the system."
        log "INFO" "Creating a shared virtual environment at the ParentGit repo level..."
        
        create_virtualenv "$GIT_ROOT/.venv" || exit 1
        source "$GIT_ROOT/.venv/bin/activate" || {
            log "ERROR" "Failed to activate virtual environment"
            exit 1
        }
        
        pip install --upgrade pip
        pip cache purge
        install_luxia "$GIT_ROOT"
        
        log "INFO" "Virtual environment setup complete. To activate it later, run:"
        log "INFO" "    source \"$GIT_ROOT/.venv/bin/activate\""
    fi
else
    log "INFO" "Project is not inside the ParentGit repository."
    log "INFO" "Creating a local virtual environment..."
    
    create_virtualenv "$GIT_PROJ/.venv" || exit 1
    source "$GIT_PROJ/.venv/bin/activate" || {
        log "ERROR" "Failed to activate virtual environment"
        exit 1
    }
    
    pip install --upgrade pip
    pip cache purge
    
    log "INFO" "Installing luxia from remote repository..."
    pip install git+https://github.com/smilies-polito/Luxia.git || {
        log "ERROR" "Failed to install luxia from remote repository"
        exit 1
    }
    
    log "INFO" "Local virtual environment setup complete. To activate it later, run:"
    log "INFO" "    source .venv/bin/activate"
fi

install_requirements "$GIT_PROJ/requirements.txt"
check_gpu

#------------------------------------------------------------------------------#
#                        Environment Verification                              #
#------------------------------------------------------------------------------#

echo ""
log "INFO" "Current virtual environment: ${VIRTUAL_ENV:-None}"

# Verify the installation
if pip show luxia &> /dev/null; then
    LUXIA_VERSION=$(pip show luxia | grep Version | awk '{print $2}')
    log "INFO" "LuxIA version $LUXIA_VERSION is successfully installed"
else
    log "ERROR" "LuxIA installation verification failed!"
    exit 1
fi

log "INFO" "Done! 🎉"
