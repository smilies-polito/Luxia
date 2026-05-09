#!/bin/bash
################################################################################
#                              System Setup Script                               #
################################################################################
# Name: system_setup.sh
#
# Description: 
#   This script verifies and installs required dependencies for LuxIA.
#   Currently supports only Debian-based systems.
#
# Dependencies Checked/Installed:
#   - apt (package manager)
#   - python3
#   - python-is-python3 (ensures python3 is default)
#   - python3-venv (for virtual environments)
#   - python3-pip (package installer)
#
# Author: Tzamn Melendez Carmona
################################################################################

#------------------------------------------------------------------------------#
#                           Debian System Verification                          #
#------------------------------------------------------------------------------#
# Verify the system is Debian-based by checking for apt
if ! command -v apt >/dev/null 2>&1; then
    echo "This script is only for debian-based systems."
    exit 1
fi

#------------------------------------------------------------------------------#
#                           Python3 Installation Check                          #
#------------------------------------------------------------------------------#
# Verify python3 installation
if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is not installed. Installing..."
    sudo apt update
    sudo apt install -y python3
else
    echo "python3 is already installed."
fi

#------------------------------------------------------------------------------#
#                        Python3 Default Configuration                          #
#------------------------------------------------------------------------------#
# Ensure python3 is the default python version
if ! command -v python >/dev/null 2>&1; then
    echo "python-is-python3 is not installed. Installing..."
    sudo apt update
    sudo apt install -y python-is-python3
else
    echo "python-is-python3 is already installed."
fi

#------------------------------------------------------------------------------#
#                    Virtual Environment Support Installation                   #
#------------------------------------------------------------------------------#
# Install python3-venv for creating virtual environments
# Required for: PhotonNN, PhotonTorch, Neurophox, Neuroptica
if ! dpkg -s python3-venv >/dev/null 2>&1; then
    echo "python3-venv is not installed. Installing..."
    sudo apt update
    sudo apt install -y python3-venv
else
    echo "python3-venv is already installed."
fi

#------------------------------------------------------------------------------#
#                           Pip Installation Check                             #
#------------------------------------------------------------------------------#
# Verify pip installation for managing Python packages
if ! command -v pip >/dev/null 2>&1; then
    echo "pip is not installed. Installing..."
    sudo apt update
    sudo apt install -y python3-pip
else
    echo "pip is already installed."
fi
