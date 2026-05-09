#!/bin/bash
# Set the Git root directory path
if [ -z "$GIT_PROJ" ]; then
    export GIT_PROJ=$(git rev-parse --show-toplevel)
    echo "setting GIT Current Project to $GIT_PROJ"
fi

# Exit on error
set -e

# bash ${GIT_PROJ}/bash_test_scripts/sota/digits_sota_lx_clements.sh
bash ${GIT_PROJ}/bash_test_scripts/sota/digits_sota_nx_clements.sh
bash ${GIT_PROJ}/bash_test_scripts/sota/digits_sota_pt_clements.sh

