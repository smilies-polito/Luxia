#------------------------------------------------------------------------------#
#                        Project Path Configuration                            #
#------------------------------------------------------------------------------#
# Set project directory paths as environment variables
if [ -z "$GIT_PROJ" ]; then
    export GIT_PROJ=$(git rev-parse --show-toplevel)
    echo "GIT_CURRENT_PROJ is not set, please check if you are in a git repository."
fi
bash ${GIT_PROJ}/bash_train_scripts/digits_bs_scenario/luxia_bs_cpu.sh
bash ${GIT_PROJ}/bash_train_scripts/digits_bs_scenario/luxia_bs_gpu.sh
bash ${GIT_PROJ}/bash_train_scripts/digits_bs_scenario/neurophox_bs_cpu.sh
bash ${GIT_PROJ}/bash_train_scripts/digits_bs_scenario/neurophox_bs_gpu.sh
bash ${GIT_PROJ}/bash_train_scripts/digits_bs_scenario/neuroptica_bs.sh
bash ${GIT_PROJ}/bash_train_scripts/digits_bs_scenario/photontorch_bs_cpu.sh
bash ${GIT_PROJ}/bash_train_scripts/digits_bs_scenario/photontorch_bs_gpu.sh

