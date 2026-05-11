#------------------------------------------------------------------------------#
#                        Project Path Configuration                            #
#------------------------------------------------------------------------------#
# Set project directory paths as environment variables
if [ -z "$GIT_PROJ" ]; then
    export GIT_PROJ=$(git rev-parse --show-toplevel)
    echo "GIT_CURRENT_PROJ is not set, please check if you are in a git repository."
fi
SRC_FOLDER="${GIT_PROJ}/examples/pnn-benchmarks"

bash ${SRC_FOLDER}/bash_train_scripts/digits_bs_scenario/luxia_bs_cpu.sh
bash ${SRC_FOLDER}/bash_train_scripts/digits_bs_scenario/luxia_bs_gpu.sh
bash ${SRC_FOLDER}/bash_train_scripts/digits_bs_scenario/neurophox_bs_cpu.sh
bash ${SRC_FOLDER}/bash_train_scripts/digits_bs_scenario/neurophox_bs_gpu.sh
bash ${SRC_FOLDER}/bash_train_scripts/digits_bs_scenario/neuroptica_bs.sh
bash ${SRC_FOLDER}/bash_train_scripts/digits_bs_scenario/photontorch_bs_cpu.sh
bash ${SRC_FOLDER}/bash_train_scripts/digits_bs_scenario/photontorch_bs_gpu.sh

