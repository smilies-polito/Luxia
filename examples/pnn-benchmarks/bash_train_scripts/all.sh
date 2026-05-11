#------------------------------------------------------------------------------#
#                        Project Path Configuration                            #
#------------------------------------------------------------------------------#
# Set project directory paths as environment variables
if [ -z "$GIT_PROJ" ]; then
    export GIT_PROJ=$(git rev-parse --show-toplevel)
    echo "GIT_CURRENT_PROJ is not set, please check if you are in a git repository."
fi
SRC_FOLDER="${GIT_PROJ}/examples/pnn-benchmarks"

bash ${SRC_FOLDER}/bash_train_scripts/digits_all_tools/train_all.sh
bash ${SRC_FOLDER}/bash_train_scripts/digits_bs_scenario/all_bs.sh
bash ${SRC_FOLDER}/bash_train_scripts/digits_n_scenario/all_n.sh
bash ${SRC_FOLDER}/bash_train_scripts/train_luxia/lx_all.sh
