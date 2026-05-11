# Description: Train all digits models
#------------------------------------------------------------------------------#
#                        Project Path Configuration                            #
#------------------------------------------------------------------------------#
# Set project directory paths as environment variables
if [ -z "$GIT_PROJ" ]; then
    export GIT_PROJ=$(git rev-parse --show-toplevel)
    echo "GIT_CURRENT_PROJ is not set, please check if you are in a git repository."
fi
SRC_FOLDER="${GIT_PROJ}/examples/pnn-benchmarks"

echo "Training all tools with the digits dataset and clements model"
bash ${SRC_FOLDER}/bash_train_scripts/digits_all_tools/train_digits_sota_lx.sh
bash ${SRC_FOLDER}/bash_train_scripts/digits_all_tools/train_digits_sota_neu.sh
bash ${SRC_FOLDER}/bash_train_scripts/digits_all_tools/train_digits_sota_nx.sh
bash ${SRC_FOLDER}/bash_train_scripts/digits_all_tools/train_digits_sota_pt.sh
