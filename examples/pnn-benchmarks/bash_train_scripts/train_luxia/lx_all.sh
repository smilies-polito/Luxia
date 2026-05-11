################################################################################
#------------------------------------------------------------------------------#
#                        Project Path Configuration                            #
#------------------------------------------------------------------------------#
# Set project directory paths as environment variables
if [ -z "$GIT_PROJ" ]; then
    export GIT_PROJ=$(git rev-parse --show-toplevel)
    echo "GIT_CURRENT_PROJ is not set, please check if you are in a git repository."
fi
SRC_FOLDER="${GIT_PROJ}/examples/pnn-benchmarks"

bash ${SRC_FOLDER}/bash_train_scripts/train_luxia/lx_digits_clements.sh 
bash ${SRC_FOLDER}/bash_train_scripts/train_luxia/lx_digits_clements_bell.sh
bash ${SRC_FOLDER}/bash_train_scripts/train_luxia/lx_digits_fldzhyan.sh
bash ${SRC_FOLDER}/bash_train_scripts/train_luxia/lx_digits_fldzhyan_bell.sh
bash ${SRC_FOLDER}/bash_train_scripts/train_luxia/lx_iris_clements.sh
bash ${SRC_FOLDER}/bash_train_scripts/train_luxia/lx_iris_clements_bell.sh
bash ${SRC_FOLDER}/bash_train_scripts/train_luxia/lx_iris_fldzhyan.sh
bash ${SRC_FOLDER}/bash_train_scripts/train_luxia/lx_iris_fldzhyan_bell.sh
bash ${SRC_FOLDER}/bash_train_scripts/train_luxia/lx_mnist_clements.sh
bash ${SRC_FOLDER}/bash_train_scripts/train_luxia/lx_mnist_clements_bell.sh
bash ${SRC_FOLDER}/bash_train_scripts/train_luxia/lx_mnist_fldzhyan.sh
bash ${SRC_FOLDER}/bash_train_scripts/train_luxia/lx_mnist_fldzhyan_bell.sh
bash ${SRC_FOLDER}/bash_train_scripts/train_luxia/lx_olivetti_clements.sh
bash ${SRC_FOLDER}/bash_train_scripts/train_luxia/lx_olivetti_clements_bell.sh
bash ${SRC_FOLDER}/bash_train_scripts/train_luxia/lx_olivetti_fldzhyan.sh
bash ${SRC_FOLDER}/bash_train_scripts/train_luxia/lx_olivetti_fldzhyan_bell.sh
