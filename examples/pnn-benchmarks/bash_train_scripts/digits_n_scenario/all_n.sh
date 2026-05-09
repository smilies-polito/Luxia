# Set project directory paths as environment variables
if [ -z "$GIT_PROJ" ]; then
    export GIT_PROJ=$(git rev-parse --show-toplevel)
    echo "GIT_CURRENT_PROJ is not set, please check if you are in a git repository."
fi

bash ${GIT_PROJ}/bash_train_scripts/digits_bs_scenario/luxia_n_cpu.sh
bash ${GIT_PROJ}/bash_train_scripts/digits_bs_scenario/luxia_n_gpu.sh
bash ${GIT_PROJ}/bash_train_scripts/digits_bs_scenario/neurophox_n_cpu.sh
bash ${GIT_PROJ}/bash_train_scripts/digits_bs_scenario/neurophox_n_gpu.sh
bash ${GIT_PROJ}/bash_train_scripts/digits_bs_scenario/neuroptica_n.sh
bash ${GIT_PROJ}/bash_train_scripts/digits_bs_scenario/photontorch_n_cpu.sh
bash ${GIT_PROJ}/bash_train_scripts/digits_bs_scenario/photontorch_n_gpu.sh
