#!/bin/bash
printf "\n%s\n" "${delimiter}"
printf "Activating conda environment"
printf "\n%s\n" "${delimiter}"
 
if [[ -f "${CONDA_DIR}/etc/profile.d/conda.sh"  ]]
then
    source "${CONDA_DIR}/etc/profile.d/conda.sh"
else
    printf "\n%s\n" "${delimiter}"
    printf "\e[1m\e[31mERROR: Cannot initialize conda, aborting...\e[0m"
    printf "\n%s\n" "${delimiter}"
    exit 1
fi 

printf "\n%s\n" "${delimiter}"
printf "Setting up conda environment..."
printf "\n%s\n" "${delimiter}"
conda activate "$VENV_DIR"
conda install -y -k  $CONDA_TORCH_CUDA_INSTALLATION

printf "\n%s\n" "${delimiter}"
printf "Handling main webui requirements..."
printf "\n%s\n" "${delimiter}"

python -m pip install -r requirements.txt
#CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 python -m pip install --upgrade llama-cpp-python

printf "\n%s\n" "${delimiter}"
printf "Finished installation"
printf "\n%s\n" "${delimiter}"
exit 0