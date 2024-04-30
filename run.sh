#!/bin/bash
printf "\n%s\n" "${delimiter}"
printf "Searching for required environments..."
printf "\n%s\n" "${delimiter}"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [[ ! -v $CONDA_DIR ]]; then
    if [[ -d "/root/anaconda3/bin" ]]; then
        CONDA_DIR="/root/anaconda3"
    elif [[ -d "/root/miniconda3/bin" ]]; then
        CONDA_DIR="/root/miniconda3"
    elif [[ -d "${SCRIPT_DIR}/conda/bin" ]]; then
        CONDA_DIR="${SCRIPT_DIR}/conda/bin"
    fi

    if [[ ! -v $CONDA_DIR ]]; then
        CONDA_DIR=$( echo $PATH | grep -o "[^:]*anaconda3[bin^:]*" | head -n 1 )
    fi
    if [[ ! -d $CONDA_DIR ]]; then
        CONDA_DIR=$( echo $PATH | grep -o "[^:]*miniconda[bin^:]*" | head -n 1 )
    fi
    if [[ ! -d $CONDA_DIR ]]; then
        printf "\n%s\n" "${delimiter}"
        printf "\e[1m\e[31mERROR: Cannot find conda installation! Please export the conda folder as CONDA_DIR ...\e[0m"
        printf "\n%s\n" "${delimiter}"
        exit 1
    fi
fi

if [[ ! -v $VENV_DIR ]]; then
    VENV_DIR="${SCRIPT_DIR}/venv/"
fi
if [[ ! -d $VENV_DIR ]]; then
    printf "\n%s\n" "${delimiter}"
    printf "\e[1m\e[31mERROR: Cannot find the conda environment! Please export the environment folder as VENV_DIR ...\e[0m"
    printf "\n%s\n" "${delimiter}"
    exit 1
fi

printf "\n%s\n" "${delimiter}"
printf "Activating conda environment and entering starting backend server..."
printf "\n%s\n" "${delimiter}"
source "${CONDA_DIR}/etc/profile.d/conda.sh" \
    && conda activate $VENV_DIR \
    && cd $SCRIPT_DIR \
    && python run_backend.py