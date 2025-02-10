#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

printf "\n%s\n" "${delimiter}"
printf "Activating environment"
printf "\n%s\n" "${delimiter}"
 
if [[ -f "${SCRIPT_DIR}/venv/bin/activate" ]]
then
    source "${SCRIPT_DIR}/venv/bin/activate"
else
    printf "\n%s\n" "${delimiter}"
    printf "\e[1m\e[31mERROR: Cannot activate environment, aborting...\e[0m"
    printf "\n%s\n" "${delimiter}"
    exit 1
fi 

printf "\n%s\n" "${delimiter}"
printf "Handling requirements..."
printf "\n%s\n" "${delimiter}"

source "${SCRIPT_DIR}/venv/bin/activate" && CMAKE_ARGS="-DLLAMA_CUDA=on" pip install --no-cache-dir -r ${SCRIPT_DIR}/requirements/cuda12.1/requirements.txt

printf "\n%s\n" "${delimiter}"
printf "Finished installation"
printf "\n%s\n" "${delimiter}"
exit 0