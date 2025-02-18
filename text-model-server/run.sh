#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

printf "\n%s\n" "${delimiter}"
printf "Activating environment"
printf "\n%s\n" "${delimiter}"
 
if [[ -f "${SCRIPT_DIR}/venv/bin/activate" ]]
then
    printf "Starting LlamaCPP server..."
    source "${SCRIPT_DIR}/venv/bin/activate"
    python run_server.py $@ &
    wait
else
    printf "\n%s\n" "${delimiter}"
    printf "\e[1m\e[31mERROR: Cannot activate environment, aborting...\e[0m"
    printf "\n%s\n" "${delimiter}"
    exit 1
fi 