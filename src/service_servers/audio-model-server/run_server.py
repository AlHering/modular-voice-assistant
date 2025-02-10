# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2025 Alexander Hering             *
****************************************************
"""
from __future__ import annotations
import os
import sys
import psutil
import signal
import subprocess
import requests
import time
import click
from uuid import uuid4
from typing import Union
from utility import load_json, save_json, get_valid_config_path, get_default_config


"""
Configuration construction
"""
DEFAULT_MODEL_CONFIGS = [
]


"""
Main functionality
"""
def load_llamacpp_server_subprocess(config: Union[dict, str], wait_for_startup: bool = True) -> subprocess.Popen:
    """
    Function for loading llamacpp-based server subprocess.
    :param config: Path to config file or config dictionary.
    :param wait_for_startup: Declares whether to wait for server startup to finish.
        Defaults to True.
    :return: Subprocess instance.
    """
    python_executable = os.environ.get("VIRTUAL_ENV")
    python_executable = os.path.realpath(sys.executable) if python_executable is None else f"{python_executable}/bin/python"
    if isinstance(config, str) and os.path.exists(config):
        data = load_json(config)
        cmd = f"{python_executable} -m llama_cpp.server --config_file {config}"
    else:
        data = config
        temp_config_path = os.path.realpath(os.path.join(os.path.dirname(__file__), f"{uuid4()}.json"))
        save_json(config, temp_config_path)
        cmd = f"{python_executable} -m llama_cpp.server --config_file {temp_config_path} & (sleep 5 && rm {temp_config_path})"
    process = subprocess.Popen(cmd, shell=True)
    
    if wait_for_startup:
        model_endpoint = f"http://{data['host']}:{data['port']}/v1/models"
        connected = False
        while not connected:
            try:
                if requests.get(model_endpoint).status_code == 200:
                    connected = True
            except requests.ConnectionError:
                time.sleep(1)
    return process


def terminate_llamacpp_server_subprocess(process: subprocess.Popen) -> None:
    """
    Function for terminating llamacpp-based server subprocess.
    :param process: Server subprocess.
    """
    process_query = str(process.args).split(" &")[0]
    for p in psutil.process_iter():
        try:
            if process_query in " ".join(p.cmdline()):
                os.kill(p.pid, signal.SIGTERM)
        except psutil.ZombieProcess:
            pass
    process.terminate()
    process.wait()


"""
Click-based entrypoint
"""
@click.command()
@click.option("--config", default=None, help="Path or name json configuration file for the LlamaCPP server.")
def run_llama_server(config: str) -> None:
    """Runner program for LlamaCPP Server."""
    config_path = get_valid_config_path(config_path=config)
    if config_path:
        print(f"\nValid config path given: {config_path}.")
        process = load_llamacpp_server_subprocess(config_path)
    else:
        print(f"\nNo valid config path given, using default configuration.")
        process = load_llamacpp_server_subprocess(get_default_config(
            fallback_model_configs=DEFAULT_MODEL_CONFIGS,
            model_path_key="models"
        ))
    try:
        while True:
            time.sleep(1)
    except:
        pass
    terminate_llamacpp_server_subprocess(process=process)


if __name__ == "__main__":
    run_llama_server()
