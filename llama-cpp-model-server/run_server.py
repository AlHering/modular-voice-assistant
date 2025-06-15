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
from utility import load_json, save_json, fix_config_path, fix_model_paths
from dotenv import dotenv_values


"""
Environment file
"""
WORK_DIR = os.path.dirname(__file__)
CONFIGS_DIR = os.path.join(WORK_DIR, "configs")
MODELS_DIR = os.path.join(WORK_DIR, "models")
ENV_PATH = os.path.join(WORK_DIR, ".env")
ENV = dotenv_values(ENV_PATH) if os.path.exists(ENV_PATH) else {}


"""
Configuration construction
"""
DEFAULT_CONFIG = {
    "host": ENV.get("HOST", "0.0.0.0"),
    "port": int(ENV.get("PORT", "8123")),
    "models": [
        {
            "model": "/llama-cpp-model-server/models/bartowski_Qwen2.5-3B-Instruct-GGUF/Qwen2.5-3B-Instruct-Q8_0.gguf",
            "model_alias": "qwen-2.5-3b-instruct-q8_0",
            "n_gpu_layers": -1,
            "offload_kqv": True,
            "n_ctx": 8192,
            "use_mlock": False
        },
        {
            "model": "/llama-cpp-model-server/models/mradermacher_Meta-Llama-3.1-8B-Instruct-i1-GGUF/Meta-Llama-3.1-8B-Instruct.i1-Q4_K_M.gguf",
            "model_alias": "llama-3.1-8b-instruct-i1",
            "chat_format": "chatml",
            "n_gpu_layers": 22,
            "offload_kqv": True,
            "n_ctx": 8192,
            "use_mlock": False
        }
    ]
}


"""
Main functionality
"""
def load_llamacpp_server_subprocess(config: dict | str, wait_for_startup: bool = True) -> subprocess.Popen:
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
@click.option("--config", "config", default=None, help="Path or name json configuration file for the LlamaCPP server.")
def run_llama_server(config: str) -> None:
    """Runner program for a configured llama-cpp model server."""
    config_path = fix_config_path(config_path=config, default_dir=CONFIGS_DIR)
    if config_path:
        print(f"\nValid config path given: {config_path}.")
        config = load_json(config_path)
    else:
        print(f"\nNo valid config path given, using default configuration.")
        config=DEFAULT_CONFIG
    fix_model_paths(config=DEFAULT_CONFIG, default_dir=MODELS_DIR)
    process = load_llamacpp_server_subprocess(config=config)
    try:
        while True:
            time.sleep(1)
    except:
        pass
    terminate_llamacpp_server_subprocess(process=process)


if __name__ == "__main__":
    run_llama_server()
