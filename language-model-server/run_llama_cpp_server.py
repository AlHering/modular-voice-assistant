# -*- coding: utf-8 -*-
"""
****************************************************
*                 Agent Workbench                  *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from __future__ import annotations
import os
import sys
import json
import psutil
import signal
import subprocess
import requests
import time
import click
from uuid import uuid4
from typing import Union
from llama_cpp.server.settings import ConfigFileSettings
from dotenv import dotenv_values


"""
Environment file
"""
WORK_DIR = os.path.dirname(__file__)
CONFIG_DIR = os.path.join(WORK_DIR, "configs")
MODEL_DIR = os.path.join(WORK_DIR, "models")
ENV_PATH = os.path.join(WORK_DIR, ".env")
ENV = dotenv_values(ENV_PATH) if os.path.exists(ENV_PATH) else {}


"""
IO helper functions
"""
def save_json(data: dict, path: str) -> None:
    """
    Function for saving dict data to path.
    :param data: Data as dictionary.
    :param path: Save path.
    """
    with open(path, "w", encoding="utf-8") as out_file:
        json.dump(data, out_file, indent=4, ensure_ascii=False)


def load_json(path: str) -> dict:
    """
    Function for loading json data from path.
    :param path: Save path.
    :return: Dictionary containing data.
    """
    with open(path, "r", encoding="utf-8") as in_file:
        return json.load(in_file)


"""
Configuration construction
"""
MODEL_CONFIGS = [
    {
        "model": "/llama-cpp-server/models/mradermacher_Meta-Llama-3.1-8B-Instruct-i1-GGUF/Meta-Llama-3.1-8B-Instruct.i1-Q4_K_M.gguf",
        "model_alias": "llama3.1-8B-i1",
        "chat_format": "chatml",
        "n_gpu_layers": -1,
        "offload_kqv": True,
#            "n_ctx": 131072,
        "n_ctx": 65536,
        "flash_attn": True,
        "use_mlock": False
    },
    {
        "model": "/llama-cpp-server/models/mradermacher_Meta-Llama-3.1-8B-Instruct-i1-GGUF/Meta-Llama-3.1-8B-Instruct.i1-Q4_K_M.gguf",
        "model_alias": "llama-3",
        "chat_format": "chatml",
        "n_gpu_layers": 22,
        "offload_kqv": True,
        "n_ctx": 8192,
        "use_mlock": False
    }
]


def get_default_model_configs() -> list:
    """
    Returns default model configs.
    :return: List of model dictionaries.
    """
    try:
        model_configs = load_json(ENV.get("MODEL_CONFIG"))
        if isinstance(model_configs, dict) and "models" in model_configs:
            model_configs = model_configs["models"]
    except:
        model_configs = MODEL_CONFIGS
    for model_config in model_configs:
        if not os.path.exists(model_config["model"]):
            rel_path = os.path.join(MODEL_DIR, model_config["model"])
            if os.path.exists(rel_path):
                model_config["model"] = rel_path
    return model_configs


SERVER_CONFIG = {
    "host": ENV.get("HOST", "0.0.0.0"),
    "port": int(ENV.get("PORT", "8123")),
    "models": get_default_model_configs()
}


def get_valid_config_path(config_path: str | None) -> str | None:
    """
    Returns valid config path.
    :param config_path: Base config path.
    :return: Valid config path or None.
    """
    if config_path is not None:
        if not config_path.lower().endswith(".json"):
            config_path += ".json"
        if os.path.exists(config_path):
            return config_path
        else:
            rel_path = os.path.join(CONFIG_DIR, config_path)
            if os.path.exists(rel_path):
                return rel_path


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
    python_exectuable = os.environ.get("VIRTUAL_ENV")
    python_exectuable = os.path.realpath(sys.executable) if python_exectuable is None else f"{python_exectuable}/bin/python"
    if isinstance(config, str) and os.path.exists(config):
        data = load_json(config)
        cmd = "{python_exectuable} -m llama_cpp.server --config_file {config}"
    else:
        data = config
        temp_config_path = os.path.realpath(os.path.join(os.path.dirname(__file__), f"{uuid4()}.json"))
        save_json(config, temp_config_path)
        cmd = f"{python_exectuable} -m llama_cpp.server --config_file {temp_config_path} & (sleep 5 && rm {temp_config_path})"
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
@click.option("--config_path", default=None, help="Path to json configuration file for the LlamaCPP server.")
def run_llama_server(config_path: str) -> None:
    """Runner program for LlamaCPP Server."""
    config_path = get_valid_config_path(config_path=config_path)
    if config_path:
        print(f"Valid config path given: {config_path}.")
        process = load_llamacpp_server_subprocess(config_path)
    else:
        print(f"No valid config path given, using default configuration.")
        process = load_llamacpp_server_subprocess(SERVER_CONFIG)
    try:
        while True:
            time.sleep(1)
    except:
        pass
    terminate_llamacpp_server_subprocess(process=process)


if __name__ == "__main__":
    run_llama_server()
