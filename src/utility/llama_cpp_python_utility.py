# -*- coding: utf-8 -*-
"""
****************************************************
*                      Utility                 
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import os
import sys
import psutil
import signal
import subprocess
import requests
import time
from uuid import uuid4
from typing import Union
from . import json_utility
try:
    import llama_cpp_cuda as llama_cpp
except ImportError:
    import llama_cpp as llama_cpp

from llama_cpp import Llama


def load_llamacpp_model(model_path: str,
                        model_file: str | None = None,
                        model_parameters: dict = {}) -> Llama:
    """
    Function for loading llamacpp-based model instance.
    :param model_path: Path to model files.
    :param model_file: Model file to load.
        Defaults to None.
    :param model_parameters: Model loading kwargs as dictionary.
        Defaults to empty dictionary.
    :return: Llama model instance.
    """
    return Llama(model_path=os.path.join(
        model_path, model_file), **model_parameters)


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
        data = json_utility.load(config)
        cmd = "{python_exectuable} -m llama_cpp.server --config_file {config}"
    else:
        data = config
        temp_config_path = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir, "data", f"{uuid4()}.json"))
        json_utility.save(config, temp_config_path)
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
    
