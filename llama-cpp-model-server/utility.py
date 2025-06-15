# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2025 Alexander Hering             *
****************************************************
"""
from __future__ import annotations
import os
import json
from typing import List
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


def get_default_model_configs(fallback_model_configs: List[dict] | None = None, model_path_key: str | None = None) -> list:
    """
    Returns default model configs.
    :param fallback_model_configs: Fallback model configs.
    :param model_path_key: Config key under which the model path is found.
    :return: List of model dictionaries.
    """
    try:
        model_configs = load_json(ENV.get("MODEL_CONFIG"))
        if isinstance(model_configs, dict) and "models" in model_configs:
            model_configs = model_configs["models"]
    except:
        model_configs = fallback_model_configs
    if model_path_key is not None:
        for model_config in model_configs:
            if not os.path.exists(model_config[model_path_key]):
                rel_path = os.path.join(MODEL_DIR, model_config[model_path_key])
                if os.path.exists(rel_path):
                    model_config[model_path_key] = rel_path
    return model_configs


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
            

def get_default_config(fallback_model_configs: List[dict] | None = None, model_path_key: str | None = None) -> dict:
    """
    Returns default config.
    :param fallback_model_configs: Fallback model configs.
    :param model_path_key: Config key under which the model path is found.
    """
    return {
        "host": ENV.get("HOST", "0.0.0.0"),
        "port": int(ENV.get("PORT", "8123")),
        "models": get_default_model_configs(
            fallback_model_configs=fallback_model_configs,
            model_path_key=model_path_key
        )
    }