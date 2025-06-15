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


def fix_config_path(config_path: str | None, default_dir: str) -> str | None:
    """
    Returns valid config path.
    :param config_path: Base config path.
    :param default_dir: Default configs directory.
    :return: Valid config path or None.
    """
    if config_path is not None:
        if not config_path.lower().endswith(".json"):
            config_path += ".json"
        if os.path.exists(config_path):
            return config_path
        else:
            rel_path = os.path.join(default_dir, config_path)
            if os.path.exists(rel_path):
                return rel_path
            

def fix_model_paths(config: dict, default_dir: str) ->None:
    """
    Fixes potentially relative model paths.
    :param config: Config.
    :param default_dir: Default models directory.
    """
    for model_config in config["models"]:
        if not os.path.exists(model_config["model"]):
            rel_path = os.path.join(default_dir, model_config["model"])
            if os.path.exists(rel_path):
                model_config["model"] = rel_path