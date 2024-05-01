# -*- coding: utf-8 -*-
"""
****************************************************
*                     Utility                      *
*            (c) 2020-2022 Alexander Hering        *
****************************************************
"""
import json
import os


def save(data: dict, path: str) -> None:
    """
    Function for saving dict data to path.
    :param data: Data as dictionary.
    :param path: Save path.
    """
    with open(path, "w", encoding="utf-8") as out_file:
        json.dump(data, out_file, indent=4, ensure_ascii=False)


def load(path: str) -> dict:
    """
    Function for loading json data from path.
    :param path: Save path.
    :return: Dictionary containing data.
    """
    with open(path, "r", encoding="utf-8") as in_file:
        return json.load(in_file)


def is_json_file(path: str) -> bool:
    """
    Function for checking whether path is json file.
    :param path: Path to file.
    :return: True if path leads to json file, else False.
    """
    if os.path.isfile(path) and os.path.splitext(path)[1] == ".json":
        return True
    else:
        return False


def is_json(text: str) -> bool:
    """
    Function for checking whether a text contains json data.
    :param text: Text.
    :return: True if text contains json data, else False.
    """
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False
    except TypeError:
        return False
