# -*- coding: utf-8 -*-
"""
****************************************************
*                     Utility                      *
*            (c) 2022 Alexander Hering             *
****************************************************
"""
import sys
from inspect import signature
import subprocess
import importlib.util
from time import sleep
from typing import Any, List, Optional
from ..bronze.hashing_utility import hash_with_sha256


def get_module(path: str, sha256: str = None) -> Optional[Any]:
    """
    Function for loading and returning module.
    :param path: Path to Python file.
    :param sha256: SHA256 hash to check file against. 
        Defaults to None in which case no check is issued.
    :return: Loaded module handle.
    """
    if sha256 is None or hash_with_sha256(path) == sha256:
        spec = importlib.util.spec_from_file_location("module", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def get_function_from_path(path: str, sha256: str = None) -> Any:
    """
    Function for loading and returning function from path.
    :param path: Path to function in the form '[path to .py-file]:[function name]'.
    :param sha256: SHA256 hash to check file against. 
        Defaults to None in which case no check is issued.
    :return: Loaded function.
    """
    module_path, function_name = path.split(":")
    module = get_module(module_path, sha256)
    if module is not None:
        return getattr(module, function_name)


def get_lambda_function_from_string(function_string: str) -> Any:
    """
    Function for loading and returning function from path.
    :param function_string: Lambda function as string.
    :return: Loaded function.
    """
    return eval(function_string)


def issue_multiple_tries(function, tries=3, *args, **kwargs) -> Any:
    """
    Function for issuing python function multiple times.
    :param function: Function to call.
    :param tries: Tries to call function.
    :param args: Arguments to forward to function call as list.
    :param kwargs: Keyword arguments to forward to function call as dictionary.
    :return: Return of successful function call, else None.
    """
    i = 0
    while i < tries:
        try:
            return function(*args, **kwargs)
        except Exception:
            sleep(1)
            i += 1


def get_object_contents(target_object: object) -> list:
    """
    Function for getting all contents of an object.
    :param target_object: Object to get contents from.
    :return: List of contents.
    """
    return dir(target_object)


def safely_import_package(package_name: str, version: str = None, import_name: str = None,
                          import_path: List[str] = None, _installation_executed: bool = False) -> Optional[Any]:
    """
    Function for safely importing pip package and installing it, if it is not already installed.
    :param package_name: Package name.
    :param version: Version string. Optional, defaults to None.
    :param import_name: Module name for import. Defaults to None in which case the package_name is used for importing.
    :param import_path: Import module path to target module.
    :param _installation_executed: Internal flag, declaring whether installation was dynamically executed.
        Defaults to False. If True, no dynamic import is tried and ImportErrors are forwarded.
    :return: The import reference, if import was successful, else None.
    """
    package_name_with_version = package_name
    if version:
        package_name_with_version += f"=={version}"
    try:
        if import_name and import_path:
            return __import__(import_name, fromlist=import_path)
        elif import_name:
            return __import__(import_name)
        else:
            return __import__(package_name)
    except ImportError as ex:
        if not _installation_executed:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package_name_with_version])
            return safely_import_package(package_name, version, import_name, import_path, True)
        else:
            raise ex


def check_module_availability(package_name: str, import_name: str = None,
                              import_path: List[str] = None) -> Optional[Any]:
    """
    Function for checking import target availability.
    :param package_name: Package name.
    :param import_name: Module name for import. Defaults to None in which case the package_name is used for importing.
    :param import_path: Import module path to target module.
    :return: True, if import target is available, else False.
    """
    try:
        if import_name and import_path:
            __import__(import_name, fromlist=import_path)
        elif import_name:
            __import__(import_name)
        else:
            __import__(package_name)
        return True
    except ImportError:
        return False


def run_function_from_string(function: Any, possible_args: list) -> Optional[Any]:
    """
    Function for running function from Action string.
    :param function: Function string.
    :param possible_args: Possible args to forward to function call.
    :return: Return of function call, if existing.
    """
    if isinstance(function, str):
        if function.startswith("lambda"):
            function = get_lambda_function_from_string(function)
        elif ".py" in function and ":" in function:
            function = get_function_from_path(function)
    parameters = []
    if function:
        expected_parameters = signature(function).parameters
        parameters = list(possible_args)[:len(expected_parameters)]

    try:
        return function(*parameters)
    except Exception:
        return None


def load_common_profile(profile: dict) -> dict:
    """
    Function for loading common profile (e.g. encoded functions).
    :param profile: Profile.
    :return: Loaded profile.
    """
    for key in profile:
        if isinstance(profile[key], dict):
            profile[key] = load_common_profile(profile[key])
        elif isinstance(profile[key], str) and profile[key].startswith("lambda"):
            profile[key] = eval(profile[key])
    return profile
