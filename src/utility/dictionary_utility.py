# -*- coding: utf-8 -*-
"""
****************************************************
*                     Utility                      *
*            (c) 2020-2021 Alexander Hering        *
****************************************************
"""
import copy
import random
from typing import Any, List, Union


def merge_data(old_data: dict, new_data: dict, create_new: bool = True) -> dict:
    """
    Function for updating old dictionary with fields and values from new dictionary.
    :param old_data: Old dictionary to update.
    :param new_data: New dictionary containing update data.
    :param create_new: Add values from new_data although not existing in old_data. Defaults to True.
    :return: Updated dictionary.
    """
    for elem in new_data:
        if elem in old_data:
            if isinstance(old_data[elem], dict):
                old_data[elem] = merge_data(old_data[elem], new_data[elem])
            else:
                old_data[elem] = new_data[elem]
        elif create_new:
            old_data[elem] = new_data[elem]
    return old_data


def extend_structure(base_data: dict, extension_data: dict) -> None:
    """
    Function for updating old dictionary structure with fields and values from other dictionary structure.
    Only sets fields, not existing in base_data!
    For updating fields, that are already existing in base_data, use merge_data function.
    :param base_data: Dictionary to extend.
    :param extension_data: New dictionary containing update data.
    :return: Extended dictionary.
    """
    for elem in extension_data:
        if elem in base_data:
            if isinstance(base_data[elem], dict):
                base_data[elem] = extend_structure(base_data[elem], extension_data[elem])
        else:
            base_data[elem] = extension_data[elem]


def set_nested_field(data: dict, key_list: list, value) -> None:
    """
    Function for setting a nested dictionary value.
    :param data: Dictionary to set field in.
    :param key_list: List of keys as path to target field.
    :param value: Value to set field to.
    """
    if len(key_list) == 1:
        data[key_list[0]] = value
    elif len(key_list) > 1:
        set_nested_field(data[key_list[0]], key_list[1:], value)


def set_and_extend_nested_field(data: dict, key_list: list, value) -> None:
    """
    Function for setting a nested dictionary value.
    :param data: Dictionary to set field in.
    :param key_list: List of keys as path to target field.
    :param value: Value to set field to.
    """
    if len(key_list) == 1:
        data[key_list[0]] = value
    elif len(key_list) > 1:
        if key_list[0] not in data:
            data[key_list[0]] = {}
        set_and_extend_nested_field(data[key_list[0]], key_list[1:], value)


def extract_nested_value(data: dict, keys: Union[list, str]) -> Any:
    """
    Function for extracting potentially nested values.
    :param data: Entity data to extract value from.
    :param keys: List of keys as path to target field or single key.
    :return: Value of potentially nested field.
    """
    if isinstance(keys, list):
        for key in keys:
            data = data[key]
    else:
        data = data[keys]
    return data


def safely_extract_nested_value(data: dict, keys: Union[list, str], default: Any) -> Any:
    """
    Function for safely extracting nested values.
    :param data: Entity data to extract value from.
    :param keys: List of keys as path to target field or single key.
    :param default: Value to return in case of extraction failure.
    :return: Value of potentially nested field or default value if not existing.
    """
    if isinstance(keys, list):
        for index, key in enumerate(keys):
            if isinstance(data, dict) and key in data:
                data = data[key]
            elif isinstance(data, list):
                return [safely_extract_nested_value(elem, keys[index:], default) for elem in data]
            else:
                data = default
    else:
        data = data.get(keys, default)
    return data


def collect_by_profile(profile: dict, target_data: dict) -> dict:
    """
    Function for collecting data from dictionary by profile.
    :param profile: Collection profile.
    :param target_data: Target data to collect from.
    :return: Collected data as dictionary.
    """
    return_data = {}
    for elem in profile:
        if isinstance(target_data[elem], dict):
            return_data[elem] = collect_by_profile(profile[elem], target_data[elem])
        else:
            return_data[elem] = target_data[elem]
    return return_data


def check_equality(data: dict, test_data: dict, exceptions: list = []) -> bool:
    """
    Function for testing fields of two dictionaries for equality, field paths for exceptions excluded.
    :param data: First dictionary.
    :param test_data: Second dictionary.
    :param exceptions: List of excluded field paths. Defaults to empty list.
    :return: True if dictionary fields (exceptions excluded) have equal values, else False.
    """
    for elem in data:
        if elem in test_data:
            if not any(elem == e[-1] for e in exceptions):
                if isinstance(data[elem], dict):
                    if isinstance(test_data[elem], dict):
                        if not check_equality(data[elem], test_data[elem], [e[1:] for e in exceptions if len(e) > 1 and (e[0] == elem or e[0] == "*")]):
                            return False
                    else:
                        return False
                elif test_data[elem] != data[elem]:
                    return False
        elif not any(elem == e[-1] for e in exceptions):
            return False
    return True


def create_test_dictionary() -> dict:
    """
    Function for creating a dummy dictionary filled with different data structures and primitives for testing purposes.
    :return: Filled dummy dictionary.
    """
    data = {}
    for i in range(random.randint(4, 10)):
        data["test_" + str(i)] = {
            "test_" + str(i) + "_list": [e for e in range(random.randint(2, 6))],
            "test_" + str(i) + "_dict": {"dict_" + str(e): e for e in range(random.randint(2, 6))},
            "test_" + str(i) + "_float": float(i) + 0.1*i
        }
        for j in range(random.randint(2, 6)):
            data["test_string_" + str(j)] = str(j)
    return data


def extract_field_paths(data: dict, field_path: List[str] = [], stop_at: List[str] = [], ignore: List[str] = [], must_contain: List[str] = []) -> List[list]:
    """
    Function for extracting all field paths from dictionary.
    :param data: Dictionary to extract field paths from.
    :param field_path: Path to start element. Defaults to empty path for dictionary root.
    :param stop_at: List of keywords that are not recursively checked. Defaults to empty list.
    :param ignore: List of keywords to fully ignore. Defaults to empty list.
    :param must_contain: List of elemts, a value must contain for its key to be added to the field paths. Defaults to empty list.
    :return: Extracted field paths.
    """
    extracted_paths = []
    for elem in [key for key in data if key not in ignore]:
        new_path = copy.deepcopy(field_path)
        new_path.append(elem)
        if all(part in data[elem] for part in must_contain):
            extracted_paths.append(new_path)
        if isinstance(data[elem], dict) and elem not in stop_at:
            extracted_paths.extend(extract_field_paths(data=data[elem], field_path=new_path, stop_at=stop_at, ignore=ignore, must_contain=must_contain))
    return extracted_paths


def filters_from_data(self, data: dict, key_path: list = [], exceptions: list = []) -> list:
    """
    Method to derive filters from data.
    :param data: Data.
    :param key_path: Current global key path. Defaults to empty path.
    :param exceptions: Exception key paths to keep out of filter masks.
    :return: Filter masks for data.
    """
    filter_masks = []
    for key in data:
        curr_key_path = copy.deepcopy(key_path)
        curr_key_path.append(key)
        if curr_key_path not in exceptions:
            if not isinstance(data[key], dict):
                filter_masks.append([curr_key_path, "==", data[key]])
            else:
                filter_masks.extend(filters_from_data(data[key], curr_key_path, exceptions=exceptions))
    return filter_masks


def get_filter_depth(filter_masks: Any) -> int:
    """
    Function for getting filter mask depth.
    :param filter_masks: Filter masks.
    """
    if not isinstance(filter_masks, list):
        return 0
    return max([get_filter_depth(filter_mask) for filter_mask in filter_masks], default=0) + 1


def exists(data: dict, field_path: List[str]) -> bool:
    """
    Function for checking, whether field path exists.
    :param data: Dictionary.
    :param field_path: Field path.
    :return: True, if field path exists, else False.
    """
    if len(field_path) == 0:
        return True
    else:
        try:
            return exists(data[field_path[0]], field_path[1:])
        except KeyError:
            return False
