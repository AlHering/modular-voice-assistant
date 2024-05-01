# -*- coding: utf-8 -*-
"""
****************************************************
*                     Utility                      *
*            (c) 2020-2023 Alexander Hering        *
****************************************************
"""


def equals(root_data, target_data) -> bool:
    """
    Function for checking condition 'equals'.
    :param root_data: Data of root entity.
    :param target_data: Data of target entity.
    :return: True, if condition is met, else False.
    """
    return root_data == target_data


def not_equals(root_data, target_data) -> bool:
    """
    Function for checking condition 'not_equals'.
    :param root_data: Data of root entity.
    :param target_data: Data of target entity.
    :return: True, if condition is met, else False.
    """
    return root_data != target_data


def contains(root_data, target_data) -> bool:
    """
    Function for checking 'contains'.
    :param root_data: Data of root entity.
    :param target_data: Data of target entity.
    :return: True, if condition is met, else False.
    """
    return target_data in root_data


def not_contains(root_data, target_data) -> bool:
    """
    Function for checking 'not_contains'.
    :param root_data: Data of root entity.
    :param target_data: Data of target entity.
    :return: True, if condition is met, else False.
    """
    return target_data not in root_data


def is_contained(root_data, target_data) -> bool:
    """
    Function for checking condition 'is_contained'.
    :param root_data: Data of root entity.
    :param target_data: Data of target entity.
    :return: True, if condition is met, else False.
    """
    return root_data in target_data


def not_is_contained(root_data, target_data) -> bool:
    """
    Function for checking condition 'not_is_contained'.
    :param root_data: Data of root entity.
    :param target_data: Data of target entity.
    :return: True, if condition is met, else False.
    """
    return root_data not in target_data


COMPARISON_METHOD_DICTIONARY = {
    "equals": lambda x, y: x == y,
    "not_equals": lambda x, y: x != y,
    "contains": lambda x, y: y in x,
    "not_contains": lambda x, y: y not in x,
    "is_contained": lambda x, y: x in y,
    "not_is_contained": lambda x, y: x not in y,
    "==": lambda x, y: x == y,
    "!=": lambda x, y: x != y,
    "has": lambda x, y: y in x,
    "not_has": lambda x, y: y not in x,
    "in": lambda x, y: x in y,
    "not_in": lambda x, y: x not in y,
    "and": lambda *x: all(x),
    "or": lambda *x: any(x),
    "not": lambda x: not x,
    "&&": lambda *x: all(x),
    "||": lambda *x: any(x),
    "!": lambda x: not x
}
