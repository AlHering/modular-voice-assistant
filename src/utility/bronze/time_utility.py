# -*- coding: utf-8 -*-
"""
****************************************************
*                     Utility                      *
*            (c) 2020-2022 Alexander Hering        *
****************************************************
"""
import datetime
import re
from datetime import datetime as dt
from typing import Any, Optional


DEFAULTS_TS_FORMAT = "%Y_%b_%d-(%H-%M-%S)"
FORMAT_PROFILES = {
    "standard": {
        "regex": r'^[0-9]{4}_[A-Z]{1}[a-z]{2,4}_[0-9]{2}\-\([0-9]{2}\-[0-9]{2}\#[0-9]{2}\)$',
        "format": "%Y_%b_%d-(%H-%M-%S)"
    },
    "old_standard": {
        "regex": r'^[0-9]{4}_[0-9]{2}_[0-9]{2}\-\([0-9]{2}\-[0-9]{2}\#[0-9]{2}\)$',
        "format": "%Y_%m_%d-(%H-%M-%S)"
    },
    "long":  {
        "regex": r'^[0-9]{2}.[0-9]{2}.[0-9]{4}$',
        "format": "%d.%m.%Y"
    },
    "short":  {
        "regex": r'^[0-9]{2}.[0-9]{2}.[0-9]{2}$',
        "format": "%d.%m.%y"
    },
    "mixed": {
        "regex": r'^[A-Z]{1}[a-z]{2,10} [0-9]{2}, [0-9]{4}$',
        "format": "%B %d, %Y"
    },
    "jobs": {
        "regex": r'^[0-9]{4}\-[0-9]{2}\-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}$',
        "format": "%Y-%m-%d %H:%M:%S"
    }
}
HOUR_DIVISION_DICT = {
    "S": 0.0002777777777777778,
    "M": 0.016666666666666666,
    "H": 1,
    "d": 24,
    "m": 730,
    "Y": 8760
}


def get_timestamp() -> str:
    """
    Function for getting current timestamp as formatted string.
    :return: Current timestamp.
    """
    return dt.now().strftime(DEFAULTS_TS_FORMAT)


def get_timestamp_by_format(fmt: str) -> str:
    """
    Function for getting current timestamp as formatted string.
    :param: fmt: Format for timestamp.
    :return: Current timestamp.
    """
    return dt.now().strftime(fmt)


def get_difference(first_timestamp: str, second_timestamp: str, metric: str = "S") -> float:
    """
    Function for getting difference between two timestamps (%Y_%b_%d-(%H-%M-%S)) by given metric.
    Attention: Years are treated as common years, not leap or mean years.
    :param first_timestamp: First timestamp to compare.
    :param second_timestamp: Second timestamp to compare.
    :param metric: Metric of return value, defaults to "S" for seconds. Other options are
    "y", "m", "d", "H", "M" for years, months, days, hours or months.
    :return: Difference between given timestamps in given metric.
    """
    first_timestamp = dt.strptime(first_timestamp, DEFAULTS_TS_FORMAT)
    second_timestamp = dt.strptime(second_timestamp, DEFAULTS_TS_FORMAT)

    first_timestamp - second_timestamp
    hours = (first_timestamp - second_timestamp).total_seconds() / 3600

    return hours / HOUR_DIVISION_DICT[metric]


def normalize_timestamp(timestamp: str, default_to_input: bool = False) -> str:
    """
    Function for normalizing timestamp to standard format.
    :param timestamp: Timestamp to normalize.
    :param default_to_input: Declares, if input shell be given as default return if no normalization was possible.
        Defaults to False.
    :return: Normalized timestamp.
    """
    for fmt in FORMAT_PROFILES:
        if re.fullmatch(FORMAT_PROFILES[fmt]["regex"], timestamp):
            return convert_format(FORMAT_PROFILES[fmt]["format"], DEFAULTS_TS_FORMAT, timestamp)
    if default_to_input:
        return timestamp


def convert_format(old_format: str, new_format: str, timestamp: str) -> str:
    """
    Function for converting timestamp from one format to another.
    :param old_format: Old format to convert from.
    :param new_format: New format to convert to.
    :param timestamp: Timestamp to convert.
    :return: Converted timestamp.
    """
    timestamp = dt.strptime(timestamp, old_format)
    return timestamp.strftime(new_format)


def get_up_to_month() -> str:
    """
    Function for getting timestamp for current year and month.
    :return: Timestamp for current year and month.
    """
    return "_".join(get_timestamp().split("_")[:2])


def get_up_to_day() -> str:
    """
    Function for getting timestamp for current year, month and day.
    :return: Timestamp for current year, month and day.
    """
    return get_timestamp().split("-")[0]


def get_delta_time(**kwargs: Optional[Any]):
    """
    Function for getting delta time.
    :param kwargs: Delta time arguments. E.g. 'days = 10'.
    :return: Delta time.
    """
    return datetime.timedelta(**kwargs)


def get_past_time(delta_time: datetime.timedelta, fmt: str = DEFAULTS_TS_FORMAT) -> str:
    """
    Function for getting past timestamp.
    :param delta_time: Delta time to substract from current timestamp.
    :param fmt: Format for timestamp, defaults to default format.
    :return: Past timestamp.
    """
    return (dt.now() - delta_time).strftime(fmt)


def get_future_time(delta_time: datetime.timedelta, fmt: str = DEFAULTS_TS_FORMAT) -> str:
    """
    Function for getting future timestamp.
    :param delta_time: Delta time to add to current timestamp.
    :param fmt: Format for timestamp, defaults to default format.
    :return: Future timestamp.
    """
    return (dt.now() + delta_time).strftime(fmt)