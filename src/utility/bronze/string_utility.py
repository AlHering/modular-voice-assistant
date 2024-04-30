# -*- coding: utf-8 -*-
"""
****************************************************
*                     Utility                      *
*            (c) 2020-2021 Alexander Hering        *
****************************************************
"""
import os
import re
from typing import Optional, List

# list of symbols
SYMBOLS = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~'"
REGULAR_SYMBOLS = "!$()*+-.?[]^{|}"
FOLDER_RESERVED = "<>:/\\|?*"
HTML_CODEC_DICT = {}


def clean_mutation(text: str) -> str:
    """
    Function for cleaning special character forms to base character (e.g. "ä" -> "a").
    :param text: Text to clean.
    :return: Cleaned text.
    """
    ret = text.replace("ä", "a").replace("ö", "o").replace("ü", "u")
    ret = re.sub(r'(â|á|à)+', 'a', ret)
    ret = re.sub(r'(ê|é|è)+', 'e', ret)
    ret = re.sub(r'(î|í|ì)+', 'i', ret)
    ret = re.sub(r'(ô|ó|ò)+', 'o', ret)
    ret = re.sub(r'(û|ú|ù)+', 'u', ret)
    return ret


def remove_symbols(text: str, exception: list = []) -> str:
    """
    Function for removing all symbols but the ones listed in exception.
    :param text: Text to remove symbols from.
    :param exception: Symbols not to remove. Defaults to empty list.
    :return: Text with removed symbols.
    """
    ret = text
    for elem in [s for s in SYMBOLS if s not in exception]:
        ret = ret.replace(elem, "")


def translate_symbols(text: str, translation: dict, exception: list = []) -> str:
    """
    Function for translating or removing symbols but the ones listed in exception.
    :param text: Text to translate and remove symbols from.
    :param translation: Translation dictionary for symbols.
    :param exception: Symbols not to remove. Defaults to empty list.
    :return: Processed text.
    """
    ret = text
    for elem in translation:
        ret = ret.replace(elem, translation[elem])
    for elem in [s for s in SYMBOLS if s not in exception and s not in translation.values()]:
        ret = ret.replace(elem, "")


def remove_multiple_spaces(text: str) -> str:
    """
    Function for condensing multiple spaces into a single space.
    :param text: Text to remove multiple spaces from.
    :return: Text with single spaces.
    """
    return " ".join(text.split(" "))


def clean_html_codec(text: str) -> str:
    """
    Cleans text from html codecs.
    :param text: Text to clean.
    :return: Cleaned text.
    """
    for codec in ["\\u", "&#", "&"]:
        if codec in text:
            for elem in HTML_CODEC_DICT[codec]:
                text = text.replace(elem, HTML_CODEC_DICT[codec][elem])
    return text


def extract_first_match(pattern: str, text: str) -> Optional[str]:
    """
    Function for extracting first match of a pattern from a text.
    :param pattern: Pattern to extract a match for.
    :param text: Text to extract pattern match from.
    :return: First match of pattern in text or None.
    """
    search = re.compile(pattern).search(text)
    if search:
        return search.group()
    else:
        return None


def extract_all_matches(pattern: str, text: str) -> Optional[List[str]]:
    """
    Function for extracting all matches of a pattern from a text.
    :param pattern: Pattern to extract matches for.
    :param text: Text to extract pattern matches from.
    :return: List of matching strings.
    """
    search = re.findall(pattern, text)
    if search:
        return ["".join(elem) for elem in search]
    else:
        return None


def extract_matches_between_bounds(start_bound: str, end_bound: str, text: str) -> Optional[List[str]]:
    """
    Function for extracting all matches of a pattern from a text.
    :param start_bound: Left bound of searched pattern.
    :param end_bound: Right bound of searched pattern.
    :param text: Text to extract pattern matches from.
    :return: First match of pattern between given boundaries in text.
    """
    search = re.findall(re.compile(escape_regular_chars(
        start_bound) + "(.+?)" + escape_regular_chars(end_bound)), text)
    print(search)
    if search:
        return [elem[1] for elem in search]
    else:
        return None


def escape_regular_chars(text: str) -> str:
    """
    Function for escaping symbols in text that are used by regular expressions.
    :param text: Text to escape symbols in.
    :return: Text with escaped symbol characters.
    """
    for elem in [s for s in REGULAR_SYMBOLS if s in text]:
        text = text.replace(elem, "\\" + elem)
    return text
