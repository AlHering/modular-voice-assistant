# -*- coding: utf-8 -*-
"""
****************************************************
*                     Utility                      *
*            (c) 2022 Alexander Hering             *
****************************************************
"""
import traceback
from time import sleep
from typing import List, Any, Union, Optional

from scrapy import Spider
from scrapy.crawler import CrawlerProcess
from scrapy.http.response import Response
from scrapy.selector import Selector


def start_crawl_process(spider: Spider, args: list, settings: dict) -> None:
    """
    Function for getting crawl process for given spider, arguments and settings.
    :param spider: Spider to execute inside process.
    :param args: Argument list.
    :param settings: Settings for crawler process.
    :return: Crawler process containing given spider, arguments and settings.
    """
    process = CrawlerProcess(settings=settings)
    process.crawl(spider, *args)
    process.start()


def safely_collect(resp: Response, data: dict) -> dict:
    """
    Function for safely collecting data by xpath into dictionary, meaning not found elements get skipped. In later cases
    the collected value will be None.
    :param resp: Response to collect from.
    :param data: Data collection dictionary.
    :return: In dict collected data.
    """
    return_data = {}
    for elem in data:
        try:
            if isinstance(data[elem], dict):
                return_data[elem] = safely_collect(resp, data[elem])
            else:
                web_element = safely_get_elements(resp, data[elem])
                if web_element:
                    return_data[elem] = web_element.getall()
                else:
                    return_data[elem] = None
        except Exception:
            return_data[elem] = None
    return return_data


def safely_get_elements(resp: Response, xpath: str) -> List[Any]:
    """
    Function for safely searching for elements in a scrapy response.
    :param resp: Response to search in.
    :param xpath: XPath of the elements to find.
    :return: List of elements if found, else None.
    """
    try:
        return resp.xpath(xpath)
    except Exception:
        return []


def safely_get_element(resp: Response, xpath: str) -> Optional[Any]:
    """
    Function for safely searching for an element in a scrapy response.
    :param resp: Response to search in.
    :param xpath: XPath of the elements to find.
    :return: Extracted element if found, else None.
    """
    try:
        return resp.xpath(xpath)[0]
    except Exception:
        return None


def safely_extract_elements(resp: Response, xpath: str) -> List[Selector]:
    """
    Function for safely extracting elements in a scrapy response.
    :param resp: Response to search in.
    :param xpath: XPath of the elements to find.
    :return: List of elements if found, else None.
    """
    try:
        return resp.xpath(xpath).getall()
    except Exception:
        return []


def safely_extract_element(resp: Response, xpath: str) -> Any:
    """
    Function for safely extracting an element in a scrapy response.
    :param resp: Response to search in.
    :param xpath: XPath of the elements to find.
    :return: Extracted element if found, else None.
    """
    try:
        return resp.xpath(xpath).get()
    except Exception:
        return None


def safely_get_text_content(resp: Response, xpath: str = ".") -> str:
    """
    Function for safely getting text content of element from response.
    :param resp: Response to search in.
    :param xpath: XPath for target element. Defaults to '.'.
    :return: Content text, if found, else empty string.
    """
    try:
        return resp.xpath("string(" + xpath + ")").getall()[0]
    except Exception:
        return ""


def safely_get_attribute(resp: Response, xpath: str, attribute: str) -> str:
    """
    Function for safely getting attribute content of an element from a scrapy response defined through given xpath.
    :param resp: Response to search in.
    :param xpath: XPath of the target element.
    :param attribute: Attribute name to search for.
    :return: Attribute content, if found, else empty string.
    """
    try:
        return resp.xpath(xpath).attrib[attribute]
    except KeyError:
        return ""
