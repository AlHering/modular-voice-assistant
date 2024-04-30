# -*- coding: utf-8 -*-
"""
****************************************************
*                     Utility                      *
*            (c) 2020-2021 Alexander Hering        *
****************************************************
"""
import os
from time import sleep
from typing import Union, List, Any, Optional
from . import json_utility
import requests
import math
from lxml import html


REQUEST_METHODS = {
    "GET": requests.get,
    "POST": requests.post,
    "PATCH": requests.patch,
    "PUT": requests.put,
    "DELETE": requests.delete
}


MEDIA_TYPES_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, "data", "media_types.json"))
if os.path.exists(MEDIA_TYPES_PATH):
    MEDIA_TYPES = json_utility.load(MEDIA_TYPES_PATH)
else:
    MEDIA_TYPES = {}


def get_page_content(url: str) -> html.HtmlElement:
    """
    Function for getting page content from URL.
    :param url: URL to get page content for.
    :return: Page content.
    """
    page = requests.get(url)
    return html.fromstring(page.content)


def get_session(proxy_dict: dict = None) -> requests.Session:
    """
    Function for getting requests session.
    :param proxy_dict: Proxy dictionary.
    :return: Session.
    """
    session = requests.session()
    if proxy_dict != None:
        session.proxies = proxy_dict

    return session


def safely_get_elements(html_element: html.HtmlElement, xpath: str) -> List[Any]:
    """
    Function for safely searching for elements in a Selenium WebElement.
    :param html_element: LXML Html Element.
    :param xpath: XPath of the elements to find.
    :return: List of elements if found, else empty list.
    """
    return html_element.xpath(xpath)


def safely_get_elements(html_element: html.HtmlElement, xpath: str) -> Optional[Any]:
    """
    Function for safely searching for elements in a Selenium WebElement.
    :param resp: Response to search in.
    :param xpath: XPath of the elements to find.
    :return: Extracted element if found, else None.
    """
    res = html_element.xpath(xpath)
    return res[0] if res else None


def safely_collect(html_element: html.HtmlElement, data: dict) -> dict:
    """
    Function for safely collecting data by xpath into dictionary, meaning not found elements get skipped. In later cases
    the collected value will be None.
    :param html_element: LXML Html Element.
    :param data: Data collection dictionary.
    :return: In dict collected data.
    """
    return_data = {}
    for elem in data:
        if isinstance(data[elem], dict):
            return_data[elem] = safely_collect(html_element, data[elem])
        elif isinstance(data[elem], str):
            return_data[elem] = safely_get_elements(html_element, data[elem])
    return return_data


def safely_request_page(url, tries: int = 5, delay: float = 2.0) -> requests.Response:
    """
    Function for safely requesting page response.
    :param url: Target page URL.
    :param tries: Maximum number of tries. Defaults to 5.
    :param delay: Delay to wait before sending off next request. Defaults to 2.0 seconds.
    :return: Response.
    """
    resp = requests.get(url)
    j = 0
    while (resp.status_code == 404 or resp.status_code == 403) and j < tries:
        j += 1
        sleep(delay)
    return resp


def download_web_asset(asset_url: str, output_path: str, add_extension: bool = False, headers: dict = None) -> None:
    """
    Function for downloading web asset.
    :param asset_url: Asset URL.
    :param output_path: Output path.
    :param add_extension: Flag, declaring whether to fetch extension from header data and add to output path.
        Defaults to False.
    :param headers: Headers to use.
        Default to None.
    """
    try:
        asset_head = requests.head(asset_url, headers=headers).headers
        asset = requests.get(
            asset_url, headers=headers, stream=True)
    except requests.exceptions.SSLError:
        asset_head = requests.head(
            asset_url, headers=headers, verify=False).headers
        asset = requests.get(
            asset_url, headers=headers, stream=True, verify=False)

    if add_extension:
        main_type, sub_type = asset_head.get(
            "Content-Type", "/").lower().split("/")
        # asset_type = asset_head.get("Content-Type")
        # asset_encoding = asset.apparent_encoding if hasattr(
        #    asset, "apparent_encoding") else asset.encoding
        asset_extension = MEDIA_TYPES.get(
            main_type, {}).get(sub_type, {}).get("extension", ".unkown")
        output_path += asset_extension

    asset_size = int(asset.headers.get("content-length", 0))
    chunk_size = 1024
    local_size = 0

    try:
        from tqdm import tqdm
        with tqdm.wrapattr(open(output_path, "wb"), "write",
                           miniters=1, desc=f"Downloading '{asset_url}' ...",
                           total=asset_size) as output_file:
            for chunk in asset.iter_content(chunk_size=chunk_size):
                output_file.write(chunk)
                local_size += len(chunk)
    except ImportError:
        with open(output_path, "wb") as output_file:
            for chunk in asset.iter_content(chunk_size=chunk_size):
                output_file.write(chunk)
                local_size += len(chunk)
    if local_size != asset_size:
        raise requests.exceptions.RequestException(
            f"Downloading '{asset_url}' failed ({local_size}/{asset_size})!")
