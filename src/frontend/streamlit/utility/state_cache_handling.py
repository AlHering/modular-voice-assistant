# -*- coding: utf-8 -*-
"""
****************************************************
*           Scraping Database Generator            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os
import streamlit as st
from typing import List, Any
from time import sleep
from src.utility import json_utility
from src.configuration import configuration as cfg
from src.frontend.streamlit.utility import backend_interaction


def wait_for_setup() -> None:
    """
    Function for waiting for setup to finish.
    """
    with st.spinner("Waiting for backend to finish startup..."):
        while "CACHE" not in st.session_state:
            try:
                populate_state_cache()
                backend_interaction.MODE = st.session_state["CACHE"].get("MODE", "direct")
                backend_interaction.setup()
                st.rerun()
            except ConnectionError:
                sleep(3)


def populate_state_cache() -> None:
    """
    Function for populating state cache.
    """
    st.session_state["CACHE"] = json_utility.load(
        cfg.PATHS.FRONTEND_CACHE
    ) if os.path.exists(cfg.PATHS.FRONTEND_CACHE) else json_utility.load(
        cfg.PATHS.FRONTEND_DEFAULT_CACHE
    )


def remove_state_cache_element(field_path: List[Any]) -> None:
    """
    Function for removing a target element from cache.
    :param field_path: Field path for traversing cache to target element.
    """
    target = field_path[-1]
    field_path.remove(target)
    data = st.session_state["CACHE"]
    for key in field_path:
        data = data[key]
    data.pop(target)


def clear_tab_config(tab_key) -> None:
    """
    Function for clearing config session state key.
    :param tab_key: Tab key.
    """
    for key in [key for key in st.session_state if key.startswith(tab_key)]:
        st.session_state.pop(key)
