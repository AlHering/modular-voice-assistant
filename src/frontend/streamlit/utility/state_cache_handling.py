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
from copy import deepcopy
from src.utility import json_utility
from src.configuration import configuration as cfg
from src.frontend.streamlit.utility import backend_interaction


def wait_for_setup() -> None:
    """
    Waits for setup to finish.
    """
    populate_state_cache()
    backend_interaction.setup()
    st.session_state["SETUP"] = True
    st.rerun()

def save_config(object_type: str) -> None:
    """
    Saves config to file system.
    :param object_type: Target object type.
    """
    config = json_utility.load(
        cfg.PATHS.FRONTEND_CACHE
    ) if os.path.exists(cfg.PATHS.FRONTEND_CACHE) else {}
    config[object_type] = st.session_state["CACHE"][object_type]
    json_utility.save(config, cfg.PATHS.FRONTEND_CACHE)


def populate_state_cache() -> None:
    """
    Populates state cache.
    """
    st.session_state["CACHE"] = json_utility.load(
        cfg.PATHS.FRONTEND_CACHE
    ) if os.path.exists(cfg.PATHS.FRONTEND_CACHE) else {}
    for key in cfg.DEFAULT_COMPONENT_CONFIG:
        if key not in st.session_state["CACHE"]:
            st.session_state["CACHE"][key] = deepcopy(cfg.DEFAULT_COMPONENT_CONFIG[key])
    st.session_state["CACHE"]["PARAM_SPECS"] = {}

def remove_state_cache_element(field_path: List[Any]) -> None:
    """
    Removes a target element from cache.
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
    Clears config session state key.
    :param tab_key: Tab key.
    """
    for key in [key for key in st.session_state if key.startswith(tab_key)]:
        st.session_state.pop(key)
