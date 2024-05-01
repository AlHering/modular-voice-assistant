# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os
from time import sleep
from typing import Any
from uuid import uuid4
import streamlit as st
from src.configuration import configuration as cfg
from src.utility.bronze import streamlit_utility, json_utility
from src.utility.silver.file_system_utility import safely_create_path
import json
import requests


def run_page() -> None:
    """
    Function for running "Home" page.
    """
    st.title("Model Control")
    st.header("Find, query and control ML models.")


PAGES = {
    "Home": run_page
}


def run_app() -> None:
    """
    Function for running the app.
    """
    st.set_page_config(
        page_title="Model Control",
        page_icon=":books:"
    )

    page = st.sidebar.selectbox(
        "Navigation",
        PAGES.keys()
    )

    PAGES[page]()

    save_cache_button = st.sidebar.button("Save state.")
    if save_cache_button:
        json_utility.save(
            st.session_state["CACHE"], cfg.PATHS.MODEL_CONTROL_FRONTEND_CACHE)


if __name__ == "__main__":
    if os.path.exists(cfg.PATHS.MODEL_CONTROL_FRONTEND_CACHE):
        st.session_state["CACHE"] = json_utility.load(
            cfg.PATHS.MODEL_CONTROL_FRONTEND_CACHE)
    else:
        st.session_state["CACHE"] = {}
    safely_create_path(cfg.PATHS.MODEL_CONTROL_FRONTEND_ASSETS)
    run_app()
