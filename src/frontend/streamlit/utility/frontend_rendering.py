# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import os
from typing import Any, List, Callable, Dict
from uuid import uuid4
import streamlit as st
import random
import tkinter as tk
from tkinter import filedialog
from src.configuration import configuration as cfg
from src.frontend.streamlit.utility.backend_interaction import AVAILABLE_MODULES
from src.frontend.streamlit.utility.state_cache_handling import clear_tab_config
import streamlit.components.v1 as st_components
from code_editor import code_editor
from barfi import st_barfi, Block
import json
from itertools import combinations


###################
# Helper functions
###################


###################
# Rendering functions
###################


def render_sidebar() -> None:
    """
    Renders the sidebar.
    """
    for key, value in st.session_state.items():
        st.sidebar.write(f"{key}: {value}")


def render_pipeline_node_plane(parent_widget: Any, block_dict: dict, session_state_key: str | None = None) -> None:
    """
    Renders a interactive node plane.
    :param parent_widget: Parent widget.
    :param block_dict: Dictionary for barfi blocks.
    """
    with parent_widget.empty():
        blocks = []
        for key in block_dict:
            new_block = Block(name=" ".join(key.split("_")).title())
            if key != "speech_recorder":
                new_block.add_input("Input")
            if key != "wave_output":
                new_block.add_output("Output")
            
        barfi_result = st_barfi(base_blocks=blocks)
        if session_state_key and barfi_result:
            st.session_state[session_state_key] = barfi_result

            
