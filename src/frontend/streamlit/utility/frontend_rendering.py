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
from src.frontend.streamlit.utility.state_cache_handling import clear_tab_config
import streamlit.components.v1 as st_components
from code_editor import code_editor
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
    Function for rendering the sidebar.
    """
    for key, value in st.session_state.items():
        st.sidebar.write(f"{key}: {value}")


            
