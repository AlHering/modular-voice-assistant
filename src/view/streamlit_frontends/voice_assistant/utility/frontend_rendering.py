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
from src.view.streamlit_frontends.voice_assistant.utility.state_cache_handling import clear_tab_config
import streamlit.components.v1 as st_components
from code_editor import code_editor
import json
from itertools import combinations


###################
# Helper functions
###################


def get_json_editor_buttons() -> List[dict]:
    """
    Function for acquiring json payload code editor buttons.
    Commands can be found at https://github.com/ajaxorg/ace/blob/v1.2.6/lib/ace/commands/default_commands.js.
    :return: Buttons as list of dictionaries.
    """
    return [
        {
            "name": "save",
            "feather": "Save",
            "hasText": True,
            "alwaysOn": True,
            "commands": [
                    "save-state",
                    [
                        "response",
                        "saved"
                    ]
            ],
            "response": "saved",
            "style": {"top": "0rem", "right": "9.6rem"}
        },
        {
            "name": "copy",
            "feather": "Copy",
            "hasText": True,
            "alwaysOn": True,
            "commands": ["copyAll"],
            "style": {"top": "0rem", "right": "5rem"}
        },
        {
            "name": "clear",
            "feather": "X",
            "hasText": True,
            "alwaysOn": True,
            "commands": ["selectall", "del", ["insertstring", "{\n\t\n}"], "save-state",
                         ["response", "saved"]],
            "style": {"top": "0rem", "right": "0.4rem"}
        },
    ]


def tkinter_folder_selector(start_folder: str = None) -> str:
   """
   Function for selecting local folder via tkinter.
   :param start_folder: Folder to start browsing in.
   """
   start_folder = cfg.PATHS.MODEL_PATH if start_folder is None else start_folder
   root = tk.Tk()
   root.withdraw()
   folder_path = filedialog.askdirectory(initialdir=start_folder, parent=root)
   root.destroy()
   return folder_path


def test():
    print("ASDisgmdfgisnigndfngsdifgnisfng")


def render_json_input(parent_widget: Any, key: str, label: str = None, default_data: dict = None) -> None:
    """
    Function for rendering JSON input.
    :param parent_widget: Parent widget.
    :param key: Widget key.
    :param label: Optional label.
    :param default_data: Default data.
    """
    if label is not None:
        parent_widget.write(label)
    with parent_widget.empty():
        widget = content = st.session_state["CACHE"].get(key)
        if widget is not None:
            content = widget["text"]
        else:
            content = json.dumps(
                default_data, 
                indent=4, 
                ensure_ascii=False
            )
        content = "{\n\t\n}" if content == "{}" else content
        code_editor(
            content,
            key=key,
            lang="json",
            allow_reset=True,
            options={"wrap": True},
            buttons=get_json_editor_buttons(),
            response_mode="debounce"
        )


###################
# Rendering functions
###################


def render_sidebar() -> None:
    """
    Function for rendering the sidebar.
    """
    for key, value in st.session_state.items():
        st.sidebar.write(f"{key}: {value}")


            
