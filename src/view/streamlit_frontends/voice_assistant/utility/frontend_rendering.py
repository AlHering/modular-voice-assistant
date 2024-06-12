# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import os
from typing import Any, List
from uuid import uuid4
import streamlit as st
import random
import tkinter as tk
from tkinter import filedialog
from src.configuration import configuration as cfg
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
    Taken from https://github.com/AlHering/scraping-database-generator/blob/development/src/view/streamlit_frontend/frontend_utility/frontend_rendering.py.
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
            "commands": ["selectall", "del", ["insertstring", "{\n\n\n\n}"], "save-state",
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


def render_json_input(parent_widget: Any, cache_field: str) -> None:
    """
    Function for rendering JSON input.
    :param parent_widget: Parent widget.
    :param cache_field: State cache field.
    """
    parent_widget.text(
        """(CTRL+ENTER or "save" to confirm)""")
    with parent_widget.empty():
        content = st.session_state["CACHE"].get(cache_field)
        code_editor(json.dumps({} if content is None else content).replace("{", "{\n\n").replace("}", "\n\n}"),
                    key=f"{cache_field}_update",
                    lang="json",
                    allow_reset=True,
                    options={"wrap": True},
                    buttons=get_json_editor_buttons()
                    )



###################
# Rendering functions
###################


def render_sidebar() -> None:
    """
    Function for rendering the sidebar.
    """
    pass


def render_object_config(object_type: str) -> None:
    """
    Function for rendering object configs.
    :param ob
    """
    available = st.session_state["CONTROLLER"].get_objects_by_type(object_type)
    if object_type == "transcriber":
        current_config_id = st.selectbox(
            key=f"{object_type}_config_selectbox",
            label="Configuration",
            options=[entry.id for entry in available]
        )

        backends = st.session_state["CONTROLLER"].Transcriber.supported_backends
        default_models = st.session_state["CONTROLLER"].Transcriber.default_models

        b_key = "new_transcriber"
        form = st.form(
            key=b_key
        )
        form.multiselect(key=f"{b_key}_backend", label="Backend", options=backends, default=backends[0])
        
        form.text_input(key=f"{b_key}_model", label="Model", value=default_models[st.session_state[f"{b_key}_backend"]])
        select_folder = form.button("Select model folder...")
        if select_folder:
            folder = tkinter_folder_selector(cfg.PATHS.SOUND_GENERATION_MODEL_PATH)
            if os.path.exists(folder):
                st.session_state[f"{b_key}_model"] = folder 

        render_json_input(form, f"{b_key}_model_parameters")
        render_json_input(form, f"{b_key}_transcription_parameters")
