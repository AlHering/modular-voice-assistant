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
    

def clear_config(tab_key) -> None:
    """
    Function for clearing config session state key.
    :param tab_key: Tab key.
    """
    for key in [key for key in st.session_state if key.startswith(tab_key)]:
        st.session_state.pop(key)
    


###################
# Rendering functions
###################


def render_sidebar() -> None:
    """
    Function for rendering the sidebar.
    """
    for key, value in st.session_state.items():
        st.sidebar.write(f"{key}: {value}")


def render_transcriber_config() -> None:
    """
    Function for rendering transcriber configs.
    """
    tab_key = "new_transcriber"
    available = {entry.id: entry for entry in st.session_state["CONTROLLER"].get_objects_by_type("transcriber")}
    options = [">>New<<"] + list(available.keys())
    
    header_columns = st.columns([.20, *[.10 for _ in range(8)]])
    header_columns[0].write("")
    current_config_id = header_columns[0].selectbox(
        key="transcriber_config_selectbox",
        label="Configuration",
        options=options,
        on_change=clear_config,
        kwargs={"tab_key": tab_key}
    )
    
    current_config = available.get(st.session_state["transcriber_config_selectbox"])
    backends = st.session_state["CLASSES"]["transcriber"].supported_backends
    default_models = st.session_state["CLASSES"]["transcriber"].default_models

    st.selectbox(
        key=f"{tab_key}_backend", 
        label="Backend", 
        options=backends, 
        index=0 if current_config is None else backends.index(current_config.backend))

            
    if f"{tab_key}_model_path" not in st.session_state:
        st.session_state[f"{tab_key}_model_path"] = default_models[st.session_state[f"{tab_key}_backend"]][0] if (
        current_config is None or current_config.model_path is None) else current_config.model_path
    st.text_input(
        key=f"{tab_key}_model_path", 
        label="Model")

    st.write("")
    render_json_input(parent_widget=st, 
                      key=f"{tab_key}_model_parameters", 
                      label="Model parameters",
                      default_data={} if current_config is None else current_config.model_parameters)
    render_json_input(parent_widget=st, 
                      key=f"{tab_key}_transcription_parameters", 
                      label="Transcription parameters",
                      default_data={} if current_config is None else current_config.transcription_parameters)

    header_columns[2].write("")
    if header_columns[2].button("Overwrite", disabled=current_config is None, 
                                help="Overwrite the current configuration"):
        print(st.session_state)
        st.session_state["CONTROLLER"].patch_object(
            "transcriber",
            current_config_id,
            ** {
                key: st.session_state[f"{tab_key}_{key}"] for key in [
                    "backend", "model_path", "model_parameters", "transcription_parameters"
                ]
            }
        )
    header_columns[3].write("#####")
    if header_columns[3].button("Add new", help="Add new entry with the below configuration if it does not exist yet."):
        pass
            
