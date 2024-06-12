# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import streamlit as st
from typing import List
from requests.exceptions import ConnectionError
import json
from time import sleep
from src.view.streamlit_frontends.voice_assistant.utility.state_cache_handling import wait_for_setup, clear_tab_config
from src.view.streamlit_frontends.voice_assistant.utility.frontend_rendering import render_sidebar, render_json_input



###################
# Main page functionality
###################
TRANSCRIBER_TAB_KEY = "new_transcriber"


def gather_transcriber_config() -> dict:
    """
    Function for gathering transcriber config.
    :return: Transcriber config.
    """
    data = {
        key: st.session_state[f"{TRANSCRIBER_TAB_KEY}_{key}"] for key in [
            "backend", "model_path"]
    }
    for key in ["model_parameters", "transcription_parameters"]:
        widget = st.session_state[f"{TRANSCRIBER_TAB_KEY}_{key}"]
        data[key] = json.loads(widget["text"] if widget is not None else "{}")
    return data


def render_transcriber_config() -> None:
    """
    Function for rendering transcriber configs.
    """
    tab_key = TRANSCRIBER_TAB_KEY
    available = {entry.id: entry for entry in st.session_state["CONTROLLER"].get_objects_by_type("transcriber")}
    options = [">>New<<"] + list(available.keys())
    default = st.session_state.get(f"{tab_key}_overwrite_config_id", st.session_state.get("transcriber_config_selectbox", ">>New<<"))
    
    header_columns = st.columns([.20, *[.10 for _ in range(8)]])
    header_columns[0].write("")
    header_columns[0].selectbox(
        key="transcriber_config_selectbox",
        label="Configuration",
        options=options,
        on_change=clear_tab_config,
        kwargs={"tab_key": tab_key},
        index=options.index(default)
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

    header_columns[2].write("#####")
    with header_columns[2].popover("Overwrite", disabled=current_config is None, 
                                help="Overwrite the current configuration"):
            st.write(f"Transcriber configuration {st.session_state['transcriber_config_selectbox']} will be overwritten.")
            popover_columns = st.columns([.30 for _ in range(3)])

            if popover_columns[1].button("Approve"):
                obj_id = st.session_state["CONTROLLER"].patch_object(
                    "transcriber",
                    st.session_state["transcriber_config_selectbox"],
                    **gather_transcriber_config()
                )
                st.info(f"Updated Transcriber configuration {obj_id}.")

            
    header_columns[3].write("#####")
    if header_columns[3].button("Add new", help="Add new entry with the below configuration if it does not exist yet."):
        obj_id = st.session_state["CONTROLLER"].put_object(
            "transcriber",
            **gather_transcriber_config()
        )
        if obj_id in available:
            st.info(f"Configuration already found under ID {obj_id}.")
        else:
            st.info(f"Created new configuration with ID {obj_id}.")
        st.session_state[f"{tab_key}_overwrite_config_id"] = obj_id
    if st.session_state.get(f"{tab_key}_overwrite_config_id", st.session_state["transcriber_config_selectbox"]) != st.session_state["transcriber_config_selectbox"]:
        st.rerun()
    

###################
# Entrypoint
###################


if __name__ == "__main__":
    # Basic metadata
    st.set_page_config(
        page_title="Voice Assistant",
        page_icon=":ocean:",
        layout="wide"
    )

    # Wait for backend and dependencies
    wait_for_setup()
        
    # Page content
    st.title("Configuration")
    
    transcriber_tab, synthesizer_tab, recorder_tab = st.tabs(["Transcribers", "Synthesizers", "Speech Recorders"])

    with transcriber_tab:
        render_transcriber_config()

    with synthesizer_tab:
        st.header("A dog")
        st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

    with recorder_tab:
        st.header("An owl")
        st.image("https://static.streamlit.io/examples/owl.jpg", width=200)
    render_sidebar()