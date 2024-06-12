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
from time import sleep
from src.view.streamlit_frontends.voice_assistant.utility.state_cache_handling import wait_for_setup, clear_tab_config
from src.view.streamlit_frontends.voice_assistant.utility.frontend_rendering import render_sidebar, render_json_input



###################
# Main page functionality
###################


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
        on_change=clear_tab_config,
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

    header_columns[2].write("#####")
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