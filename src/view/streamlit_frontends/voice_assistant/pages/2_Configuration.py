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
SYNTHESIZER_TAB_KEY = "new_synthesizer"
SPEECH_RECORDER_TAB_KEY = "new_speech_recorder"


def get_json_parameters(object_type: str) -> List[str]:
    """
    Function for getting object json parameters.
    :param object_type: Target object type.
    :return: Object json parameters.
    """
    if object_type == "transcriber":
        return ["model_parameters", "transcription_parameters"]
    elif object_type == "synthesizer":
        return ["model_parameters", "synthesis_parameters"]
    elif object_type == "speech_recorder":
        return ["recognizer_parameters", "microphone_parameters"]
    

def gather_config(object_type: str) -> dict:
    """
    Function for gathering object config.
    :param object_type: Target object type.
    :return: Object config.
    """
    if object_type in ["transcriber", "synthesizer"]: 
        data = {
            key: st.session_state[f"new_{object_type}_{key}"] for key in [
                "backend", "model_path"]
        }
    elif object_type == "speech_recorder":
        pass
    for key in get_json_parameters(object_type):
        widget = st.session_state[f"new_{object_type}_{key}"]
        data[key] = json.loads(widget["text"]) if widget is not None else None
    return data


def render_config(object_type: str) -> None:
    """
    Function for rendering configs.
    :param object_type: Target object type.
    """
    tab_key = f"new_{object_type}"
    available = {entry.id: entry for entry in st.session_state["CONTROLLER"].get_objects_by_type(object_type)}
    options = [">>New<<"] + list(available.keys())
    default = st.session_state.get(f"{tab_key}_overwrite_config_id", st.session_state.get(f"{object_type}_config_selectbox", ">>New<<"))
    
    header_columns = st.columns([.25, .10, .65])
    header_button_columns = header_columns[2].columns([.30, .30, .30])
    header_columns[0].write("")
    header_columns[0].selectbox(
        key=f"{object_type}_config_selectbox",
        label="Configuration",
        options=options,
        on_change=clear_tab_config,
        kwargs={"tab_key": tab_key},
        index=options.index(default)
    )
    
    current_config = available.get(st.session_state[f"{object_type}_config_selectbox"])
    backends = st.session_state["CLASSES"][object_type].supported_backends if hasattr(st.session_state["CLASSES"][object_type], "supported_backends") else None
    default_models = st.session_state["CLASSES"][object_type].default_models if hasattr(st.session_state["CLASSES"][object_type], "default_models") else None

    if object_type in ["transcriber", "synthesizer"]:
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
    elif object_type == "speech_recorder":
        pass

    st.write("")
    for parameter in get_json_parameters(object_type):
        render_json_input(parent_widget=st, 
                        key=f"{tab_key}_{parameter}", 
                        label=" ".join(parameter.split("_")).capitalize(),
                        default_data={} if current_config is None else getattr(current_config, parameter))

    header_button_columns[0].write("#####")
    object_title = " ".join(object_type.split("_")).title()
    with header_button_columns[0].popover("Overwrite", disabled=current_config is None, 
                                help="Overwrite the current configuration"):
            st.write(f"{object_title} configuration {st.session_state[f'{object_type}_config_selectbox']} will be overwritten.")
            
            if st.button("Approve"):
                obj_id = st.session_state["CONTROLLER"].patch_object(
                    object_type,
                    st.session_state[f"{object_type}_config_selectbox"],
                    **gather_config(object_type)
                )
                st.info(f"Updated {object_title} configuration {obj_id}.")

    header_button_columns[1].write("#####")
    if header_button_columns[1].button("Add new", help="Add new entry with the below configuration if it does not exist yet."):
        obj_id = st.session_state["CONTROLLER"].put_object(
            object_type,
            **gather_config(object_type)
        )
        if obj_id in available:
            st.info(f"Configuration already found under ID {obj_id}.")
        else:
            st.info(f"Created new configuration with ID {obj_id}.")
        st.session_state[f"{tab_key}_overwrite_config_id"] = obj_id
    if st.session_state.get(f"{tab_key}_overwrite_config_id", st.session_state[f"{object_type}_config_selectbox"]) != st.session_state[f"{object_type}_config_selectbox"]:
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
        render_config("transcriber")

    with synthesizer_tab:
        st.header("A dog")
        st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

    with recorder_tab:
        st.header("An owl")
        st.image("https://static.streamlit.io/examples/owl.jpg", width=200)
    render_sidebar()