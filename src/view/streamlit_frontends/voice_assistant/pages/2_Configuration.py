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
        params = ["backend", "model_path"]
    elif object_type == "speech_recorder":
        params = ["input_device_index", "loop_pause"]
    data = {key: st.session_state[f"new_{object_type}_{key}"] for key in params}
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
    options = [">> New <<"] + list(available.keys())
    default = st.session_state.get(f"{tab_key}_overwrite_config_id", st.session_state.get(f"{object_type}_config_selectbox", ">> New <<"))
    
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
    
    if object_type in ["transcriber", "synthesizer"]:
        backends = st.session_state["CLASSES"][object_type].supported_backends
        default_models = st.session_state["CLASSES"][object_type].default_models

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
        input_devices = {entry["name"]: entry["index"] 
                         for entry in sorted(
                             st.session_state["CLASSES"][object_type].supported_input_devices,
                             key=lambda x: x["index"])}
        input_device_column, loop_pause_column, _ = st.columns([.25, .25, .50])
        current_device_index = 0 
        if current_config is not None:
            for device_name in input_devices:
                if current_config.input_device_index == input_devices[device_name]:
                    current_device_index = input_devices[device_name]
                    break
        device_name = input_device_column.selectbox(
            key=f"{tab_key}_input_device_name", 
            label="Input device", 
            options=list(input_devices.keys()), 
            index=current_device_index)
        st.session_state[f"{tab_key}_input_device_index"] = input_devices[device_name]
        st.markdown("""
        <style>
            button.step-up {display: none;}
            button.step-down {display: none;}
            div[data-baseweb] {border-radius: 4px;}
        </style>""",
        unsafe_allow_html=True)
        loop_pause_column.number_input(
            "Loop pause",
            key=f"{tab_key}_loop_pause", 
            format="%0.2f",
            step=0.1,
            min_value=0.01,
            max_value=10.1,
            value=.1 if current_config is None else current_config.loop_pause
        )


    st.write("")
    for parameter in get_json_parameters(object_type):
        render_json_input(parent_widget=st, 
                        key=f"{tab_key}_{parameter}", 
                        label=" ".join(parameter.split("_")).capitalize(),
                        default_data={} if current_config is None else getattr(current_config, parameter))

    header_button_columns[0].write("#####")
    object_title = " ".join(object_type.split("_")).title()
    with header_button_columns[0].popover("Overwrite",
                                          disabled=current_config is None, 
                                          help="Overwrite the current configuration"):
            st.write(f"{object_title} configuration {st.session_state[f'{object_type}_config_selectbox']} will be overwritten.")
            
            if st.button("Approve", key=f"{tab_key}_approve_btn",):
                obj_id = st.session_state["CONTROLLER"].patch_object(
                    object_type,
                    st.session_state[f"{object_type}_config_selectbox"],
                    **gather_config(object_type)
                )
                st.info(f"Updated {object_title} configuration {obj_id}.")

    header_button_columns[1].write("#####")
    if header_button_columns[1].button("Add new", 
                                       key=f"{tab_key}_add_btn",
                                       help="Add new entry with the below configuration if it does not exist yet."):
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
    
    tabs = ["transcriber", "synthesizer", "speech_recorder"]
    for index, tab in enumerate(st.tabs([" ".join(elem.split("_")).title()+"s" for elem in tabs])):
        with tab:
            render_config(tabs[index])
            
    render_sidebar()