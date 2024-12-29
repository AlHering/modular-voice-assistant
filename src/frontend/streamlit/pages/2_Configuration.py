# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import streamlit as st
from typing import List, Any
from inspect import getfullargspec
import json
from copy import deepcopy
from src.utility.streamlit_utility import render_json_input
from src.frontend.streamlit.utility.backend_interaction import AVAILABLE_MODULES, fetch_default_config
from src.frontend.streamlit.utility.state_cache_handling import save_config
from src.frontend.streamlit.utility.frontend_rendering import render_sidebar


###################
# Main page functionality
###################
def gather_config(object_type: str) -> dict:
    """
    Gathers object config.
    :param object_type: Target object type.
    :return: Object config.
    """
    object_class = AVAILABLE_MODULES[object_type]
    data = {}
    param_spec = retrieve_parameter_specification(object_class.__init__, ignore=["self"])
    for param in param_spec:
        if param_spec[param]["type"] == dict:
            widget = st.session_state[f"new_{object_type}_{param}"]
            data[param] = json.loads(widget["text"]) if widget is not None else None
        else:
            if f"new_{object_type}_{param}" in st.session_state:
                data[param] = param_spec[param]["type"](st.session_state[f"new_{object_type}_{param}"])
            else:
                data[param] = param_spec[param].get("default")
    return data


def retrieve_type(input_type: Any) -> Any:
    """
    Retrieves type of input.
    :param input_type: Inspected input type hint.
    :return: Target type.
    """
    base_data_types = [str, bool, int, float, complex, list, tuple, range, dict, set, frozenset, bytes, bytearray, memoryview]
    if input_type in base_data_types:
        return input_type
    if input_type == callable:
        return callable
    elif str(input_type).startswith("typing.List"):
        return list
    elif str(input_type).startswith("typing.Dict"):
        return dict
    elif str(input_type).startswith("typing.Tuple"):
        return tuple
    elif str(input_type).startswith("typing.Set"):
        return set
    else:
        for data_type in base_data_types:
            string_representation = str(data_type).split("'")[1]
            if string_representation in str(input_type):
                return data_type
            

def retrieve_parameter_specification(func: callable, ignore: List[str] | None = None) -> dict:
    """
    Retrieves parameter specification.
    :param func: Callable to retrieve parameter specs from.
    :param ignore: Parameters to ignore.
    :return: Specification of parameters.
    """
    ignore = [] if ignore is None else ignore

    spec = {}
    arg_spec = getfullargspec(func)
    default_offset = len(arg_spec.args) - len(arg_spec.defaults)

    for param_index, param in enumerate(arg_spec.args):
        spec[param] = {}
        if param in arg_spec.annotations:
            spec[param]["type"] = retrieve_type(arg_spec.annotations[param])
        if param_index < default_offset:
            spec[param]["default"] = arg_spec.defaults[param_index]
    for ignored_param in ignore:
        if ignored_param in spec:
            spec.pop(ignored_param)
    return spec


def render_config_inputs(parent_widget: Any, 
                         tab_key: str, 
                         object_type: str) -> None:
    """
    Renders config inputs.
    :param parent_widget: Parent widget.
    :param tab_key: Current tab key.
    :param object_type: Target object type.
    """
    object_class = AVAILABLE_MODULES[object_type]
    backends = object_class.supported_backends if hasattr(object_class, "supported_backends") else None
    default_models = object_class.default_models if hasattr(object_class, "default_models") else None
    if object_type == "speech_recorder":
        input_devices = {entry["name"]: entry["index"] 
                         for entry in sorted(
                             AVAILABLE_MODULES[object_type].supported_input_devices,
                             key=lambda x: x["index"])}
        input_device_column, loop_pause_column, _ = parent_widget.columns([.25, .25, .50])
        device_name = input_device_column.selectbox(
            key=f"{tab_key}_input_device_name", 
            label="Input device", 
            options=list(input_devices.keys()))
        st.session_state[f"{tab_key}_input_device_index"] = input_devices[device_name]
        parent_widget.markdown("""
        <style>
            button.step-up {display: none;}
            button.step-down {display: none;}
            div[data-baseweb] {border-radius: 4px;}
        </style>""",
        unsafe_allow_html=True)
        loop_pause_column.number_input(
            "Loop pause",
            key=f"{tab_key}_recorder_loop_pause", 
            format="%0.2f",
            step=0.1,
            min_value=0.01,
            max_value=10.1,
        )
    elif object_type in AVAILABLE_MODULES:
        if backends is not None:
            parent_widget.selectbox(
                key=f"{tab_key}_backend", 
                label="Backend", 
                options=backends)
        if default_models is not None:
            if f"{tab_key}_model_path" not in st.session_state:
                st.session_state[f"{tab_key}_model_path"] = default_models[st.session_state[f"{tab_key}_backend"]][0]
            parent_widget.text_input(
                key=f"{tab_key}_model_path", 
                label="Model (Model name or path)")

        parent_widget.write("")
        
    param_spec = retrieve_parameter_specification(object_class.__init__, ignore=["self", "backend", "model_path", "recorder_loop_pause", "input_device_index"])
    for param in param_spec:
        if param_spec[param]["type"] == str:
            parent_widget.text_input(
                key=f"{tab_key}_{param}", 
                label=" ".join(param.split("_")).title(),
                value=param_spec[param].get("default", ""))
        elif param_spec[param]["type"] in [int, float]:
            parent_widget.number_input(
                key=f"{tab_key}_{param}", 
                label=" ".join(param.split("_")).title(),
                value=param_spec[param].get("default", .0 if param_spec[param]["type"] == float else 0))
        elif param_spec[param]["type"]  == dict:
            render_json_input(parent_widget=parent_widget, 
                    key=f"{tab_key}_{param}", 
                    label=" ".join(param.split("_")).title(),
                    default_data={})
        

def render_header_buttons(parent_widget: Any, 
                          tab_key: str, 
                          object_type: str) -> None:
    """
    Renders header buttons.
    :param tab_key: Current tab key.
    :param parent_widget: Parent widget.
    :param object_type: Target object type.
    """
    changed = False
    notification_status = parent_widget.empty()
    notification_info = parent_widget.empty()
    header_button_columns = parent_widget.columns([.2, .2, .2, .2, .2])

    object_title = " ".join(object_type.split("_")).title()
    header_button_columns[0].write("#####")
    if header_button_columns[0].button("Save", key=f"{tab_key}_approve_btn", help="Saves the current configuration"):
        st.session_state["CACHE"][object_type] = deepcopy(gather_config(object_type=object_type))
        save_config(object_type=object_type)
        st.info(f"Updated {object_title} configuration.")
        changed = True

    header_button_columns[1].write("#####")
    with header_button_columns[1].popover("Validate",
                                          help="Reset the current configuration"):
        st.write(f"{object_title} configuration will be reset!")
        if st.button("Approve", 
                                        key=f"{tab_key}_validate_btn",
                                        help="Validate the current configuration."):
            pass
    
    header_button_columns[2].write("#####")
    with header_button_columns[2].popover("Reset",
                                          help="Reset the current configuration"):
        st.write(f"{object_title} configuration will be reset!")
        
        if st.button("Approve", key=f"{tab_key}_delapprove_btn",):
            st.session_state["CACHE"][object_type] = deepcopy(fetch_default_config()[object_type])
            save_config(object_type=object_type)
            st.info(f"{object_title} configuration was reset.")
            changed = True

    if changed:
        st.rerun()


def render_config(object_type: str) -> None:
    """
    Renders configs.
    :param object_type: Target object type.
    """
    tab_key = f"new_{object_type}"
    header = st.empty()
    
    render_config_inputs(parent_widget=st,
                         tab_key=tab_key,
                         object_type=object_type)

    render_header_buttons(parent_widget=header,
                          tab_key=tab_key,
                          object_type=object_type)
    
    

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
        
    # Page content
    st.title("Configuration")

     # Wait for backend and dependencies
    if "SETUP" not in st.session_state or not st.session_state["SETUP"]:
        st.write("Please choose a mode in the sidebar and press the setup button.")
    else:
        tabs = list(AVAILABLE_MODULES.keys())
        for index, tab in enumerate(st.tabs([" ".join(elem.split("_")).title()+"s" for elem in tabs])):
            with tab:
                render_config(tabs[index])
            
    render_sidebar()