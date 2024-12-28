# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import streamlit as st
from typing import List, Any, Optional
from requests.exceptions import ConnectionError
from typing import get_type_hints
from inspect import getfullargspec
import json
from copy import deepcopy
from time import sleep
from src.utility.streamlit_utility import render_json_input
from src.frontend.streamlit.utility.backend_interaction import AVAILABLE_MODULES, fetch_default_config
from src.frontend.streamlit.utility.state_cache_handling import wait_for_setup, clear_tab_config
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
    for parameter in param_spec:
        if param_spec[parameter]["type"] == dict:
            widget = st.session_state[f"new_{object_type}_{key}"]
            data[parameter] = json.loads(widget["text"]) if widget is not None else None
        else:
            data[parameter] = param_spec[parameter]["type"](st.session_state[f"new_{object_type}_{key}"])
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
            if str(data_type) in str(input_type):
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
        spec[param] = {
            "type": retrieve_type(arg_spec.annotations[param])
        }
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
    current_config = st.session_state.get(f"{tab_key}_current")
    if object_type == "speech_recorder":
        input_devices = {entry["name"]: entry["index"] 
                         for entry in sorted(
                             st.session_state["CLASSES"][object_type].supported_input_devices,
                             key=lambda x: x["index"])}
        input_device_column, loop_pause_column, _ = parent_widget.columns([.25, .25, .50])
        current_device_index = 0 
        if current_config is not None:
            for device_name in input_devices:
                if current_config["input_device_index"] == input_devices[device_name]:
                    current_device_index = input_devices[device_name]
                    break
        device_name = input_device_column.selectbox(
            key=f"{tab_key}_input_device_name", 
            label="Input device", 
            options=list(input_devices.keys()), 
            index=current_device_index)
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
            key=f"{tab_key}_loop_pause", 
            format="%0.2f",
            step=0.1,
            min_value=0.01,
            max_value=10.1,
            value=.1 if current_config is None else current_config["loop_pause"]
        )
    elif object_type in AVAILABLE_MODULES:
        object_class = AVAILABLE_MODULES[object_type]
        backends = object_class.supported_backends if hasattr(object_class, "supported_backends") else None
        default_models = object_class.default_models if hasattr(object_class, "default_models") else None
        if backends is not None:
            parent_widget.selectbox(
                key=f"{tab_key}_backend", 
                label="Backend", 
                options=backends, 
                index=0 if current_config is None else backends.index(current_config["backend"]))
        if default_models is not None:
            if f"{tab_key}_model_path" not in st.session_state:
                st.session_state[f"{tab_key}_model_path"] = default_models[st.session_state[f"{tab_key}_backend"]][0] if (
                current_config is None or current_config["model_path"] is None) else current_config["model_path"]
            parent_widget.text_input(
                key=f"{tab_key}_model_path", 
                label="Model (Model name or path)")

        parent_widget.write("")
        
        param_spec = retrieve_parameter_specification(object_class.__init__, ignore=["self", "backend", "model_path", "loop_pause"])
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
                        default_data={} if current_config is None or not not current_config.get(param, {}) else current_config[param])
        

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
    current_config = st.session_state.get(f"{tab_key}_current")
    header_button_columns = parent_widget.columns([.30, .30, .30])

    object_title = " ".join(object_type.split("_")).title()
    header_button_columns[0].write("#####")
    with header_button_columns[0].popover("Overwrite",
                                          disabled=current_config is None, 
                                          help="Overwrite the current configuration"):
            st.write(f"{object_title} configuration {st.session_state[f'{object_type}_config_selectbox']} will be overwritten.")
            
            if st.button("Approve", key=f"{tab_key}_approve_btn",):
                st.session_state["CACHE"][object_type] = deepcopy(gather_config(object_type=object_type))
                st.info(f"Updated {object_title} configuration.")
                changed = True

    header_button_columns[1].write("#####")
    if header_button_columns[1].button("Add new", 
                                       key=f"{tab_key}_add_btn",
                                       help="Add new entry with the below configuration if it does not exist yet."):
        pass
    
    header_button_columns[2].write("#####")
    with header_button_columns[2].popover("Reset",
                                          disabled=current_config is None, 
                                          help="Reset the current configuration"):
            st.write(f"{object_title} configuration will be reset!")
            
            if st.button("Approve", key=f"{tab_key}_delapprove_btn",):
                st.session_state["CACHE"][object_type] = deepcopy(fetch_default_config()[object_type])
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

    # Wait for backend and dependencies
    wait_for_setup()
        
    # Page content
    st.title("Configuration")
    
    tabs = list(AVAILABLE_MODULES.keys())
    for index, tab in enumerate(st.tabs([" ".join(elem.split("_")).title()+"s" for elem in tabs])):
        with tab:
            render_config(tabs[index])
            
    render_sidebar()