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
from src.utility.streamlit_utility import render_json_input
from src.frontend.streamlit.utility.backend_interaction import AVAILABLE_SERVICES, SERVICE_TITLES, validate_config, put_config, delete_config, get_configs, patch_config
from src.frontend.streamlit.utility.state_cache_handling import clear_tab_config
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
    object_class = AVAILABLE_SERVICES[object_type]
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
    default_offset = len(arg_spec.args) - len(arg_spec.defaults) if arg_spec.defaults else None

    for param_index, param in enumerate(arg_spec.args):
        spec[param] = {"title": " ".join(param.split("_")).title()}
        if param in arg_spec.annotations:
            spec[param]["type"] = retrieve_type(arg_spec.annotations[param])
        if default_offset and param_index > default_offset:
            spec[param]["default"] = arg_spec.defaults[param_index-default_offset]
    for ignored_param in ignore:
        if ignored_param in spec:
            spec.pop(ignored_param)
    return spec


def get_default_value(key: str, current_config: dict | None, default: Any, options: List[Any] | None = None) -> Any:
    """
    Retrieves default value for configuration input widget.
    :param key: Target key.
    :param current_config: Current config.
    :param default: Default value or index.
    :param options: Options in case of selectbox.
    """
    if options is None:
        return default if (current_config is None 
                           or key not in current_config) else current_config[key]
    else:
        return default if (current_config is None 
                           or key not in current_config 
                           or current_config[key] not in options) else options.index(current_config[key])


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
    object_class = AVAILABLE_SERVICES[object_type]
    backends = object_class.supported_backends if hasattr(object_class, "supported_backends") else None
    default_models = object_class.default_models if hasattr(object_class, "default_models") else None
    if object_type == "speech_recorder":
        input_devices = {entry["name"]: entry["index"] 
                         for entry in sorted(
                             AVAILABLE_SERVICES[object_type].supported_input_devices,
                             key=lambda x: x["index"])}
        input_device_column, loop_pause_column, _ = parent_widget.columns([.25, .25, .50])
        current_device_index = 0 
        if current_config is not None:
            for device_name in input_devices:
                if current_config.get("input_device_index") == input_devices[device_name]:
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
            "Recorder loop pause",
            key=f"{tab_key}_recorder_loop_pause", 
            format="%0.2f",
            step=0.1,
            min_value=0.01,
            max_value=10.1,
            value=get_default_value(key="recorder_loop_pause",
                                    current_config=current_config,
                                    default=.1)
        )
    elif object_type in AVAILABLE_SERVICES:
        if backends is not None:
            parent_widget.selectbox(
                key=f"{tab_key}_backend", 
                label="Backend", 
                options=backends,
                index=get_default_value(key="backend",
                                        current_config=current_config,
                                        default=0,
                                        options=backends))
        if default_models is not None:
            if f"{tab_key}_model_path" not in st.session_state:
                st.session_state[f"{tab_key}_model_path"] = get_default_value(
                    key="model_path",
                    current_config=current_config,
                    default=default_models[st.session_state[f"{tab_key}_backend"]][0]
                )
            parent_widget.text_input(
                key=f"{tab_key}_model_path", 
                label="Model (Model name or path)")

        parent_widget.write("")
    
    if object_type not in st.session_state["CACHE"]["PARAM_SPECS"]:
        st.session_state["CACHE"]["PARAM_SPECS"][object_type] = retrieve_parameter_specification(
            object_class.__init__, ignore=["self", "backend", "model_path", "recorder_loop_pause", "input_device_index"])

    param_spec = st.session_state["CACHE"]["PARAM_SPECS"][object_type]
    for param in param_spec:
        if param_spec[param]["type"] == str:
            parent_widget.text_input(
                key=f"{tab_key}_{param}", 
                label=param_spec[param]["title"],
                value=get_default_value(
                    key=param,
                    current_config=current_config,
                    default=param_spec[param].get("default", "")
                ))
        elif param_spec[param]["type"] in [int, float]:
            parent_widget.number_input(
                key=f"{tab_key}_{param}", 
                label=param_spec[param]["title"],
                value=get_default_value(
                    key=param,
                    current_config=current_config,
                    default=param_spec[param].get("default", .0 if param_spec[param]["type"] == float else 0)
                ))
        elif param_spec[param]["type"]  == dict:
            render_json_input(parent_widget=parent_widget, 
                    key=f"{tab_key}_{param}", 
                    label=param_spec[param]["title"],
                    default_data={} if current_config is None or not current_config.get(param, {}) else current_config[param])
        

def render_header_buttons(parent_widget: Any, 
                          tab_key: str, 
                          object_type: str) -> None:
    """
    Renders header buttons.
    :param tab_key: Current tab key.
    :param parent_widget: Parent widget.
    :param object_type: Target object type.
    """
    current_config = st.session_state.get(f"{tab_key}_current")
    
    header_button_columns = parent_widget.columns([.2, .2, .2, .2, .2])

    object_title = SERVICE_TITLES[object_type]
    header_button_columns[0].write("#####")
    with header_button_columns[0].popover("Validate",
                                          help="Validates the current configuration"):
            st.write(f"Validations can result in errors or warnings.")
            
            if st.button("Approve", key=f"{tab_key}_validate_approve_btn",):
                config = gather_config(object_type)
                result = validate_config(config_type=object_type, config=config)
                if result[0] is None:
                    st.warning("Status: Warning")
                    st.warning("Reason: " + result[1])
                elif result[0]:
                    st.info("Status: Success")
                    st.info("Reason: " + result[1])
                else:
                    st.error("Status: Error")
                    st.error("Reason: " + result[1])
                
    header_button_columns[1].write("#####")
    with header_button_columns[1].popover("Overwrite",
                                          disabled=current_config is None, 
                                          help="Overwrite the current configuration"):
            st.write(f"{object_title} configuration {st.session_state[f'{object_type}_config_selectbox']} will be overwritten.")
            
            if st.button("Approve", key=f"{tab_key}_overwrite_approve_btn",):
                obj_id = patch_config(
                    config_type=object_type,
                    config_data=gather_config(object_type),
                    config_id=st.session_state[f"{object_type}_config_selectbox"]
                ).get("id")
                st.info(f"Updated {object_title} configuration {obj_id}.")

    header_button_columns[2].write("#####")
    if header_button_columns[2].button("Add new", 
                                       key=f"{tab_key}_add_btn",
                                       help="Add new entry with the below configuration if it does not exist yet."):
        obj_id = put_config(
            config_type=object_type,
            config_data=gather_config(object_type)
        ).get("id")
        if obj_id in st.session_state[f"{tab_key}_available"]:
            st.info(f"Configuration already found under ID {obj_id}.")
        else:
            st.info(f"Created new configuration with ID {obj_id}.")
        st.session_state[f"{tab_key}_overwrite_config_id"] = obj_id
    
    header_button_columns[3].write("#####")
    with header_button_columns[3].popover("Delete",
                                          disabled=current_config is None, 
                                          help="Delete the current configuration"):
            st.write(f"{object_title} configuration {st.session_state[f'{object_type}_config_selectbox']} will be deleted!")
            
            if st.button("Approve", key=f"{tab_key}_delete_approve_btn",):
                obj_id = delete_config(
                    config_type=object_type,
                    config_id=st.session_state[f"{object_type}_config_selectbox"]
                ).get("id")
                st.info(f"Deleted {object_title} configuration {obj_id}.")
                ids = [st.session_state[f"{tab_key}_available"][elem]["id"] 
                       for elem in st.session_state[f"{tab_key}_available"]]
                deleted_index = ids.index(st.session_state[f"{object_type}_config_selectbox"])
                if len(ids) > deleted_index+1:
                    st.session_state[f"{tab_key}_overwrite_config_id"] = ids[deleted_index+1]
                elif len(ids) > 1:
                    st.session_state[f"{tab_key}_overwrite_config_id"] = ids[deleted_index-1]
                else:
                    st.session_state[f"{tab_key}_overwrite_config_id"] = ">> New <<"

    if st.session_state.get(f"{tab_key}_overwrite_config_id", st.session_state[f"{object_type}_config_selectbox"]) != st.session_state[f"{object_type}_config_selectbox"]:
        st.rerun()
        

def render_config(object_type: str) -> None:
    """
    Renders configs.
    :param object_type: Target object type.
    """
    tab_key = f"new_{object_type}"
    st.session_state[f"{tab_key}_available"] = {
        entry["id"]: entry for entry in get_configs(config_type=object_type)
        if not entry["inactive"]}
    options = [">> New <<"] + list(st.session_state[f"{tab_key}_available"].keys())
    default = st.session_state.get(f"{tab_key}_overwrite_config_id", st.session_state.get(f"{object_type}_config_selectbox", ">> New <<"))
    
    header_columns = st.columns([.25, .10, .65])
    header_columns[0].write("")
    header_columns[0].selectbox(
        key=f"{object_type}_config_selectbox",
        label="Configuration",
        options=options,
        on_change=clear_tab_config,
        kwargs={"tab_key": tab_key},
        index=options.index(default)
    )
    st.session_state[f"{tab_key}_current"] = st.session_state[f"{tab_key}_available"].get(st.session_state[f"{object_type}_config_selectbox"])
    
    render_config_inputs(parent_widget=st,
                         tab_key=tab_key,
                         object_type=object_type)

    render_header_buttons(parent_widget=header_columns[2],
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
        st.info("System inactive. Please choose a Setup Mode in the sidebar and press the Setup button.")
    else:
        tabs = list(AVAILABLE_SERVICES.keys())
        tabs.remove("wave_output")
        for index, tab in enumerate(st.tabs([SERVICE_TITLES[elem]+"s" for elem in tabs])):
            with tab:
                render_config(tabs[index])
            
    render_sidebar()