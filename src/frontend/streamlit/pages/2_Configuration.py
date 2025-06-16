# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import streamlit as st
from typing import List, Any
import json
import copy
from src.utility.streamlit_utility import render_json_input
from src.frontend.streamlit.utility.backend_interaction import AVAILABLE_SERVICES, DEFAULTS, CONFIGURATION_PARAMETERS, validate_config, put_config, delete_config, get_configs, patch_config
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
    data = {}
    if isinstance(CONFIGURATION_PARAMETERS[object_type], list):
        current_sub_type = st.session_state["configuration_subtype"]
        param_spec = copy.deepcopy([entry for entry in CONFIGURATION_PARAMETERS[object_type] if entry["#option"] == current_sub_type][0])
        param_spec.pop("#option")
    else:
        param_spec = CONFIGURATION_PARAMETERS[object_type]
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
    backends = DEFAULTS.get(object_type, {}).get("backends")
    default_models = DEFAULTS.get(object_type, {}).get("defaults")
    if object_type in AVAILABLE_SERVICES:
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
    
    if isinstance(CONFIGURATION_PARAMETERS[object_type], list):
        if current_config:
            current_entry = sorted([[len([param for param in current_config if param in entry]), entry]for entry in CONFIGURATION_PARAMETERS[object_type]])[-1][1]["#option"]
            if "configuration_subtype" in st.session_state and st.session_state["configuration_subtype"] != current_entry:
                    st.session_state["configuration_subtype"] = current_entry

        current_option = parent_widget.selectbox(
            key="configuration_subtype", 
            label=f"{object_type} Config Type", 
            options=[entry["#option"] for entry in CONFIGURATION_PARAMETERS[object_type]])
        param_spec = [entry for entry in CONFIGURATION_PARAMETERS[object_type] if entry["#option"] == current_option][0]
    else:
        param_spec = CONFIGURATION_PARAMETERS[object_type]
    for param in [param for param in param_spec if param not in ["#option"]]:
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

    object_title = object_type
    header_button_columns[0].write("#####")
    with header_button_columns[0].popover("Validate",
                                          help="Validates the current configuration"):
            st.write(f"Validations can result in errors or warnings.")
            
            if st.button("Approve", key=f"{tab_key}_validate_approve_btn", disabled=True):
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
                ).get("uuid")
                st.info(f"Updated {object_title} configuration {obj_id}.")

    header_button_columns[2].write("#####")
    if header_button_columns[2].button("Add new", 
                                       key=f"{tab_key}_add_btn",
                                       help="Add new entry with the below configuration if it does not exist yet."):
        obj_id = put_config(
            config_type=object_type,
            config_data=gather_config(object_type)
        ).get("uuid")
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
                ).get("uuid")
                st.info(f"Deleted {object_title} configuration {obj_id}.")
                ids = [st.session_state[f"{tab_key}_available"][elem]["uuid"] 
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
        entry["uuid"]: entry for entry in get_configs(config_type=object_type)
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
        st.info("System inactive. Please enter a correct backend server API in the sidebar (Local example: 'http://127.0.0.1:7861/api/v1').")
    else:
        tabs = list(AVAILABLE_SERVICES.keys())
        for index, tab in enumerate(st.tabs(tabs)):
            with tab:
                render_config(tabs[index])
            
    render_sidebar()