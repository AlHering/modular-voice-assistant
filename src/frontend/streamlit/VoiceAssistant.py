# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from typing import Any
import streamlit as st
from src.frontend.streamlit.utility.backend_interaction import AVAILABLE_SERVICES, SERVICE_TITLES, unload_service, get_configs, chat, chat_streamed, load_service, get_loaded_service
from src.frontend.streamlit.utility.frontend_rendering import render_sidebar


###################
# Main page functionality
###################

def render_service_control(parent_widget: Any, service_type: str) -> None:
    """
    Renders service control.
    :param parent_widget: Parent widget.
    :param service_type: Service type.
    """
    parent_widget.selectbox(
        SERVICE_TITLES[service_type],
        key=f"active_{service_type}",
        options=st.session_state[f"available_services"][service_type],
        index=st.session_state[f"available_services"][service_type].index(st.session_state["loaded_services"][service_type])
    )
    if st.session_state["loaded_services"][service_type]:
        parent_widget.write("   " + st.session_state["loaded_services"][service_type] + " is active.")
    else:
        parent_widget.write("No active configuration.")

    loaded = st.session_state[f"active_{service_type}"] == st.session_state["loaded_services"][service_type]
    control_columns = parent_widget.columns([.3, .3, .4])
    if control_columns[0].button("Unload", key=f"unload_{service_type}_btn", disabled=not loaded):
        if st.session_state[f"active_{service_type}"] is not None:
            if not "error" in unload_service(service_type=service_type, config_uuid=st.session_state[f"active_{service_type}"]):
                st.session_state["loaded_services"][service_type] = None
                st.rerun()
    if control_columns[1].button("Load", key=f"load_{service_type}_btn", disabled=st.session_state[f"active_{service_type}"] is None or loaded):
        if st.session_state[f"active_{service_type}"] is not None:
            if not "error" in load_service(service_type=service_type, config_uuid=st.session_state[f"active_{service_type}"]):
                st.session_state["loaded_services"][service_type] = st.session_state[f"active_{service_type}"]
                st.rerun()


def main_page_content() -> None:
    """
    Renders main page content.
    """
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
        st.session_state["loaded_services"] = get_loaded_service() 
    st.session_state["available_services"] = {
        service_type: [None] + [entry["id"] for entry in get_configs(config_type=service_type) if not entry["inactive"]]
            for service_type in AVAILABLE_SERVICES
        }
    column_count = (len(list(AVAILABLE_SERVICES.keys()))+1) // 2
    upper_service_columns = st.columns(column_count)
    lower_service_columns = st.columns(column_count)
    control_service_columns = st.columns(10)
    for service_index, service_type in enumerate([key for key in AVAILABLE_SERVICES][:column_count]):
        render_service_control(parent_widget=upper_service_columns[service_index], service_type=service_type)
    for service_index, service_type in enumerate([key for key in AVAILABLE_SERVICES][column_count:]):
        render_service_control(parent_widget=lower_service_columns[service_index], service_type=service_type)

    streamed = control_service_columns[0].checkbox("Stream responses.")

    st.title("Chat")
    for message in st.session_state["chat_history"]:
        message_box = st.chat_message(message["role"])
        message_box.write(message["content"])
    prompt_box = st.empty()
    new_response_box = st.empty()

    active_worker = st.session_state["loaded_services"]["remote_chat"]
    if active_worker is None:
        active_worker = st.session_state["loaded_services"]["local_chat"]

    prompt = st.chat_input("Say something", disabled=active_worker is None)

    if prompt:
        prompt_message = prompt_box.chat_message("user")
        prompt_message.write(prompt)
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        new_response = new_response_box.chat_message("assistant")
        chat_kwargs = {
            "config_uuid": active_worker,
            "prompt": prompt,
            "local": active_worker in st.session_state["available_services"]["local_chat"]
        }
        if streamed:
            response_content = new_response.write_stream(chat_streamed(**chat_kwargs))
        else:
            response_content = chat(**chat_kwargs)
            new_response.write(response_content)
        st.session_state["chat_history"].append({"role": "assistant", "content": response_content})



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
    st.title("Voice Assistant")        

    # Wait for backend and dependencies
    if "SETUP" not in st.session_state or not st.session_state["SETUP"]:
        st.info("System inactive. Please choose a Setup Mode in the sidebar and press the Setup button.")
    else:    
        main_page_content()
    
    render_sidebar()