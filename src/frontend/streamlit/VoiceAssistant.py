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
        placeholder="None",
        options=st.session_state[f"available_services"][service_type],
        index=st.session_state[f"available_services"][service_type].index(st.session_state["loaded_services"][service_type])
    )

    loaded = st.session_state[f"active_{service_type}"] == st.session_state["loaded_services"][service_type]
    if not loaded:
        if st.session_state[f"active_{service_type}"] is not None:
            with st.spinner("Loading service..."):
                if not "error" in load_service(service_type=service_type, config_uuid=st.session_state[f"active_{service_type}"]):
                    st.session_state["loaded_services"][service_type] = st.session_state[f"active_{service_type}"]
                    st.rerun()
        elif st.session_state["loaded_services"][service_type] is not None:
            if not "error" in unload_service(service_type=service_type, config_uuid=st.session_state[f"active_{service_type}"]):
                st.session_state["loaded_services"][service_type] = None
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
    for service_index, service_type in enumerate([key for key in AVAILABLE_SERVICES][:column_count]):
        render_service_control(parent_widget=upper_service_columns[service_index], service_type=service_type)
    for service_index, service_type in enumerate([key for key in AVAILABLE_SERVICES][column_count:]):
        render_service_control(parent_widget=lower_service_columns[service_index], service_type=service_type)
        
    st.divider()
    st.write("##### Flags")
    control_service_columns = st.columns(6)
    streamed = control_service_columns[0].checkbox("Streamed Generation")

    st.divider()
    st.write("##### Chat")
    for message in st.session_state["chat_history"]:
        message_box = st.chat_message(message["role"])
        message_box.write(message["content"])
    prompt_box = st.empty()
    new_response_box = st.empty()

    interaction_columns = st.columns([.8, .2])
    # text input
    active_worker = st.session_state["loaded_services"]["remote_chat"]
    if active_worker is None:
        active_worker = st.session_state["loaded_services"]["local_chat"]
    prompt = interaction_columns[0].chat_input("🖊️ Write something")
    if prompt:
        if active_worker:
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
        else:
            st.error("Remote or Local Chat need to be loaded.")
    # speech input
    active_speech_recorder = st.session_state["loaded_services"]["speech_recorder"]
    active_transcriber = st.session_state["loaded_services"]["transcriber"]

    voice_input = interaction_columns[1].button("🎙️ Say something")
    if voice_input:
        if active_speech_recorder is not None and active_transcriber is not None:
            pass
        else:
            st.error("Speech Recorder and Transcriber need to be loaded.")


    



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
        st.info("System inactive. Please enter a correct backend server API in the sidebar (Local example: 'http://127.0.0.1:7861/api/v1').")
    else:    
        main_page_content()
    
    render_sidebar()