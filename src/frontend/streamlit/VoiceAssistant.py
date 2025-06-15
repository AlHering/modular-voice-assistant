# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from typing import Any
import streamlit as st
import requests
from src.frontend.streamlit.utility.backend_interaction import AVAILABLE_SERVICES, unload_service, get_configs, chat, chat_streamed, load_service, get_loaded_service
from src.frontend.streamlit.utility.backend_interaction import record_and_transcribe_speech, reset_service
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
        service_type,
        key=f"active_{service_type}",
        placeholder="None",
        options=st.session_state["available_services"][service_type],
        index=st.session_state["available_services"][service_type].index(st.session_state["loaded_services"][service_type])
    )

    loaded = st.session_state[f"active_{service_type}"] == st.session_state["loaded_services"][service_type]
    if not loaded:
        if st.session_state[f"active_{service_type}"] is not None:
            with st.spinner("Loading service..."):
                if not "error" in load_service(service_type=service_type, config_uuid=st.session_state[f"active_{service_type}"]):
                    st.session_state["loaded_services"][service_type] = st.session_state[f"active_{service_type}"]
                    st.rerun()
        elif st.session_state["loaded_services"][service_type] is not None:
            if not "error" in unload_service(service_type=service_type):
                st.session_state["loaded_services"][service_type] = None
                st.rerun()


def handle_prompting(prompt_box: Any, 
                     response_box: Any, 
                     chat_kwargs: dict,
                     stream: bool) -> str:
    """
    Handles prompt interaction.
    :param prompt_box: Prompt box widget.
    :param response_box: Response box widget.
    :param chat_kwargs: Chat keyword arguments.
    :param stream: Flag declaring whether or not to stream response.
    """
    prompt_message = prompt_box.chat_message("user")
    prompt_message.write(chat_kwargs["prompt"])
    st.session_state["chat_history"].append({"role": "user", "content": chat_kwargs["prompt"]})
    new_response = response_box.chat_message("assistant")
    if stream:
        response_content = new_response.write_stream(chat_streamed(**chat_kwargs))
    else:
        response_content = chat(**chat_kwargs)
        new_response.write(response_content)
    st.session_state["chat_history"].append({"role": "assistant", "content": response_content})


def main_page_content() -> None:
    """
    Renders main page content.
    """
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    st.session_state["loaded_services"] = get_loaded_service() 
    st.session_state["available_services"] = {
        service_type: [None] + [entry["uuid"] for entry in get_configs(config_type=service_type) if not entry["inactive"]]
            for service_type in AVAILABLE_SERVICES
        }
    
    service_buttons = st.columns(6)
    if service_buttons[0].button("Load default services..."):
        for service_type in st.session_state["available_services"]:
            with st.spinner(f"Loading {service_type} service..."):
                target_uuid = st.session_state["available_services"][service_type][1]
                if not "error" in load_service(service_type=service_type, config_uuid=target_uuid):
                    st.session_state["loaded_services"][service_type] = target_uuid
                    del st.session_state[f"active_{service_type}"]
        st.rerun()
    if service_buttons[1].button("Unload all services..."):
        for service_type in st.session_state["loaded_services"]:
            if st.session_state["loaded_services"][service_type] is not None:
                with st.spinner(f"Unloading {service_type} service..."):
                    if not "error" in unload_service(service_type=service_type):
                        st.session_state["loaded_services"][service_type] = None
                        del st.session_state[f"active_{service_type}"]
        st.rerun()
    upper_service_columns = st.columns(3)
    for service_index, service_type in enumerate([key for key in AVAILABLE_SERVICES]):
        render_service_control(parent_widget=upper_service_columns[service_index], service_type=service_type)
        
    st.divider()
    chat_flag_columns = st.columns(6)
    stream_response = chat_flag_columns[0].checkbox("Stream Generation",)
    output_as_audio = chat_flag_columns[1].checkbox("Output Speech")
    if chat_flag_columns[2].button("Interrupt"):
        _ = requests.post(st.session_state["API_BASE"] + "/interrupt")
    if chat_flag_columns[3].button("Clear Chat"):
        st.session_state["chat_history"] = []
        if st.session_state["loaded_services"]["chat"]:
            with st.spinner("Resetting service..."):
                reset_service("Chat", st.session_state["loaded_services"]["chat"])
    if not st.session_state["loaded_services"]["synthesizer"] and output_as_audio:
        output_as_audio = False
        st.error("Synthesizer service needs to be loaded.")

    st.divider()
    st.write("##### Chat")
    for message in st.session_state["chat_history"]:
        message_box = st.chat_message(message["role"])
        message_box.write(message["content"])
    prompt_box = st.empty()
    response_box = st.empty()
    interaction_box = st.empty()
    interaction_columns = interaction_box.columns([.8, .2])

    # text input
    active_worker = st.session_state["loaded_services"]["chat"]
    prompt = interaction_columns[0].chat_input("🖊️ Write something")
    if prompt:
        if active_worker:
            chat_kwargs = {
                "prompt": prompt,
                "output_as_audio": output_as_audio
            }
            handle_prompting(prompt_box=prompt_box,
                             response_box=response_box,
                             chat_kwargs=chat_kwargs,
                             stream=stream_response)
        else:
            st.error("Chat service needs to be loaded.")

    # speech input
    active_transcriber = st.session_state["loaded_services"]["transcriber"]
    voice_input = interaction_columns[1].button("🎙️ Say something")
    if voice_input:
        if active_transcriber is not None:
            prompt = record_and_transcribe_speech()
            chat_kwargs = {
                "prompt": prompt,
                "output_as_audio": output_as_audio
            }
            handle_prompting(prompt_box=prompt_box,
                             response_box=response_box,
                             chat_kwargs=chat_kwargs,
                             stream=stream_response)
        else:
            st.error("Transcriber service needs to be loaded.")


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