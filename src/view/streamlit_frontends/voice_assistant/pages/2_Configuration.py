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
from src.view.streamlit_frontends.voice_assistant.utility.state_cache_handling import wait_for_setup
from src.view.streamlit_frontends.voice_assistant.utility.frontend_rendering import render_sidebar, render_transcriber_config


###################
# Main page functionality
###################


def fetch_available_configs(object_type: str) -> List[dict]:
    """
    Function for fetching available confis.
    :param object_type: Target object type.
    :return: List of config dictionaries.
    """
    return st.session_state["CLIENT"].get_configurations(object_type)


def fetch_available_object_types() -> dict:
    """
    Function for fetching available object types.
    :return: Dictionary, mapping object titles to internal object type strings.
    """
    return st.session_state["CACHE"]["object_types"]
    

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