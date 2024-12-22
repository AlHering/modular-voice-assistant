# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import streamlit as st

from requests.exceptions import ConnectionError
from time import sleep
from src.frontend.streamlit.utility.frontend_rendering import render_sidebar, render_pipeline_node_plane
from src.frontend.streamlit.utility.backend_interaction import get_components
from src.frontend.streamlit.utility.state_cache_handling import wait_for_setup


###################
# Main page functionality
###################


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
    st.title("Voice Assistant")
    render_sidebar()
    render_pipeline_node_plane(parent_widget=st.container(),
                               block_entries=get_components(),
                               session_state_key="pipeline_schema")