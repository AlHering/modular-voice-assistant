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
from .frontend_rendering import render_sidebar


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
    with st.spinner("Waiting for backend to finish startup..."):#
        pass
                
        
    # Page content
    st.title("Voice Assistant")
    render_sidebar()
