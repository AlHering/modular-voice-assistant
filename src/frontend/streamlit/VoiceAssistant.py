# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import streamlit as st

from src.frontend.streamlit.utility.frontend_rendering import render_sidebar, render_pipeline_node_plane
from src.frontend.streamlit.utility.backend_interaction import AVAILABLE_MODULES


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
        
    # Page content
    st.title("Voice Assistant")        

    # Wait for backend and dependencies
    if "SETUP" not in st.session_state or not st.session_state["SETUP"]:
        st.write("Please choose a mode in the sidebar and press the setup button.")
    else:    
        render_pipeline_node_plane(parent_widget=st.container(),
                                block_dict=AVAILABLE_MODULES,
                                session_state_key="pipeline_schema")
    
    render_sidebar()