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
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode
from streamlit_flow.layouts import TreeLayout
from src.configuration import configuration as cfg
from src.frontend.streamlit.utility.backend_interaction import AVAILABLE_SERVICES, SERVICE_TITLES, get_configs
from src.frontend.streamlit.utility.state_cache_handling import wait_for_setup
from streamlit_flow.state import StreamlitFlowState


###################
# Helper functions
###################


###################
# Rendering functions
###################


def render_sidebar() -> None:
    """
    Renders the sidebar.
    """
    
    mode = st.sidebar.selectbox(
        label="Setup Mode",
        options=["DIRECT", "API"])
    if st.sidebar.button(" Setup " + "(Status: " + ("Active)" if st.session_state.get("SETUP") else "Inactive)")):
        with st.spinner("Waiting for backend to finish startup..."):
            wait_for_setup(mode.lower())
    if mode == "API":
        api_base = f"http://{cfg.BACKEND_HOST}:{cfg.BACKEND_PORT}{cfg.BACKEND_ENDPOINT_BASE}/check"
        try:
            if requests.get(api_base).status_code == 200:
                st.sidebar.info("Backend server is running!")
        except requests.exceptions.ConnectionError:
            st.sidebar.error("Backend server is not available!")

    st.sidebar.write("#")
    st.sidebar.write("#")
    show_cache = st.sidebar.selectbox(
        label="Show Cache (Debug Mode)",
        options=["HIDE", "SHOW"])
    if show_cache == "SHOW":
        for key, value in st.session_state.items():
            st.sidebar.write(f"{key}: {value}")


def setup_default_flow() -> None:
    """
    Sets up default flow.
    """
    #TODO: Implement
    pass


def render_pipeline_node_plane(parent_widget: Any, block_dict: dict, session_state_key: str | None = None) -> None:
    """
    Renders a interactive node plane.
    :param parent_widget: Parent widget.
    :param block_dict: Dictionary for blocks.
    """
    if "flow" not in st.session_state:
        st.session_state["flow"] = StreamlitFlowState(nodes=[], edges=[])
    if "flow_modules" not in st.session_state:
        st.session_state["flow_modules"] = {key: {
            "available": {entry["id"]: entry for entry in get_configs(config_type=key)
                        if not  entry["inactive"]},
            "active": []
            } for key in AVAILABLE_SERVICES
        }


    node_menu_columns = parent_widget.columns([.25, .25, .10, .10, .10, .10, .10])
    
    node_menu_columns[0].write("")
    node_object_type = node_menu_columns[0].selectbox(
                key=f"flow_node_object_type", 
                label="Module type", 
                options=list(st.session_state["flow_modules"].keys()))
    node_menu_columns[1].write("")
    node_object_id = node_menu_columns[1].selectbox(
                key=f"flow_node_object_id", 
                label="Module UUID", 
                options=st.session_state["flow_modules"][node_object_type]["available"])
    target_node_flow_id = f"{node_object_type}_{node_object_id}"
    node_menu_columns[3].write("#####")
    if node_menu_columns[3].button(
        "Add", 
        key=f"add_node_btn", 
        disabled=node_object_id in st.session_state["flow_modules"][node_object_type]["active"]):
        node_type = "input" if node_object_type in [
            "speech_recorder"] else "output" if node_object_type in ["wave_output"] else "default"
        node_content = str(node_object_id)
        new_node = StreamlitFlowNode(
            id=target_node_flow_id, 
            pos=(0, 0), 
            data={"content": f"{SERVICE_TITLES[node_object_type]}\n\n{node_content}"}, 
            node_type=node_type, 
            source_position="right",
            target_position="left",
            selectable=True,
            connectable=True,
            draggable=True,
            resizing=True,
            deletable=True)
        st.session_state["flow"].nodes.append(new_node)
        st.session_state["flow_modules"][node_object_type]["active"].append(node_object_id)
        st.rerun()

    node_menu_columns[4].write("#####")
    if node_menu_columns[4].button(
        "Remove", 
        key=f"remove_node_btn", 
        disabled=node_object_id not in st.session_state["flow_modules"][node_object_type]["active"]):

        st.session_state["flow"].nodes = [node for node in st.session_state["flow"].nodes if node.id != target_node_flow_id]
        st.session_state["flow"].edges = [edge for edge in st.session_state["flow"].edges if edge.source != target_node_flow_id and edge.target != target_node_flow_id]
        st.session_state["flow_modules"][node_object_type]["active"].remove(node_object_id)
        st.rerun()

    st.session_state["flow"] = streamlit_flow(
        key="voice_assistant_flow", 
        state=st.session_state["flow"], 
        layout=TreeLayout(direction="right"), 
        fit_view=True, 
        height=500, 
        enable_node_menu=False,
        enable_edge_menu=True,
        enable_pane_menu=False,
        get_edge_on_click=True,
        get_node_on_click=True, 
        show_minimap=True, 
        hide_watermark=True, 
        allow_new_edges=True,
        min_zoom=0.1)
    if "flow_initiated" not in st.session_state:
        st.session_state["flow_initiated"] = True
        node_type = "input" if node_object_type in [
            "speech_recorder"] else "output" if node_object_type in ["wave_output"] else "default"
        node_content = str(node_object_id)
        new_node = StreamlitFlowNode(
            id=target_node_flow_id, 
            pos=(0, 0), 
            data={"content": f"{SERVICE_TITLES[node_object_type]}\n\n{node_content}"}, 
            node_type=node_type, 
            source_position="right",
            target_position="left",
            selectable=True,
            connectable=True,
            draggable=True,
            resizing=True,
            deletable=True)
        st.session_state["flow"].nodes.append(new_node)
        st.session_state["flow_modules"][node_object_type]["active"].append(node_object_id)
        st.rerun()
