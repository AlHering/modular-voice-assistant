# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from typing import Any
from uuid import uuid4
import streamlit as st
import random
import requests
from src.configuration import configuration as cfg
from src.frontend.streamlit.utility.state_cache_handling import wait_for_setup
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.layouts import TreeLayout
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
    if st.sidebar.button(" Setup " + "(Status: " + ("Active)" if st.session_state.get("SETUP") else "Inactive)"), help="Sets up assistant infrastructure."):
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


def render_pipeline_node_plane(parent_widget: Any, block_dict: dict, session_state_key: str | None = None) -> None:
    """
    Renders a interactive node plane.
    :param parent_widget: Parent widget.
    :param block_dict: Dictionary for blocks.
    """
    if 'curr_state' not in st.session_state:
        nodes = [StreamlitFlowNode("1", (0, 0), {'content': 'Node 1'}, 'input', 'right'),
                StreamlitFlowNode("2", (1, 0), {'content': 'Node 2'}, 'default', 'right', 'left'),
                StreamlitFlowNode("3", (2, 0), {'content': 'Node 3'}, 'default', 'right', 'left'),
                # StreamlitFlowNode("4", (2, 1), {'content': 'Node 4'}, 'output', target_position='left'),
                # StreamlitFlowNode("5", (3, 0), {'content': 'Node 5'}, 'output', target_position='left'),
                # StreamlitFlowNode("6", (3, 1), {'content': 'Node 6'}, 'output', target_position='left'),
                # StreamlitFlowNode("7", (4, 0), {'content': 'Node 7'}, 'output', target_position='left'),
                ]

        edges = [StreamlitFlowEdge("1-2", "1", "2", animated=True, marker_start={}, marker_end={'type': 'arrow'}),
                StreamlitFlowEdge("1-3", "1", "3", animated=True),
                # StreamlitFlowEdge("2-4", "2", "4", animated=True),
                # StreamlitFlowEdge("2-5", "2", "5", animated=True),
                # StreamlitFlowEdge("3-6", "3", "6", animated=True),
                # StreamlitFlowEdge("3-7", "3", "7", animated=True)
                ]
        
        st.session_state.curr_state = StreamlitFlowState(nodes, edges)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Add node"):
            new_node = StreamlitFlowNode(str(f"st-flow-node_{uuid4()}"), (0, 0), {'content': f'Node {len(st.session_state.curr_state.nodes) + 1}'}, 'default', 'right', 'left')
            st.session_state.curr_state.nodes.append(new_node)
            st.rerun()

    with col2:
        if st.button("Delete Random Node"):
            if len(st.session_state.curr_state.nodes) > 0:
                node_to_delete = random.choice(st.session_state.curr_state.nodes)
                st.session_state.curr_state.nodes = [node for node in st.session_state.curr_state.nodes if node.id != node_to_delete.id]
                st.session_state.curr_state.edges = [edge for edge in st.session_state.curr_state.edges if edge.source != node_to_delete.id and edge.target != node_to_delete.id]
                st.rerun()

    with col3:
        if st.button("Add Edge"):
            if len(st.session_state.curr_state.nodes) > 1:
                source = random.choice(st.session_state.curr_state.nodes)
                target = random.choice([node for node in st.session_state.curr_state.nodes if node.id != source.id])
                new_edge = StreamlitFlowEdge(f"{source.id}-{target.id}", source.id, target.id, animated=True)
                st.session_state.curr_state.edges.append(new_edge)
                st.rerun()
        
    with col4:
        if st.button("Delete Random Edge"):
            if len(st.session_state.curr_state.edges) > 0:
                edge_to_delete = random.choice(st.session_state.curr_state.edges)
                st.session_state.curr_state.edges = [edge for edge in st.session_state.curr_state.edges if edge.id != edge_to_delete.id]
                st.rerun()


    st.session_state.curr_state = streamlit_flow('example_flow', 
                                    st.session_state.curr_state, 
                                    layout=TreeLayout(direction='right'), 
                                    fit_view=True, 
                                    height=500, 
                                    enable_node_menu=True,
                                    enable_edge_menu=True,
                                    enable_pane_menu=True,
                                    get_edge_on_click=True,
                                    get_node_on_click=True, 
                                    show_minimap=True, 
                                    hide_watermark=True, 
                                    allow_new_edges=True,
                                    min_zoom=0.1)

            
