# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import os
from typing import Any, List, Callable, Dict
from uuid import uuid4
import streamlit as st
import random
import tkinter as tk
from tkinter import filedialog
from src.configuration import configuration as cfg
from src.frontend.streamlit.utility.backend_interaction import AVAILABLE_MODULES
from src.frontend.streamlit.utility.state_cache_handling import clear_tab_config
import streamlit.components.v1 as st_components
from code_editor import code_editor
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.layouts import TreeLayout, RadialLayout
from streamlit_flow.state import StreamlitFlowState
import json
from itertools import combinations


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

            
