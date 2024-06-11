# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from typing import Any, List
from uuid import uuid4
import streamlit as st
import random
import streamlit.components.v1 as st_components
from itertools import combinations


###################
# Helper functions
###################


def get_json_editor_buttons() -> List[dict]:
    """
    Function for acquiring json payload code editor buttons.
    Taken from https://github.com/AlHering/scraping-database-generator/blob/development/src/view/streamlit_frontend/frontend_utility/frontend_rendering.py.
    :return: Buttons as list of dictionaries.
    """
    return [
        {
            "name": "save",
            "feather": "Save",
            "hasText": True,
            "alwaysOn": True,
            "commands": [
                    "save-state",
                    [
                        "response",
                        "saved"
                    ]
            ],
            "response": "saved",
            "style": {"top": "0rem", "right": "9.6rem"}
        },
        {
            "name": "copy",
            "feather": "Copy",
            "hasText": True,
            "alwaysOn": True,
            "commands": ["copyAll"],
            "style": {"top": "0rem", "right": "5rem"}
        },
        {
            "name": "clear",
            "feather": "X",
            "hasText": True,
            "alwaysOn": True,
            "commands": ["selectall", "del", ["insertstring", "{\n\n\n\n}"], "save-state",
                         ["response", "saved"]],
            "style": {"top": "0rem", "right": "0.4rem"}
        },
    ]



###################
# Rendering functions
###################


def render_sidebar() -> None:
    """
    Function for rendering the sidebar.
    """
    pass