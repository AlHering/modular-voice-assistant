# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import sys
import os
import streamlit
from src.configuration import configuration as cfg


def run_streamlit_frontend() -> None:
    """
    Runner function for streamlit frontend.
    """
    import streamlit.web.bootstrap as streamlit_bootstrap
    if not os.path.exists(cfg.PATHS.FRONTEND_PATH):
        os.makedirs(cfg.PATHS.FRONTEND_PATH)
    streamlit_bootstrap.run(os.path.join(cfg.PATHS.SOURCE_PATH, "frontend", "streamlit", "Voice_Assistant.py"),
                            is_hello=False, args=[], flag_options=[],)

def run_commandline_frontend() -> None:
    """
    Runner function for commandline frontend.
    """
    pass


if __name__ == "__main__":
    if "--cli" in sys.argv:
         run_commandline_frontend()
    else:
         run_streamlit_frontend()