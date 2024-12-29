# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import os
import streamlit as st
from src.voice_assistant import BasicVoiceAssistant, setup_default_voice_assistant
from src.voice_assistant import AVAILABLE_MODULES as AVAILABLE_MODULES
from src.interface_client import VoiceAssistantClient
from src.interface import VoiceAssistantInterface
from src.configuration import configuration as cfg


# MODES:
#   api: Backend is running on default network address
#   direct: Controller in session cache
#
MODE: str = "direct"


def setup() -> bool:
    """
    Sets up and assistant.
    :return: True, if successful, else False.
    """
    if "ASSISTANT" in st.session_state:
        st.session_state.pop("ASSISTANT")
    st.session_state["WORKDIR"] = os.path.join(cfg.PATHS.DATA_PATH, "frontend")
    if MODE == "direct":
        st.session_state["CLIENT"] = VoiceAssistantInterface()
    elif MODE == "api":
        st.session_state["CLIENT"] = VoiceAssistantClient()
        if not st.session_state["CLIENT"].check_connection():
            return False
    else:
        raise NotImplementedError(f"Mode '{MODE}' is not implemented.")
    return True


def fetch_default_config() -> dict:
    """
    Fetches default config.
    :return: Config.
    """
    return cfg.DEFAULT_COMPONENT_CONFIG


def load_voice_assistant(
    config: dict
) -> BasicVoiceAssistant:
    """
    Loads a basic voice assistant.
    :param module_set: Module set.
    :param loop_pause: Loop pause for modules.
    :return: Conversation handler.
    """
    return setup_default_voice_assistant(config=config)