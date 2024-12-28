# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import os
from typing import List, Any, Optional, Dict
import json
from enum import Enum
from copy import deepcopy
from asyncio import sleep
import streamlit as st
import traceback
import httpx
from httpx import RequestError, ConnectError, ConnectTimeout
from http.client import responses as status_codes
from src.modules.abstractions import VAModule
from src.voice_assistant import BaseModuleSet, BasicVoiceAssistant, ModularConversationHandler, AVAILABLE_MODULES, setup_default_voice_assistant
from src.configuration import configuration as cfg


# MODES:
#   default: Backend is running on default network address
#   direct: Controller in session cache
#
MODE: str = "direct"


def setup() -> None:
    """
    Sets up and assistant.
    """
    if "ASSISTANT" in st.session_state:
        st.session_state.pop("ASSISTANT")
    if MODE == "direct":
        st.session_state["WORKDIR"] = cfg.PATHS.DATA_PATH
    else:
        raise NotImplementedError("API mode is not implemented yet.")


def load_conversation_handler(module_set: BaseModuleSet, loop_pause: float = .1) -> BasicVoiceAssistant:
    """
    Loads a basic voice assistant.
    :param module_set: Module set.
    :param loop_pause: Loop pause for modules.
    :return: Conversation handler.
    """
    return ModularConversationHandler(
        working_directory=os.path.join(st.session_state["WORKDIR"], "conversation_handler"),
        module_set=module_set,
        loop_pause=loop_pause
    )

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