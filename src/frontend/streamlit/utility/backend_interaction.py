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
from src.backend.database.basic_sqlalchemy_interface import BasicSQLAlchemyInterface, FilterMask
from src.backend.database.data_model import populate_data_infrastructure, get_default_entries
from src.backend.voice_assistant.modular_voice_assistant_abstractions_v2 import BasicVoiceAssistant, ChatModelInstance, RemoteChatModelInstance, Transcriber, Synthesizer, SpeechRecorder, BaseModuleSet, BasicHandlerModule, WaveOutputModule, SpeechRecorderModule, ModularConversationHandler
from run_backend import setup_default_voice_assistant
from src.configuration import configuration as cfg


# MODES:
#   default: Backend is running on default network address
#   direct: Controller in session cache
#
MODE: str = "direct"
OBJECT_STRUCTURE = {
    "transcriber": {
        "core_parameters": ["backend", "model_path"],
        "json_parameters": ["model_parameters", "transcription_parameters"],
    },
    "synthesizer": {
        "core_parameters": ["backend", "model_path"],
        "json_parameters": ["model_parameters", "synthesis_parameters"],
    },
    "speech_recorder": {
        "core_parameters": ["input_device_index", "loop_pause"],
        "json_parameters": ["recognizer_parameters", "microphone_parameters"],
    }
}


def setup() -> None:
    """
    Sets up and assistant.
    """
    if "ASSISTANT" in st.session_state:
        st.session_state.pop("ASSISTANT")
    if MODE == "direct":
        st.session_state["WORKDIR"] = os.path.join(cfg.PATHS.DATA_PATH, "voice_assistant_interface")
        st.session_state["DATABASE"] = BasicSQLAlchemyInterface(
            working_directory=os.path.join(st.session_state["WORKDIR"], "database"),
            population_function=populate_data_infrastructure,
            default_entries=get_default_entries(),
            handle_objects_as_dicts=True
        )       
    else:
        raise NotImplementedError("API mode is not implemented yet.")
    st.session_state["CLASSES"] = {
        "transcriber": Transcriber,
        "synthesizer": Synthesizer,
        "speech_recorder": SpeechRecorder
    }


def load_conversation_handler(module_set: BaseModuleSet, loop_pause: float = .1) -> ModularConversationHandler:
    """
    Loads a basic voice assistant.
    :param module_set: Module set.
    :param loop_pause: Loop pause for modules.
    :return: Conversation handler.
    """
    return ModularConversationHandler(working_directory=os.path.join(st.session_state["WORKDIR"], "conversation_handler"),
                                      module_set=module_set,
                                      loop_pause=loop_pause)


def get_components() -> List[Dict[str, dict]]:
    """
    Retrieves available components.
    :returns: Dictionary of components.
    """
    return {
        "transcriber": st.session_state["DATABASE"].get_objects_by_type(object_type="transcriber"),
        "synthesizer": st.session_state["DATABASE"].get_objects_by_type(object_type="synthesizer"),
        "speech_recorder": st.session_state["DATABASE"].get_objects_by_type(object_type="speech_recorder"),
        "chat_model": st.session_state["DATABASE"].get_objects_by_type(object_type="chat_model"),
        "remote_chat_model": st.session_state["DATABASE"].get_objects_by_type(object_type="remote_chat_model"),
    }


def load_component(object_type: str, object_config: dict) -> Transcriber | Synthesizer | SpeechRecorder | ChatModelInstance | RemoteChatModelInstance:
    """
    Loads a component.
    :param object_type: Object type.
    :param object_config: Object config. 
    """
    return {
        "transcriber": Transcriber,
        "synthesizer": Synthesizer,
        "speech_recorder": SpeechRecorder,
        "chat_model": ChatModelInstance,
        "remote_chat_model": RemoteChatModelInstance,
    }[object_type](**object_config)