# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import os
from typing import List, Any, Optional
import json
from enum import Enum
from asyncio import sleep
import streamlit as st
import traceback
import httpx
from httpx import RequestError, ConnectError, ConnectTimeout
from http.client import responses as status_codes
from src.backend.database.basic_sqlalchemy_interface import BasicSQLAlchemyInterface, FilterMask
from src.backend.database.data_model import populate_data_infrastructure
from src.backend.voice_assistant.modular_voice_assistant_abstractions_v2 import BasicVoiceAssistant, Transcriber, Synthesizer, SpeechRecorder
from run_backend import setup_default_voice_assistant
from src.configuration import configuration as cfg


# MODES:
#   default: Backend is running on default network address
#   direct: Controller in session cache
#
MODE: str = "direct"


def setup() -> None:
    """
    Function for setting up and assistant.
    """
    if "ASSISTANT" in st.session_state:
        st.session_state.pop("ASSISTANT")
    if MODE == "direct":
        st.session_state["WORKDIR"] = os.path.join(cfg.PATHS.DATA_PATH, "voice_assistant_interface")
        st.session_state["DATABASE"] = BasicSQLAlchemyInterface(
            working_directory=os.path.join(st.session_state["WORKDIR"], "database"),
            population_function=populate_data_infrastructure,
            default_entries={
                
            }
        )
        st.session_state["ASSISTANT"] = setup_default_voice_assistant(
            use_remote_llm=st.session_state.get("use_remote_llm", True),
            download_model_files=st.session_state.get("download_model_files", False),
            llm_parameters=st.session_state.get("llm_parameters"),
            speech_recorder_parameters=st.session_state.get("speech_recorder_parameters"),
            transcriber_parameters=st.session_state.get("transcriber_parameters"),
            synthesizer_parameters=st.session_state.get("synthesizer_parameters"),
            voice_assistant_parameters=st.session_state.get("voice_assistant_parameters")
        )
        
    else:
        raise NotImplementedError("API mode is not implemented yet.")


