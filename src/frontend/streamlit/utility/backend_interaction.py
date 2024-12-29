# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import os
from typing import List, Tuple
import traceback
import streamlit as st
from uuid import UUID
from src.voice_assistant import BasicVoiceAssistant, setup_default_voice_assistant
from src.voice_assistant import AVAILABLE_MODULES as AVAILABLE_MODULES
from src.interface_client import RemoteVoiceAssistantClient
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
    st.session_state["WORKDIR"] = os.path.join(cfg.PATHS.DATA_PATH, "frontend")
    if MODE == "direct":
        st.session_state["CLIENT"] = VoiceAssistantInterface()
    elif MODE == "api":
        st.session_state["CLIENT"] = RemoteVoiceAssistantClient()
        if not st.session_state["CLIENT"].check_connection():
            return False
    else:
        raise NotImplementedError(f"Mode '{MODE}' is not implemented.")
    return True


def validate_config(config_type: str, config: dict) -> Tuple[bool | None, str]:
    """
    Validates an configuration.
    :param config_type: Config type.
    :param config: Module configuration.
    :return: True or False and validation report depending on validation success. 
        None and validation report in case of warnings. 
    """
    try:
        return AVAILABLE_MODULES[config_type].validate_configuration(config=config)
    except Exception as ex:
        return False, f"Exception {ex} appeared: {traceback.format_exc()}."


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


def flatten_config(config: dict) -> None:
    """
    Flattens config for further usage.
    :param config: Database config entry.
    :return: Flattened config entry.
    """
    config.update(config.pop("config"))
    return config


def get_configs(config_type: str) -> List[dict]:
    """
    Fetches configs from database.
    :param config_type: Config type.
    :return: Config entries.
    """
    return [flatten_config(entry) for entry in st.session_state["CLIENT"].get_configs(module_type=config_type)]


def patch_config(config_type: str, config_data: dict, config_id: str | UUID | None = None) -> dict:
    """
    Patches config in database.
    :param config_type: Config type.
    :param config_data: Config data.
    :param config_id: Config UUID, if available.
    :return: Config entry.
    """
    patch = {"config": config_data}
    if config_id is not None:
        patch["id"] = config_id
    return flatten_config(st.session_state["CLIENT"].overwrite_config(module_type=config_type, config=patch))


def put_config(config_type: str, config_data: dict, config_id: str | None = None) -> dict:
    """
    Puts config into database.
    :param config_type: Config type.
    :param config_data: Config data.
    :param config_id: Config UUID, if available.
    :return: Config entry.
    """
    patch = {"config": config_data}
    if config_id is not None:
        patch["id"] = config_id
    return flatten_config(st.session_state["CLIENT"].add_config(module_type=config_type, config=patch))


def delete_config(config_type: str, config_id: str) -> dict:
    """
    Puts config into database.
    :param config_type: Config type.
    :param config_id: Config ID.
    :return: Config entry.
    """
    deletion_patch = {"id": config_id, "inactive": True}
    return flatten_config(st.session_state["CLIENT"].overwrite_config(module_type=config_type, config=deletion_patch))