# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import os
from typing import List, Tuple, Any, Generator, Dict
import traceback
import streamlit as st
from inspect import getfullargspec
from uuid import UUID
from src.services.services import TranscriberService, ChatService, SynthesizerService, Service
from src.services.services import Transcriber, Synthesizer
from src.services.fastapi.service_registry_client import VoiceAssistantClient
from src.configuration import configuration as cfg

AVAILABLE_SERVICES: Dict[str, Service] = {
    "transcriber": TranscriberService,
    "chat": ChatService,
    "synthesizer": SynthesizerService
}
DEFAULTS = {
     "transcriber": {
        "backends": Transcriber.supported_backends,
        "defaults": Transcriber.default_models
    },
    "synthesizer": {
        "backends": Synthesizer.supported_backends,
        "defaults": Synthesizer.default_models
    }
}
CONFIGURATION_PARAMETERS =  {
    "transcriber": {
        "model_parameters": {"title": "Model Parameters", "type": dict, "default": None}, 
        "transcription_parameters": {"title": "Transcription Parameters", "type": dict, "default": None}
    },
    "chat": [
        {
            "#option": "local",
            "language_model": {"title": "Language Model", "type": dict}, 
            "chat_parameters": {"title": "Chat Parameters", "type": dict}, 
            "system_prompt": {"title": "System Prompt", "type": str, "default": None}, 
            "prompt_maker": {"title": "Prompt Maker", "type": str, "default": None}, 
            "use_history": {"title": "Use History", "type": bool, "default": True}, 
            "history": {"title": "History", "type": str, "default": None}
        }, {
            "#option": "remote",
            "api_base": {"title": "Api Base", "type": str}, 
            "api_token": {"title": "Api Token", "type": str}, 
            "chat_parameters": {"title": "Chat Parameters", "type": dict, "default": None}, 
            "system_prompt": {"title": "System Prompt", "type": str, "default": None}, 
            "prompt_maker": {"title": "Prompt Maker", "type": None, "default": None}, 
            "use_history": {"title": "Use History", "type": bool, "default": True}, 
            "history": {"title": "History", "type": str, "default": None}
        }
    ],
    "synthesizer": {
        "model_parameters": {"title": "Model Parameters", "type": dict, "default": None}, 
        "synthesis_parameters": {"title": "Synthesis Parameters", "type": dict, "default": None},
        "replace_symbols": {"title": "Replace Symbols", "type": dict, "default": None}
    }
}


def setup() -> None:
    """
    Sets up and assistant.
    """
    st.session_state["WORKDIR"] = os.path.join(cfg.PATHS.DATA_PATH, "frontend")
    st.session_state["CLIENT"] = VoiceAssistantClient(api_base=st.session_state["API_BASE"])


def validate_config(config_type: str, config: dict) -> Tuple[bool | None, str]:
    """
    Validates an configuration.
    :param config_type: Config type.
    :param config: Module configuration.
    :return: True or False and validation report depending on validation success. 
        None and validation report in case of warnings. 
    """
    try:
        return AVAILABLE_SERVICES[config_type].validate_configuration(config=config)
    except Exception as ex:
        return False, f"Exception {ex} appeared: {traceback.format_exc()}."


def fetch_default_config() -> dict:
    """
    Fetches default config.
    :return: Config.
    """
    return cfg.DEFAULT_COMPONENT_CONFIG


"""
API based interaction
"""
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
    return [flatten_config(entry) for entry in st.session_state["CLIENT"].get_configs(service=config_type)["results"]]


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
    return flatten_config(st.session_state["CLIENT"].patch_config(service=config_type, config=patch)["results"][0])


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
    return flatten_config(st.session_state["CLIENT"].add_config(service=config_type, config=patch)["results"][0])


def delete_config(config_type: str, config_id: str) -> dict:
    """
    Puts config into database.
    :param config_type: Config type.
    :param config_id: Config ID.
    :return: Config entry.
    """
    deletion_patch = {"id": config_id, "inactive": True}
    return flatten_config(st.session_state["CLIENT"].patch_config(service=config_type, config=deletion_patch)["results"][0])


def get_loaded_service() -> dict:
    """
    Retrieves loaded services.
    :return: Response.
    """
    return st.session_state["CLIENT"].get_services()["results"][0]


def load_service(service_type: str,
                 config_uuid: str | UUID | None = None) -> dict:
    """
    Loads a service from the given config UUID.
    :param service_type: Target service type.
    :param config_uuid: Config UUID.
    :return: Response.
    """
    return st.session_state["CLIENT"].setup_and_run_service(service=service_type, config_uuid=config_uuid)


def reset_service(service_type: str) -> dict:
    """
    Unloads a service.
    :param service_type: Target service type.
    :return: Response.
    """
    return st.session_state["CLIENT"].reset_service(service=service_type)


def unload_service(service_type: str) -> dict:
    """
    Unloads a service.
    :param service_type: Target service type.
    :return: Response.
    """
    return st.session_state["CLIENT"].stop_service(service=service_type)


def reset_service(service_type: str,
                   config_uuid: str | UUID | None = None) -> dict:
    """
    Loads a service from the given config UUID.
    :param service_type: Target service type.
    :param config_uuid: Config UUID.
    :return: Response.
    """
    return st.session_state["CLIENT"].reset_service(service=service_type, config_uuid=config_uuid)


def record_and_transcribe_speech() -> str:
        """
        Records and transcribes a speech input.
        """
        return st.session_state["CLIENT"].record_and_transcribe_speech()[0]


def synthesize_and_output_speech(text: str) -> None:
    """
    Synthesizes and outputs speech.
    :param text: Text input.
    """
    st.session_state["CLIENT"].synthesize_and_output_speech(text=text)


def chat(prompt: str, 
         output_as_audio: bool = False) -> str:
    """
    Fetches chat response from st.session_state["CLIENT"] interface.
    :param prompt: User prompt.
    :param output_as_audio: Outputting response as audio.
    :return: Chat response.
    """
    for response in st.session_state["CLIENT"].chat(prompt=prompt, stream=False, output_as_audio=output_as_audio):
        return response[0]
        

def chat_streamed(prompt: str, 
                  output_as_audio: bool = False) -> Generator[str, None, None]:
    """
    Fetches streamed chat response from st.session_state["CLIENT"] interface.
    :param prompt: User prompt.
    :param output_as_audio: Outputting response as audio.
    :return: Chat response generator.
    """
    for response in st.session_state["CLIENT"].chat(prompt=prompt, stream=True, output_as_audio=output_as_audio):
        yield response[0]


"""
Parameter dict creation
"""
def retrieve_type(input_type: Any) -> Any:
    """
    Retrieves type of input.
    :param input_type: Inspected input type hint.
    :return: Target type.
    """
    base_data_types = [str, bool, int, float, complex, list, tuple, range, dict, set, frozenset, bytes, bytearray, memoryview]
    if input_type in base_data_types:
        return input_type
    if input_type == callable:
        return callable
    elif str(input_type).startswith("typing.List"):
        return list
    elif str(input_type).startswith("typing.Dict"):
        return dict
    elif str(input_type).startswith("typing.Tuple"):
        return tuple
    elif str(input_type).startswith("typing.Set"):
        return set
    else:
        for data_type in base_data_types:
            string_representation = str(data_type).split("'")[1]
            if string_representation in str(input_type):
                return data_type
            

def retrieve_parameter_specification(func: callable, ignore: List[str] | None = None) -> dict:
    """
    Retrieves parameter specification.
    :param func: Callable to retrieve parameter specs from.
    :param ignore: Parameters to ignore.
    :return: Specification of parameters.
    """
    ignore = [] if ignore is None else ignore

    spec = {}
    arg_spec = getfullargspec(func)
    default_offset = len(arg_spec.args) - len(arg_spec.defaults) if arg_spec.defaults else None

    for param_index, param in enumerate(arg_spec.args):
        spec[param] = {"title": " ".join(param.split("_")).title()}
        if param in arg_spec.annotations:
            spec[param]["type"] = retrieve_type(arg_spec.annotations[param])
        if default_offset and param_index > default_offset:
            spec[param]["default"] = arg_spec.defaults[param_index-default_offset]
    for ignored_param in ignore:
        if ignored_param in spec:
            spec.pop(ignored_param)
    return spec