# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from typing import List, Any, Optional
import json
from enum import Enum
from asyncio import sleep
import streamlit as st
import traceback
import httpx
from httpx import RequestError, ConnectError, ConnectTimeout
from http.client import responses as status_codes
from src_legacy.control.voice_assistant_controller import VoiceAssistantController, Transcriber, Synthesizer, SpeechRecorder
from src_legacy.configuration import configuration as cfg


# MODES:
#   default: Backend is running on default network address
#   direct: Controller in session chache
#
MODE: str = "direct"
CLIENT: httpx.AsyncClient = None
METHODS: dict | None = None 


VA_BASE = f"{cfg.VOICE_ASSISTANT_BACKEND_HOST}:{cfg.VOICE_ASSISTANT_BACKEND_PORT}/{cfg.VOICE_ASSISTANT_BACKEND_ENDPOINT_BASE}"
TG_BASE = f"{cfg.TEXT_GENERATION_BACKEND_HOST}:{cfg.TEXT_GENERATION_BACKEND_PORT}/{cfg.TEXT_GENERATION_BACKEND_ENDPOINT_BASE}"
class Endpoints(str, Enum):
    # Text Generation Interface
    lm_instance = f"{TG_BASE}/lm_instance"
    kb_instance = f"{TG_BASE}/kb_instance"
    tool_argument = f"{TG_BASE}/tool_argument"
    agent_tool = f"{TG_BASE}/agent_tool"
    agent_memory = f"{TG_BASE}/agent_memory"
    agent = f"{TG_BASE}/agent"

    # Voice Assistant Interface
    transcriber = f"{VA_BASE}/transcriber"
    synthesizer = f"{VA_BASE}/synthesizer"
    speech_recorder = f"{VA_BASE}/speech_recorder"

    transcribe_appendix = f"/transcribe"
    synthesize_appendix = f"/synthesize"
    srecor_speech_appendix = f"/record"


def setup() -> None:
    """
    Function for setting up backend interaction.
    """
    global MODE, CLIENT, METHODS
    if MODE == "default":
        CLIENT = httpx.AsyncClient()
        METHODS = {
            "get": httpx.AsyncClient.get,
            "post": httpx.AsyncClient.post,
            "put": httpx.AsyncClient.put,
            "patch": httpx.AsyncClient.patch,
            "delete": httpx.AsyncClient.delete,
        }
        connected_tg = False
        connected_va = False
        while not (connected_tg and connected_va):
            try:
                if connected_tg or CLIENT.get(Endpoints.lm_instance).status_code == 200:
                    connected_tg = True
                if connected_va or CLIENT.get(Endpoints.transcriber).status_code == 200:
                    connected_va = True
            except (ConnectError, ConnectTimeout):
                sleep(2)
    if MODE == "direct":
        st.session_state["CONTROLLER"] = VoiceAssistantController()
        st.session_state["CONTROLLER"].setup()
    st.session_state["CLASSES"] = {
        "transcriber": Transcriber,
        "synthesizer": Synthesizer,
        "speech_recorder": SpeechRecorder
    }



def send_request(method: str, url: str, headers: Optional[dict] = None, params: Optional[dict] = None, json_payload: Optional[dict] = None) -> dict:
    """
    Function for sending off request.
    :param method: Request method.
    :param url: Target URL.
    :param headers: Request headers.
        Defaults to None.
    :param params: Request parameters.
        Defaults to None.
    :param json_payload: JSON payload.
        Defaults to None.
    :return: Response data.
    """
    global METHODS
    response_content = {}
    response_status = -1
    response_status_message = "An unknown error appeared"
    response_headers = {}

    response = None
    try:
        response = METHODS[method](
            url=url,
            params=params,
            headers=headers,
            json=json_payload
        )
        response_status = response.status_code
        response_status_message = f"Status description: {status_codes[response_status]}"
        response_headers = dict(response.headers)
    except RequestError as ex:
        response_status_message = f"Exception '{ex}' appeared.\n\nTrace:{traceback.format_exc()}"

    if response is not None:
        try:
            response_content = response.json()
        except json.decoder.JSONDecodeError:
            response_content = response.text

    return {
        "request_method": method,
        "request_url": url,
        "request_headers": headers,
        "request_params": params,
        "request_json_payload": json_payload,
        "response": response_content,
        "response_status": response_status,
        "response_status_message": response_status_message,
        "response_headers": response_headers}


def get_objects(object_type: str) -> List[dict]:
    """
    Function for getting objects.
    :param object_type: Object type.
    :return: Object dictionaries.
    """
    global MODE
    if MODE == "default":
        response = send_request(
            method="get",
            url=getattr(Endpoints, object_type)
        )
        try:
            return response["response"][object_type]
        except KeyError:
            return []
    if MODE == "direct":
        return st.session_state["CONTROLLER"].get_objects_by_type(
            object_type
        )


def get_object(object_type: str, object_id: int) -> Optional[dict]:
    """
    Function for getting an object.
    :param object_type: Object type.
    :param object_id: Object ID.
    :return: Object dictionary.
    """
    global MODE
    if MODE == "default":
        response = send_request(
            method="get",
            url=f"{getattr(Endpoints, object_type)}/{object_id}",
        )
        try:
            return response["response"][object_type]
        except KeyError:
            return None
    if MODE == "direct":
        return st.session_state["CONTROLLER"].get_object_by_id(
            object_type,
            object_id
        )


def post_object(object_type: str, object_data: dict) -> Optional[dict]:
    """
    Function for patching objects.
    :param object_type: Object type.
    :param object_id: Object ID.
    :param object_data: Object data.
    :return: Object dictionary.
    """
    global MODE
    if MODE == "default":
        response = send_request(
            method="post",
            url=getattr(Endpoints, object_type),
            json_payload={"data": object_data}
        )
        try:
            return response["response"][object_type]
        except KeyError:
            return None
    if MODE == "direct":
        return st.session_state["CONTROLLER"].post_object(
            object_type,
            **object_data
        )


def patch_object(object_type: str, object_id: int, object_data: dict) -> Optional[dict]:
    """
    Function for patching objects.
    :param object_type: Object type.
    :param object_id: Object ID.
    :param object_data: Object data.
    :return: Object dictionary.
    """
    global MODE
    if MODE == "default":
        response = send_request(
            method="patch",
            url=f"{getattr(Endpoints, object_type)}/{object_id}",
            json_payload={"patch": object_data}
        )
        try:
            return response["response"][object_type]
        except KeyError:
            return None
    if MODE == "direct":
        return st.session_state["CONTROLLER"].patch_object(
            object_type,
            object_id,
            **object_data
        )
    

def put_object(object_type: str, object_data: dict) -> Optional[dict]:
    """
    Function for putting objects.
    :param object_type: Object type.
    :param object_data: Object data.
    :return: Object dictionary.
    """
    global MODE
    if MODE == "default":
        response = send_request(
            method="put",
            url=getattr(Endpoints, object_type),
            json_payload={"data": object_data}
        )
        try:
            return response["response"][object_type]
        except KeyError:
            return None
    if MODE == "direct":
        return st.session_state["CONTROLLER"].put_object(
            object_type,
            **object_data
        )
    

def delete_object(object_type: str, object_id: int) -> Optional[dict]:
    """
    Function for deleting objects.
    :param object_type: Object type.
    :param object_id: Object ID.
    :return: Object dictionary.
    """
    global MODE
    if MODE == "default":
        response = send_request(
            method="delete",
            url=f"{getattr(Endpoints, object_type)}/{object_id}",
        )
        try:
            return response["response"][object_type]
        except KeyError:
            return None
    if MODE == "direct":
        return st.session_state["CONTROLLER"].delete_object(
            object_type,
            object_id
        )
