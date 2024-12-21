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
from src.backend.voice_assistant.modular_voice_assistant_abstractions_v2 import BasicVoiceAssistant, Transcriber, Synthesizer, SpeechRecorder
from run_backend import setup_default_voice_assistant
from src.configuration import configuration as cfg


def setup() -> None:
    """
    Function for setting up and assistant.
    """
    if "ASSISTANT" in st.session_state:
        st.session_state.pop("ASSISTANT")
    st.session_state["ASSISTANT"] = setup_default_voice_assistant(
        use_remote_llm=st.session_state.get("use_remote_llm", True),
        download_model_files=st.session_state.get("download_model_files", True),
        llm_parameters=st.session_state.get("llm_parameters", True),
        speech_recorder_parameters=st.session_state.get("speech_recorder_parameters", True),
        transcriber_parameters=st.session_state.get("transcriber_parameters", True),
        synthesizer_parameters=st.session_state.get("synthesizer_parameters", True),
        voice_assistant_parameters=st.session_state.get("voice_assistant_parameters", True)
    )


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
