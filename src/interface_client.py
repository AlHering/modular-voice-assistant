# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from enum import Enum
from typing import Generator
from uuid import UUID
import requests
import json
from src.configuration import configuration as cfg


class Endpoints(str, Enum):
    """
    Endpoints config.
    """
    check_connection = "/check"
    add_configs = "/configs/add"
    patch_configs = "/configs/patch"
    get_configs = "/configs/get"
    get_loaded_services = "/services/get"
    load_services = "/services/load"
    unload_services = "/services/unload"

    run_input_pipeline = "/assistant/run_input_pipeline"
    run_output_pipeline = "/assistant/run_output_pipeline"

    record_speech = "/services/record"
    play_audio = "/services/play-audio"
    transcribe = "/services/transcribe"
    synthesize = "/services/synthesize"
    local_chat = "/services/local-chat"
    local_chat_streamed = "/services/local-chat-stream"
    remote_chat = "/services/remote-chat"
    remote_chat_streamed = "/services/remote-chat-stream"

    def __str__(self) -> str:
        """
        Returns string representation.
        """
        return str(self.value)


class VoiceAssistantClient(object):
    """
    Remote voice assistant client.
    """
    def __init__(self, api_base: str | None = None) -> None:
        """
        Initiation method.
        :param api_base: API Base, e.g. http://127.0.0.1:7861/api/v1.
        """
        self.api_base = f"http://{cfg.BACKEND_HOST}:{cfg.BACKEND_PORT}{cfg.BACKEND_ENDPOINT_BASE}" if api_base is None else api_base
        

    def check_connection(self) -> bool:
        """
        Checks connection to backend.
        :return: True, if available, else False.
        """
        try:
            resp = requests.get(Endpoints.check_connection).json()
            return True
        except Exception as ex:
            return False

    """
    Config handling
    """

    def add_config(self,
                   service_type: str,
                   config: dict) -> dict:
        """
        Adds a config to the database.
        :param service_type: Target service type.
        :param config: Config.
        :return: Response.
        """
        return requests.post(self.api_base + Endpoints.add_configs, params={
            "service_type": service_type,
            "config": config
        }).json().get("result")

    def overwrite_config(self,
                   service_type: str,
                   config: dict) -> dict:
        """
        Overwrites a config in the database.
        :param service_type: Target service type.
        :param config: Config.
        :return: Response.
        """
        return requests.post(self.api_base + Endpoints.patch_configs, params={
            "service_type": service_type,
            "config": config
        }).json().get("result")
    
    def get_configs(self,
                    service_type: str = None) -> dict:
        """
        Adds a config to the database.
        :param service_type: Target service type.
            Defaults to None in which case all configs are returned.
        :return: Response.
        """
        return requests.post(
            self.api_base + Endpoints.get_configs, params={"service_type": service_type}).json().get("result")

    """
    Service handling
    """
    def get_loaded_services(self) -> dict:
        """
        Retrieves loaded services.
        :return: Response.
        """
        return requests.get(self.api_base + Endpoints.get_loaded_services).json()

    def load_service(self,
                    service_type: str,
                    config_uuid: str) -> dict:
        """
        Loads a service from the given config UUID.
        :param service_type: Target service type.
        :param config_uuid: Config UUID.
        :return: Response.
        """
        return requests.post(self.api_base + Endpoints.load_services, params={
                "service_type": service_type,
                "config_uuid": config_uuid
            }).json()
            
    def unload_service(self,
                      service_type: str,
                      config_uuid: str) -> dict:
        """
        Unloads a service from the given config UUID.
        :param service_type: Target service type.
        :param config_uuid: Config UUID.
        :return: Response.
        """
        return requests.post(self.api_base + Endpoints.unload_services, params={
                "service_type": service_type,
                "config_uuid": config_uuid
            }).json()

    """
    Assistant handling
    """

    """
    Direct service access
    """
    def record_speech(self, 
                      config_uuid: str | UUID,
                      recognizer_parameters: dict | None = None,
                      microphone_parameters: dict | None = None) -> dict:
        return requests.post(self.api_base + Endpoints.record_speech, json={
                "config_uuid": config_uuid,
                "recognizer_parameters": recognizer_parameters,
                "microphone_parameters": microphone_parameters
            }).json()

    def transcribe(self, 
                   config_uuid: str | UUID,
                   audio_input: str | list, 
                   transcription_parameters: dict | None = None) -> dict:
        return requests.post(self.api_base + Endpoints.transcribe, json={
                "config_uuid": config_uuid,
                "audio": audio_input,
                "parameters": transcription_parameters
            }).json()

    def synthesize(self, 
                   config_uuid: str | UUID,
                   text: str,
                   synthesis_parameters: dict | None = None) -> dict:
        return requests.post(self.api_base + Endpoints.synthesize, json={
                "config_uuid": config_uuid,
                "text": text,
                "parameters": synthesis_parameters
            }).json()

    def play_audio(self, 
                    config_uuid: str | UUID,
                    audio_input: str | list, 
                    playback_parameters: dict | None = None) -> dict:
        return requests.post(self.api_base + Endpoints.play_audio, json={
                "config_uuid": config_uuid,
                "audio": audio_input,
                "parameters": playback_parameters
            }).json()
        
    def local_chat(self, 
                   config_uuid: str | UUID,
                   prompt: str, 
                   chat_parameters: dict | None = None) -> dict:
        return requests.post(self.api_base + Endpoints.local_chat, json={
                "config_uuid": config_uuid,
                "text": prompt,
                "parameters": chat_parameters
            }).json()
        
    def remote_chat(self, 
                   config_uuid: str | UUID,
                   prompt: str, 
                   chat_parameters: dict | None = None) -> dict:
        return requests.post(self.api_base + Endpoints.remote_chat, json={
                "config_uuid": config_uuid,
                "text": prompt,
                "parameters": chat_parameters
            }).json()

    def local_chat_streamed(self, 
                   config_uuid: str | UUID,
                   prompt: str, 
                   chat_parameters: dict | None = None) -> Generator[dict, None, None]:
        with requests.post(self.api_base + Endpoints.local_chat_streamed, json={
                "config_uuid": config_uuid,
                "text": prompt,
                "parameters": chat_parameters
            }, stream=True) as response:
            accumulated = ""
            for chunk in response.iter_content():
                decoded_chunk = chunk.decode("utf-8")
                accumulated += decoded_chunk
                if accumulated.endswith("}"):
                    try:
                        json_chunk = json.loads(accumulated)
                        yield json_chunk
                        accumulated = ""
                    except json.JSONDecodeError:
                        pass

    def remote_chat_streamed(self, 
                   config_uuid: str | UUID,
                   prompt: str, 
                   chat_parameters: dict | None = None) -> Generator[dict, None, None]:
        with requests.post(self.api_base + Endpoints.remote_chat_streamed, json={
                "config_uuid": config_uuid,
                "text": prompt,
                "parameters": chat_parameters
            }, stream=True) as response:
            accumulated = ""
            for chunk in response.iter_content():
                decoded_chunk = chunk.decode("utf-8")
                accumulated += decoded_chunk
                if accumulated.endswith("}"):
                    try:
                        json_chunk = json.loads(accumulated)
                        yield json_chunk
                        accumulated = ""
                    except json.JSONDecodeError:
                        pass