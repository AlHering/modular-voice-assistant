# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2025 Alexander Hering             *
****************************************************
"""
from __future__ import annotations
import numpy as np
from typing import Generator, Tuple
from multiprocessing import Queue, Event
import requests
from uuid import UUID
import time
import json
from src.configuration import configuration as cfg
from src.services.service_abstractions import ServicePackage
from src.model.abstractions.sound_model_abstractions import SpeechRecorder, AudioPlayer
from src.services.fastapi.service_registry_server import BaseResponse, ServicePackage, Endpoints


class ServiceRegistryClient(object):
    """
    Service registry client.
    """

    def __init__(self, api_base: str) -> None:
        """
        Initiation method.
        :param api_base: API base.
        """
        self.api_base = api_base

    """
    Service interaction
    """
    def interrupt(self) -> dict:
        """
        Interrupts services.
        """
        return requests.post(self.api_base + Endpoints.interrupt).json()
    
    def get_services(self) -> dict:
        """
        Responds available services.
        """
        return requests.get(self.api_base + Endpoints.services_get).json()

    def setup_and_run_service(self, service: str, config_uuid: str | UUID) -> dict:
        """
        Sets up and runs a service.
        :param service: Target service name.
        :param config_uuid: Config UUID.
        :return: Response.
        """
        return requests.post(self.api_base + Endpoints.service_run, params={
            "service": service,
            "config_uuid": config_uuid
        }).json()
    
    def reset_service(self, service: str, config_uuid: str | UUID) -> dict:
        """
        Resets a service.
        :param service: Target service name.
        :param config_uuid: Config UUID.
        :return: Response.
        """
        return requests.post(self.api_base + Endpoints.service_reset, params={
            "service": service,
            "config_uuid": config_uuid
        }).json()
    
    def stop_service(self, service: str) -> BaseResponse:
        """
        Stops a service.
        :param service: Target service name.
        :return: Response.
        """
        return requests.post(self.api_base + Endpoints.service_stop, params={
            "service": service
        }).json()
    
    def process(self, service: str, input_package: ServicePackage, timeout: float | None = None) -> dict | None:
        """
        Runs a service process.
        :param service: Service name.
        :param input_package: Input service package.
        :param timeout: Timeout.
        :return: Response.
        """
        try:
            return requests.post(self.api_base + Endpoints.service_process, json={
                "service": service,
                "input_package": input_package.model_dump(),
                "timeout": timeout
            }).json()
        except json.JSONDecodeError: 
            return None

    def stream(self, service: str, input_package: ServicePackage, timeout: float | None = None) -> Generator[dict, None, None]:
        """
        Runs a service process.
        :param service: Service name.
        :param input_package: Input service package.
        :param timeout: Timeout.
        :return: Response generator.
        """
        with requests.post(self.api_base + Endpoints.service_stream, json={
                "service": service,
                "input_package": input_package.model_dump(),
                "timeout": timeout
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

    """
    Config handling
    """
    def add_config(self, service: str, config: dict) -> dict:
        """
        Adds a config to the database.
        :param service: Target service.
        :param config: Config.
        :return: Response.
        """
        return requests.post(self.api_base + Endpoints.configs_add, json={
            "service": service,
            "config": config
        }).json()
    
    def patch_config(self, service: str, config: dict) -> dict:
        """
        Adds a config to the database.
        :param service: Target service.
        :param config: Config.
        :return: Response.
        """
        return requests.post(self.api_base + Endpoints.configs_patch, json={
            "service": service,
            "config": config
        }).json()
    
    def get_configs(self, service: str) -> dict:
        """
        Adds a config to the database.
        :param service: Target service.
        :return: Response.
        """
        return requests.post(self.api_base + Endpoints.configs_get, params={
            "service": service
        }).json()


class VoiceAssistantClient(ServiceRegistryClient):
    """
    Voice assistant client.
    """

    def __init__(self, api_base: str,
                 speech_recorder_parameters: dict | None = None,
                 audio_player_parameters: dict | None = None) -> None:
        """
        Initiation method.
        :param api_base: API base.
        :param speech_recorder_parameters: Speech recorder keyword arguments.
        :param audio_player_parameters: Audio player keyword arguments.
        """
        super().__init__(api_base=api_base)
        self.speech_recorder = SpeechRecorder(**cfg.DEFAULT_SPEECH_RECORDER if speech_recorder_parameters is None else speech_recorder_parameters)
        self.audio_player = AudioPlayer(**cfg.DEFAULT_AUDIO_PLAYER if audio_player_parameters is None else audio_player_parameters)
        self.audio_input_queue = Queue()
        self.audio_stop_event = Event()
        self.audio_thread = self.audio_player.spawn_output_thread(
            input_queue=self.audio_input_queue,
            stop_event=self.audio_stop_event,
            loop_pause=0.4
        )

    def interrupt(self) -> dict:
        """
        Interrupts services.
        """
        self.audio_stop_event.set()
        while not self.audio_input_queue.empty():
            self.audio_input_queue.get_nowait() 
        self.audio_input_queue.put((np.array([]), {}))
        self.audio_thread.join()
        self.audio_input_queue = Queue()
        self.audio_stop_event = Event()
        self.audio_thread = self.audio_player.spawn_output_thread(
            input_queue=self.audio_input_queue,
            stop_event=self.audio_stop_event,
            loop_pause=0.4
        )
        return requests.post(self.api_base + Endpoints.interrupt).json()


    def transcribe(self, audio_input: np.ndarray) -> Tuple[str, dict]:
        """
        Fetches transcription response from interface.
        :param audio_input: Audio input.
        :param transcription_parameters: Transcription parameters.
        :return: Output text and process metadata.
        """
        input_package = ServicePackage(content=audio_input.tolist())
        input_package.metadata_stack[-1]["dtype"] = str(audio_input.dtype)
        result = self.process(
            service="transcriber", 
            input_package=input_package
            )
        return result["content"], result["metadata_stack"][-1]

    def synthesize(self, text: str) -> Tuple[np.ndarray, dict]:
        """
        Fetches synthesis response from interface.
        :param text: Text input.
        :return: Output file path and metadata.
        """
        result = self.process(
            service="synthesizer", 
            input_package=ServicePackage(content=text)
            )
        return np.array(result["content"], dtype=result["metadata_stack"][-1].pop("dtype")), result["metadata_stack"][-1]

    def record_and_transcribe_speech(self) -> str:
        """
        Records and transcribes a speech input.
        """
        audio_input, _ = self.speech_recorder.record_single_input()
        return self.transcribe(audio_input=audio_input)

    def synthesize_and_output_speech(self, text: str) -> None:
        """
        Synthesizes and outputs speech.
        :param text: Text input.
        """
        audio_input, playback_parameters = self.synthesize(text=text)
        time.sleep(1.5)
        self.audio_input_queue.put((audio_input, playback_parameters))
    
    def chat(self,
             prompt: str, 
             stream: bool = True,
             output_as_audio: bool = False) -> Generator[Tuple[str, dict], None, None]:
        """
        Fetches chat response from st.session_state["CLIENT"] interface.
        :param prompt: User prompt.
        :param chat_parameters: Chat parameters.
        :param output_as_audio: Outputting response as audio.
        :return: Chat response.
        """
        input_package = ServicePackage(content=prompt)
        input_package.metadata_stack[-1]["chat_parameters"] = {"stream": stream}

        if stream:
            for response in self.stream(service="chat", 
                                        input_package=input_package):
                response_chunk = response.get("content", "") 
                if response_chunk and output_as_audio:
                    self.synthesize_and_output_speech(text=response_chunk)
                yield response_chunk, response["metadata_stack"][-1] 
        else:
            response = self.process(
                service="chat", 
                input_package=input_package
            )
            if response and output_as_audio:
                self.synthesize_and_output_speech(text=response["content"])
            yield response["content"], response["metadata_stack"][-1] 

    def __del__(self) -> None:
        """
        Deconstructs instance.
        """
        self.audio_stop_event.set()
        self.audio_thread.join()