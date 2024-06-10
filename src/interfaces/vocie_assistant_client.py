# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import os
from enum import Enum
import httpx
import asyncio
import numpy as np
from requests import session
from typing import Optional, Any, Union, List, Tuple
from src.configuration import configuration as cfg


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


class Client(object):
    """
    Client class for easier interface interaction.
    """
    def __init__(self, **kwargs) -> None:
        """
        Initiation method:
        :param kwargs: Keyword arguments for defining relevant objects as either dictionary or id.
            Can contain
                lm_instance: Optional[Union[int, dict]] = None
                kb_instance: Optional[Union[int, dict]] = None
                tool_argument: Optional[Union[int, dict]] = None
                agent_tool: Optional[Union[int, dict]] = None
                agent_memory: Optional[Union[int, dict]] = None
                agent: Optional[Union[int, dict]] = None
                transcriber: Optional[Union[int, dict]] = None
                synthesizer: Optional[Union[int, dict]] = None
                speech_recorder: Optional[Union[int, dict]] = None
        """
        self.kwargs = kwargs
        for config in ["lm_instance", "kb_instance", "tool_argument", "agent_tool", "agent_memory",
                       "agent", "transcriber", "synthesizer", "speech_recorder"]:
            self.kwargs = self.kwargs.get(config)
        to_gather = [
            (key, self.kwargs[key]) for key in self.kwargs if isinstance(self.kwargs[key], dict)
        ]
        self.set_configurations(to_gather)

    async def process_objects(self, method: str, object_list: List[Tuple[str, dict]]) -> List[dict]:
        """
        Method for processing a list of objects.
        :param method: Target method.
        :param object_list: Objects as list of tuples of object type and object data.
        :return: List of responses.
        """
        async with httpx.AsyncClient() as client:
            tasks = [await getattr(client, method)(getattr(Endpoints, obj[0], json=obj[1])) for obj in object_list]
            results = await asyncio.gather(*tasks)
        return [res.json() for res in results]
    
    async def set_configurations(self, object_list: List[Tuple[str, dict]]) -> List[dict]:
        """
        Method for setting configurations for objects.
        :param object_list: Objects as list of tuples of object type and object data.
        :return: List of responses.
        """
        results = await self.process_objects("put", object_list)
        for index, value in enumerate("to_gather"):
            self.kwargs[value[0]] = [res[value[0]].get("id") for res in results if value[0] in res][0]


    """
    Voice Assistant

    def synthesize(text):
        resp = requests.post(f"http://127.0.0.1:7862/api/v1/synthesizer/1/synthesize",
                             json={
                                "text": text
                             })

        data = resp.json()
        
        audio = np.asanyarray(data["synthesis"], dtype=data["dtype"])
        play_wave(audio, data["metadata"])
        return audio, data["metadata"]

    def transcribe(audio):
        resp = requests.post("http://127.0.0.1:7862/api/v1/transcriber/1/transcribe",
                             json={
                                 "audio_data": audio.tolist(), 
                                 "audio_dtype": str(audio.dtype)
                             })
        data = resp.json()
        
        print(data)
    """
    async def transcribe(self, audio: np.ndarray, transcriber_id: Optional[int] = None) -> Tuple[str, dict]:
        """
        Method for transcribing audio data to text.
        :param audio: Audio data.
        :param transcriber_id: Optional transcriber ID.
            Defaults to client transcriber or 1 if none is set.
        :return: Transcript and metadata.
        """
        transcriber_id = self.kwargs["transcriber"] if transcriber_id is None else transcriber_id
        response = httpx.post(
            f"{Endpoints.transcriber}/{1 if transcriber_id is None 
                                       else transcriber_id}/{Endpoints.transcribe_appendix}",
            json={
                "audio_data": audio.tolist(), 
                "audio_dtype": str(audio.dtype)
            }
        ).json()
        return response["transcript"], response["metadata"]

    async def synthesize(self, text: str, synthesizer_id: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """
        Method for synthesizing text to audio data.
        :param text: Text data.
        :param synthesizer_id: Optional synthesizer ID.
            Defaults to client synthesizer or 1 if none is set.
        :return: Synthesis and metadata.
        """
        synthesizer_id = self.kwargs["synthesizer"] if synthesizer_id is None else synthesizer_id
        response = httpx.post(
            f"{Endpoints.synthesizer}/{1 if synthesizer_id is None 
                                       else synthesizer_id}/{Endpoints.synthesize_appendix}",
            json={
                "text": text
            }
        ).json()
        return np.asanyarray(response["synthesis"], dtype=response["dtype"]), response["metadata"]








