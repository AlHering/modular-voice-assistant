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
        to_gather = {
            key: self.kwargs[key] for key in self.kwargs if isinstance(self.kwargs[key], dict)
        }
        
        resps = self.process_objects(
            "put",
            to_gather
        )
        for index, key in enumerate(list(to_gather.keys())):
            self.kwargs[key] = resps[index][key]


    async def process_objects(self, method: str, object_list: List[Tuple[str, dict]]) -> List[dict]:
        """
        Method for processing a list of objects ().
        :param method: Target method.
        :param object_list: Objects as list of tuples of object type and object data.
        :return: List of responses.
        """
        async with httpx.AsyncClient() as client:
            tasks = [await getattr(client, method)(getattr(Endpoints, obj[0], json=obj[1])) for obj in object_list]
            results = await asyncio.gather(*tasks)
        return [res.json() for res in results]








