# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from typing import Callable, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from src.control.backend_controller import BackendController


class SpeechRecorder(BaseModel):
    """
    SpeechRecorder dataclass.
    """
    input_device_index: Optional[int] = None
    recognizer_parameters: Optional[dict] = None
    microphone_parameters: Optional[dict] = None
    loop_pause: Optional[float] = None


def register_endpoints(backend: FastAPI,
                       interaction_decorator: Callable,
                       controller: BackendController,
                       endpoint_base: str) -> None:
    """
    Function for registering endpoints to given FastAPI based backend.
    :param backend: backend to register endpoints under. 
    :param interaction_decorator: Decorator function for wrapping endpoint functions.
    :param controller: Backend controller to handle endpoint accesses.
    :param endpoint_base: Endpoint base.
    """
    if endpoint_base[-1] == "/":
        endpoint_base = endpoint_base[:-1]
    endpoint_base += "/speech_recorder"

    @backend.get(f"{endpoint_base}")
    @interaction_decorator()
    async def get_speech_recorders() -> dict:
        """
        Endpoint for getting all speech_recorder entries.
        :return: Response.
        """
        return {"speech_recorders": controller.get_objects_by_type("speech_recorder")}

    @backend.post(f"{endpoint_base}")
    @interaction_decorator()
    async def post_speech_recorder(speech_recorder: SpeechRecorder) -> dict:
        """
        Endpoint for posting speech_recorder entries.
        :param speech_recorder: SpeechRecorder instance.
        :return: Response.
        """
        return {"speech_recorder": controller.post_object("speech_recorder", **dict(speech_recorder))}

    @backend.get(f"{endpoint_base}/{{id}}")
    @interaction_decorator()
    async def get_speech_recorder(id: int) -> dict:
        """
        Endpoint for getting an speech_recorder entry.
        :param id: SpeechRecorder ID.
        :return: Response.
        """
        return {"speech_recorder": controller.get_object_by_id("speech_recorder", id)}

    @backend.delete(f"{endpoint_base}/{{id}}")
    @interaction_decorator()
    async def delete_speech_recorder(id: int) -> dict:
        """
        Endpoint for deleting an speech_recorder entry.
        :param id: SpeechRecorder ID.
        :return: Response.
        """
        return {"speech_recorder": controller.delete_object("speech_recorder", id)}

    @backend.patch(f"{endpoint_base}/{{id}}")
    @interaction_decorator()
    async def patch_speech_recorder(id: int, patch: dict) -> dict:
        """
        Endpoint for patching an speech_recorder entry.
        :param id: SpeechRecorder ID.
        :param patch: Patch payload.
        :return: Response.
        """
        return {"speech_recorder": controller.patch_object("speech_recorder", id, **patch)}

    @backend.put(f"{endpoint_base}")
    @interaction_decorator()
    async def put_speech_recorder(speech_recorder: SpeechRecorder) -> dict:
        """
        Endpoint for posting or updating an speech_recorder entry.
        :param speech_recorder: SpeechRecorder.
        :return: Response.
        """
        return {"speech_recorder": controller.put_document(**dict(speech_recorder))}
