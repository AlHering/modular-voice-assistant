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


class Synthesizer(BaseModel):
    """
    Synthesizer dataclass.
    """
    backend: str
    model_path: str
    model_path: Optional[str] = None
    model_parameters: Optional[dict] = None
    synthesis_parameters: Optional[dict] = None


class Transcriber(BaseModel):
    """
    Transcriber dataclass.
    """
    backend: str
    model_path: str
    model_path: Optional[str] = None
    model_parameters: Optional[dict] = None
    transcription_parameters: Optional[dict] = None


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
    endpoint_base += "/transcriber"

    @backend.get(f"{endpoint_base}")
    @interaction_decorator()
    async def get__transcribers() -> dict:
        """
        Endpoint for getting all transcriber entries.
        :return: Response.
        """
        return {"transcribers": controller.get_objects_by_type("transcriber")}

    @backend.post(f"{endpoint_base}")
    @interaction_decorator()
    async def post_transcriber(transcriber: Transcriber) -> dict:
        """
        Endpoint for posting transcriber entries.
        :param transcriber: Transcriber instance.
        :return: Response.
        """
        return {"transcriber": controller.post_object("transcriber", **dict(transcriber))}

    @backend.get(f"{endpoint_base}/{{id}}")
    @interaction_decorator()
    async def get_transcriber(id: int) -> dict:
        """
        Endpoint for getting an transcriber entry.
        :param id: Transcriber ID.
        :return: Response.
        """
        return {"transcriber": controller.get_object_by_id("transcriber", id)}

    @backend.delete(f"{endpoint_base}/{{id}}")
    @interaction_decorator()
    async def delete_transcriber(id: int) -> dict:
        """
        Endpoint for deleting an transcriber entry.
        :param id: Transcriber ID.
        :return: Response.
        """
        return {"transcriber": controller.delete_object("transcriber", id)}

    @backend.patch(f"{endpoint_base}/{{id}}")
    @interaction_decorator()
    async def patch_transcriber(id: int, patch: dict) -> dict:
        """
        Endpoint for patching an transcriber entry.
        :param id: Transcriber ID.
        :param patch: Patch payload.
        :return: Response.
        """
        return {"transcriber": controller.patch_object("transcriber", id, **patch)}

    @backend.put(f"{endpoint_base}")
    @interaction_decorator()
    async def put_transcriber(transcriber: Transcriber) -> dict:
        """
        Endpoint for posting or updating an transcriber entry.
        :param transcriber: Trancriber.
        :return: Response.
        """
        return {"transcriber": controller.put_document(**dict(transcriber))}
