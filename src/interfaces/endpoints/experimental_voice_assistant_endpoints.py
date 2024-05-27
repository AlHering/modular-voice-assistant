# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from typing import Callable, Optional, Union
from fastapi import FastAPI
from pydantic import BaseModel
from src.control.backend_controller import BackendController


class Transcriber(BaseModel):
    """
    Transcriber dataclass.
    """
    backend: str
    model_path: str
    model_path: Optional[str] = None
    model_parameters: Optional[dict] = None
    transcription_parameters: Optional[dict] = None


class Synthesizer(BaseModel):
    """
    Synthesizer dataclass.
    """
    backend: str
    model_path: str
    model_path: Optional[str] = None
    model_parameters: Optional[dict] = None
    synthesis_parameters: Optional[dict] = None


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

    target_classes = {
        "transcriber": Transcriber,
        "synthesizer": Synthesizer,
        "speech_recorder": SpeechRecorder
    }

    for target in target_classes:
        constructed_endpoint = endpoint_base + f"/{target}"

        @backend.get(f"{constructed_endpoint}")
        @interaction_decorator()
        async def get_all() -> dict:
            f"""
            Endpoint for getting all {target} entries.
            :return: Response.
            """
            return {f"{target}s": controller.get_objects_by_type(target)}

        @backend.post(f"{constructed_endpoint}")
        @interaction_decorator()
        async def post(data: Union[Transcriber, Synthesizer, SpeechRecorder]) -> dict:
            f"""
            Endpoint for posting {target} entries.
            :param {target}: {target_classes[target].__name__} data.
            :return: Response.
            """
            return {target: controller.post_object(target, **dict(data))}

        @backend.get(f"{constructed_endpoint}/{{id}}")
        @interaction_decorator()
        async def get(id: int) -> dict:
            f"""
            Endpoint for getting an {target} entry.
            :param id: {target_classes[target].__name__} ID.
            :return: Response.
            """
            return {target: controller.get_object_by_id(target, id)}

        @backend.delete(f"{constructed_endpoint}/{{id}}")
        @interaction_decorator()
        async def delete(id: int) -> dict:
            f"""
            Endpoint for deleting {target} entries.
            :param id: {target_classes[target].__name__} ID.
            :return: Response.
            """
            return {target: controller.delete_object(target, id)}

        @backend.patch(f"{constructed_endpoint}/{{id}}")
        @interaction_decorator()
        async def patch(id: int, patch: dict) -> dict:
            f"""
            Endpoint for patching {target} entries.
            :param id: {target_classes[target].__name__} ID.
            :param patch: Patch payload.
            :return: Response.
            """
            return {target: controller.patch_object(target, id, **patch)}

        @backend.put(f"{constructed_endpoint}")
        @interaction_decorator()
        async def put(data: Union[Transcriber, Synthesizer, SpeechRecorder]) -> dict:
            f"""
            Endpoint for posting or updating an transcriber entry.
            :param {target}: {target_classes[target].__name__} data.
            :return: Response.
            """
            return {target: controller.put_object(target, **dict(data))}
        
        get_all.__name__ = f"get_{target}s"
        post.__name__ = f"post_{target}"
        get.__name__ = f"get_{target}"
        delete.__name__ = f"delete_{target}"
        patch.__name__ = f"patch_{target}"
        put.__name__ = f"put_{target}"
        
    print(backend.openapi_schema)
