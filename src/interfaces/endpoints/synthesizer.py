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
    endpoint_base += "/synthesizer"

    @backend.get(f"{endpoint_base}")
    @interaction_decorator()
    async def get_synthesizers() -> dict:
        """
        Endpoint for getting all synthesizer entries.
        :return: Response.
        """
        return {"synthesizers": controller.get_objects_by_type("synthesizer")}

    @backend.post(f"{endpoint_base}")
    @interaction_decorator()
    async def post_synthesizer(synthesizer: Synthesizer) -> dict:
        """
        Endpoint for posting synthesizer entries.
        :param synthesizer: Synthesizer instance.
        :return: Response.
        """
        return {"synthesizer": controller.post_object("synthesizer", **dict(synthesizer))}

    @backend.get(f"{endpoint_base}/{{id}}")
    @interaction_decorator()
    async def get_synthesizer(id: int) -> dict:
        """
        Endpoint for getting an synthesizer entry.
        :param id: Synthesizer ID.
        :return: Response.
        """
        return {"synthesizer": controller.get_object_by_id("synthesizer", id)}

    @backend.delete(f"{endpoint_base}/{{id}}")
    @interaction_decorator()
    async def delete_synthesizer(id: int) -> dict:
        """
        Endpoint for deleting an synthesizer entry.
        :param id: Synthesizer ID.
        :return: Response.
        """
        return {"synthesizer": controller.delete_object("synthesizer", id)}

    @backend.patch(f"{endpoint_base}/{{id}}")
    @interaction_decorator()
    async def patch_synthesizer(id: int, patch: dict) -> dict:
        """
        Endpoint for patching an synthesizer entry.
        :param id: Synthesizer ID.
        :param patch: Patch payload.
        :return: Response.
        """
        return {"synthesizer": controller.patch_object("synthesizer", id, **patch)}

    @backend.put(f"{endpoint_base}")
    @interaction_decorator()
    async def put_synthesizer(synthesizer: Synthesizer) -> dict:
        """
        Endpoint for posting or updating an synthesizer entry.
        :param synthesizer: Synthesizer.
        :return: Response.
        """
        return {"synthesizer": controller.put_document(**dict(synthesizer))}
