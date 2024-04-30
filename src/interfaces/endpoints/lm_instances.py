# -*- coding: utf-8 -*-
"""
****************************************************
*               Command Line Agents                *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
from typing import Callable, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from src.control.backend_controller import BackendController


class LMInstance(BaseModel):
    """
    LMinstance dataclass.
    """
    backend: str
    model_path: str

    model_file: Optional[str] = None
    model_parameters: Optional[dict] = None
    tokenizer_path: Optional[str] = None
    tokenizer_parameters: Optional[dict] = None
    config_path: Optional[str] = None
    config_parameters: Optional[dict] = None

    default_system_prompt: Optional[str] = None
    use_history: Optional[dict] = True
    encoding_parameters: Optional[dict] = None
    generating_parameters: Optional[dict] = None
    decoding_parameters: Optional[dict] = None

    resource_requirements: Optional[dict] = None


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
    endpoint_base += "/lm_instance"

    @backend.get(f"{endpoint_base}")
    @interaction_decorator()
    async def get_lm_instances() -> dict:
        """
        Endpoint for getting all LM instance entries.
        :return: Response.
        """
        return {"lm_instances": controller.get_object_count_by_type("lm_instance")}

    @backend.post(f"{endpoint_base}")
    @interaction_decorator()
    async def post_lm_instances(lm_instance: LMInstance) -> dict:
        """
        Endpoint for posting an LM instance entries.
        :param lm_instance: LM instance.
        :return: Response.
        """
        return {"lm_instances": controller.post_object("lm_instance", **dict(lm_instance))}

    @backend.get(f"{endpoint_base}/{{id}}")
    @interaction_decorator()
    async def get_lm_instance(id: int) -> dict:
        """
        Endpoint for getting an LM instance entry.
        :param id: LM instance ID.
        :return: Response.
        """
        return {"lm_instance": controller.get_object_by_id("lm_instance", id)}

    @backend.delete(f"{endpoint_base}/{{id}}")
    @interaction_decorator()
    async def delete_lm_instance(id: int) -> dict:
        """
        Endpoint for deleting an LM instance entry.
        :param id: LM instance ID.
        :return: Response.
        """
        return {"lm_instance": controller.delete_object("lm_instance", id)}

    @backend.patch(f"{endpoint_base}/{{id}}")
    @interaction_decorator()
    async def patch_lm_instance(id: int, patch: dict) -> dict:
        """
        Endpoint for deleting an LM instance entry.
        :param id: LM instance ID.
        :param patch: Patch payload.
        :return: Response.
        """
        return {"lm_instance": controller.patch_object("lm_instance", id, **patch)}

    @backend.put(f"{endpoint_base}")
    @interaction_decorator()
    async def put_lm_instance(lm_instance: LMInstance) -> dict:
        """
        Endpoint for posting or updating an LM instance entry.
        :param lm_instance: LM instance.
        :return: Response.
        """
        return {"lm_instance": controller.put_document(**dict(lm_instance))}
