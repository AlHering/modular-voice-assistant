# -*- coding: utf-8 -*-
"""
****************************************************
*               Command Line Agents                *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
from typing import Callable, Optional, List
from fastapi import FastAPI
from pydantic import BaseModel
from src.control.backend_controller import BackendController


class AgentMemory(BaseModel):
    """
    AgentMemory dataclass.
    """
    backend: str
    path: str
    paramters: Optional[dict] = None


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
    endpoint_base += "/agent_memory"

    @backend.get(f"{endpoint_base}")
    @interaction_decorator()
    async def get_agent_memory() -> dict:
        """
        Endpoint for getting all AgentMemory entries.
        :return: Response.
        """
        return {"agent_memory": controller.get_object_count_by_type("agent_memory")}

    @backend.post(f"{endpoint_base}")
    @interaction_decorator()
    async def post_agent_memory(agent_memory: AgentMemory) -> dict:
        """
        Endpoint for posting an AgentMemory entries.
        :param agent_memory: AgentMemory.
        :return: Response.
        """
        return {"agent_memory": controller.post_object("agent_memory", **dict(agent_memory))}

    @backend.get(f"{endpoint_base}/{{id}}")
    @interaction_decorator()
    async def get_agent_memory(id: int) -> dict:
        """
        Endpoint for getting an AgentMemory entry.
        :param id: AgentMemory ID.
        :return: Response.
        """
        return {"agent_memory": controller.get_object_by_id("agent_memory", id)}

    @backend.delete(f"{endpoint_base}/{{id}}")
    @interaction_decorator()
    async def delete_agent_memory(id: int) -> dict:
        """
        Endpoint for deleting an AgentMemory entry.
        :param id: AgentMemory ID.
        :return: Response.
        """
        return {"agent_memory": controller.delete_object("agent_memory", id)}

    @backend.patch(f"{endpoint_base}/{{id}}")
    @interaction_decorator()
    async def patch_agent_memory(id: int, patch: dict) -> dict:
        """
        Endpoint for deleting an AgentMemory entry.
        :param id: AgentMemory ID.
        :param patch: Patch payload.
        :return: Response.
        """
        return {"agent_memory": controller.patch_object("agent_memory", id, **patch)}

    @backend.put(f"{endpoint_base}")
    @interaction_decorator()
    async def put_agent_memory(agent_memory: AgentMemory) -> dict:
        """
        Endpoint for posting or updating an AgentMemory entry.
        :param agent_memory: AgentMemory.
        :return: Response.
        """
        return {"agent_memory": controller.put_document(**dict(agent_memory))}
