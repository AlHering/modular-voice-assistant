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
from src.interfaces.endpoints.lm_instances import LMInstance
from src.interfaces.endpoints.agent import Agent
from src.interfaces.endpoints.agent_memory import AgentMemory
from src.interfaces.endpoints.tooling import AgentTool
from src.control.backend_controller import BackendController


class Agent(BaseModel):
    """
    Agent dataclass.
    """
    # TODO: Include AgentTools
    name: str
    description: str

    memory: Optional[AgentMemory] = None

    general_lm: LMInstance
    dedicated_planner_lm: Optional[LMInstance] = None
    dedicated_actor_lm: Optional[LMInstance] = None
    dedicated_oberserver_lm: Optional[LMInstance] = None


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
    endpoint_base += "/agent"

    @backend.get(f"{endpoint_base}")
    @interaction_decorator()
    async def get_agent() -> dict:
        """
        Endpoint for getting all Agent entries.
        :return: Response.
        """
        return {"agent": controller.get_object_count_by_type("agent")}

    @backend.post(f"{endpoint_base}")
    @interaction_decorator()
    async def post_agent(agent: Agent) -> dict:
        """
        Endpoint for posting an Agent entries.
        :param agent: Agent.
        :return: Response.
        """
        return {"agent": controller.post_object("agent", **dict(agent))}

    @backend.get(f"{endpoint_base}/{{id}}")
    @interaction_decorator()
    async def get_agent(id: int) -> dict:
        """
        Endpoint for getting an Agent entry.
        :param id: Agent ID.
        :return: Response.
        """
        return {"agent": controller.get_object_by_id("agent", id)}

    @backend.delete(f"{endpoint_base}/{{id}}")
    @interaction_decorator()
    async def delete_agent(id: int) -> dict:
        """
        Endpoint for deleting an Agent entry.
        :param id: Agent ID.
        :return: Response.
        """
        return {"agent": controller.delete_object("agent", id)}

    @backend.patch(f"{endpoint_base}/{{id}}")
    @interaction_decorator()
    async def patch_agent(id: int, patch: dict) -> dict:
        """
        Endpoint for deleting an Agent entry.
        :param id: Agent ID.
        :param patch: Patch payload.
        :return: Response.
        """
        return {"agent": controller.patch_object("agent", id, **patch)}

    @backend.put(f"{endpoint_base}")
    @interaction_decorator()
    async def put_agent(agent: Agent) -> dict:
        """
        Endpoint for posting or updating an Agent entry.
        :param agent: Agent.
        :return: Response.
        """
        return {"agent": controller.put_document(**dict(agent))}
