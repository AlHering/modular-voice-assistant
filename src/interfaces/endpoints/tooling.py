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


class ToolArgument(BaseModel):
    """
    ToolArgument dataclass.
    """
    name: str
    type: str
    description: Optional[str] = None
    value: Optional[str] = None


class AgentTool(BaseModel):
    """
    AgentTool dataclass.
    """
    name: str
    description: str
    func: str = None
    return_type: str = None

    tool_arguments: Optional[List[ToolArgument]] = None


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
    endpoint_base += "/agent_tool"

    @backend.get(f"{endpoint_base}")
    @interaction_decorator()
    async def get_agent_tool() -> dict:
        """
        Endpoint for getting all AgentTool entries.
        :return: Response.
        """
        return {"agent_tool": controller.get_object_count_by_type("agent_tool")}

    @backend.post(f"{endpoint_base}")
    @interaction_decorator()
    async def post_agent_tool(agent_tool: AgentTool) -> dict:
        """
        Endpoint for posting an AgentTool entries.
        :param agent_tool: AgentTool.
        :return: Response.
        """
        return {"agent_tool": controller.post_object("agent_tool", **dict(agent_tool))}

    @backend.get(f"{endpoint_base}/{{id}}")
    @interaction_decorator()
    async def get_agent_tool(id: int) -> dict:
        """
        Endpoint for getting an AgentTool entry.
        :param id: AgentTool ID.
        :return: Response.
        """
        return {"agent_tool": controller.get_object_by_id("agent_tool", id)}

    @backend.delete(f"{endpoint_base}/{{id}}")
    @interaction_decorator()
    async def delete_agent_tool(id: int) -> dict:
        """
        Endpoint for deleting an AgentTool entry.
        :param id: AgentTool ID.
        :return: Response.
        """
        return {"agent_tool": controller.delete_object("agent_tool", id)}

    @backend.patch(f"{endpoint_base}/{{id}}")
    @interaction_decorator()
    async def patch_agent_tool(id: int, patch: dict) -> dict:
        """
        Endpoint for deleting an AgentTool entry.
        :param id: AgentTool ID.
        :param patch: Patch payload.
        :return: Response.
        """
        return {"agent_tool": controller.patch_object("agent_tool", id, **patch)}

    @backend.put(f"{endpoint_base}")
    @interaction_decorator()
    async def put_agent_tool(agent_tool: AgentTool) -> dict:
        """
        Endpoint for posting or updating an AgentTool entry.
        :param agent_tool: AgentTool.
        :return: Response.
        """
        return {"agent_tool": controller.put_document(**dict(agent_tool))}
