# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2025 Alexander Hering             *
****************************************************
"""
from typing import Callable, List, Any
from smolagents import OpenAIServerModel, CodeAgent, Tool, DuckDuckGoSearchTool


def get_remote_agent(
    api_base: str,
    api_token: str | None = None,
    model_id: str = "default",
    model_parameters: dict | None = None,
    tools: List[Tool] | None = None,
    agent_class: Callable | None = None,
    agent_parameters: dict | None = None
) -> Any:
    """
    Sets up and returns an agent based on a remote language model.
    """
    model_parameters = {} if model_parameters is None else model_parameters
    agent_parameters = {} if agent_parameters is None else agent_parameters
    agent_parameters["model"] = OpenAIServerModel(
        model_id=model_id,
        api_base=api_base,
        api_key=api_token,
        **model_parameters
    )
    agent_parameters["tools"] = [DuckDuckGoSearchTool()] if tools is None else tools
    if agent_class is None:
        return CodeAgent(**agent_parameters)
    else:
        return agent_class(**agent_parameters)
