# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2025 Alexander Hering             *
****************************************************
"""
from typing import Callable, List, Any
from agno.agent import Agent, Toolkit
from agno.models.openai import OpenAIChat
#from agno.embedder.openai import OpenAIEmbedder
#from agno.vectordb.lancedb import LanceDb, SearchType
from agno.tools.duckduckgo import DuckDuckGoTools


def get_remote_agent(
    api_base: str,
    api_token: str | None = None,
    model_id: str = "default",
    model_parameters: dict | None = None,
    tools: List[Toolkit] | None = None,
    agent_class: Callable | None = None,
    agent_parameters: dict | None = None
) -> Any:
    """
    Sets up and returns an agent based on a remote language model.
    """
    model_parameters = {} if model_parameters is None else model_parameters
    agent_parameters = {} if agent_parameters is None else agent_parameters
    agent_parameters["model"] = OpenAIChat(
        id=model_id,
        base_url=api_base,
        api_key=api_token,
        **model_parameters
    )
    agent_parameters["tools"] = [DuckDuckGoTools()] if tools is None else tools
    return Agent(**agent_parameters)