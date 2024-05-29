# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from typing import Callable, Optional, Union, List, Dict
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

    encoding_parameters: Optional[dict] = None
    generating_parameters: Optional[dict] = None
    decoding_parameters: Optional[dict] = None

    resource_requirements: Optional[dict] = None


class ChatInstance(BaseModel):
    """
    ChatInstance dataclass.
    """
    language_model_instance: Union[LMInstance, int]
    chat_parameters: Optional[dict] = None
    default_system_prompt: Optional[dict] = None
    prompt_maker: Optional[str] = None
    use_history: bool = True
    history: Optional[List[Dict[str, Union[str, dict]]]] = None


class KBInstance(BaseModel):
    """
    KBinstance dataclass.
    """
    backend: str
    knowledgebase_path: str

    knowledgebase_parameters: Optional[dict] = None
    preprocessing_parameters: Optional[dict] = None
    embedding_parameters: Optional[dict] = None

    encoding_parameters: Optional[dict] = None
    generating_parameters: Optional[dict] = None
    decoding_parameters: Optional[dict] = None

    default_retrieval_method: Optional[str] = None
    retrieval_parameters: Optional[dict] = None


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
    function: str = None
    return_type: str = None
    tool_arguments: Optional[List[Union[ToolArgument, int]]] = None


class AgentMemory(BaseModel):
    """
    AgentMemory dataclass.
    """
    backend: str
    path: str
    paramters: Optional[dict] = None


class Agent(BaseModel):
    """
    Agent dataclass.
    """
    # TODO: Include AgentTools
    name: str
    description: str

    memory: Optional[Union[AgentMemory, int]] = None

    general_lm: Union[LMInstance, int]
    dedicated_planner_lm: Optional[Union[LMInstance, int]] = None
    dedicated_actor_lm: Optional[Union[LMInstance, int]] = None
    dedicated_oberserver_lm: Optional[Union[LMInstance, int]] = None


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
        "lm_instance": LMInstance,
        "kb_instance": KBInstance,
        "tool_argument": ToolArgument,
        "agent_tool": AgentTool,
        "agent_memory": AgentMemory,
        "agent": Agent
    }

    for target in target_classes:
        constructed_endpoint = endpoint_base + f"/{target}"

        @backend.get(f"{constructed_endpoint}")
        @interaction_decorator()
        async def get_all() -> dict:
            """
            Endpoint for getting all entries.
            :return: Response.
            """
            return {f"{target}s": controller.get_objects_by_type(target)}

        @backend.post(f"{constructed_endpoint}")
        @interaction_decorator()
        async def post(data: Union[LMInstance, ToolArgument, AgentTool, AgentMemory, Agent]) -> dict:
            """
            Endpoint for posting entries.
            :param data: Instance data.
            :return: Response.
            """
            return {target: controller.post_object(target, **dict(data))}

        @backend.get(f"{constructed_endpoint}/{{id}}")
        @interaction_decorator()
        async def get(id: int) -> dict:
            """
            Endpoint for getting entries.
            :param id: Instance ID.
            :return: Response.
            """
            return {target: controller.get_object_by_id(target, id)}

        @backend.delete(f"{constructed_endpoint}/{{id}}")
        @interaction_decorator()
        async def delete(id: int) -> dict:
            """
            Endpoint for deleting entries.
            :param id: Instance ID.
            :return: Response.
            """
            return {target: controller.delete_object(target, id)}

        @backend.patch(f"{constructed_endpoint}/{{id}}")
        @interaction_decorator()
        async def patch(id: int, patch: dict) -> dict:
            """
            Endpoint for patching entries.
            :param id: Instance ID.
            :param patch: Patch payload.
            :return: Response.
            """
            return {target: controller.patch_object(target, id, **patch)}

        @backend.put(f"{constructed_endpoint}")
        @interaction_decorator()
        async def put(data: Union[LMInstance, ToolArgument, AgentTool, AgentMemory, Agent]) -> dict:
            """
            Endpoint for posting or updating entries.
            :param data: Instance data.
            :return: Response.
            """
            return {target: controller.put_object(target, **dict(data))}
        
    descriptions = {
        "get": "Endpoint for getting entries.",
        "post": "Endpoint for posting entries.",
        "patch": "Endpoint for patching entries.",
        "put": "Endpoint for putting entries.",
        "delete": "Endpoint for deleting entries."
    }
        
    scheme = backend.openapi()
    for path in scheme["paths"]:
        target = path.replace(f"{endpoint_base}/", "").split("/")[0]
        if target in target_classes:
            for method in ["post", "put"]:
                # 'requestBody': {'content': {'application/json': {'schema': {'$ref': '#/components/schemas/Transcriber'}}}, 'required': True}
                if "requestBody" in scheme["paths"][path].get(method, {}):
                    scheme["paths"][path][method]["requestBody"]["content"]["application/json"]["schema"] = {"$ref": f"#/components/schemas/{target_classes[target].__name__}"}
                    scheme["paths"][path][method]["summary"] += f" {target_classes[target].__name__}"
            for method in scheme["paths"][path]:
                scheme["paths"][path][method]["description"] = descriptions[method]
    backend.openapi_schema = scheme

