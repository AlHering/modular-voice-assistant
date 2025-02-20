# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2025 Alexander Hering             *
****************************************************
"""
from typing import Callable, Any
from pydantic import BaseModel


class Memory(object):
    pass


class AgentTool(object):
    """
    Class, representing a agent tool.
    """

    def __init__(self,
                 name: str,
                 description: str,
                 func: Callable,
                 input_declaration: BaseModel,
                 output_declaration: BaseModel) -> None:
        """
        Initiation method.
        :param name: Name of the tool.
        :param description: Description of the tool.
        :param func: Function of the tool.
        :param input_declaration: Pydantic based input declaration.
        :param output_declaration: Pydantic based output declaration.
        """
        self.name = name
        self.description = description
        self.func = func
        self.input_declaration = input_declaration
        self.output_declaration = output_declaration

    def get_openai_function_representation(self) -> dict:
        """
        Method for acquiring openai function representation.
        :return: Tool as openai function dictionary.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_declaration.model_json_schema()
        }
    
    def __call__(self) -> Any:
        """
        Call method for running tool function with arguments.
        """
        return self.func(**self.input_declaration.model_dump())