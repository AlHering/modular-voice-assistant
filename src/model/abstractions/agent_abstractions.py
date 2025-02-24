# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2025 Alexander Hering             *
****************************************************
"""
from typing import Callable, Any
from pydantic import BaseModel
from src.utility.foreign.json_schema_to_grammar import SchemaConverter


def convert_pydantic_model_to_grammar(model: BaseModel) -> str:
    """
    Converts pydantic BaseModel class to GBNF grammar.
    :param model: Pydantic BaseModel class.
    :return: GBNF grammar as string.
    """
    conv = SchemaConverter(prop_order={},
                           allow_fetch=False,
                           dotall=False,
                           raw_pattern=False)
    schema = conv.resolve_refs(model.model_json_schema(), "")
    conv.visit(schema, "")
    return conv.format_grammar()


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
    
    def get_input_grammar(self) -> str:
        """
        Returns input grammar.
        """
        return convert_pydantic_model_to_grammar(model=self.input_declaration)

    def get_output_grammar(self) -> str:
        """
        Returns output grammar.
        """
        return convert_pydantic_model_to_grammar(model=self.output_declaration)
    
    def __call__(self) -> Any:
        """
        Call method for running tool function with arguments.
        """
        return self.func(**self.input_declaration.model_dump())