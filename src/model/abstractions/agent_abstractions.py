# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2025 Alexander Hering             *
****************************************************
"""
from typing import Callable, Any, List
from pydantic import BaseModel
from uuid import UUID, uuid4
from enum import Enum
from src.model.abstractions.knowledgebase_abstractions import Entry, ChromaKnowledgeBase, Neo4jKnowledgebase, Knowledgebase
from src.utility.filter_mask_utility import FilterMask
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


class MemoryType(str, Enum):
    chroma = "chroma"
    neo4j = "neo4j"


class Memory(object):
    """
    Class, representing an agent's memory.
    """
    def __init__(self, 
                 memory_type: MemoryType = MemoryType.chroma, 
                 memory_parameters: dict | None = None,
                 retrieval_parameters: dict | None = None) -> None:
        """
        Initiation method.
        :param memory_type: Memory type.
        :param memory_parameters: Memory instantiation parameters.
        :param retrieval_parameters: Retrieval parameters. 
        """
        self.memory_parameters = {} if memory_parameters is None else memory_parameters
        self.retrieval_parameters = {} if retrieval_parameters is None else retrieval_parameters

        self.memory_type = {
            "chroma": ChromaKnowledgeBase,
            "neo4j": Neo4jKnowledgebase
        }[memory_type.value()]
        self.memory: Knowledgebase = self.memory_type(**self.memory_parameters)

    def store(self, content: str, metadata: dict) -> Entry:
        """
        Stores a memory.
        :param content: Textual content.
        :param metadata: Memory metadata.
        """
        entry = Entry(id=uuid4(),
                      content=content,
                      metadata=metadata)
        self.memory.store_entry(entry=entry)
        return entry
    
    def prepare_prompt_addition(self, one_or_multiple_entries: Entry | List[Entry]) -> str:
        """
        Prepares prompt addition for incorporating memory entries.
        :param one_or_multiple_entries: Memory entry or list of memory entries.
        :return: Prompt memory addition as string.
        """
        if isinstance(one_or_multiple_entries, Entry):
            one_or_multiple_entries = [one_or_multiple_entries]
        return "\nHere is a list of memories:\n" + "\n".join(
            f"Memory '{entry.id}':\nContent: '{entry.content}'\nMetadata{entry.metadata}" 
            for entry in one_or_multiple_entries) * "\n\n"

    def retrieve_by_id(self, entry_id: int | str | UUID) -> str:
        """
        Retrieves a memory by its ID.
        :param entry_id: Entry ID.
        :return: Prompt memory addition as string.
        """
        return self.prepare_prompt_addition(self.memory.get_entry_by_id(entry_id=entry_id))

    def retrieve_by_metadata(self, filtermasks: List[FilterMask]) -> str:
        """
        Retrieves a memories by metadata filters.
        :param filtermasks: List of filter masks.
        :return: Prompt memory addition as string.
        """
        return self.prepare_prompt_addition(self.memory.retrieve_entries(filtermasks=filtermasks))

    def retrieve_by_similarity(self, reference: str, filtermasks: List[FilterMask] | None = None) -> str:
        """
        Retrieves a memories by similarity.
        :param reference: Textual reference.
        :param filtermasks: List of filter masks.
        :param metadata: Prompt memory addition as string.
        """
        return self.prepare_prompt_addition(self.memory.retrieve_entries(query=reference, filtermasks=filtermasks))


class AgentTool(object):
    """
    Class, representing an agent tool.
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

    @classmethod
    def from_function(cls, func: Callable) -> Any:
        """
        Returns a tool instance from a function.
        :param func: Target function.
        :return: AgentTool instance.
        """
        raise NotImplementedError("Function to tool conversion not yet implemented.")
        parameters = {}
        return cls(**parameters)

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
    
    def __call__(self, *args: Any | None, **kwargs: Any | None) -> Any:
        """
        Call method for running tool function with arguments.
        :param args: Arbitrary function arguments.
        :param kwargs: Arbitrary function keyword arguments.
        :return: Function output.
        """
        return self.func(*args, **kwargs)