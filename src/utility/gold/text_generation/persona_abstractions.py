# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from typing import List, Any, Union
from pydantic import BaseModel
from .language_model_abstractions import ChatModelInstance, RemoteChatModelInstance
from .knowledgebase_abstractions import MemoryMetadata, MemoryEntry, Memory


class PersonaConfiguration(BaseModel):
    """
    Persona configuration dataclass.
    """
    persona_description: str
    conversation_example: str | None = None
    memories: List[MemoryEntry] | None = None


class Persona(object): 
    """
    Persona class.
    """
    def __init__(self, 
                 persona_description: str,
                 chat_model_instance: Union[ChatModelInstance, RemoteChatModelInstance],
                 conversation_example: str | None = None,
                 memories: List[MemoryEntry] | None = None) -> None:
        """
        Initiation method.
        :param persona_description: Persona character description.
        :param chat_model_instance: Chat model instance.
        :param conversation_example: An conversation example.
        :param memories: List of memories to initate the Persona with.
        :param remote: Flag that declares, whether to use an 
        """
        self.persona_description = persona_description
        self.conversation_example = conversation_example
        self.memories = memories
        
    @classmethod
    def from_configuration(cls, config: PersonaConfiguration) -> Any:
        """
        Returns a persona instance from configuration.
        :param config: Persona configuration.
        :return: Persona instance.
        """
        return cls(description=config.persona_description,
                   conversation_example=config.conversation_example,
                   memories=config.memories)