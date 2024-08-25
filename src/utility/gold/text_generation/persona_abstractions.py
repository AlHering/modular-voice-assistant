# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from typing import List, Any, Union
from datetime import datetime as dt
from pydantic import BaseModel
from .language_model_abstractions import ChatModelInstance, RemoteChatModelInstance, ChatModelConfig, RemoteChatModelConfig
from .knowledgebase_abstractions import MemoryMetadata, MemoryEntry, Memory, ChromaKnowledgebase


class PersonaConfiguration(BaseModel):
    """
    Persona configuration dataclass.
    """
    persona_description: str
    chat_model_config: Union[ChatModelConfig, RemoteChatModelConfig]
    conversation_example: str | None = None
    welcome_message: str | None = None
    memory: Memory | None = None


class Persona(object): 
    """
    Persona class.
    """
    def __init__(self, 
                 persona_description: str,
                 chat_model_config: Union[ChatModelConfig, RemoteChatModelConfig],
                 conversation_example: str | None = None,
                 welcome_message: str | None = None,
                 memory: Memory | None = None) -> None:
        """
        Initiation method.
        :param persona_description: Persona character description.
        :param chat_model_config: (Remote) chat model config.
        :param conversation_example: An conversation example.
        :param welcome_message: Welcome message, put into the chat model history after the system prompt. 
        :param memory: Memory.
        """
        self.persona_description = persona_description
        self.chat_model_config = chat_model_config
        self.conversation_example = conversation_example
        self.welcome_message = welcome_message

        self.chat_model_config.system_prompt = f"{self.persona_description}.\n{self.conversation_example if self.conversation_example else ''}"
        if isinstance(self.chat_model_config, ChatModelConfig):
            self.chat_model = ChatModelInstance.from_configuration(config=self.chat_model_config)
        elif isinstance(self.chat_model_config, RemoteChatModelConfig):
            self.chat_model = RemoteChatModelConfig.from_configuration(config=self.chat_model_config)
        
        if self.welcome_message is not None:
            self.chat_model.history.append({
                "role": "assistant", 
                "content": self.welcome_message, 
                "metadata": {"intitated": dt.now()}
            })

        self.memory = memory

    @classmethod
    def from_configuration(cls, config: PersonaConfiguration) -> Any:
        """
        Returns a persona instance from configuration.
        :param config: Persona configuration.
        :return: Persona instance.
        """
        return cls(**config.model_dump())