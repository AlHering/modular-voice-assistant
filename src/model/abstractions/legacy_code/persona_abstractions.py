# -*- coding: utf-8 -*-
"""

WARNING: LEGACY CODE - just for reference

****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from typing import Any, Union, Optional
from datetime import datetime as dt
from pydantic import BaseModel
from src.model.abstractions.language_model_abstractions import ChatModelInstance, ChatModelConfig, RemoteChatModelConfig
from .knowledgebase_abstractions import Memory


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
                "metadata": {"initiated": dt.now()}
            })

        self.memory = memory

    def talk(self, text: str = None) -> Optional[str]:
        """
        Prompts a response of the persona wrapped language model.
        :param text: Optional text content.
        :return: Text response.
        """
        pass

    def consolidate(self) -> None:
        """
        Consolidates memory.
        """
        pass

    @classmethod
    def from_configuration(cls, config: PersonaConfiguration) -> Any:
        """
        Returns a persona instance from configuration.
        :param config: Persona configuration.
        :return: Persona instance.
        """
        return cls(**config.model_dump())