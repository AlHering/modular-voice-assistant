# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from typing import Any, List, Callable, Dict
from src.utility.language_model_abstractions import LlamaCPPModelInstance, ChatModelInstance, RemoteChatModelInstance
from src.modules.abstractions import BasicHandlerModule


class LocalChatModule(BasicHandlerModule):
    """
    Local chat module.
    Generates a response for a given input.
    """
    def __init__(self, 
                 model_path: str,
                 model_file: str | None = None,
                 model_parameters: dict | None = None,
                 encoding_parameters: dict | None = None,
                 embedding_parameters: dict | None = None,
                 generating_parameters: dict | None = None,
                 decoding_parameters: dict | None = None,
                 chat_parameters: dict | None = None,
                 system_prompt: str | None = None,
                 use_history: bool = True,
                 history: List[Dict[str, str | dict]] | None = None,
                 stream: bool = True,
                 *args: Any | None, 
                 **kwargs: Any | None) -> None:
        """
        Initiates an instance.
        :param model_path: Path to model files.
        :param model_file: Model file to load.
            Defaults to None.
        :param model_parameters: Model loading kwargs as dictionary.
            Defaults to None.
        :param encoding_parameters: Kwargs for encoding in the generation process as dictionary.
            Defaults to None in which case an empty dictionary is created and can be filled depending on the backend in the 
            different initiation methods.
        :param embedding_parameters: Kwargs for embedding as dictionary.
            Defaults to None in which case an empty dictionary is created and can be filled depending on the backend in the 
            different initiation methods.
        :param generating_parameters: Kwargs for generating in the generation process as dictionary.
            Defaults to None in which case an empty dictionary is created and can be filled depending on the backend in the 
            different initiation methods.
        :param decoding_parameters: Kwargs for decoding in the generation process as dictionary.
            Defaults to None in which case an empty dictionary is created and can be filled depending on the backend in the 
            different initiation methods.
        :param chat_parameters: Kwargs for chatting in the chatting process as dictionary.
            Defaults to None in which case an empty dictionary is created and can be filled depending on the language instance's
            model backend.
        :param system_prompt: Default system prompt.
            Defaults to a None in which case no system prompt is used.
            Only necessary, if the backend does not support chat interaction via message list.
        :param use_history: Flag, declaring whether to use the history.
            Defaults to True.
        :param history: Interaction history as list of {"role": <role>, "content": <message>, "metadata": <metadata>}-dictionaries.
            Defaults to None.
        :param stream: Flag for declaring whether to stream responses.
        :param args: Arbitrary arguments.
        :param kwargs: Arbitrary keyword arguments.
        """
        self.language_model = LlamaCPPModelInstance(
            model_path=model_path,
            model_file=model_file,
            model_parameters=model_parameters,
            encoding_parameters=encoding_parameters,
            embedding_parameters=embedding_parameters,
            generating_parameters=generating_parameters,
            decoding_parameters=decoding_parameters
        )
        self.chat_model = ChatModelInstance(
            language_model=self.language_model,
            chat_parameters=chat_parameters,
            system_prompt=system_prompt,
            use_history=use_history,
            history=history
        )
        super().__init__(handler_method=self.chat_model.chat_stream if stream else self.chat_model.chat, *args, **kwargs)


class RemoteChatModule(BasicHandlerModule):
    """
    Remote chat module.
    Generates a response for a given input.
    """
    def __init__(self, 
                 api_base: str,
                 api_token: str | None = None,
                 chat_parameters: dict | None = None,
                 system_prompt: str | None = None,
                 use_history: bool = True,
                 history: List[Dict[str, str | dict]] | None = None,
                 stream: bool = True,
                 *args: Any | None, 
                 **kwargs: Any | None) -> None:
        """
        Initiates an instance.
        :param api_base: API base URL in the format http://<host>:<port>/v1.
        :param api_token: API token, if necessary.
        :param chat_parameters: Kwargs for chatting in the chatting process as dictionary.
            Defaults to None in which case an empty dictionary is created and can be filled depending on the language instance's
            model backend.
        :param system_prompt: Default system prompt.
            Defaults to a None in which case no system prompt is used.
        :param use_history: Flag, declaring whether to use the history.
            Defaults to True.
        :param history: Interaction history as list of {"role": <role>, "content": <message>, "metadata": <metadata>}-dictionaries.
            Defaults to None.
        :param stream: Flag for declaring whether to stream responses.
        :param args: Arbitrary arguments.
        :param kwargs: Arbitrary keyword arguments.
        """
        self.chat_model = RemoteChatModelInstance(
            api_base=api_base,
            api_token=api_token,
            chat_parameters=chat_parameters,
            system_prompt=system_prompt,
            use_history=use_history,
            history=history
        )
        super().__init__(handler_method=self.chat_model.chat_stream if stream else self.chat_model.chat, *args, **kwargs)