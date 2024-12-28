# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from typing import Any, List, Callable, Dict
from src.utility.language_model_abstractions import LanguageModelConfig, ChatModelInstance, RemoteChatModelInstance
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
                 prompt_maker: Callable | str | None = None,
                 use_history: bool = True,
                 history: List[Dict[str, str | dict]] | None = None) -> None:
        """
        Initiates an instance.
        :param backend: Backend for model loading.
            Check Transcriber.supported_backends for supported backends.
        :param model_path: Path to model files.
            Defaults to None in which case a default model is used.
            The latter will most likely result in it being downloaded.
        :param model_parameters: Model loading parameters as dictionary.
            Defaults to None.
        :param synthesis_parameters: Synthesis parameters as dictionary.
            Defaults to None.
        :param args: Arbitrary arguments.
        :param kwargs: Arbitrary keyword arguments.
        """
        self.language_model = LanguageModelConfig(

        )
        self.chat_model = ChatModelInstance(
            backend=backend,
            model_path=model_path,
            model_parameters=model_parameters,
            synthesis_parameters=synthesis_parameters
        )
        super().__init__(handler_method=self.synthesizer.synthesize, *args, **kwargs)


class RemoteChatModule(BasicHandlerModule):
    """
    Remote chat module.
    Generates a response for a given input.
    """
    def __init__(self, 
                 backend: str,
                 model_path: str | None = None,
                 model_parameters: dict | None = None,
                 synthesis_parameters: dict | None = None, 
                 *args: Any | None, 
                 **kwargs: Any | None) -> None:
        """
        Initiates an instance.
        :param backend: Backend for model loading.
            Check Transcriber.supported_backends for supported backends.
        :param model_path: Path to model files.
            Defaults to None in which case a default model is used.
            The latter will most likely result in it being downloaded.
        :param model_parameters: Model loading parameters as dictionary.
            Defaults to None.
        :param synthesis_parameters: Synthesis parameters as dictionary.
            Defaults to None.
        :param args: Arbitrary arguments.
        :param kwargs: Arbitrary keyword arguments.
        """
        self.synthesizer = Synthesizer(
            backend=backend,
            model_path=model_path,
            model_parameters=model_parameters,
            synthesis_parameters=synthesis_parameters
        )
        super().__init__(handler_method=self.synthesizer.synthesize, *args, **kwargs)