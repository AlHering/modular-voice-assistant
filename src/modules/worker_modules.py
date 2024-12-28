# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from typing import Any, Tuple, List, Callable, Generator
import numpy as np
from queue import Empty
from pydantic import BaseModel, ConfigDict
from src.utility.commandline_utility import silence_stderr
from src.utility.pyaudio_utility import play_wave
from src.utility.language_model_abstractions import ChatModelConfig, ChatModelInstance, RemoteChatModelConfig, RemoteChatModelInstance
from src.utility.sound_model_abstractions import Transcriber, Synthesizer, SpeechRecorder
from src.modules.abstractions import VAModule, VAPackage, VAModuleConfig, BasicHandlerModule


class ChatModule(BasicHandlerModule):
    """
    Chat module.
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