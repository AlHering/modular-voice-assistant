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


class SynthesizerModule(BasicHandlerModule):
    """
    Synthesizer module.
    Synthesizes and forwards audio data.
    """
    supported_backends = Synthesizer.supported_backends
    default_models = Synthesizer.default_models
    
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


class WaveOutputModule(VAModule):
    """
    Wave output module.
    Takes in VAPackages with wave audio data and outputs it. 
    Note, that the last metadata stack element must be stream parameters for outputting.
    """
    def process(self) -> VAPackage | Generator[VAPackage, None, None] | None:
        """
        Module processing method.
        Note, that the last metadata stack element must be stream parameters for outputting.
        Follow the "play_wave"-function for more information.
        """
        if not self.pause.is_set():
            try:
                input_package: VAPackage = self.input_queue.get(block=True, timeout=self.input_timeout)
                self.pause.set()
                self.add_uuid(self.received, input_package.uuid)
                self.log_info(f"Received input:\n'{input_package.content}'")
                with silence_stderr():
                    play_wave(input_package.content, input_package.metadata_stack[-1])
                self.pause.clear()
            except Empty:
                pass
