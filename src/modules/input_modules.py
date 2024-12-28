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


class SpeechRecorderModule(VAModule):
    """
    Speech recorder module.
    Records a speech snipped from the user and forwards it as VAPackage.
    """
    supported_input_devices = SpeechRecorder.supported_input_devices

    def __init__(self, 
                 input_device_index: int | None = None,
                 recognizer_parameters: dict | None = None,
                 microphone_parameters: dict | None = None,
                 recorder_loop_pause: float = .1,
                 *args: Any | None, 
                 **kwargs: Any | None) -> None:
        """
        Initiates an instance.
        :param input_device_index: Input device index.
            Check SpeechRecorder.supported_input_devices for available input device profiles.
            Defaults to None in which case the default input device index is fetched.
        :param recognizer_parameters: Keyword arguments for setting up recognizer instances.
            Defaults to None in which case default values are used.
        :param microphone_parameters: Keyword arguments for setting up microphone instances.
            Defaults to None in which case default values are used.
        :param loop_pause: Forced pause between loops in seconds.
            Defaults to 0.1.
        :param args: Arbitrary arguments.
        :param kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.speech_recorder = SpeechRecorder(
            input_device_index=input_device_index,
            recognizer_parameters=recognizer_parameters,
            microphone_parameters=microphone_parameters,
            loop_pause=recorder_loop_pause
        )

    def process(self) -> VAPackage | Generator[VAPackage, None, None] | None:
        """
        Module processing method.
        :returns: Voice assistant package, containing the recorded audio data and a metadata stack with the
            recording metadata as first element.
        """
        if not self.pause.is_set():
            recorder_output, recorder_metadata = self.speech_recorder.record_single_input()
            self.log_info(f"Got voice input.")
            yield VAPackage(content=recorder_output, metadata_stack=[recorder_metadata])


class TranscriberModule(BasicHandlerModule):
    """
    Transcriber module.
    Transcribes audio data.
    """
    supported_backends = Transcriber.supported_backends
    default_models = Transcriber.default_models

    def __init__(self, 
                 backend: str,
                 model_path: str | None = None,
                 model_parameters: dict | None = None,
                 transcription_parameters: dict | None = None,
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
        :param transcription_parameters: Transcription parameters as dictionary.
            Defaults to None.
        :param args: Arbitrary arguments.
        :param kwargs: Arbitrary keyword arguments.
        """
        self.transcriber = Transcriber(
            backend=backend,
            model_path=model_path,
            model_parameters=model_parameters,
            transcription_parameters=transcription_parameters
        )
        super().__init__(handler_method=self.transcriber.transcribe, *args, **kwargs)