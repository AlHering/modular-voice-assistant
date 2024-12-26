# -*- coding: utf-8 -*-
"""
****************************************************
*                      Utility                 
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from typing import Any, Tuple, List, Callable, Generator
import numpy as np
from queue import Empty
from pydantic import BaseModel, ConfigDict
from src.utility.commandline_utility import silence_stderr
from src.utility.pyaudio_utility import play_wave
from src.sound_model_abstractions import Transcriber, Synthesizer, SpeechRecorder
from src.modules.module_abstractions import VAModule, VAPackage, VAModuleConfig


class SpeechRecorderConfig(VAModuleConfig):
    """
    Speech recorder config class.
    """
    model_config: ConfigDict = ConfigDict(protected_namespaces=())

    input_device_index: int | None = None
    recognizer_parameters: dict | None = None
    microphone_parameters: dict | None = None
    recorder_loop_pause: float = .1


class SpeechRecorderModule(VAModule):
    """
    Speech recorder module.
    Records a speech snipped from the user and forwards it as VAPackage.
    """
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
        

WaveOutputConfig = VAModuleConfig


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


class BasicHandlerModule(VAModule):
    """
    Basic handler module.
    """
    def __init__(self, 
                 handler_method: Callable, 
                 *args: Any | None, 
                 **kwargs: Any | None) -> None:
        """
        Initiates an instance.
        :param handler_method: Handler method.
        :param args: Arbitrary arguments.
        :param kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.handler_method = handler_method

    def process(self) -> VAPackage | Generator[VAPackage, None, None] | None:
        """
        Module processing method.
        :returns: Voice assistant package, package generator or None.
        """
        if not self.pause.is_set():
            try:
                input_package: VAPackage = self.input_queue.get(block=True, timeout=self.input_timeout)
                self.add_uuid(self.received, input_package.uuid)
                self.log_info(f"Received input:\n'{input_package.content}'")
                valid_input = (isinstance(input_package.content, np.ndarray) and input_package.content.size > 0) or input_package.content

                if valid_input:
                    result = self.handler_method(input_package.content)
                    if isinstance(result, Generator):
                        for response_tuple in result:
                            self.log_info(f"Received response shard\n'{response_tuple[0]}'.")   
                            yield VAPackage(uuid=input_package.uuid, content=response_tuple[0], metadata_stack=input_package.metadata_stack + [response_tuple[1]])
                    else:
                        self.log_info(f"Received response\n'{result[0]}'.")             
                        yield VAPackage(uuid=input_package.uuid, content=result[0], metadata_stack=input_package.metadata_stack + [result[1]])
            except Empty:
                pass


class TranscriberConfig(VAModuleConfig):
    """
    Transcriber config class.
    """
    model_config: ConfigDict = ConfigDict(protected_namespaces=())

    backend: str
    model_path: str | None = None
    model_parameters: dict | None = None
    transcription_parameters: dict | None = None


class TranscriberModule(BasicHandlerModule):
    """
    Transcriber module.
    Transcribes audio data.
    """
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


class SynthesizerConfig(VAModuleConfig):
    """
    Synthesizer config class.
    """
    model_config: ConfigDict = ConfigDict(protected_namespaces=())

    backend: str
    model_path: str | None = None
    model_parameters: dict | None = None
    synthesis_parameters: dict | None = None


class SynthesizerModule(BasicHandlerModule):
    """
    Synthesizer module.
    Synthesizes and forwards audio data.
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