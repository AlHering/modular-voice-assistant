# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import os
from typing import Any, Generator, Tuple
from src.utility.sound_model_abstractions import Transcriber, SpeechRecorder
from src.modules.abstractions import PipelineModule, PipelinePackage, BasicHandlerModule


class SpeechRecorderModule(PipelineModule):
    """
    Speech recorder module.
    Records a speech snipped from the user and forwards it as PipelinePackage.
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

    @classmethod
    def validate_configuration(cls, config: dict) -> Tuple[bool | None, str]:
        """
        Validates an configuration.
        :param config: Module configuration.
        :return: True or False and validation report depending on validation success. 
            None and validation report in case of warnings. 
        """
        return True, "Validation succeeded."

    def process(self) -> PipelinePackage | Generator[PipelinePackage, None, None] | None:
        """
        Module processing method.
        :returns: Pipeline package, containing the recorded audio data and a metadata stack with the
            recording metadata as first element.
        """
        if not self.pause.is_set():
            recorder_output, recorder_metadata = self.speech_recorder.record_single_input()
            self.log_info(f"Got voice input.")
            yield PipelinePackage(content=recorder_output, metadata_stack=[recorder_metadata])


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

    @classmethod
    def validate_configuration(cls, config: dict) -> Tuple[bool | None, str]:
        """
        Validates an configuration.
        :param config: Module configuration.
        :return: True or False and validation report depending on validation success. 
            None and validation report in case of warnings. 
        """
        if not config["backend"] in cls.supported_backends:
            return False, f"Transcriber backend '{config['backend']}' is not supported."
        model_path = Transcriber.default_models[config["backend"]][0] if config.get("model_path") is None else config.get("model_path")
        if not os.path.exists(model_path):
            return None, f"Model path '{model_path}' is not in local filesystem.\nModel access must be handled remotely by the chosen backend."
        return True, "Validation succeeded."