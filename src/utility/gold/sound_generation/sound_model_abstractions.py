# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import os
import traceback
from typing import List, Tuple, Any, Callable, Optional, Union
from datetime import datetime as dt
from enum import Enum
import numpy as np
from .sound_model_instantiation import load_whisper_model, load_faster_whisper_model, load_coqui_tts_model
from .speech_to_text_utility import transcribe_with_faster_whisper, transcribe_with_whisper
from .text_to_speech_utility import synthesize_with_coqui_tts, synthesize_with_coqui_tts_to_file


class Transcriber(object):
    """
    Represents transcriber.
    """

    supported_backends: List[str] = ["faster-whisper", "whisper"]

    def __init__(self,
                 backend: str,
                 model_path: str,
                 model_parameters: dict = None,
                 transcription_parameters: dict = None) -> None:
        """
        Initiation method.
        :param backend: Backend for model loading.
            Check Transcriber.supported_backends for supported backends.
        :param model_path: Path to model files.
        :param model_parameters: Model loading kwargs as dictionary.
            Defaults to None.
        :param transcription_parameters: Transcription kwargs as dictionary.
            Defaults to None.
        """
        self.backend = backend

        self.model_parameters = {} if model_parameters is None else model_parameters
        self.transcription_parameters = {} if transcription_parameters is None else transcription_parameters

        self.model = {
            "faster-whisper": load_faster_whisper_model,
            "whisper": load_whisper_model
        }[self.backend](
            model_path=model_path,
            model_parameters=model_parameters
        )

        self.transcription_function = {
            "faster-whisper": transcribe_with_faster_whisper,
            "whisper": transcribe_with_whisper
        }[self.backend]

    def transcribe(self, audio_input: str, transcription_parameters: dict = None) -> Tuple[str, dict]:
        """
        Transcribes audio to text.
        :param audio_input: Wave file path or waveform.
        :param transcription_parameters: Transcription kwargs as dictionary.
            Defaults to None.
        """
        return self.transcription_function(
            audio_input=audio_input,
            model=self.model,
            transcription_kwargs=self.transcription_parameters if transcription_parameters is None else transcription_parameters
        )


class Synthesizer(object):
    """
    Represents synthesizer.
    """

    supported_backends: List[str] = ["coqui-tts"]

    def __init__(self,
                 backend: str,
                 model_path: str,
                 model_parameters: dict = None,
                 synthesis_parameters: dict = None) -> None:
        """
        Initiation method.
        :param backend: Backend for model loading.
            Check Transcriber.supported_backends for supported backends.
        :param model_path: Path to model files.
        :param model_parameters: Model loading kwargs as dictionary.
            Defaults to None.
        :param synthesis_parameters: Synthesis kwargs as dictionary.
            Defaults to None.
        """
        self.backend = backend

        self.model_parameters = {} if model_parameters is None else model_parameters
        self.synthesis_parameters = {} if synthesis_parameters is None else synthesis_parameters

        self.model = {
            "coqui-tts": load_coqui_tts_model,
        }[self.backend](
            model_path=model_path,
            model_parameters=model_parameters
        )

        self.sound_out_snythesis_functions = {
            "coqui-tts": synthesize_with_coqui_tts
        }[self.backend]
        self.file_out_snythesis_functions = {
            "coqui-tts": synthesize_with_coqui_tts_to_file
        }[self.backend]

    def synthesize(self, text: str, synthesis_parameters: dict = None) -> Tuple[np.ndarray, dict]:
        """
        Synthesize text to audio.
        :param audio_input: Wave file path or waveform.
        :param synthesis_parameters: Transcription kwargs as dictionary.
            Defaults to None.
        :return: File path and metadata.
        """
        return self.sound_out_snythesis_functions(
            text=text, 
            model=self.model,
            synthesis_kwargs=self.synthesis_parameters if synthesis_parameters is None else synthesis_parameters)

    def synthesize_to_file(self, text: str, output_path: str, synthesis_parameters: dict = None) -> Tuple[np.ndarray, dict]:
        """
        Synthesize text to audio.
        :param audio_input: Wave file path or waveform.
        :param output_path: Path for output file.
        :param synthesis_parameters: Transcription kwargs as dictionary.
            Defaults to None.
        :return: Output file path and metadata.
        """
        return self.file_out_snythesis_functions(
            text=text, 
            output_path=output_path,
            model=self.model,
            synthesis_kwargs=self.synthesis_parameters if synthesis_parameters is None else synthesis_parameters)
        