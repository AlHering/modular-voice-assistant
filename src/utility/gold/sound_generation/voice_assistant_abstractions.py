# -*- coding: utf-8 -*-
"""
****************************************************
*                      Utility                 
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from src.configuration import configuration as cfg
from ...bronze.audio_utility import get_input_devices, get_output_devices
from . import speech_to_text_utility, text_to_speech_utility
from typing import Any, Union, Tuple, List
import os
import pyaudio


class ConversationHandler(object):
    """
    Represents a conversation handler for handling audio based interaction.
    """
    supported_stt_engines: List[str] = ["faster-whisper", "whsiper"]
    supported_tts_engines: List[str] = ["coqui-tts"]

    def __init__(self, 
                 working_directory: str,
                 stt_engine: str = None,
                 stt_model: str = None,
                 stt_instantiation_kwargs: dict = None,
                 tts_engine: str = None,
                 tts_model: str = None,
                 tts_instantiation_kwargs: dict = None) -> None:
        """
        Initiation method.
        :param working_directory: Directory for productive files.
        :param stt_engine: STT engine.
            See AudioHandler.supported_stt_engines for supported engines.
            Defaults to None in which case the first supported engine is used.
        :param stt_model: STT model name or path.
        :param stt_instantiation_kwargs: STT model instantiation keyword arguments.
        :param tts_engine: TTS engine.
            See AudioHandler.supported_tts_engines for supported engines.
            Defaults to None in which case the first supported engine is used.
        :param tts_model: TTS model name or path.
        :param tts_instantiation_kwargs: TTS model instantiation keyword arguments.
        """
        if not os.path.exists(working_directory):
            os.makedirs(working_directory)
        self.working_directory = working_directory
        self.input_path = os.path.join(self.working_directory, "input.wav")
        self.output_path = os.path.join(self.working_directory, "output.wav")

        pya = pyaudio.PyAudio()
        self.input_device_index = pya.get_default_input_device_info().get("index")
        self.output_device_index = pya.get_default_output_device_info().get("index")
        pya.terminate()

        self.stt_processor = None
        self.tts_processor = None

        self.interrupt = False