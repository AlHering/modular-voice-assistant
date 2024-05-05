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

        self.stt_engine = None
        self.tts_engine = None
        self.stt_processor = None
        self.tts_processor = None

        self.set_stt_processor(
            stt_engine=stt_engine,
            stt_model=stt_model,
            stt_instantiation_kwargs=stt_instantiation_kwargs
        )
        self.set_tts_processor(
            tts_engine=tts_engine,
            tts_model=tts_model,
            tts_instantiation_kwargs=stt_instantiation_kwargs
        )

        self.interrupt = False

    def set_input_device(self, input_device: Union[int, str] = None) -> None:
        """
        Sets input device for handler.
        :param input_device: Name or index of input device.
        """
        if input_device is None:
            pya = pyaudio.PyAudio()
            self.input_device_index = pya.get_default_input_device_info().get("index")
            pya.terminate()
        else:
            try:
                self.input_device_index = input_device if isinstance(input_device, int) else [device for device in get_input_devices(include_metadata=True) if device["name"] == input_device][0]
            except IndexError:
                cfg.LOGGER.warning(f"Setting input device failed. Could not find device '{input_device}'.")

    def set_output_device(self, output_device: Union[int, str] = None) -> None:
        """
        Sets output device for handler.
        :param output_device: Name or index of output device.
        """
        if output_device is None:
            pya = pyaudio.PyAudio()
            self.output_device_index = pya.get_default_input_device_info().get("index")
            pya.terminate()
        else:
            try:
                self.input_device_index = output_device if isinstance(output_device, int) else [device for device in get_output_devices(include_metadata=True) if device["name"] == output_device][0]
            except IndexError:
                cfg.LOGGER.warning(f"Setting output device failed. Could not find device '{output_device}'.")

    def set_stt_processor(self,
                          stt_engine: str = None,
                          stt_model: str = None,
                          stt_instantiation_kwargs: dict = None) -> None:
        """
        Sets STT processor.
        :param stt_engine: STT engine.
            See AudioHandler.supported_stt_engines for supported engines.
        :param stt_model: STT model name or path.
        :param stt_instantiation_kwargs: STT model instantiation keyword arguments.
        """
        self.sst_engine = self.supported_stt_engines[0] if stt_engine is None else stt_engine
        self.stt_processor = {
            "whisper": speech_to_text_utility.get_whisper_model,
            "faster-whisper": speech_to_text_utility.get_faster_whisper_model
        }[self.sst_engine](
            model_name_or_path=stt_model,
            instantiation_kwargs=stt_instantiation_kwargs
        )
        
    def set_tts_processor(self,
                          tts_engine: str = None,
                          tts_model: str = None,
                          tts_instantiation_kwargs: dict = None) -> None:
        """
        Sets TTS processor.
        :param tts_engine: TTS engine.
            See AudioHandler.supported_tts_engines for supported engines.
        :param tts_model: TTS model name or path.
        :param tts_instantiation_kwargs: TTS model instantiation keyword arguments.
        """
        self.tts_engine = self.supported_tts_engines[0] if tts_engine is None else tts_engine
        self.tts_processor = {
            "coqui-tts": text_to_speech_utility.get_coqui_tts_model
        }[self.tts_engine](
            model_name_or_path=tts_model,
            instantiation_kwargs=tts_instantiation_kwargs
        )

    def run_stt_process(self) -> None:
        """
        Runs STT process.
        """
        pass
        
    def run_tts_process(self) -> None:
        """
        Runs TTS process.
        :param tts_engine: TTS engine.
        """
        pass
             