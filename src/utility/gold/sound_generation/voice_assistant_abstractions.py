# -*- coding: utf-8 -*-
"""
****************************************************
*                      Utility                 
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from enum import Enum
from typing import Any, Union, Tuple, List, Optional
import os
import numpy as np
import pyaudio
import speech_recognition
from datetime import datetime as dt
from src.configuration import configuration as cfg
from threading import Thread, Event as TEvent
from queue import Empty, Queue as TQueue
from ...bronze.audio_utility import get_input_devices, get_output_devices
from ...bronze.time_utility import get_timestamp
from . import speech_to_text_utility, text_to_speech_utility


class InputMethod(Enum):
    """
    Represents input methods.
    """
    SPEECH_TO_TEXT = 0
    COMMAND_LINE = 1
    TEXT_FILE = 2


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
                 tts_instantiation_kwargs: dict = None,
                 history: List[dict] = None,
                 loop_pause: float = 0.1) -> None:
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
        :param history: History as list of dictionaries of the structure
            {"process": <"tts"/"stt">, "text": <text content>, "metadata": {...}}
        :param loop_pause: Pause in seconds between processing loops.
            Defaults to 0.1.
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

        self.stt_engine = stt_engine
        self.stt_model = stt_model
        self.stt_instantiation_kwargs = stt_instantiation_kwargs
        self.tts_engine = tts_engine
        self.tts_model = tts_model
        self.tts_instantiation_kwargs = tts_instantiation_kwargs

        self.set_stt_processor(
            stt_engine=stt_engine,
            stt_model=stt_model,
            stt_instantiation_kwargs=stt_instantiation_kwargs
        )
        self.set_tts_processor(
            tts_engine=tts_engine,
            tts_model=tts_model,
            tts_instantiation_kwargs=tts_instantiation_kwargs
        )

        self.input_queue = None
        self.input_interrupt = None
        self.llm_thread = None
        self.llm_interrupt = None
        self.output_thread = None
        self.output_queue = None
        self.output_interrupt = None

        self.history = history
        self.loop_pause = loop_pause
        self.cache = None

    def _reset(self, delete_history: bool = False) -> None:
        """
        Method for setting up and resetting handler. 
        :param delete_history: Flag for declaring whether to delete history.    
            Defaults to None.
        """
        self.set_stt_processor(
            stt_engine=self.stt_engine,
            stt_model=self.stt_model,
            stt_instantiation_kwargs=self.stt_instantiation_kwargs
        )
        self.set_tts_processor(
            tts_engine=self.tts_engine,
            tts_model=self.tts_model,
            tts_instantiation_kwargs=self.tts_instantiation_kwargs
        )

        self.input_queue = TQueue()
        self.input_interrupt = TEvent()
        self.llm_interrupt = TEvent()
        self.llm_thread = Thread(
            target=self.run_llm_process
        )
        self.output_queue = TQueue()
        self.output_interrupt = TEvent()
        self.output_thread = Thread(
            target=self.run_tts_process
        )
        self.output_thread.daemon = True
        self.output_thread.start()

        self.history = [] if self.history is None or delete_history else self.history
        self.cache = {}


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

    def handle_stt_input(self) -> Tuple[Optional[str], Optional[dict]]:
        """
        Acquires input based on STT.
        :return: Transcribed input and list of metadata entries.
        """
        recognizer = speech_recognition.Recognizer()
        recognizer_kwargs = {
            "energy_threshold": 1000,
            "dynamic_energy_threshold": False,
            "pause_threshold": 1.0
        }
        for key in recognizer_kwargs:
            setattr(recognizer, key, recognizer_kwargs[key])
        microphone_kwargs = {
            "device_index": self.input_device_index,
            "sample_rate": 16000,
            "chunk_size": 1024
        }
        microphone = speech_recognition.Microphone(**microphone_kwargs)
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(
            source=microphone
        )
        audio_as_numpy_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        text, metadata_entries = {
            "whisper": speech_to_text_utility.transcribe_with_whisper,
            "faster-whisper": speech_to_text_utility.transcribe_with_faster_whisper
        }[self.sst_engine](
            audio_input=audio_as_numpy_array,
            model=self.stt_processor,
            transcription_kwargs=None
        )
        return text, {"timestamp": get_timestamp(), "input_method": "speech_to_text", "transcription_metadata": metadata_entries}

    def handle_cli_input(self) -> Tuple[Optional[str], Optional[dict]]:
        """
        Acquires input based on command line interaction.
        :return: Transcribed input and list of metadata entries.
        """
        text = input("User > ")
        metadata = {"timestamp": get_timestamp(), "input_method": "command_line"}
        return text, metadata


    def handle_file_input(self) -> Tuple[Optional[str], Optional[dict]]:
        """
        Acquires input based on text files.
        :return: Transcribed input and list of metadata entries.
        """
        if self.cache["text_input"]:
            text = self.cache["text_input"][0]
            self.cache["text_input"] = self.cache["text_input"][1:]
            metadata = {"timestamp": get_timestamp(), "input_method": "text_file"}
        return text, metadata

    def run_stt_process(self, input_method: InputMethod = InputMethod.SPEECH_TO_TEXT) -> None:
        """
        Runs STT process.
        :param inut_method: Input method out of SPEECH_TO_TEXT, COMMAND_LINE, TEXT_FILE.
            Defaults to SPEECH_TO_TEXT.
        """
        if input_method == InputMethod.TEXT_FILE:
            input_file = os.path.join(self.working_directory, "input.txt")
            self.cache["text_input"] = self.cache.get("text_input", open(input_file, "r").readlines() if os.path.exists(input_file) else [])
        input_handling = {
            InputMethod.SPEECH_TO_TEXT: self.handle_stt_input,
            InputMethod.COMMAND_LINE: self.handle_cli_input,
            InputMethod.TEXT_FILE: self.handle_file_input
        }[input_method]
        while not self.input_interrupt.is_set():
            new_input = input_handling()
            if new_input[0]:
                self.input_queue.put(new_input)

    def run_llm_process(self) -> None:
        """
        Runs LLM process.
        Should run in a separate thread to allow for continuous interaction.
        """
        while not self.llm_interrupt.is_set():
            try:
                input_text, input_metadata = self.output_queue.get(self.loop_pause)
                if input_text:
                    # Handle input text
                    pass
                else:
                    # Handle empty input text
                    pass
            except Empty:
                # Handle empty queue stopping
                pass

    def run_tts_process(self) -> None:
        """
        Runs TTS process.
        Should run in a separate thread to allow for continuous interaction.
        """
        while not self.output_interrupt.is_set():
            try:
                output_text, output_metadata = self.output_queue.get(self.loop_pause)
                if output_text == "<EOS>":
                    # Handle EOS stopping
                    pass
                elif output_text:
                    # Handle output text
                    pass
                else:
                    # Handle empty output text
                    pass
            except Empty:
                # Handle empty queue stopping
                pass