# -*- coding: utf-8 -*-
"""
****************************************************
*                      Utility                 
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from enum import Enum
from inspect import getfullargspec
from typing import Any, Union, Tuple, List, Optional, Callable, Dict
import os
import gc
import time
import numpy as np
import pyaudio
import speech_recognition
from datetime import datetime as dt
from src.configuration import configuration as cfg
from threading import Thread, Event as TEvent
from queue import Empty, Queue as TQueue
from ..text_generation.language_model_abstractions import LanguageModelInstance
from ...bronze.concurrency_utility import PipelineComponentThread
from ...bronze.audio_utility import get_input_devices, get_output_devices
from ...bronze.time_utility import get_timestamp
from . import speech_to_text_utility, text_to_speech_utility
from .sound_model_abstractions import Transcriber, Synthesizer, SpeechRecorder


class IOMethod(Enum):
    """
    Represents input or output methods.
    """
    SPEECH = 0
    COMMAND_LINE = 1
    TEXT_FILE = 2


class ConversationHandler(object):
    """
    Represents a conversation handler for handling audio based interaction.
    """
    supported_stt_engines: List[str] = ["faster-whisper", "whsiper"]

    def __init__(self, 
                 working_directory: str,
                 speech_recorder: SpeechRecorder = None,
                 transcriber: Transcriber = None,
                 synthesizer: Synthesizer = None,
                 llm: LanguageModelInstance = None,
                 history: List[dict] = None,
                 input_method: IOMethod = IOMethod.SPEECH,
                 output_method: IOMethod = IOMethod.SPEECH,
                 loop_pause: float = 0.1) -> None:
        """
        Initiation method.
        :param working_directory: Directory for productive files.
        :param speech_recorder: Speech recorder for STT processes.
        :param transcriber: Transcriber for STT processes.
            Defaults to None.
        :param synthesizer: Synthesizer for TTS processes.
            Defaults to None.
        :param llm: Language model instance.
            Defaults to None.
        :param history: History as list of dictionaries of the structure
            {"process": <"tts"/"stt">, "text": <text content>, "metadata": {...}}
        :param input_method: Input method.
            Defaults to SPEECH (STT).
        :param output_method: Output method.
            Defaults to SPEECH (TTS).
        :param loop_pause: Pause in seconds between processing loops.
            Defaults to 0.1.
        """
        cfg.LOGGER.info("Initiating Conversation Handler...")
        if not os.path.exists(working_directory):
            os.makedirs(working_directory)
        self.working_directory = working_directory
        self.input_path = os.path.join(self.working_directory, "input.wav") if input_method == IOMethod.SPEECH else os.path.join(self.working_directory, "input.txt")
        self.output_path = os.path.join(self.working_directory, "output.wav") if input_method == IOMethod.SPEECH else os.path.join(self.working_directory, "output.txt")

        pya = pyaudio.PyAudio()
        self.input_device_index = pya.get_default_input_device_info().get("index")
        self.output_device_index = pya.get_default_output_device_info().get("index")
        pya.terminate()

        self.speech_recorder = speech_recorder
        self.transcriber = transcriber
        self.synthesizer = synthesizer
        self.llm = llm

        self.interrupt = None
        self.llm_input_queue = None
        self.llm_output_queue = None
        self.llm_interrupt = None
        self.llm_thread = None

        self.tts_input_queue = None
        self.tts_output_queue = None
        self.tts_interrupt = None
        self.tts_thread = None

        self.history = history
        self.loop_pause = loop_pause
        self.input_method = input_method
        self.output_method = output_method
        self.input_method_handle = None
        self.cache = None

        self._reset()

    def _reset(self, delete_history: bool = False) -> None:
        """
        Method for setting up and resetting handler. 
        :param delete_history: Flag for declaring whether to delete history.    
            Defaults to None.
        """
        cfg.LOGGER.info("(Re)setting Conversation Handler...")
        self.interrupt = TEvent()

        self.llm_input_queue = TQueue()
        self.llm_output_queue = TQueue()
        self.llm_interrupt = TEvent()
        llm_function = lambda x: (x, {"dummy_method": True, "input": x}) if self.llm is None else self.llm.generate
        self.llm_thread = PipelineComponentThread(
            pipeline_function=llm_function,
            input_queue=self.llm_input_queue,
            output_queue=self.llm_output_queue,
            interrupt=self.llm_interrupt,
            validation_function=lambda x: x[0]
        )
        self.llm_thread.daemon = True
        self.llm_thread.start()

        self.tts_input_queue = TQueue()
        self.tts_output_queue = TQueue()
        self.tts_interrupt = TEvent()
        tts_function = self.synthesizer.synthesize
        self.tts_thread = PipelineComponentThread(
            pipeline_function=tts_function,
            input_queue=self.tts_input_queue,
            output_queue=self.tts_output_queue,
            interrupt=self.tts_interrupt,
            validation_function=lambda x: x[0]
        )
        self.tts_thread.daemon = True
        self.tts_thread.start()

        self.history = [] if self.history is None or delete_history else self.history
        if self.input_method == IOMethod.TEXT_FILE:
            self.cache["text_input"] = self.cache.get("text_input", open(self.input_path, "r").readlines() if os.path.exists(self.input_path) else [])
        self.input_method_handle = {
            IOMethod.SPEECH: self.handle_stt_input,
            IOMethod.COMMAND_LINE: self.handle_cli_input,
            IOMethod.TEXT_FILE: self.handle_file_input
        }[self.input_method]
        self.cache = {}
        cfg.LOGGER.info("Setup is done.")

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

    def handle_stt_input(self) -> Tuple[Optional[str], Optional[dict]]:
        """
        Acquires input based on STT.
        :return: Transcribed input and list of metadata entries.
        """
        return self.speech_recorder.record_single_input()

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

    def stt_gateway(self) -> Optional[Tuple[str, dict]]:
        """
        Collects and refines STT outputs.
        :return: Refined STT output.
        """
        result = self.input_method_handle()
        cfg.LOGGER.info(f"STT Gateway handles {result[0]}.")
        return result if result[0] else None
    
    def llm_gateway(self) -> Optional[Tuple[str, dict]]:
        """
        Collects and refines LLM output.
        :return: Refined LLM output.
        """
        try:
            llm_output = self.llm_output_queue.get(self.loop_pause)
            cfg.LOGGER.info(f"LLM Gateway handles {llm_output[0]}.")
            return llm_output
        except Empty:
            return None
        
    def tts_gateway(self) -> Optional[Tuple[Any, dict]]:
        """
        Collects and refines TTS output.
        :return: Refined TTS output.
        """
        try:
            tts_output = self.tts_output_queue.get(self.loop_pause)
            cfg.LOGGER.info(f"TTS Gateway handles {tts_output[0]}.")
            return tts_output
        except Empty:
            return None        

    def run_conversation_loop(self) -> None:
        """
        Runs conversation loop.
        """
        cfg.LOGGER.info(f"Starting conversation loop...")
        while not self.interrupt.is_set():
            try:
                cfg.LOGGER.info(f"[1/3] Waiting for user input...")
                stt_output = self.stt_gateway()
                if stt_output is not None:
                    self.llm_input_queue.put(stt_output[0])
                
                cfg.LOGGER.info(f"[2/3] Waiting for LLM output...")
                llm_output = self.llm_gateway()
                if llm_output is not None:
                    self.tts_input_queue.put(llm_output[0])

                cfg.LOGGER.info(f"[3/3] Generating output...")
                tts_output = self.tts_gateway()
                if tts_output is not None:
                    if self.output_method == IOMethod.SPEECH:
                        text_to_speech_utility.play_wave(tts_output[0], tts_output[1])
                    elif self.output_method == IOMethod.COMMAND_LINE:
                        print(f"Assistant: {tts_output[0]}")
                    elif self.output_method == IOMethod.TEXT_FILE:
                        open(self.output_path, "a" if os.path.exists(self.output_path) else "w").write(
                            f"\nAssistant: {tts_output[0]}"
                        )
                time.sleep(self.loop_pause)
            except KeyboardInterrupt:
                cfg.LOGGER.info(f"Recieved keyboard interrupt, shutting down handler ...")
                self.interrupt.set()
        self.llm_interrupt.set()
        self.tts_interrupt.set()

        self.llm_thread.join(1)
        self.tts_thread.join(1)

    def experimental_live_conversation(self) -> None:
        """
        Runs a live conversation.
        """
        cfg.LOGGER.info(f"Starting live conversation...")
        
        speech_input_queue = TQueue()
        speech_recorder_thread = Thread(
            target=self.speech_recorder.record,
            args=(speech_input_queue,)
        )
        speech_recorder_thread.daemon = True
        speech_recorder_thread.start()
        input_accumulator = []
        recieved_valid = False
        recieved_empty = None

        while not self.interrupt.is_set():
            try:
                result = speech_input_queue.get(self.loop_pause)
                if result[0]:
                    recieved_valid = True
                    recieved_empty = None
                    input_accumulator.append(result[0])

                llm_output = self.llm_gateway()
                if llm_output is not None:
                    self.tts_input_queue.put(llm_output[0])

                tts_output = self.tts_gateway()
                if tts_output is not None:
                    if self.output_method == IOMethod.SPEECH:
                        text_to_speech_utility.play_wave(tts_output[0], tts_output[1])
                    elif self.output_method == IOMethod.COMMAND_LINE:
                        print(f"Assistant: {tts_output[0]}")
                    elif self.output_method == IOMethod.TEXT_FILE:
                        open(self.output_path, "a" if os.path.exists(self.output_path) else "w").write(
                            f"\nAssistant: {tts_output[0]}"
                        )
                time.sleep(self.loop_pause)
            except Empty:
                if recieved_valid:
                    if recieved_empty is None:
                        recieved_empty = time.time()
                    elif time.time() - recieved_empty >= 3.0:
                        self.llm_input_queue.put(" ".join(input_accumulator))
                        input_accumulator = []
            except KeyboardInterrupt:
                cfg.LOGGER.info(f"Recieved keyboard interrupt, shutting down handler ...")
                self.interrupt.set()
        self.llm_interrupt.set()
        self.tts_interrupt.set()
        self.interrupt.set()

        self.llm_thread.join(1)
        self.tts_thread.join(1)
            