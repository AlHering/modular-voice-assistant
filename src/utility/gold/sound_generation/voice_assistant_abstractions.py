# -*- coding: utf-8 -*-
"""
****************************************************
*                      Utility                 
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from enum import Enum
from typing import Any, Union, Tuple, List, Optional, Callable
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


class ThreadedTTSHandler(Thread):
    """
    Represents a threaded TTS output handler.
    """
    supported_tts_engines: List[str] = ["coqui-tts"]

    def __init__(self, 
                 input_queue: TQueue,
                 output_queue: TQueue,
                 interrupt: TEvent,
                 loop_pause: float = .1,
                 tts_engine: str = None,
                 tts_model: str = None,
                 tts_instantiation_kwargs: dict = None,
                 tts_synthesis_kwargs: dict = None,
                 *thread_args: Optional[Any], 
                 **thread_kwargs: Optional[Any]) -> None:
        """
        Initiation method.
        :param input_queue: Input queue.
        :param output_queue: Output queue.
        :param interrupt: Interrupt event.
        :param loop_pause: Processing loop pause.
        :param tts_engine: TTS engine.
            See TTSThread.supported_tts_engines for supported engines.
            Defaults to None in which case the first supported engine is used.
        :param tts_model: TTS model name or path.
        :param tts_instantiation_kwargs: TTS model instantiation keyword arguments.
        :param tts_synthesis_kwargs: TTS synthesis keyword arguments.
        :param thread_args: Thread constructor arguments.
        :param thread_kwargs: Thread constructor keyword arguments.
        """
        super().__init__(*thread_args, **thread_kwargs)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.interrupt = interrupt
        self.loop_pause = loop_pause

        self.tts_engine = tts_engine
        self.tts_model = tts_model
        self.tts_instantiation_kwargs = tts_instantiation_kwargs
        self.tts_synthesis_kwargs = tts_synthesis_kwargs

        self.setup_process()

    def setup_process(self) -> None:
        """
        Sets up process.
        """
        self.tts_engine = self.supported_tts_engines[0] if self.tts_engine is None else self.tts_engine
        self.tts_processor = {
            "coqui-tts": text_to_speech_utility.get_coqui_tts_model
        }[self.tts_engine](
            model_name_or_path=self.tts_model,
            instantiation_kwargs=self.tts_instantiation_kwargs
        )
        self.interrupt.clear()

    def unload(self) -> None:
        """
        Stop process and unload resource expensive components.
        Call setup_process() to reload and resume operations.
        """
        self.interrupt.set()
        self.tts_processor = None
        gc.collect()


    def run(self) -> None:
        """
        Main runner method.
        """
        while not self.interrupt.is_set():
            try:
                input_text, input_metadata = self.input_queue.get(self.loop_pause)
                if input_text:
                    # Handle output text
                    result = text_to_speech_utility.synthesize_with_coqui_tts(
                        text=input_text,
                        model=self.tts_processor,
                        synthesis_kwargs=self.tts_synthesis_kwargs
                    )
                    self.output_queue.put(result)
                else:
                    # Handle empty output text
                    pass
            except Empty:
                # Handle empty queue stopping
                pass


class TemporaryThreadedLLMHandler(Thread):
    """
    Represents a threaded LLM handler.
    Used as static testing component for now.
    """

    def __init__(self, 
                 input_queue: TQueue,
                 output_queue: TQueue,
                 interrupt: TEvent,
                 loop_pause: float = .1,
                 *thread_args: Optional[Any], 
                 **thread_kwargs: Optional[Any]) -> None:
        """
        Initiation method.
        :param input_queue: Input queue.
        :param output_queue: Output queue.
        :param interrupt: Interrupt event.
        :param loop_pause: Processing loop pause.
        :param tts_engine: TTS engine.
            See TTSThread.supported_tts_engines for supported engines.
            Defaults to None in which case the first supported engine is used.
        :param tts_model: TTS model name or path.
        :param tts_instantiation_kwargs: TTS model instantiation keyword arguments.
        :param thread_args: Thread constructor arguments.
        :param thread_kwargs: Thread constructor keyword arguments.
        """
        super().__init__(*thread_args, **thread_kwargs)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.interrupt = interrupt
        self.loop_pause = loop_pause

        self.llm = None
        self.setup_process()

    def setup_process(self) -> None:
        """
        Sets up process.
        """
        self.interrupt.clear()

    def unload(self) -> None:
        """
        Stop process and unload resource expensive components.
        Call setup_process() to reload and resume operations.
        """
        self.interrupt.set()
        self.llm = None
        gc.collect()


    def run(self) -> None:
        """
        Main runner method.
        """
        while not self.interrupt.is_set():
            try:
                input_text, input_metadata = self.input_queue.get(self.loop_pause)
                if input_text:
                    # Handle output text
                    self.output_queue.put(tuple(f"Response for '{input_text}'", {"timestamp": get_timestamp()}))
                else:
                    # Handle empty output text
                    pass
            except Empty:
                # Handle empty queue stopping
                pass


class ConversationHandler(object):
    """
    Represents a conversation handler for handling audio based interaction.
    """
    supported_stt_engines: List[str] = ["faster-whisper", "whsiper"]

    def __init__(self, 
                 working_directory: str,
                 stt_engine: str = None,
                 stt_model: str = None,
                 stt_instantiation_kwargs: dict = None,
                 tts_engine: str = None,
                 tts_model: str = None,
                 tts_instantiation_kwargs: dict = None,
                 history: List[dict] = None,
                 input_method: InputMethod = InputMethod.SPEECH_TO_TEXT,
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
        :param input_method: Input method out of SPEECH_TO_TEXT, COMMAND_LINE, TEXT_FILE.
            Defaults to SPEECH_TO_TEXT.
        :param loop_pause: Pause in seconds between processing loops.
            Defaults to 0.1.
        """
        cfg.LOGGER.info("Initiating Conversation Handler...")
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
        self.set_stt_processor(
            stt_engine=self.stt_engine,
            stt_model=self.stt_model,
            stt_instantiation_kwargs=self.stt_instantiation_kwargs
        )

        self.llm_input_queue = TQueue()
        self.llm_output_queue = TQueue()
        self.llm_interrupt = TEvent()
        self.llm_thread = TemporaryThreadedLLMHandler(
            input_queue=self.llm_input_queue,
            output_queue=self.llm_output_queue,
            interrupt=self.llm_interrupt
        )
        self.llm_thread.daemon = True
        self.llm_thread.start()

        self.tts_input_queue = TQueue()
        self.tts_output_queue = TQueue()
        self.tts_interrupt = TEvent()
        self.tts_thread = ThreadedTTSHandler(
            input_queue=self.tts_input_queue,
            output_queue=self.tts_output_queue,
            interrupt=self.tts_interrupt,
            tts_engine=self.tts_engine,
            tts_model=self.tts_model,
            tts_instantiation_kwargs=self.tts_instantiation_kwargs
        )
        self.tts_thread.daemon = True
        self.tts_thread.start()

        self.history = [] if self.history is None or delete_history else self.history
        if self.input_method == InputMethod.TEXT_FILE:
            input_file = os.path.join(self.working_directory, "input.txt")
            self.cache["text_input"] = self.cache.get("text_input", open(input_file, "r").readlines() if os.path.exists(input_file) else [])
        self.input_method_handle = {
            InputMethod.SPEECH_TO_TEXT: self.handle_stt_input,
            InputMethod.COMMAND_LINE: self.handle_cli_input,
            InputMethod.TEXT_FILE: self.handle_file_input
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

    def stt_gateway(self) -> Optional[Tuple[str, dict]]:
        """
        Collects and refines STT outputs.
        :return: Refined STT output.
        """
        result = self.input_method_handle()
        cfg.LOGGER.info(f"STT Gateway handles {result}.")
        return result if result[0] else None
    
    def llm_gateway(self) -> Optional[Tuple[str, dict]]:
        """
        Collects and refines LLM output.
        :return: Refined LLM output.
        """
        try:
            llm_output = self.llm_output_queue.get(self.loop_pause)
            cfg.LOGGER.info(f"LLM Gateway handles {llm_output}.")
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
            cfg.LOGGER.info(f"TTS Gateway handles {tts_output}.")
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
                stt_output = self.stt_gateway()
                if stt_output is not None:
                    self.llm_input_queue.put(stt_output)
                
                llm_output = self.llm_gateway()
                if llm_output is not None:
                    self.tts_input_queue.put(llm_output)
                
                tts_output = self.tts_gateway()
                if tts_output is not None:
                    text_to_speech_utility.play_wave(
                        tts_output[0], tts_output[1]
                    )
                time.sleep(self.loop_pause)
            except KeyboardInterrupt:
                cfg.LOGGER.info(f"Recieved keyboard interrupt, shutting down handler ...")
                self.llm_interrupt.set()
                self.tts_interrupt.set()
                self.interrupt.set()

                self.llm_thread.unload()
                self.tts_thread.unload()
                self.llm_thread.join(1)
                self.tts_thread.join(1)

        
            