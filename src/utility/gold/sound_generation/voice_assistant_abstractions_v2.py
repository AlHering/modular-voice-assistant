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
    A conversation handler manages the following components:
        - speech_recorder: A recorder for spoken input.
        - transcriber: A transcriber to transcribe spoken input into text.
        - worker: A worker to compute an output for the given input.
        - synthesizer: A synthesizer to convert output texts to sound.
    Depending on the input and output methods, not all components are used.
    """

    def __init__(self, 
                 working_directory: str,
                 speech_recorder: SpeechRecorder = None,
                 transcriber: Transcriber = None,
                 synthesizer: Synthesizer = None,
                 worker_function: Callable = None,
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
        :param worker_function: Worker function for handling cleaned input.
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
        self.worker_function = worker_function
        self.synthesizer = synthesizer
        self.component_functions = {
            "transcriber": None if self.transcriber is None else self.transcriber.transcribe,
            "worker": None if self.worker_function is None else self.worker_function,
            "synthesizer": None if self.synthesizer is None else self.synthesizer.synthesize,
        }

        self.interrupt = None
        self.interrupts = {}
        self.queues = {}
        self.threads = {}

        self.history = history
        self.loop_pause = loop_pause

        self.input_method = input_method
        self.output_method = output_method

        self.cache = None
        self._reset()

    def _stop(self) -> None:
        """
        Method for stopping processes.
        """
        if self.interrupt is not None:
            self.interrupt.set()
        for component in self.interrupts:
            self.interrupts[component].set()
        for component in self.threads:
            self.threads[component].join(1) 

    def _setup_components(self) -> None:
        """
        Method for setting up components.
        """  
        needed_components = ["transcriber", "worker", "synthesizer"]
        needed_queues = ["transcriber_in",
                         "transcriber_out",
                         "worker_in",
                         "worker_out",
                         "synthesizer_in",
                         "synthesizer_out"]
        if self.input_method != IOMethod.SPEECH:
            needed_components.remove("transcriber")
        if self.output_method != IOMethod.SPEECH:
            needed_components.remove("synthesizer")

        self.interrupts = {component: TEvent() for component in needed_components}
        self.queues = {component: TQueue() for component in needed_queues}

        for component in needed_components:
            self.threads[component] = PipelineComponentThread(
                pipeline_function=self.component_functions[component],
                input_queue=self.queues.get(f"{component}_in"),
                output_queue=self.queues.get(f"{component}_out"),
                interrupt=self.interrupt["component"],
                loop_pause=self.loop_pause
            )
            self.threads[component].daemon = True

        for component in self.threads:
            self.threads[component].start()


    def _reset(self, delete_history: bool = False) -> None:
        """
        Method for setting up and resetting handler. 
        :param delete_history: Flag for declaring whether to delete history.    
            Defaults to None.
        """
        cfg.LOGGER.info("(Re)setting Conversation Handler...")
        self._stop()
        self.interrupt = None
        self.interrupts = None
        self.queues = None
        self.threads = None
        gc.collect()

        self._setup_components()
        self.history = [] if self.history is None or delete_history else self.history
        self.cache = {
            "input": [line.replace("\n", "") 
                      for line in open(self.input_path, "r").readlines()] 
                      if os.path.exists(self.input_path) else []} if self.input_method == IOMethod.TEXT_FILE else {}
        cfg.LOGGER.info("Setup is done.")

    def handle_input(self) -> None:
        """
        Acquires and reroutes input based on configured input method.
        """
        if self.input_method == IOMethod.SPEECH:
            result = self.speech_recorder.record_single_input()
            if result[0]:
                self.queues["transcriber_in"].put(result)
        elif self.input_method == IOMethod.COMMAND_LINE:
            result = input("User > ")
            self.queues["transcriber_out"].put((result, {"timestamp": get_timestamp()}))
        elif self.input_method == IOMethod.TEXT_FILE:
            if self.cache["input"]:
                result = self.cache["input"][0]
                self.cache["input"] = self.cache["input"][1:]
                self.queues["transcriber_out"].put((result, {"timestamp": get_timestamp()}))
            
        return self.speech_recorder.record_single_input()
    
    def input_to_worker_gateway(self) -> None:
        """
        Collects and refines input before passing on to LLM.
        """
        try:
            raw_input, raw_input_metadata = self.queues["transcriber_out"].get(self.loop_pause)
            # TODO: Refine input
            self.queues["worker_in"].put((raw_input, raw_input_metadata))
        except Empty:
            pass
        
    def worker_to_output_gateway(self) -> None:
        """
        Collects and refines LLM output.
        """
        try:
            raw_output, raw_output_metadata = self.queues["worker_out"].get(self.loop_pause)
            # TODO: Refine output
            if self.output_method == IOMethod.SPEECH:
                self.queues["synthesizer_in"].put((raw_output, raw_output_metadata))
            else:
                self.queues["synthesizer_out"].put((raw_output, raw_output_metadata))
        except Empty:
            return None    

    def handle_output(self) -> None:
        """
        Acquires and reroutes generated output based on configured output method.
        """  
        output, output_metadata = self.queues["synthesizer_out"].get(self.loop_pause)
        if output:
            if self.output_method == IOMethod.SPEECH:
                text_to_speech_utility.play_wave(output, output_metadata)
            elif self.output_method == IOMethod.COMMAND_LINE:
                print(f"Assistant: {output}")
            elif self.output_method == IOMethod.TEXT_FILE:
                open(self.output_path, "a" if os.path.exists(self.output_path) else "w").write(
                    f"\nAssistant: {output}"
                )  

    def run_conversation_loop(self) -> None:
        """
        Runs conversation loop.
        """
        cfg.LOGGER.info(f"Starting conversation loop...")
        while not self.interrupt.is_set():
            try:
                cfg.LOGGER.info(f"[1/4] Handling input...")
                self.handle_input()
                cfg.LOGGER.info(f"[2/4] Preparing worker input...")
                self.input_to_worker_gateway()
                cfg.LOGGER.info(f"[3/4] Preparing worker output...")
                self.worker_to_output_gateway()
                cfg.LOGGER.info(f"[4/4] Handling output...")
                self.handle_output()
               
                time.sleep(self.loop_pause)
            except KeyboardInterrupt:
                cfg.LOGGER.info(f"Recieved keyboard interrupt, shutting down handler ...")
                self._stop()
        
