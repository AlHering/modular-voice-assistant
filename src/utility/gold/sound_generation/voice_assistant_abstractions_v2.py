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
from threading import Thread, Event as TEvent, Lock
from queue import Empty, Queue as TQueue
from ..text_generation.language_model_abstractions import LanguageModelInstance
from ..text_generation.agent_abstractions import Agent
from ...bronze.concurrency_utility import PipelineComponentThread, timeout
from ...bronze.time_utility import get_timestamp
from . import speech_to_text_utility, text_to_speech_utility
from .sound_model_abstractions import Transcriber, Synthesizer, SpeechRecorder


class ConversationHandler(object):
    """
    Represents a conversation handler for handling audio based interaction.
    A conversation handler manages the following components:
        - speech_recorder: A recorder for spoken input.
        - transcriber: A transcriber to transcribe spoken input into text.
        - worker: A worker to compute an output for the given input.
        - synthesizer: A synthesizer to convert output texts to sound.
    """

    def __init__(self, 
                 working_directory: str,
                 speech_recorder: SpeechRecorder,
                 transcriber: Transcriber,
                 synthesizer: Synthesizer,
                 worker_function: Callable,
                 history: List[dict] = None,
                 loop_pause: float = 0.1) -> None:
        """
        Initiation method.
        :param working_directory: Directory for productive files.
        :param speech_recorder: Speech recorder for STT processes.
        :param transcriber: Transcriber for STT processes.
        :param synthesizer: Synthesizer for TTS processes.
        :param worker_function: Worker function for handling cleaned input.
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
        self.input_path = os.path.join(self.working_directory, "input.wav") 
        self.output_path = os.path.join(self.working_directory, "output.wav") 

        pya = pyaudio.PyAudio()
        self.input_device_index = pya.get_default_input_device_info().get("index")
        self.output_device_index = pya.get_default_output_device_info().get("index")
        pya.terminate()

        self.speech_recorder = speech_recorder
        self.transcriber = transcriber
        self.worker_function = worker_function
        self.synthesizer = synthesizer

        self.interrupt: TEvent = None
        self.pause_input: TEvent = None
        self.pause_worker: TEvent = None
        self.pause_output: TEvent = None
        self.queues: Dict[str, TQueue] = {}
        self.threads: Dict[str, PipelineComponentThread] = {}

        self.history = history
        self.loop_pause = loop_pause
        self._setup_components()

    def _stop(self) -> None:
        """
        Stops processes.
        """
        for event in [
            self.interrupt,
            self.pause_input,
            self.worker_function,
            self.pause_output
        ]:
            if event is not None:
                event.set()
        for component in self.threads:
            self.threads[component].join(self.loop_pause) 

    def _setup_components(self) -> None:
        """
        Sets up components.
        """
        self.interrupt = TEvent()
        self.pause_input = TEvent()
        self.worker_function = TEvent()
        self.pause_output = TEvent()

        self.queues = {f"worker_in": TQueue(), "worker_out": TQueue() }

        self.threads = {
            "input": Thread(target=self.handle_step_in_loop, args=("input", )),
            "worker": Thread(target=self.handle_step_in_loop, args=("worker", )),
            "output": Thread(target=self.handle_step_in_loop, args=("output", ))
        }
        for thread in self.threads:
            self.threads[thread].daemon = True

    def reset(self, delete_history: bool = False) -> None:
        """
        Sets up and resets handler. 
        :param delete_history: Flag for declaring whether to delete history.    
            Defaults to None.
        """
        cfg.LOGGER.info("(Re)setting Conversation Handler...")
        self._stop()
        gc.collect()
        self._setup_components()
        self.history = [] if self.history is None or delete_history else self.history
        cfg.LOGGER.info("Setup is done.")

    def handle_input(self) -> None:
        """
        Acquires and forwards user input.
        """
        if not self.pause_input.is_set():
            recorder_data, recorder_metadata = self.speech_recorder.record_single_input()
            cfg.LOGGER.info(f"Got voice input.")
            cfg.LOGGER.info(f"Transcribing input...")
            input_data, transcriber_metadata = self.transcriber.transcribe(recorder_data)
            cfg.LOGGER.info(f"Forwarding {input_data} to worker...")
            self.queues["worker_in"].put(input_data)

    def handle_work(self, timeout: float = None) -> None:
        """
        Acquires input, computes and reroutes worker output.
        :param timeout: Timeout for blocking methods.
        """  
        self.queues["worker_out"].put(self.worker_function(self.queues["worker_in"].get(block=True, timeout=timeout)))

    def handle_output(self, timeout: float = None) -> None:
        """
        Acquires and reroutes generated output based.
        :param timeout: Timeout for blocking methods.
        """  
        if not self.pause_output.is_set():
            try:
                worker_output, worker_metadata = self.queues["worker_out"].get(block=True, timeout=timeout)
                cfg.LOGGER.info(f"Fetched worker output {worker_output}.")
                cfg.LOGGER.info(f"Synthesizing output...")
                synthesizer_output, synthesizer_metadata = self.synthesizer.synthesize(worker_output)
                cfg.LOGGER.info(f"Ouputting synthesized response...")
                self.pause_output.set()
                text_to_speech_utility.play_wave(synthesizer_output, synthesizer_metadata)
                self.pause_output.clear()
            except Empty:
                pass

    def handle_step_in_loop(self, step: str) -> None:
        """
        Runs work handling in loop.
        :param step: Pipeline step to handle in loop.
        """
        target_function = {
            "input": self.handle_input,
            "work": self.handle_work,
            "output": self.handle_output
        }[step]
        kwargs = {} if step == "input" else {"timeout": self.loop_pause}
        while not self.interrupt.is_set():
            target_function(**kwargs)

    def pipeline_is_busy(self) -> bool:
        """
        Returns pipeline status:
        :return: True, if pipeline is busy, else False.
        """
        return (any(self.queues[queue].qsize() > 0 for queue in self.queues) or
                any(self.threads[thread].is_alive() for thread in self.threads))

    def run_conversation_loop(self, blocking: bool = True) -> None:
        """
        Runs conversation loop.
        :param blocking: Flag which declares whether or not to wait for each step.
            Defaults to True.
        """
        cfg.LOGGER.info(f"Starting conversation loop...")
        self.queues["worker_out"].put(("Hello there, how may I help you today?", {}))
        self.handle_output()
        try:
            if not blocking:
                self.threads["input"].start()
                self.threads["worker"].start()
                self.threads["output"].start()
            else:
                self.handle_input()
                self.handle_work()
                self.handle_output()
        except KeyboardInterrupt:
            cfg.LOGGER.info(f"Recieved keyboard interrupt, shutting down handler ...")
            self._stop()
        
