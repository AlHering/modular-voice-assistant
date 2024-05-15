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
from ..text_generation.agent_abstractions import Agent
from ...bronze.concurrency_utility import PipelineComponentThread
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
        self.component_functions: Dict[str, Callable] = {
            "transcriber": self.transcriber.transcribe,
            "worker": self.worker_function,
            "synthesizer": self.synthesizer.synthesize,
        }

        self.interrupt: TEvent = None
        self.pause_inputting: TEvent = None
        self.pause_forwarding: TEvent = None
        self.pause_outputting: TEvent = None
        self.queues: Dict[str, TQueue] = {}
        self.threads: Dict[str, PipelineComponentThread] = {}

        self.history = history
        self.loop_pause = loop_pause
        self._reset()

    def _stop(self) -> None:
        """
        Method for stopping processes.
        """
        for event in [
            self.interrupt,
            self.pause_inputting,
            self.pause_forwarding,
            self.pause_outputting
        ]:
            if event is not None:
                event.set()
        for component in self.threads:
            self.threads[component].join(self.loop_pause) 

    def _setup_components(self) -> None:
        """
        Method for setting up components.
        """
        self.interrupt = TEvent()
        self.pause_inputting = TEvent()
        self.pause_forwarding = TEvent()
        self.pause_outputting = TEvent()

        self.queues = {f"{component}_in": TQueue() for component in self.component_functions}
        self.queues.update({f"{component}_out": TQueue() for component in self.component_functions})

        self.threads = {}
        for component in self.component_functions:
            self.threads[component] = PipelineComponentThread(
                pipeline_function=self.component_functions[component],
                input_queue=self.queues.get(f"{component}_in"),
                output_queue=self.queues.get(f"{component}_out"),
                interrupt=self.interrupt,
                loop_pause=self.loop_pause/8
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
        gc.collect()

        self._setup_components()
        self.history = [] if self.history is None or delete_history else self.history
        cfg.LOGGER.info("Setup is done.")

    def handle_input(self) -> Tuple[Optional[np.ndarray], Optional[dict]]:
        """
        Acquires and forwards user input.
        :return: User input audio data and metadata.
        """
        if not self.pause_inputting.is_set():
            return self.speech_recorder.record_single_input()
        return None, None
    
    def input_to_worker_gateway(self) -> bool:
        """
        Collects and refines input before passing on to LLM.
        :return: True, if data was successfully forwarded, else False.
        """
        if not self.pause_forwarding.is_set():
            try:
                cfg.LOGGER.info(f"Trying to fetch 'transcriber_out' data ...")
                raw_input, raw_input_metadata = self.queues["transcriber_out"].get(self.loop_pause/4)
                # Refine input
                cfg.LOGGER.info(f"Forwarding {raw_input} to worker...")
                self.queues["worker_in"].put(raw_input)
                return True
            except Empty:
                cfg.LOGGER.info(f"'transcriber_out'-queue is empty.")
        return False
        
    def worker_to_output_gateway(self) -> bool:
        """
        Collects and refines LLM output.
        :return: True, if data was successfully forwarded, else False.
        """
        if not self.pause_forwarding.is_set():
            try:
                cfg.LOGGER.info(f"Trying to fetch 'worker_out' data ...")
                response_text, response_metadata = self.queues["worker_out"].get(self.loop_pause/4)
                if response_text:
                    cfg.LOGGER.info(f"Forwarding {response_text} to synthesizer...")
                    self.queues["synthesizer_in"].put(response_text)
                return True
            except Empty:
                cfg.LOGGER.info(f"'worker_out'-queue is empty.")
        return False

    def handle_output(self) -> bool:
        """
        Acquires and reroutes generated output based.
        :return: True, if data was successfully outputted, else False.
        """  
        if not self.pause_outputting.is_set():
            try:
                cfg.LOGGER.info(f"Trying to fetch 'synthesizer_out' data ...")
                output, output_metadata = self.queues["synthesizer_out"].get(self.loop_pause/4)
                if output.any():
                    cfg.LOGGER.info(f"Ouputting synthesized response...")
                    self.pause_outputting.set()
                    text_to_speech_utility.play_wave(output, output_metadata)
                    self.pause_outputting.clear()
                    cfg.LOGGER.info(f"Ouput finished.")
                return True
            except Empty:
                cfg.LOGGER.info(f"'synthesizer_out'-queue is empty.")
        return False
        
    def _forward_queued_elements(self) -> None:
        """
        Method for forwarding queued elements.
        """
        self.input_to_worker_gateway()
        self.worker_to_output_gateway()
        self.handle_output()
        
    def _run_blocking_conversation_steps(self) -> None:
        """
        Runs conversation steps for step.
        """
        cfg.LOGGER.info(f"[1/4] Handling input...")
        audio_data, audio_metadata = self.handle_input()
        if audio_data is not None and audio_data.any():
            cfg.LOGGER.info(f"Forwarding audio data to transcriber...")
            self.queues["transcriber_in"].put(audio_data)
            cfg.LOGGER.info(f"[2/4] Preparing worker input...")
            while not self.input_to_worker_gateway():
                time.sleep(self.loop_pause/4)
            cfg.LOGGER.info(f"[3/4] Preparing worker output...")
            while not self.worker_to_output_gateway():
                time.sleep(self.loop_pause/4)
            cfg.LOGGER.info(f"[4/4] Handling output...")
            while not self.handle_output():
                time.sleep(self.loop_pause/4)

    def _busy_threads(self) -> bool:
        """
        Returns flag which declares whether there is currently a process running.
        """
        return any(self.threads[thread].busy.is_set() for thread in self.threads if 
                   thread != "_forward_queued_elements" and
                   thread != "print_component_status")

    def print_component_status(self) -> None:
        """
        Prints out component status.
        """
        print("="*20 + "\nCOMPONENT STATUS START\n" + "="*18)
        print("THREADS")
        for thread in self.threads:
            print(f"{thread}: {'busy' if self.threads[thread].busy.is_set() else 'waiting'}")
        print("QUEUES")
        for queue in self.queues:
            print(f"{queue}: {self.queues[queue].qsize()} elements waiting")
        print("FLAGS")
        print(f"interrupt: {self.interrupt.is_set()}")
        print(f"pause_inputting: {self.pause_inputting.is_set()}")
        print(f"pause_forwarding: {self.pause_forwarding.is_set()}")
        print(f"pause_outputting: {self.pause_outputting.is_set()}")
        print("="*20 + "\nCOMPONENT STATUS END\n" + "="*20)

    def _run_non_blocking_conversation_steps(self) -> None:
        """
        Runs conversation in a non-blocking manner.
        """
        if "_forward_queued_elements" not in self.threads:
            cfg.LOGGER.info(f"Setting up forwarder thread...")
            self.threads["_forward_queued_elements"] = PipelineComponentThread(
                pipeline_function=self._forward_queued_elements,
                interrupt=self.interrupt,
                loop_pause=self.loop_pause
            )
            self.threads["_forward_queued_elements"].daemon = True

            self.threads["_forward_queued_elements"].start()
            cfg.LOGGER.info(f"Setting up forwarder thread...")
            self.threads["print_component_status"] = PipelineComponentThread(
                pipeline_function=self.print_component_status,
                interrupt=self.interrupt,
                loop_pause=3
            )
            self.threads["print_component_status"].daemon = True
            self.threads["print_component_status"].start()
        
        cfg.LOGGER.info(f"[1/4] Handling input...")
        audio_data, audio_metadata = self.handle_input()
        if audio_data is not None and audio_data.any():
            cfg.LOGGER.info(f"Clearing out artefacts from last input...")
            self.pause_inputting.set()
            self.pause_forwarding.set()
            for queue in ["transcriber_in", "worker_in", "synthesizer_in"]:
                with self.queues[queue].mutex:
                    self.queues[queue].queue.clear()
            while self._busy_threads():
                print("busy threads")
                time.sleep(self.loop_pause/16)
            for queue in ["transcriber_out", "worker_out", "synthesizer_out"]:
                with self.queues[queue].mutex:
                    self.queues[queue].queue.clear()
            self.pause_forwarding.clear()
            self.pause_inputting.clear()
            cfg.LOGGER.info(f"Finished clearing...")
            cfg.LOGGER.info(f"Forwarding audio data to transcriber...")
            self.queues["transcriber_in"].put(audio_data)

    def run_conversation_loop(self, blocking: bool = True) -> None:
        """
        Runs conversation loop.
        :param blocking: Flag which declares whether or not to wait for each step.
            Defaults to True.
        """
        cfg.LOGGER.info(f"Starting conversation loop...")
        self.queues["synthesizer_out"].put(self.synthesizer.synthesize(
            "Hello there, how may I help you today?"
        ))
        while not self.handle_output():
            time.sleep(self.loop_pause/4)
        while not self.interrupt.is_set():
            try:
                cfg.LOGGER.info(f"Looping main loop...")
                if blocking:
                    self._run_blocking_conversation_steps()
                else:
                   self._run_non_blocking_conversation_steps()
               
                time.sleep(self.loop_pause)
            except KeyboardInterrupt:
                cfg.LOGGER.info(f"Recieved keyboard interrupt, shutting down handler ...")
                self._stop()
        
