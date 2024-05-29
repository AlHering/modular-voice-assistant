# -*- coding: utf-8 -*-
"""
****************************************************
*                      Utility                 
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from abc import ABC, abstractmethod
from enum import Enum
from inspect import getfullargspec
from typing import Any, Union, Tuple, List, Optional, Callable, Dict
import os
import gc
import time
import numpy as np
import pyaudio
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding.key_bindings import KeyBindings
from rich import print as rich_print
from rich.style import Style as RichStyle
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.styles import Style as PTStyle
import speech_recognition
from datetime import datetime as dt
from src.configuration import configuration as cfg
from threading import Thread, Event as TEvent, Lock
from queue import Empty, Queue as TQueue
from ..text_generation.language_model_abstractions import LanguageModelInstance, ChatModelInstance
from ..text_generation.agent_abstractions import ToolArgument, AgentTool
from ..text_generation.agent_abstractions import Agent
from ...bronze.concurrency_utility import PipelineComponentThread, timeout
from ...bronze.time_utility import get_timestamp
from ...bronze.json_utility import load as load_json
from ...silver.file_system_utility import safely_create_path
from ...bronze.pyaudio_utility import play_wave
from . import speech_to_text_utility, text_to_speech_utility
from .sound_model_abstractions import Transcriber, Synthesizer, SpeechRecorder


def setup_prompt_session(bindings: KeyBindings = None) -> PromptSession:
    """
    Function for setting up prompt session.
    :param bindings: Key bindings.
        Defaults to None.
    :return: Prompt session.
    """
    return PromptSession(
        bottom_toolbar=[
        ("class:bottom-toolbar",
         "ctl-c to exit, ctl-d to save cache and exit",)
    ],
        style=PTStyle.from_dict({
        "bottom-toolbar": "#333333 bg:#ffcc00"
    }),
        auto_suggest=AutoSuggestFromHistory(),
        key_bindings=bindings
    )


class ConversationHandler(object):
    """
    Represents a conversation handler for handling audio based interaction.
    A conversation handler manages the following components:
        - speech_recorder: A recorder for spoken input.
        - transcriber: A transcriber to transcribe spoken input into text.
        - worker: A worker to compute an output for the given user input.
        - synthesizer: A synthesizer to convert output texts to sound.
    """

    def __init__(self, 
                 working_directory: str,
                 speech_recorder: SpeechRecorder,
                 transcriber: Transcriber,
                 synthesizer: Synthesizer,
                 worker_function: Callable = None,
                 loop_pause: float = 0.1) -> None:
        """
        Initiation method.
        :param working_directory: Directory for productive files.
        :param speech_recorder: Speech recorder for STT processes.
        :param transcriber: Transcriber for STT processes.
        :param synthesizer: Synthesizer for TTS processes.
        :param worker_function: Worker function for handling cleaned input.
            Default to None.
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

        if worker_function is None:
            def worker_function(worker_input: Any) -> Any:
                """
                Dummy worker function, logging and forwarding input.
                :param worker_input: Worker input.
                :return: Worker input.
                """
                cfg.LOGGER.info(f"Worker recieved: {worker_input}")
                return worker_input

        self.speech_recorder = speech_recorder
        self.transcriber = transcriber
        self.worker_function = worker_function
        self.synthesizer = synthesizer

        self.interrupt: TEvent = None
        self.pause_input: TEvent = None
        self.pause_worker: TEvent = None
        self.pause_output: TEvent = None
        self.queues: Dict[str, TQueue] = {}
        self.threads: Dict[str, Thread] = {}

        self.loop_pause = loop_pause
        self._setup_components()

    def stop(self) -> None:
        """
        Stops processes.
        """
        for event in [
            self.interrupt,
            self.pause_input,
            self.pause_worker,
            self.pause_output
        ]:
            if event is not None:
                event.set()
        for component in self.threads:
            try:
                self.threads[component].join(self.loop_pause) 
            except RuntimeError:
                pass

    def _setup_components(self) -> None:
        """
        Sets up components.
        """
        self.interrupt = TEvent()
        self.pause_input = TEvent()
        self.pause_worker = TEvent()
        self.pause_output = TEvent()

        self.queues = {f"worker_in": TQueue(), "worker_out": TQueue() }

        self.threads = {
            "input": Thread(target=self.loop, kwargs={"task": "input"}),
            "worker": Thread(target=self.loop, kwargs={"task": "worker", "parameters": {"timeout": self.loop_pause/2}}),
            "output": Thread(target=self.loop, kwargs={"task": "output", "parameters": {"timeout": self.loop_pause/2}})
        }
        for thread in self.threads:
            self.threads[thread].daemon = True

    def reset(self) -> None:
        """
        Sets up and resets handler. 
        """
        cfg.LOGGER.info("(Re)setting Conversation Handler...")
        self.stop()
        gc.collect()
        self._setup_components()
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

    def handle_work(self, timeout: float = None, stream: bool = False) -> None:
        """
        Acquires input, computes and reroutes worker output.
        :param timeout: Timeout for blocking methods.
        :param stream: Declares, whether worker function return should be handled as a generator.
        """  
        if not self.pause_worker.is_set():
            try:
                if stream:
                    for elem in self.worker_function(self.queues["worker_in"].get(block=True, timeout=timeout)):
                        self.queues["worker_out"].put(elem)
                else:
                    self.queues["worker_out"].put(self.worker_function(self.queues["worker_in"].get(block=True, timeout=timeout)))
            except Empty:
                pass

    def handle_output(self, timeout: float = None) -> None:
        """
        Acquires and reroutes generated output based.
        :param timeout: Timeout for blocking methods.
        """  
        if not self.pause_output.is_set():
            try:
                worker_output, worker_metadata = self.queues["worker_out"].get(block=True, timeout=timeout)
                self.pause_output.set()
                cfg.LOGGER.info(f"Fetched worker output {worker_output}.")
                cfg.LOGGER.info(f"Synthesizing output...")
                synthesizer_output, synthesizer_metadata = self.synthesizer.synthesize(worker_output)
                cfg.LOGGER.info(f"Ouputting synthesized response...")
                self.pause_output.set()
                play_wave(synthesizer_output, synthesizer_metadata)
                self.pause_output.clear()
            except Empty:
                pass

    def loop(self, task: str = "output", parameters: dict = None) -> None:
        """
        Method for looping task.
        :param task: Task out of "input", "worker" and "output".
        :param parameters: Parameters for the method call.
            Defaults to None.
        """
        parameters = {} if parameters is None else parameters
        method = {
            "input": self.handle_input,
            "worker": self.handle_work,
            "output": self.handle_output
        }[task]
        while not self.interrupt.is_set():
            method(**parameters)

    def queues_are_busy(self) -> bool:
        """
        Returns queue status:
        :return: True, if any queue contains elements, else False.
        """
        return any(self.queues[queue].qsize() > 0 for queue in self.queues)

    def pipeline_is_busy(self) -> bool:
        """
        Returns pipeline status:
        :return: True, if pipeline is busy, else False.
        """
        return (self.queues_are_busy or
                any(self.threads[thread].is_alive() for thread in self.threads))

    def _run_nonblocking_conversation(self, loop: bool, stream: bool = False) -> None:
        """
        Internal method for running non-blocking conversation.
        :param loop: Delcares, whether to loop conversation or stop after a single interaction.
            Defaults to True.
        :param stream: Declares, whether worker function streams response.
            Defaults to False.
        """
        if stream:
            self.threads["worker"]._kwargs["parameters"]["stream"] = True
        self.threads["input"].start()
        self.threads["worker"].start()
        self.threads["output"].start()
        if not loop:
            while self.queues["worker_in"].qsize() == 0:
                time.sleep(self.loop_pause/16)
            self.pause_input.set()
            while self.queues["worker_out"].qsize() == 0:
                time.sleep(self.loop_pause/16)
            while self.queues["worker_out"].qsize() > 0 or self.pause_output.is_set():
                time.sleep(self.loop_pause/16)
            self.reset()
    
    def _run_blocking_conversation(self, loop: bool, stream: bool = False) -> None:
        """
        Internal method for running blocking conversation.
        :param loop: Delcares, whether to loop conversation or stop after a single interaction.
            Defaults to True.
        :param stream: Declares, whether worker function streams response.
            Defaults to False.
        """
        while not self.interrupt.is_set():
            self.handle_input()
            self.handle_work(stream=stream)
            self.handle_output()
            if not loop:
                self.interrupt.set()
        self.reset()

    def run_conversation(self, blocking: bool = True, loop: bool = True, stream: bool = False) -> None:
        """
        Runs conversation.
        :param blocking: Declares, whether or not to wait for each step.
            Defaults to True.
        :param loop: Delcares, whether to loop conversation or stop after a single interaction.
            Defaults to True.
        :param stream: Declares, whether worker function streams response.
            Defaults to False.
        """
        cfg.LOGGER.info(f"Starting conversation loop...")
        self.queues["worker_out"].put(("Hello there, how may I help you today?", {}))
        self.handle_output()
        try:
            if not blocking:
                self._run_nonblocking_conversation(loop=loop, stream=stream)
            else:
                self._run_blocking_conversation(loop=loop, stream=stream)
        except KeyboardInterrupt:
            cfg.LOGGER.info(f"Recieved keyboard interrupt, shutting down handler ...")
            self.stop()

class ConversationHandlerSession(object):
    """
    Represents a conversation handler session.
    """
    transcriber_supported_backends: List[str] = Transcriber.supported_backends
    synthesizer_supported_backends: List[str] = Synthesizer.supported_backends

    def __init__(self,
        working_directory: str = None,
        loop_pause: float = .1,
        transcriber_backend: str = None,
        transcriber_model_path: str = None,
        transcriber_model_parameters: dict = None,
        transcriber_transcription_parameters: dict = None,
        synthesizer_backend: str = None,
        synthesizer_model_path: str = None,
        synthesizer_model_parameters: dict = None,
        synthesizer_synthesis_parameters: dict = None,
        speechrecorder_input_device_index: int = None,
        speechrecorder_recognizer_parameters: dict = None,
        speechrecorder_microphone_parameters: dict = None,
        speechrecorder_loop_pause: float = .1 
    ) -> None:
        """
        Initiation method.
        :param working_directory: Working directory.
        :param loop_pause: Conversation handler loop pause.
        :param transcriber_backend: Transcriber backend.
        :param transcriber_model_path: Transcriber model path.
        :param transcriber_model_parameters: Transcriber model parameters.
        :param transcriber_transcription_parameters: Transcription parameters.
        :param synthesizer_backend: Synthesizer backend.
        :param synthesizer_model_path: Synthesizer model path.
        :param synthesizer_model_parameters: Synthesizer model parameters.
        :param synthesizer_synthesis_parameters: Synthesizer synthesis parameters.
        :param speechrecorder_input_device_index: Speech Recorder input device index.
        :param speechrecorder_recognizer_parameters: Speech Recorder recognizer parameters.
        :param speechrecorder_microphone_parameters: Speech Recorder microphone parameters.
        :param speechrecorder_loop_pause: Speech Recorder loop pause.
        """
        self.working_directory = working_directory
        self.loop_pause = loop_pause
        self.transcriber_backend = transcriber_backend
        self.transcriber_model_path = transcriber_model_path
        self.transcriber_model_parameters = transcriber_model_parameters
        self.transcriber_transcription_parameters = transcriber_transcription_parameters
        self.synthesizer_backend = synthesizer_backend
        self.synthesizer_model_path = synthesizer_model_path
        self.synthesizer_model_parameters = synthesizer_model_parameters
        self.synthesizer_synthesis_parameters = synthesizer_synthesis_parameters
        self.speechrecorder_input_device_index = speechrecorder_input_device_index
        self.speechrecorder_recognizer_parameters = speechrecorder_recognizer_parameters
        self.speechrecorder_microphone_parameters = speechrecorder_microphone_parameters
        self.speechrecorder_loop_pause = speechrecorder_loop_pause

    @classmethod
    def from_dict(cls, parameters: dict) -> Any:
        """
        Returns session instance from dict.
        :param parameters: Session parameters dictionary.
        :returns: VoiceAssistantSession instance.
        """
        return cls(**parameters)
    
    @classmethod
    def from_file(cls, file_path: str) -> Any:
        """
        Returns session instance from a json file.
        :param file_path: Path to json file.
        :returns: VoiceAssistantSession instance.
        """
        return cls(**load_json(file_path))
    
    def to_dict(self) -> dict:
        """
        Returns a parameter dictionary for later instantiation.
        :returns: Parameter dictionary.
        """
        return {
            "working_directory": self.working_directory,
            "loop_pause": self.loop_pause,
            "transcriber_backend": self.transcriber_backend,
            "transcriber_model_path": self.transcriber_model_path,
            "transcriber_model_parameters": self.transcriber_model_parameters,
            "transcriber_transcription_parameters": self.transcriber_transcription_parameters,
            "synthesizer_backend": self.synthesizer_backend,
            "synthesizer_model_path": self.synthesizer_model_path,
            "synthesizer_model_parameters": self.synthesizer_model_parameters,
            "synthesizer_synthesis_parameters": self.synthesizer_synthesis_parameters,
            "speechrecorder_input_device_index": self.speechrecorder_input_device_index,
            "speechrecorder_recognizer_parameters": self.speechrecorder_recognizer_parameters,
            "speechrecorder_microphone_parameters": self.speechrecorder_microphone_parameters,
            "speechrecorder_loop_pause": self.speechrecorder_loop_pause,
        }

    def spawn_conversation_handler(self, worker_function: Callable) -> ConversationHandler:
        """
        Spawns a conversation handler based on the session parameters
        :param worker_function: Worker function.
        """
        return ConversationHandler(
            working_directory=safely_create_path(self.working_directory),
            speech_recorder=SpeechRecorder(
                input_device_index=self.speechrecorder_input_device_index,
                recognizer_parameters=self.speechrecorder_recognizer_parameters,
                microphone_parameters=self.speechrecorder_microphone_parameters,
                loop_pause=self.speechrecorder_loop_pause
            ),
            transcriber=Transcriber(
                backend=self.transcriber_backend,
                model_path=self.transcriber_model_path,
                model_parameters=self.transcriber_model_parameters,
                transcription_parameters=self.transcriber_transcription_parameters
            ),
            synthesizer=Synthesizer(
                backend=self.synthesizer_backend,
                model_path=self.synthesizer_model_path,
                model_parameters=self.synthesizer_model_parameters,
                synthesis_parameters=self.synthesizer_synthesis_parameters
            ),
            worker_function=worker_function,
            loop_pause=self.loop_pause
        )


class BasicVoiceAssistant(object):
    """
    Represents a basic voice assistant.
    """

    def __init__(self,
                 handler_session: ConversationHandlerSession,
                 chat_model: ChatModelInstance) -> None:
        """
        Initiation method.
        :param handler_session: Conversation handler session.
        :param chat_model: Chat model to handle interaction.
        """
        self.session = handler_session
        self.chat_model = chat_model
        self.handler = None

    def setup(self) -> None:
        """
        Method for setting up conversation handler.
        """
        if self.handler is None:
            self.handler = self.session.spawn_conversation_handler(worker_function=self.chat_model.chat)
        else:
            self.handler.reset()

    def stop(self) -> None:
        """
        Method for setting up conversation handler.
        """
        self.handler.stop()

    def run_conversation(self, blocking: bool = True, stream: bool = False) -> None:
        """
        Method for running a looping conversation.
        :param blocking: Flag which declares whether or not to wait for each conversation step.
            Defaults to True.
        :param stream: Declares, whether worker function streams response.
            Defaults to False.
        """
        self.handler.run_conversation(blocking=blocking, stream=stream)

    def run_interaction(self, blocking: bool = True, stream: bool = False) -> None:
        """
        Method for running an conversational interaction.
        :param blocking: Flag which declares whether or not to wait for each conversation step.
            Defaults to True.
        :param stream: Declares, whether worker function streams response.
            Defaults to False.
        """
        self.handler.run_conversation(blocking=blocking, loop=False, stream=stream)

    def run_terminal_conversation(self, stream: bool = False) -> None:
        """
        Runs conversation loop with terminal input.
        :param stream: Declares, whether worker function streams response.
            Defaults to False.
        """
        stop = TEvent()
        bindings = KeyBindings()

        @bindings.add("c-c")
        @bindings.add("c-d")
        def exit_session(event: KeyPressEvent) -> None:
            """
            Function for exiting session.
            :param event: Event that resulted in entering the function.
            """
            cfg.LOGGER.info(f"Recieved keyboard interrupt, shutting down handler ...")
            self.handler.reset()
            rich_print("[bold]\nBye [white]...")
            event.app.exit()
            stop.set()

        session = setup_prompt_session(bindings)
        cfg.LOGGER.info(f"Starting conversation loop...")
        self.handler.pause_input.set()
        self.handler.run_conversation(blocking=False, loop=True, stream=stream)
        
        while not stop.is_set():
            user_input = session.prompt(
                "User: ")
            if user_input is not None:
                self.queues["worker_in"].put(user_input)
