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
from pydantic import BaseModel, Field
from logging import Logger
from uuid import uuid4
from typing import Any, Union, Tuple, List, Optional, Callable, Dict, Generator
import os
import gc
import time
import numpy as np
import pyaudio
from prompt_toolkit import PromptSession, HTML, print_formatted_text
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding.key_bindings import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style as PTStyle
from datetime import datetime as dt
from src.configuration import configuration as cfg
from threading import Thread, Event as TEvent, Lock
from queue import Empty, Queue as TQueue
from ..text_generation.language_model_abstractions import ChatModelInstance, RemoteChatModelInstance
from ..text_generation.agent_abstractions import ToolArgument, AgentTool
from ..text_generation.agent_abstractions import Agent
from ...bronze.concurrency_utility import PipelineComponentThread, timeout
from ...bronze.time_utility import get_timestamp
from ...bronze.json_utility import load as load_json
from ...silver.file_system_utility import safely_create_path
from ...bronze.pyaudio_utility import play_wave
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



def create_default_metadata() -> List[dict]:
    """
    Creates a default VA package dictionary.
    :return: Default VA package dictionary
    """
    return [{"created": get_timestamp()}]


def create_uuid() -> str:
    """
    Creates an UUID for a VA package.
    :return: UUID as string.
    """
    return str(uuid4())


class VAPackage(BaseModel):
    """
    Voice assistant package for exchanging data between modules.
    """
    uuid: str = Field(default_factory=create_uuid)
    content: Any
    metadata_stack: List[dict] = Field(default_factory=create_default_metadata)


class VAModule(ABC):
    """
    Voice assistant module.
    """
    def __init__(self, 
                 interrupt: TEvent | None = None,
                 pause: TEvent | None = None,
                 loop_pause: float = 0.1,
                 input_timeout: float | None = None, 
                 input_queue: TQueue | None = None,
                 output_queue: TQueue | None = None,
                 stream_result: bool = False,
                 logger: Logger | None = None) -> None:
        """
        Initiates an instance.
        :param interrupt: Interrupt event.
        :param pause: Pause event.
        :param loop_pause: Time to wait between looped runs.
        :param input_timeout: Time to wait for inputs in a single run.
        :param input_queue: Input queue.
        :param output_queue: Output queue.
        :param stream_result: Flag for declaring, whether to handle the result as a generator object.
            Defaults to None.
        :param logger: Logger.
        """
        self.interrupt = TEvent() if interrupt is None else interrupt
        self.pause = TEvent() if pause is None else pause
        self.loop_pause = loop_pause
        self.input_timeout = input_timeout
        self.input_queue = TQueue() if input_queue is None else input_queue
        self.output_queue = TQueue() if output_queue is None else output_queue
        self.stream_result = stream_result
        self.logger = logger

        self.received = []
        self.sent = []

    def _flush_queue(self, queue: TQueue) -> None:
        """
        Flushes queue.
        :param queue: Queue to flush.
        """
        with queue.mutex:
            queue.clear()
            queue.notify_all()

    def flush_inputs(self) -> None:
        """
        Flushes input queue.
        """
        self._flush_queue(self.input_queue)
        

    def flush_outputs(self) -> None:
        """
        Flushes output queue.
        """
        self._flush_queue(self.output_queue)

    def queues_are_busy(self) -> bool:
        """
        Returns queue status.
        :return: True, if any queue contains elements, else False.
        """
        return self.input_queue.qsize() > 0 or self.output_queue.qsize > 0
    
    def log_info(self, text: str) -> None:
        """
        Logs info, if logger is available.
        :param text: Text content to log.
        """
        if self.logger is not None:
            self.logger.info(text)

    def to_thread(self) -> Thread:
        """
        Returns a thread for running module process in loop.
        """
        thread = Thread(target=self.loop)
        thread.daemon = True
        return thread

    def loop(self) -> None:
        """
        Starts processing cycle loop.
        """
        while not self.interrupt.is_set():
            self.run()
            time.sleep(self.loop_pause)
        
    def run(self) -> None:
        """
        Runs a single processing cycle..
        """
        result = self.run()
        if result is not None:
            if self.stream_result:
                for elem in result:
                    self.output_queue.put(elem)
                    self.sent.append(elem.uuid)
            else:
                self.output_queue.put(result)
                self.sent.append(result.uuid)

    @abstractmethod
    def process(self) -> VAPackage | Generator[VAPackage, None, None] | None:
        """
        Module processing method.
        :returns: Voice assistant package, a package generator in case of streaming or None.
        """
        pass


class SpeechRecorderModule(VAModule):
    """
    Speech recorder module.
    """
    def __init__(self, 
                 speech_recorder: SpeechRecorder, 
                 *args: Any | None, 
                 **kwargs: Any | None) -> None:
        """
        Initiates an instance.
        :param speech_recorder: Speech recorder instance.
        :param args: Arbitrary arguments.
        :param kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.speech_recorder = speech_recorder

    def process(self) -> VAPackage | Generator[VAPackage, None, None] | None:
        """
        Module processing method.
        :returns: Voice assistant package, a package generator in case of streaming or None.
        """
        if not self.pause.is_set():
            recorder_output, recorder_metadata = self.speech_recorder.record_single_input()
            self.log_info(f"Got voice input.")
            return VAPackage(content=recorder_output, metadata_stack=[recorder_metadata])
        

class WaveOutputModule(VAModule):
    """
    Wave output module.
    """
    def __init__(self, 
                 *args: Any | None, 
                 **kwargs: Any | None) -> None:
        """
        Initiates an instance.
        :param handler_method: Handler method.
        :param streamed_handler_method: Handler method for streamed responses.
        :param args: Arbitrary arguments.
        :param kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)

    def process(self) -> VAPackage | Generator[VAPackage, None, None] | None:
        """
        Module processing method.
        :returns: Voice assistant package, a package generator in case of streaming or None.
        """
        if not self.pause.is_set():
            try:
                input_package: VAPackage = self.input_queue.get(block=True, timeout=self.input_timeout)
                self.pause.set()
                self.received.append(input_package.uuid)
                self.log_info(f"Received input:\n'{input_package.content}'")
                play_wave(input_package.content, input_package.metadata_stack[-1])
                self.pause.clear()
            except Empty:
                pass
        

class BasicHandlerModule(VAModule):
    """
    Basic handler module.
    """
    def __init__(self, 
                 handler_method: Callable, 
                 streamed_handler_method: Callable | None = None,
                 *args: Any | None, 
                 **kwargs: Any | None) -> None:
        """
        Initiates an instance.
        :param handler_method: Handler method.
        :param streamed_handler_method: Handler method for streamed responses.
        :param args: Arbitrary arguments.
        :param kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.handler_method = handler_method
        self.streamed_handler_method = streamed_handler_method

    def process(self) -> VAPackage | Generator[VAPackage, None, None] | None:
        """
        Module processing method.
        :returns: Voice assistant package, a package generator in case of streaming or None.
        """
        if not self.pause.is_set():
            try:
                input_package: VAPackage = self.input_queue.get(block=True, timeout=self.input_timeout)
                self.received.append(input_package.uuid)
                self.log_info(f"Received input:\n'{input_package.content}'")
                if self.stream_result:
                    for response_tuple in self.streamed_handler_method(input_package.content):
                        self.log_info(f"Received response part\n'{response_tuple[0]}'.")
                        yield VAPackage(uuid=input_package.uuid, content=response_tuple[0], metadata_stack=input_package.metadata_stack + [response_tuple[1]])
                else:
                    response, response_metadata = self.handler_method(input_package.content)
                    self.log_info(f"Received response\n'{response}'.")             
                    return VAPackage(uuid=input_package.uuid, content=response, metadata_stack=input_package.metadata_stack + [response_metadata])
            except Empty:
                pass


class TranscriberModule(BasicHandlerModule):
    """
    Transcriber module.
    """
    def __init__(self, 
                 transcriber: Transcriber, 
                 *args: Any | None, 
                 **kwargs: Any | None) -> None:
        """
        Initiates an instance.
        :param transcriber: Transciber instance.
        :param args: Arbitrary arguments.
        :param kwargs: Arbitrary keyword arguments.
        """
        super().__init__(handler_method=transcriber.transcribe,
                         *args, 
                         **kwargs)


class ChatModelModule(BasicHandlerModule):
    """
    Chat model module.
    """
    def __init__(self, 
                 chat_model: ChatModelInstance | RemoteChatModelInstance, 
                 *args: Any | None, 
                 **kwargs: Any | None) -> None:
        """
        Initiates an instance.
        :param chat_model: Chat model instance.
        :param args: Arbitrary arguments.
        :param kwargs: Arbitrary keyword arguments.
        """
        super().__init__(handler_method=chat_model.chat,
                         streamed_handler_method=chat_model.chat_stream,
                         *args, 
                         **kwargs)


class SynthesizerModule(BasicHandlerModule):
    """
    Synthesizer module.
    """
    def __init__(self, 
                 synthesizer: Synthesizer, 
                 *args: Any | None, 
                 **kwargs: Any | None) -> None:
        """
        Initiates an instance.
        :param synthesizer: Synthesizerinstance.
        :param args: Arbitrary arguments.
        :param kwargs: Arbitrary keyword arguments.
        """
        super().__init__(handler_method=synthesizer.synthesize,
                         *args, 
                         **kwargs)
        

class BaseModuleSet(object):
    """
    Base module set.
    """
    input_modules: List[VAModule] = []
    worker_modules: List[VAModule] = []
    output_modules: List[VAModule] = []
    additional_modules: List[VAModule] = []

    @classmethod
    def get_all(cls) -> List[VAModule]:
        """
        Returns all available modules.
        :returns: List of VA modules.
        """
        return cls.input_modules + cls.worker_modules + cls.output_modules + cls.additional_modules


class ModularConversationHandler(object):

    """
    Represents a modular conversation handler for handling audio based interaction.
    A conversation handler manages the following modules:
        - speech_recorder: A recorder for spoken input.
        - transcriber: A transcriber to transcribe spoken input into text.
        - worker: A worker to compute an output for the given user input.
        - synthesizer: A synthesizer to convert output texts to sound.
    """

    def __init__(self, 
                 working_directory: str,
                 module_set: BaseModuleSet) -> None:
        """
        Initiation method.
        :param working_directory: Directory for productive files.
        :param module_set: Module set.
        """
        cfg.LOGGER.info("Initiating Conversation Handler...")
        self.working_directory = working_directory
        os.makedirs(self.working_directory, exist_ok=True)

        self.module_set = module_set
        self.input_threads = None
        self.worker_threads = None
        self.output_threads = None
        self.additional_threads = None
        self.stop = None
        self.setup_modules()

    def get_all_threads(self) -> List[Thread]:
        """
        Returns all threads.
        :returns: List of threads.
        """
        res = []
        for threads in [self.input_threads, self.worker_threads, self.output_threads, self.additional_threads]:
            if threads is not None:
                res.extend(threads)
        return threads

    def stop_modules(self) -> None:
        """
        Stops process modules.
        """
        self.stop.set()
        for module in self.module_set.get_all():
            module.interrupt.set()
            module.pause.set()
        for thread in self.get_all_threads():
            try:
                thread.join(.12) 
            except RuntimeError:
                pass

    def setup_modules(self) -> None:
        """
        Sets up modules.
        """
        self.stop = TEvent()
        for module in self.module_set.get_all():
            module.pause.clear()
            module.interrupt.clear()
        self.input_threads = [module.to_thread() for module in self.module_set.input_modules]
        self.worker_threads = [module.to_thread() for module in self.module_set.worker_modules]
        self.output_threads = [module.to_thread() for module in self.module_set.output_modules]
        self.additional_threads = [module.to_thread() for module in self.module_set.additional_modules]

    def reset(self) -> None:
        """
        Sets up and resets handler. 
        """
        cfg.LOGGER.info("(Re)setting Conversation Handler...")
        self.stop_modules()
        gc.collect()
        self.setup_modules()
        cfg.LOGGER.info("Setup is done.")

    def _run_nonblocking_conversation(self, loop: bool) -> None:
        """
        Runs a non-blocking conversation.
        :param loop: Delcares, whether to loop conversation or stop after a single interaction.
        """
        for thread in self.worker_threads + self.input_threads:
            thread.start()   
        if not loop:
            while self.module_set.input_modules[0].output_queue.qsize() == 0:
                time.sleep(self.loop_pause/16)
            self.module_set.input_modules[0].pause.set()
            while self.module_set.worker_modules[-1].output_queue.qsize() == 0:
                time.sleep(self.loop_pause/16)
            while self.module_set.output_modules[-1].input_queue.qsize() > 0 or self.module_set.output_modules[-1].pause.is_set():
                time.sleep(self.loop_pause/16)
            self.reset()
    
    def _run_blocking_conversation(self, loop: bool) -> None:
        """
        Runs a blocking conversation.
        :param loop: Delcares, whether to loop conversation or stop after a single interaction.
        """
        while not self.stop.is_set():
            for module in self.module_set.input_modules:
                module.run()
            for module in self.module_set.worker_modules:
                module.run()
            while any(module.queues_are_busy() for module in self.module_set.output_modules):
                time.sleep(.12)
            if not loop:
                self.stop.set()
        self.reset()

    def run_conversation(self, 
                         blocking: bool = True, 
                         loop: bool = True, 
                         greeting: str = "Hello there, how may I help you today?") -> None:
        """
        Runs conversation.
        :param blocking: Declares, whether or not to wait for each step.
            Defaults to True.
        :param loop: Delcares, whether to loop conversation or stop after a single interaction.
            Defaults to True.
        :param greeting: Assistant greeting.
            Defaults to "Hello there, how may I help you today?".
        """
        cfg.LOGGER.info(f"Starting conversation loop...")
        for thread in self.output_threads:
            thread.start()
        if self.module_set.output_modules:
            self.module_set.output_modules[0].input_queue.put(VAPackage(content=greeting))

        try:
            if not blocking:
                self._run_nonblocking_conversation(loop=loop)
            else:
                self._run_blocking_conversation(loop=loop)
        except KeyboardInterrupt:
            cfg.LOGGER.info(f"Recieved keyboard interrupt, shutting down handler ...")
            self.stop_modules()


class BasicVoiceAssistant(object):
    """
    Represents a basic voice assistant.
    """

    def __init__(self,
                 working_directory: str,
                 speech_recorder: SpeechRecorder,
                 transcriber: Transcriber,
                 chat_model: ChatModelInstance,
                 synthesizer: Synthesizer,
                 stream: bool = False) -> None:
        """
        Initiation method.
        :param working_directory: Working directory.
        :param speech_recorder: Speech Recorder.
        :param transcriber: Transcriber.
        :param chat_model: Chat model to handle interaction.
        :param synthesizer: Synthesizer.
        :param stream: Declares, whether chat model should stream its response.
            Defaults to False.
        """
        self.working_directory = working_directory
        self.speech_recorder = speech_recorder
        self.transcriber = transcriber
        self.chat_model = chat_model
        self.synthesizer = synthesizer
        self.stream = stream

        self.module_set = BaseModuleSet()
        self.module_set.input_modules.append(
            SpeechRecorderModule(speech_recorder=self.speech_recorder))
        self.module_set.input_modules.append(
            TranscriberModule(transcriber=self.transcriber, 
                              input_queue=self.module_set.input_modules[0].output_queue))
        self.module_set.worker_modules.append(
            ChatModelModule(chat_model=self.chat_model,
                            input_queue=self.module_set.input_modules[-1].output_queue,
                            stream_result=self.stream)
        )
        self.module_set.output_modules.append(
            SynthesizerModule(synthesizer=self.synthesizer,
                              input_queue=self.module_set.worker_modules[-1].output_queue)
        )
        self.module_set.output_modules.append(
            WaveOutputModule(input_queue=self.module_set.output_modules[-1].output_queue)
        )

        self.conversation_kwargs = {}
        if self.chat_model.history[-1]["role"] == "assistant":
            self.conversation_kwargs["greeting"] = chat_model.history[-1]["content"]

        self.handler = None

    def setup(self) -> None:
        """
        Method for setting up conversation handler.
        """
        if self.handler is None:
            self.handler = ModularConversationHandler(working_directory=self.working_directory,
                                                      module_set=self.module_set)
        else:
            self.handler.reset()

    def stop(self) -> None:
        """
        Method for stopping conversation handler.
        """
        self.handler.stop_modules()

    def run_conversation(self, blocking: bool = True) -> None:
        """
        Method for running a looping conversation.
        :param blocking: Flag which declares whether or not to wait for each conversation step.
            Defaults to True.
        """
        self.handler.run_conversation(blocking=blocking, **self.conversation_kwargs)

    def run_interaction(self, blocking: bool = True) -> None:
        """
        Method for running an conversational interaction.
        :param blocking: Flag which declares whether or not to wait for each conversation step.
            Defaults to True.
        """
        self.handler.run_conversation(blocking=blocking, loop=False, **self.conversation_kwargs)

    def run_terminal_conversation(self) -> None:
        """
        Runs conversation loop with terminal input.
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
            print_formatted_text(HTML("<b>Bye...</b>"))
            event.app.exit()
            stop.set()

        session = setup_prompt_session(bindings)
        cfg.LOGGER.info(f"Starting conversation loop...")
        self.module_set.input_modules[0].pause.set()
        self.handler.run_conversation(blocking=False, loop=True, **self.conversation_kwargs)
        
        while not stop.is_set():
            with patch_stdout():
                user_input = session.prompt(
                    "User: ")
                if user_input is not None:
                    self.module_set.worker_modules[0].input_queue.put(VAPackage(content=user_input))