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


class VAPackage(BaseModel):
    """
    Voice assistant package for exchanging data between modules.
    """
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
        Returns a thread for handling module operations.
        """
        thread = Thread(target=self.loop)
        thread.daemon = True
        return thread

    def loop(self) -> None:
        """
        Module looping method.
        """
        while not self.interrupt.is_set():
            result = self.run()
            if result is not None:
                if self.stream_result:
                    for elem in result:
                        self.output_queue.put(elem)
                else:
                    self.output_queue.put(result)
            time.sleep(self.loop_pause)

    @abstractmethod
    def run(self) -> VAPackage | Generator[VAPackage, None, None] | None:
        """
        Module runner method.
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

    def run(self) -> VAPackage | Generator[VAPackage, None, None] | None:
        """
        Module runner method.
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

    def run(self) -> VAPackage | Generator[VAPackage, None, None] | None:
        """
        Module runner method.
        :returns: Voice assistant package, a package generator in case of streaming or None.
        """
        if not self.pause.is_set():
            try:
                input_package: VAPackage = self.input_queue.get(block=True, timeout=self.input_timeout)
                self.log_info(f"Received input:\n'{input_package.content}'")
                if self.stream_result:
                    for response_tuple in self.streamed_handler_method(input_package.content):
                        self.log_info(f"Received response part\n'{response_tuple[0]}'.")
                        yield VAPackage(content=response_tuple[0], metadata_stack=input_package.metadata_stack + [response_tuple[1]])
                else:
                    response, response_metadata = self.handler_method(input_package.content)
                    self.log_info(f"Received response\n'{response}'.")             
                    return VAPackage(content=response, metadata_stack=input_package.metadata_stack + [response_metadata])
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

    def run(self) -> VAPackage | Generator[VAPackage, None, None] | None:
        """
        Module runner method.
        :returns: Voice assistant package, a package generator in case of streaming or None.
        """
        if not self.pause.is_set():
            try:
                input_package: VAPackage = self.input_queue.get(block=True, timeout=self.input_timeout)
                self.pause.set()
                cfg.LOGGER.info(f"Outputting wave response...")
                play_wave(input_package.content, input_package.metadata_stack[-1])
                self.pause.clear()
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
                 modules: List[VAModule],
                 loop_pause: float = 0.1) -> None:
        """
        Initiation method.
        :param working_directory: Directory for productive files.
        :param modules: List of modules to handle.
            Defaults to SPEECH (TTS).
        :param loop_pause: Pause in seconds between processing loops.
            Defaults to 0.1.
        """
        cfg.LOGGER.info("Initiating Conversation Handler...")
        safely_create_path(working_directory)
        self.working_directory = working_directory
        self.input_path = os.path.join(self.working_directory, "input.wav") 
        self.output_path = os.path.join(self.working_directory, "output.wav") 

        pya = pyaudio.PyAudio()
        self.input_device_index = pya.get_default_input_device_info().get("index")
        self.output_device_index = pya.get_default_output_device_info().get("index")
        pya.terminate()

        self.modules = modules
        self.threads = None
        self.stop = None
        self.loop_pause = loop_pause
        self.setup_components()

    def stop_components(self) -> None:
        """
        Stops process components.
        """
        self.stop.set()
        for module in self.modules:
            module.interrupt.set()
            module.pause.set()
        for thread in self.threads:
            try:
                thread.join(self.loop_pause) 
            except RuntimeError:
                pass

    def setup_components(self) -> None:
        """
        Sets up components.
        """
        self.stop = TEvent()
        self.threads = [module.to_thread() for module in self.modules]

    def reset(self) -> None:
        """
        Sets up and resets handler. 
        """
        cfg.LOGGER.info("(Re)setting Conversation Handler...")
        self.stop_components()
        gc.collect()
        self.setup_components()
        cfg.LOGGER.info("Setup is done.")

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
        if stream:
            self.threads["output"].start()
        while not self.stop.is_set():
            self.handle_input()
            self.handle_work(stream=stream)
            if stream:
                while self.queues["worker_out"].qsize() > 0 or self.pause_output.is_set():
                    time.sleep(self.loop_pause)
            else:
                self.handle_output()
            if not loop:
                self.stop.set()
        self.reset()

    def run_conversation(self, 
                         blocking: bool = True, 
                         loop: bool = True, 
                         stream: bool = False,
                         greeting: str = "Hello there, how may I help you today?") -> None:
        """
        Runs conversation.
        :param blocking: Declares, whether or not to wait for each step.
            Defaults to True.
        :param loop: Delcares, whether to loop conversation or stop after a single interaction.
            Defaults to True.
        :param stream: Declares, whether worker function streams response.
            Defaults to False.
        :param greeting: Assistant greeting.
            Defaults to "Hello there, how may I help you today?".
        """
        cfg.LOGGER.info(f"Starting conversation loop...")
        self.queues["worker_out"].put(self.prepare_worker_output(greeting, {}))
        self.handle_output()
        try:
            if not blocking:
                self._run_nonblocking_conversation(loop=loop, stream=stream)
            else:
                self._run_blocking_conversation(loop=loop, stream=stream)
        except KeyboardInterrupt:
            cfg.LOGGER.info(f"Recieved keyboard interrupt, shutting down handler ...")
            self.stop_components()

    def report(self) -> str:
        """
        Return a report, formatted as string.
        :return: Report string.
        """
        thread_info = "        \n".join(f"Thread '{thread}: {self.threads[thread].is_alive()}'" for thread in self.threads)
        queue_info = "        \n".join(f"Queue '{queue}: {self.queues[queue].qsize()}'" for queue in self.queues)
        events = {
            "interrupt": self.stop,
            "pause_input": self.pause_input,
            "pause_worker": self.pause_worker,
            "pause_output": self.pause_output
        }
        event_info = "\n".join(f"Event '{event}: {events[event].is_set()}'" for event in events)
        return "\n".join([
            "==========================================================",
            "#                    {get_timestamp()}                   #",
            "==========================================================",
            thread_info,
            "----------------------------------------------------------",
            queue_info,
            "----------------------------------------------------------",
            event_info,
            "----------------------------------------------------------"
        ])


class BasicVoiceAssistant(object):
    """
    Represents a basic voice assistant.
    """

    def __init__(self,
                 handler: Union[ConversationHandlerSession, ConversationHandler],
                 chat_model: ChatModelInstance,
                 stream: bool = False) -> None:
        """
        Initiation method.
        :param handler: Conversation handler or session.
        :param chat_model: Chat model to handle interaction.
        :param stream: Declares, whether chat model should stream its response.
            Defaults to False.
        """
        if isinstance(handler, ConversationHandlerSession):
            self.session = handler
            self.handler = None
        elif isinstance(handler, ConversationHandler):
            self.session = ConversationHandlerSession.from_handler(handler=handler)
            self.handler = handler
        self.chat_model = chat_model
        self.stream = stream

        self.conversation_kwargs = {}
        if self.chat_model.history[-1]["role"] == "assistant":
            self.conversation_kwargs["greeting"] = self.chat_model.history[-1]["content"]

    def setup(self) -> None:
        """
        Method for setting up conversation handler.
        """
        if self.handler is None:
            self.handler = self.session.spawn_conversation_handler(worker_function=self.chat_model.chat_stream if self.stream else self.chat_model.chat)
        else:
            self.handler.reset()

    def stop(self) -> None:
        """
        Method for stopping conversation handler.
        """
        self.handler.stop_components()

    def run_conversation(self, blocking: bool = True) -> None:
        """
        Method for running a looping conversation.
        :param blocking: Flag which declares whether or not to wait for each conversation step.
            Defaults to True.
        """
        self.handler.run_conversation(blocking=blocking, stream=self.stream, **self.conversation_kwargs)

    def run_interaction(self, blocking: bool = True) -> None:
        """
        Method for running an conversational interaction.
        :param blocking: Flag which declares whether or not to wait for each conversation step.
            Defaults to True.
        """
        self.handler.run_conversation(blocking=blocking, loop=False, stream=self.stream, **self.conversation_kwargs)

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
        self.handler.pause_input.set()
        self.handler.run_conversation(blocking=False, loop=True, stream=self.stream, **self.conversation_kwargs)
        
        while not stop.is_set():
            with patch_stdout():
                user_input = session.prompt(
                    "User: ")
                if user_input is not None:
                    self.handler.queues["worker_in"].put(self.handler.prepare_worker_input(user_input, {}))
                    """while True:
                        print(self.handler.report())
                        time.sleep(3)"""
