# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from logging import Logger
from uuid import uuid4
from typing import Any, Tuple, List, Callable, Generator
import os
import gc
import time
import numpy as np
from prompt_toolkit import PromptSession, HTML, print_formatted_text
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding.key_bindings import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style as PTStyle
from src.configuration import configuration as cfg
from threading import Thread, Event as TEvent
from queue import Empty, Queue as TQueue
from src.modules.language_model_abstractions import ChatModelInstance, RemoteChatModelInstance
from src.modules.language_model_abstractions import ChatModelConfig, RemoteChatModelConfig
from src.utility.time_utility import get_timestamp
from src.utility.string_utility import separate_pattern_from_text, extract_matches_between_bounds, remove_multiple_spaces, EMOJI_PATTERN
from src.modules.worker_modules import BasicHandlerModule, SpeechRecorderModule, TranscriberModule, SynthesizerModule, WaveOutputModule
from src.modules.worker_modules import SpeechRecorderConfig, TranscriberConfig, SynthesizerConfig, WaveOutputConfig
from src.modules.abstractions import BaseModuleSet, VAPackage
from src.modules.sound_model_abstractions import Transcriber, Synthesizer, SpeechRecorder


def setup_prompt_session(bindings: KeyBindings = None) -> PromptSession:
    """
    Function for setting up a command line prompt session.
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
                 module_set: BaseModuleSet,
                 loop_pause: float = 0.1) -> None:
        """
        Initiation method.
        :param working_directory: Directory for productive files.
        :param module_set: Module set.
        :param loop_pause: Loop pause.
        """
        cfg.LOGGER.info("Initiating Conversation Handler...")
        self.working_directory = working_directory
        os.makedirs(self.working_directory, exist_ok=True)
        self.loop_pause = loop_pause

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
        cfg.LOGGER.info("Reset is done.")

    def _run_nonblocking_conversation(self, loop: bool) -> None:
        """
        Runs a non-blocking conversation.
        :param loop: Declares, whether to loop conversation or stop after a single interaction.
        """
        for thread in self.worker_threads + self.input_threads:
            thread.start()         
        if not loop:
            while self.module_set.input_modules[0].output_queue.qsize() == 0 and self.module_set.input_modules[-1].output_queue.qsize() == 0:
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
        :param loop: Declares, whether to loop conversation or stop after a single interaction.
        """
        # TODO: Trace inputs via VAPackage UUIDs
        while not self.stop.is_set():
            for module in self.module_set.input_modules:
                while not module.run():
                    time.sleep(self.loop_pause//16)
            for module in self.module_set.worker_modules:
                while not module.run():
                    time.sleep(self.loop_pause//16)
            while any(module.queues_are_busy() for module in self.module_set.output_modules):
                time.sleep(self.loop_pause//16)
            if not loop:
                self.stop.set()
        self.reset()

    def run_conversation(self, 
                         blocking: bool = True, 
                         loop: bool = True, 
                         greeting: str = "Hello there, how may I help you today?",
                         report: bool = False) -> None:
        """
        Runs conversation.
        :param blocking: Declares, whether or not to wait for each step.
            Defaults to True.
        :param loop: Declares, whether to loop conversation or stop after a single interaction.
            Defaults to True.
        :param greeting: Assistant greeting.
            Defaults to "Hello there, how may I help you today?".
        :param report: Flag for logging reports.
        """
        cfg.LOGGER.info(f"Starting conversation loop...")
        if report:
            self.run_report_thread()

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
            cfg.LOGGER.info(f"Received keyboard interrupt, shutting down handler ...")
            self.stop_modules()

    def run_report_thread(self) -> None:
        """
        Runs a thread for logging reports.
        """ 
        def log_report(wait_time: float = 10.0) -> None:
            while not self.stop.is_set():
                module_info = "\n".join([
                    "==========================================================",
                    f"#                    {get_timestamp()}                   ",
                    f"#                    {self}                              ",
                    f"#                Running: {not self.stop.is_set()}       ",
                    "=========================================================="
                ])
                for threads in ["input_threads", "worker_threads", "output_threads", "additional_threads"]:
                    for thread_index, thread in enumerate(getattr(self, threads)):
                        module = getattr(self.module_set, f"{threads.split('_')[0]}_modules")[thread_index]
                        module_info += f"\n\t[{type(module).__name__}<{module.name}>] Thread '{thread}: {thread.is_alive()}'"
                        module_info += f"\n\t\t Inputs: {module.input_queue.qsize()}'"
                        module_info += f"\n\t\t Outputs: {module.output_queue.qsize()}'"
                        module_info += f"\n\t\t Received: {module.received}'"
                        module_info += f"\n\t\t Sent: {module.sent}'"
                        module_info += f"\n\t\t Pause: {module.pause.is_set()}'"
                        module_info += f"\n\t\t Interrupt: {module.interrupt.is_set()}'"
                cfg.LOGGER.info(module_info)
                time.sleep(wait_time)
        thread = Thread(target=log_report)
        thread.daemon = True
        thread.start()    
        

def clean_worker_output(text: str) -> Tuple[str, dict]:
    """
    Cleanse worker output from emojis and emotional hints.
    :param text: Worker output.
    :return: Cleaned text and metadata.
    """
    metadata = {"full_text": text}
    metadata["text_without_emojis"], metadata["emojis"] = separate_pattern_from_text(text=text, pattern=EMOJI_PATTERN)
    metadata["emotional_hints"] = [f"*{hint}*" for hint in extract_matches_between_bounds(start_bound=r"*", end_bound=r"*", text=metadata["text_without_emojis"])]
    metadata["text_without_emotional_hints"] = metadata["text_without_emojis"]
    if metadata["emotional_hints"]:
        for hint in metadata["emotional_hints"]:
            metadata["text_without_emotional_hints"] = metadata["text_without_emotional_hints"].replace(hint, "")
    return remove_multiple_spaces(text=metadata["text_without_emotional_hints"]), metadata


class BasicVoiceAssistant(object):
    """
    Represents a basic voice assistant.
    """

    def __init__(self,
                 working_directory: str,
                 speech_recorder_config: SpeechRecorderConfig,
                 transcriber_config: TranscriberConfig,
                 chat_model_config: ChatModelConfig | RemoteChatModelConfig,
                 synthesizer_config: SynthesizerConfig,
                 stream: bool = False,
                 forward_logging: bool = False,
                 report: bool = False) -> None:
        """
        Initiation method.
        :param working_directory: Working directory.
        :param speech_recorder_config: Speech Recorder config.
        :param transcriber_config: Transcriber config.
        :param chat_model_config: Config of chat model to handle interaction.
        :param synthesizer_config: Synthesizer config.
        :param stream: Declares, whether chat model should stream its response.
        :param forward_logging: Flag for forwarding logger to modules.
        :param report: Flag for running report thread.
        """
        self.working_directory = working_directory
        self.chat_model = ChatModelInstance.from_configuration(
            config=chat_model_config) if isinstance(
                chat_model_config, ChatModelConfig) else RemoteChatModelConfig.from_configuration(config=chat_model_config)
        self.stream = stream

        forward_logging = cfg.LOGGER if forward_logging else None
        for config in [speech_recorder_config, transcriber_config, synthesizer_config]:
            config.logger = forward_logging

        self.module_set = BaseModuleSet()
        self.module_set.input_modules.append(
            SpeechRecorderModule.from_configuration(speech_recorder_config))
        self.module_set.input_modules.append(
            TranscriberModule.from_configuration(transcriber_config))
        self.module_set.worker_modules.append(
            BasicHandlerModule(handler_method=self.chat_model.chat_stream if stream else self.chat_model.chat,
                            input_queue=self.module_set.input_modules[-1].output_queue,
                            logger=forward_logging,
                            name="Chat")
        )
        self.module_set.output_modules.append(
            BasicHandlerModule(handler_method=clean_worker_output,
                               input_queue=self.module_set.worker_modules[-1].output_queue,
                               logger=forward_logging,
                               name="Cleaner")
        )
        self.module_set.output_modules.append(
            SynthesizerModule.from_configuration(synthesizer_config))
        self.module_set.output_modules.append(
            WaveOutputModule.from_configuration(WaveOutputConfig))

        self.conversation_kwargs = {}
        if self.chat_model.history[-1]["role"] == "assistant":
            self.conversation_kwargs["greeting"] = self.chat_model.history[-1]["content"]
        self.conversation_kwargs["report"] = report
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

    def inject_prompt(self, prompt: str) -> None:
        """
        Injects a prompt into a running conversation.
        :param prompt: Prompt to inject.
        """
        self.module_set.input_modules[-1].output_queue.put(VAPackage(content=prompt))

    def run_terminal_conversation(self) -> None:
        """
        Runs conversation loop with terminal input.
        """
        run_terminal_conversation(handler=self.handler, conversation_kwargs=self.conversation_kwargs)


def run_terminal_conversation(handler: ModularConversationHandler, conversation_kwargs: dict = None) -> None:
    """
    Runs conversation loop with terminal input.
    :param handler: Conversation handler.
    :param conversation_kwargs: Conversation keyword arguments, such as a 'greeting'.
    """
    conversation_kwargs = {} if conversation_kwargs is None else conversation_kwargs
    stop = TEvent()
    bindings = KeyBindings()

    @bindings.add("c-c")
    @bindings.add("c-d")
    def exit_session(event: KeyPressEvent) -> None:
        """
        Function for exiting session.
        :param event: Event that resulted in entering the function.
        """
        cfg.LOGGER.info(f"Received keyboard interrupt, shutting down handler ...")
        handler.reset()
        print_formatted_text(HTML("<b>Bye...</b>"))
        event.app.exit()
        stop.set()

    session = setup_prompt_session(bindings)
    handler.module_set.input_modules[0].pause.set()
    handler.run_conversation(blocking=False, loop=True, **conversation_kwargs)
    
    while not stop.is_set():
        with patch_stdout():
            user_input = session.prompt(
                "User: ")
            if user_input is not None:
                handler.module_set.input_modules[-1].output_queue.put(VAPackage(content=user_input))