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
from src.utility.time_utility import get_timestamp
from src.utility.string_utility import separate_pattern_from_text, extract_matches_between_bounds, remove_multiple_spaces, EMOJI_PATTERN
from src.modules.abstractions import BaseModuleSet, VAPackage
from src.modules.input_modules import SpeechRecorderModule, TranscriberModule
from src.modules.worker_modules import BasicHandlerModule, LocalChatModule, RemoteChatModule
from src.modules.output_modules import SynthesizerModule, WaveOutputModule
from src.conversation_handler import ModularConversationHandler


AVAILABLE_MODULES = {
    "speech_recorder": SpeechRecorderModule,
    "transcriber": TranscriberModule,
    "local_chat": LocalChatModule,
    "remote_chat": RemoteChatModule,
    "synthesizer": SynthesizerModule,
    "wave_output": WaveOutputModule
}


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
                 speech_recorder: SpeechRecorderModule,
                 transcriber: TranscriberModule,
                 worker: BasicHandlerModule,
                 synthesizer: SynthesizerModule,
                 wave_output: WaveOutputModule,
                 stream: bool = True,
                 forward_logging: bool = False,
                 report: bool = False) -> None:
        """
        Initiation method.
        :param working_directory: Working directory.
        :param speech_recorder: Speech Recorder module.
        :param transcriber: Transcriber module.
        :param worker: Worker module, e.g. LocalChatModule or RemoteChatModule.
        :param synthesizer: Synthesizer module.
        :param wave_output: Wave output module.
        :param stream: Declares, whether chat model should stream its response.
        :param forward_logging: Flag for forwarding logger to modules.
        :param report: Flag for running report thread.
        """
        self.working_directory = working_directory
        self.stream = stream

        forward_logging = cfg.LOGGER if forward_logging else None
        for va_module in [speech_recorder, transcriber, worker, synthesizer, wave_output]:
            va_module.logger = forward_logging

        self.module_set = BaseModuleSet()
        self.module_set.input_modules.append(speech_recorder)
        self.module_set.input_modules.append(transcriber)
        self.module_set.worker_modules.append(worker)
        self.module_set.output_modules.append(
            BasicHandlerModule(handler_method=clean_worker_output,
                               input_queue=self.module_set.worker_modules[-1].output_queue,
                               logger=forward_logging,
                               name="Cleaner")
        )
        self.module_set.output_modules.append(synthesizer)
        self.module_set.output_modules.append(wave_output)

        self.module_set.reroute_pipeline_queues()

        self.conversation_kwargs = {}
        if isinstance(worker, LocalChatModule) or isinstance(worker, RemoteChatModule):
            if worker.chat_model.history[-1]["role"] == "assistant":
                self.conversation_kwargs["greeting"] = worker.chat_model.history[-1]["content"]
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


def setup_default_voice_assistant(config: dict | None = None) -> BasicVoiceAssistant:
    """
    Sets up a default voice assistant for reference.
    :param config: Config to overwrite voice assistant default config with.
    :return: Basic voice assistant.
    """
    config = cfg.DEFAULT_COMPONENT_CONFIG if config is None else config
    if config.get("download_model_files"):
        raise NotImplementedError("Downloading models is not yet implemented!")

    return BasicVoiceAssistant(
        working_directory=os.path.join(cfg.PATHS.DATA_PATH, "voice_assistant"),
        speech_recorder=SpeechRecorderModule(**config.get("speech_recorder", cfg.DEFAULT_SPEECH_RECORDER)),
        transcriber=TranscriberModule(**config.get("transcriber", cfg.DEFAULT_TRANSCRIBER)),
        worker=RemoteChatModule(**config.get("remote_chat", cfg.DEFAULT_REMOTE_CHAT)) if config.get("use_remote_llm") else LocalChatModule(**config.get("local_chat", cfg.DEFAULT_LOCAL_CHAT)),
        synthesizer=SynthesizerModule(**config.get("synthesizer", cfg.DEFAULT_SYNTHESIZER)),
        wave_output=WaveOutputModule(**config.get("wave_output", cfg.DEFAULT_WAVE_OUTPUT)),
        **config.get("voice_assistant", cfg.DEFAULT_VOICE_ASSISTANT)
    )