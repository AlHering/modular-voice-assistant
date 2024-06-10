# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import os
import traceback
from abc import ABC, abstractmethod
from queue import Queue
import speech_recognition
from src.utility.bronze import json_utility
from src.utility.bronze import time_utility
from typing import List, Tuple, Any, Callable, Optional, Union, Dict, Generator
from datetime import datetime as dt
from enum import Enum
import pyaudio
import numpy as np
import time
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding.key_bindings import KeyBindings
from rich import print as rich_print
from rich.style import Style as RichStyle
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style as PTStyle


class VoiceAssistantModule(ABC):
    """
    Abstract voice assistant module.
    """

    def __init__(self, name: str, description: str, skip_validation: bool = False) -> None:
        """
        Initiation method.
        :param name: Module name.
        :param description: Module description.
        :param skip_validation: Flag for declaring whether or not to skip validation.
        """
        self.name = name
        self.description = description
        self.skip_validation = skip_validation

    @abstractmethod
    def _validate(self, *args: Optional[Any], **kwargs: Optional[Any]) -> Tuple[bool, dict]:
        """
        Abstract method for validating module functionality.
        :param args: Arbitrary arguments.
        :param kwargs: Arbitrary keyword arguments.
        :return: Functionality as boolean and debug metadata as dictionary.
        """
        pass

    def validate(self, *args: Optional[Any], **kwargs: Optional[Any]) -> Tuple[bool, dict]:
        """
        Method for validating module functionality.
        :param args: Arbitrary arguments.
        :param kwargs: Arbitrary keyword arguments.
        :return: Functionality as boolean and debug metadata as dictionary.
        """
        return (True, {"reason": "skipped"}) if self.skip_validation else self._validate(*args, **kwargs)

    @abstractmethod
    def run(self, *args: Optional[Any], **kwargs: Optional[Any]) -> Tuple[Any, dict]:
        """
        Abstract method for running module.
        :param args: Arbitrary arguments.
        :param kwargs: Arbitrary keyword arguments.
        :return: Return value and metadata as dictionary.
        """
        pass

    def stream(self, *args: Optional[Any], **kwargs: Optional[Any]) -> Generator[Tuple[Any, dict], None, None]:
        """
        Abstract method for running module in a streaming manner.
        Note, that if not implemented, it will yield the result of the default "run" method.
        :param args: Arbitrary arguments.
        :param kwargs: Arbitrary keyword arguments.
        :return: Return value and metadata as dictionary.
        """
        yield self.run(*args, **kwargs)



class FileInputModule(VoiceAssistantModule):
    """
    File input module.
    """

    def __init__(self, file_path: str, skip_validation: bool = False) -> None:
        """
        Initiation method.
        :param file_path: File path to load inputs from.
        :param skip_validation: Flag for declaring whether or not to skip validation.
        """
        super().__init__(
            name="FileInputModule",
            description="A module that returns line after line of a text file. If a JSON file can be found under the same path and name, it will be loaded as array for returning metadata.",
            skip_validation=skip_validation
        )
        self.file_path = file_path
        self.lines = None
        self.metadatas = None
        self.index = -1

        if os.path.exists(file_path):
            self.lines = [line.replace("\n", "") for line in open(file_path, "r").readlines()]
            root, _ = os.path.splitext(file_path)
            json_path = f"{root}.json"
            if os.path.exists(json_path):
                self.metadatas = json_utility.load(json_path)
    
    def _validate(self) -> Tuple[bool, Dict]:
        """
        Method for validating module functionality.
        """
        if self.lines is None:
            return (False, {"reason": f"path {self.file_path} does not exist"})
        elif self.metadatas is not None and len(self.lines != len(self.metadatas)):
            return (False, {"reason": f"number of metadata entries does not match number of text inputs"})
        else:
            return (True, {"reason": "validation successful"})
        
    def run(self) -> Tuple[str, dict]:
        """
        Method for validating module functionality.
        """
        self.index += 1
        return (self.lines[self.index], {
                self.name: {
                    "file_path": self.file_path, 
                    "index": self.index,
                    "timestamp": time_utility.get_timestamp()
                }
            } if self.metadatas is None else self.metadatas[self.index])
    

class PromptInputModule(VoiceAssistantModule):
    """
    Prompt input module.
    """

    def __init__(self, prompt_message: str = "User :") -> None:
        """
        Initiation method.
        :param prompt_message: Prompt message to ask user for input.
            Defaults to "User: ".
        """
        super().__init__(
            name="PromptInputModule",
            description="A module that prompts the user for the next textual input.",
            skip_validation=True
        )
        self.prompt_query = prompt_message
        self.bindings = KeyBindings()

        @self.bindings.add("c-c")
        @self.bindings.add("c-d")
        def exit_session(event: KeyPressEvent) -> None:
            """
            Function for exiting session.
            :param event: Event that resulted in entering the function.
            """
            rich_print("[bold]\nBye [white]...")
            event.app.exit()

        self.session = self._setup_prompt_session()

    def _setup_prompt_session(self) -> PromptSession:
        """
        Function for setting up prompt session.
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
            key_bindings=self.bindings
        )
    
    def _validate(self) -> Tuple[bool, Dict]:
        """
        Method for validating module functionality.
        """
        pass
        
    def run(self) -> Tuple[str, dict]:
        """
        Method for validating module functionality.
        """
        user_input = ""
        with patch_stdout():
            user_input = self.session.prompt(self.prompt_query)
        return user_input, {self.name: {"timestamp": time_utility.get_timestamp()}}
    