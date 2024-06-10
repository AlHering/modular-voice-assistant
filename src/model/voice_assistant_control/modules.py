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
from typing import List, Tuple, Any, Callable, Optional, Union, Dict, Generator
from datetime import datetime as dt
from enum import Enum
import pyaudio
import numpy as np
import time


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
        return (self.lines[self.index], {"file_path": self.file_path, "index": self.index}
                if self.metadatas is None else self.metadatas[self.index])