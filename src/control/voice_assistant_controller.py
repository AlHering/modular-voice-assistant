# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import os
from time import sleep
from datetime import datetime as dt
import numpy as np
from typing import Optional, Any, List, Dict, Union, Tuple
from src.configuration import configuration as cfg
from src.utility.gold.basic_sqlalchemy_interface import BasicSQLAlchemyInterface
from src.control.text_generation_controller import TextGenerationController
from src.model.voice_assistant_control.data_model import populate_data_instrastructure
from src.utility.gold.sound_generation.sound_model_abstractions import Transcriber, Synthesizer, SpeechRecorder


class VoiceAssistantController(BasicSQLAlchemyInterface):
    """
    Controller class for handling voice assistant interface requests.
    """
    
    def __init__(self, working_directory: str = None, database_uri: str = None) -> None:
        """
        Initiation method.
        :param working_directory: Working directory.
            Defaults to folder 'processes' folder under standard backend data path.
        :param database_uri: Database URI.
            Defaults to 'backend.db' file under configured backend folder.
        """
        # Main instance variables
        self._logger = cfg.LOGGER
        self.working_directory = cfg.PATHS.BACKEND_PATH if working_directory is None else working_directory
        if not os.path.exists(self.working_directory):
            os.makedirs(self.working_directory)
        self.database_uri = f"sqlite:///{os.path.join(cfg.PATHS.BACKEND_PATH, 'voice_assistant.db')}" if database_uri is None else database_uri

        # Database infrastructure
        super().__init__(self.working_directory, self.database_uri,
                         populate_data_instrastructure, "voice_assistant.", self._logger)
        
        # Orchestrated workers
        self.workers = {
            "transcribers": {},
            "synthesizers": {},
            "speech_recorders": {}
        }

        
    """
    Setup and shutdown methods
    """
    def setup(self) -> None:
        """
        Method for running setup process.
        """
        pass


    def shutdown(self) -> None:
        """
        Method for running shutdown process.
        """
        pass


    """
    Base interaction
    """
    def log(self, data: dict) -> None:
        """
        Method for adding a log entry.
        :param data: Log entry data.
        """
        self.post_object("log", **data)

    """
    Orchestration interaction
    """
    def transcribe(self, transcriber_id: int, audio_input: np.ndarray) -> Tuple[str, dict]:
        """
        Method for transcribing audio data with specific transriber.
        :param transcriber_id: Transcriber ID.
        :param audio_input: Audio input data.
        :return: Tuple of transcription and metadata.
        """
        if str(transcriber_id) not in self.workers["transcibers"]:
            entry = self.get_object_by_id("transcriber", transcriber_id)
            self.workers["transcribers"][str(transcriber_id)] = Transcriber(
                backend=entry.backend,
                model_path=entry.model_path,
                model_parameters=entry.model_paramters,
                transcription_parameters=entry.transcription_parameters
            )
        return self.workers["transcribers"][str(transcriber_id)].transcribe(audio_input=audio_input)

    def synthesize(self, synthesizer_id: int, text: str) -> Tuple[np.ndarray, dict]:
        """
        Endpoint for synthesis.
        :param synthesizer_id: Synthesizer ID.
        :param text: Text to synthesize audio for.
        :return: Tuple of synthesis and metadata.
        """
        if str(synthesizer_id) not in self.workers["synthesizers"]:
            entry = self.get_object_by_id("synthesizers", synthesizer_id)
            self.workers["synthesizers"][str(synthesizer_id)] = Synthesizer(
                backend=entry.backend,
                model_path=entry.model_path,
                model_parameters=entry.model_paramters,
                synthesis_parameters=entry.synthesis_parameters
            )
        return self.workers["synthesizers"][str(synthesizer_id)].synthesize(text=text)


