# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import os
from src.configuration import configuration as cfg
from src.backend.database.basic_sqlalchemy_interface import BasicSQLAlchemyInterface, FilterMask
from src.backend.database.data_model import populate_data_infrastructure
from src.backend.voice_assistant.modular_voice_assistant_abstractions_v2 import BasicVoiceAssistant, SpeechRecorder, Transcriber, Synthesizer

class VoiceAssistantInterface(object):
    """
    Voice assistant interface.
    """
    def __init__(self, working_directory: str = None) -> None:
        """
        Initiation method.
        :param working_directory: Working directory.
        """
        self.working_directory = working_directory | os.path.join(cfg.PATHS.DATA_PATH, "voice_assistant_interface")
        self.database = BasicSQLAlchemyInterface(
            working_directory=os.path.join(self.working_directory, "database"),
            population_function=populate_data_infrastructure)
        self.assistant: BasicVoiceAssistant | None = None

     
