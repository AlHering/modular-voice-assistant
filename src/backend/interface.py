# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import os
from fastapi import APIRouter
from src.configuration import configuration as cfg
from src.backend.database.basic_sqlalchemy_interface import BasicSQLAlchemyInterface, FilterMask
from src.backend.database.data_model import populate_data_infrastructure
from src.backend.voice_assistant.modular_voice_assistant_abstractions_v2 import BasicVoiceAssistant, SpeechRecorder, Transcriber, Synthesizer, ChatModelInstance, RemoteChatModelInstance

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
        
        self.current_assistant_config = {
            "speech_recorder": None,
            "transcriber": None,
            "synthesizer": None,
            "chat_model": None,
        }
        self.assistant: BasicVoiceAssistant | None = None

    def setup_assistant(self, 
                        speech_recorder_id: int| None = None,
                        transcriber_id: int| None = None,
                        synthesizer_id: int| None = None,
                        chat_model_id: int| None = None) -> dict:
        """
        Sets up a voice assistant.
        :param speech_recorder_id: Config ID for the speech recorder.
        :param transcriber_id: Config ID for the transcriber.
        :param synthesizer_id: Config ID for the synthesizer.
        :param chat_model_id: Config ID for the chat model.
        """
        speech_recorder_configs = self.database.get_objects_by_filtermasks(
                object_type="config", 
                filtermasks=FilterMask(expressions=[["type", "==", "speech_recorder"]] if speech_recorder_id is None else 
                        [["type", "==", "speech_recorder", ["id", "==", speech_recorder_id]]])
            )
        transcriber_configs = self.database.get_objects_by_filtermasks(
            object_type="config", 
            filtermasks=FilterMask(expressions=[["type", "==", "transcriber"]] if transcriber_id is None else 
                    [["type", "==", "transcriber", ["id", "==", transcriber_id]]])
        )
        synthesizer_configs = self.database.get_objects_by_filtermasks(
            object_type="config", 
            filtermasks=FilterMask(expressions=[["type", "==", "synthesizer"]] if synthesizer_id is None else 
                    [["type", "==", "synthesizer", ["id", "==", synthesizer_id]]])
        )
        chat_model_configs = self.database.get_objects_by_filtermasks(
            object_type="config", 
            filtermasks=FilterMask(expressions=[["type", "==", "chat_model"]] if chat_model_id is None else 
                    [["type", "==", "chat_model", ["id", "==", chat_model_id]]])
        )

        speech_recorder_config = speech_recorder_configs[0]
        transcriber_config = transcriber_configs[0]
        synthesizer_config = synthesizer_configs[0]
        chat_model_config = chat_model_configs[0]
        
        if self.assistant is None:
            self.assistant = BasicVoiceAssistant(
                working_directory=os.path.join(self.working_directory, "voice_assistant"),
                speech_recorder=SpeechRecorder(**speech_recorder_config.config),
                transcriber=Transcriber(**transcriber_config.config),
                synthesizer=Synthesizer(**synthesizer_config.config),
                chat_model=ChatModelInstance(**chat_model_config.config) if chat_model_config.mode == "local"
                    else RemoteChatModelInstance(**chat_model_config.config)
            )
        else:
            if speech_recorder_id and self.current_assistant_config["speech_recorder"] != speech_recorder_id:
                self.assistant.speech_recorder = SpeechRecorder(**speech_recorder_config.config)
            if transcriber_id and self.current_assistant_config["transcriber"] != transcriber_id:
                self.assistant.speech_recorder = SpeechRecorder(**speech_recorder_config.config)
            if speech_recorder_id and self.current_assistant_config["speech_recorder"] != speech_recorder_id:
                self.assistant.speech_recorder = SpeechRecorder(**speech_recorder_config.config)
            if speech_recorder_id and self.current_assistant_config["speech_recorder"] != speech_recorder_id:
                self.assistant.speech_recorder = SpeechRecorder(**speech_recorder_config.config)
            
        self.current_assistant_config = {
            "speech_recorder": speech_recorder_config.id,
            "transcriber": transcriber_config.id,
            "synthesizer": synthesizer_config.id,
            "chat_model": chat_model_config.id,
        }
            

