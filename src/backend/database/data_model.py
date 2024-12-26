# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import os
from sqlalchemy.orm import relationship, mapped_column, declarative_base
from sqlalchemy import Engine, Column, String, JSON, Float, ForeignKey, Integer, DateTime, func, Uuid, Text, event, Boolean
from uuid import uuid4, UUID
from typing import Any
from src.configuration import configuration as cfg


def populate_data_infrastructure(engine: Engine, schema: str, model: dict) -> None:
    """
    Function for populating data infrastructure.
    :param engine: Database engine.
    :param schema: Schema for tables.
    :param model: Model dictionary for holding data classes.
    """
    schema = str(schema)
    if schema and not schema.endswith("."):
        schema += "."
    base = declarative_base()

    class Log(base):
        """
        Log class, representing an log entry, connected to a backend interaction.
        """
        __tablename__ = f"{schema}log"
        __table_args__ = {
            "comment": "Log table.", "extend_existing": True}

        id = Column(Integer, primary_key=True, autoincrement=True, unique=True, nullable=False,
                    comment="ID of the logging entry.")
        request = Column(JSON, nullable=False,
                         comment="Request, sent to the backend.")
        response = Column(JSON, comment="Response, given by the backend.")
        requested = Column(DateTime, server_default=func.now(),
                           comment="Timestamp of request receive.")
        responded = Column(DateTime, server_default=func.now(), server_onupdate=func.now(),
                           comment="Timestamp of response transmission.")
        
    class Transcriber(base):
        """
        Config class, representing a transcriber instance.
        """
        __tablename__ = f"{schema}transcriber"
        __table_args__ = {
            "comment": "Transcriber instance table.", "extend_existing": True}

        id = Column(Integer, primary_key=True, unique=True, nullable=False, autoincrement=True,
                    comment="ID of the model instance.")
        backend = Column(String, nullable=False,
                         comment="Backend for model instantiation.")
        model_path = Column(String, nullable=False,
                            comment="Path of the model folder.")
        model_parameters = Column(JSON, default={},
                                  comment="Parameters for the model instantiation.")
        transcription_parameters = Column(JSON, default={},
                                comment="Parameters for transcribing.")

        created = Column(DateTime, server_default=func.now(),
                         comment="Timestamp of creation.")
        updated = Column(DateTime, server_default=func.now(), server_onupdate=func.now(),
                         comment="Timestamp of last update.")
        inactive = Column(Boolean, nullable=False, default=False,
                          comment="Inactivity flag.")
    
    class Synthesizer(base):
        """
        Config class, representing a synthesizer instance.
        """
        __tablename__ = f"{schema}synthesizer"
        __table_args__ = {
            "comment": "Synthesizer instance table.", "extend_existing": True}

        id = Column(Integer, primary_key=True, unique=True, nullable=False, autoincrement=True,
                    comment="ID of the model instance.")
        backend = Column(String, nullable=False,
                         comment="Backend for model instantiation.")
        model_path = Column(String, nullable=False,
                            comment="Path of the model folder.")
        model_parameters = Column(JSON, default={},
                                  comment="Parameters for the model instantiation.")
        synthesis_parameters = Column(JSON, default={},
                                comment="Parameters for synthesizing.")

        created = Column(DateTime, server_default=func.now(),
                         comment="Timestamp of creation.")
        updated = Column(DateTime, server_default=func.now(), server_onupdate=func.now(),
                         comment="Timestamp of last update.")
        inactive = Column(Boolean, nullable=False, default=False,
                          comment="Inactivity flag.")
        
    class SpeechRecorder(base):
        """
        Config class, representing a speech recorder instance.
        """
        __tablename__ = f"{schema}speech_recorder"
        __table_args__ = {
            "comment": "Speech recorder instance table.", "extend_existing": True}

        id = Column(Integer, primary_key=True, unique=True, nullable=False, autoincrement=True,
                    comment="ID of the speech recorder instance.")
        input_device_index = Column(Integer,
                         comment="Input device index for recording.")
        recognizer_parameters = Column(JSON, default={},
                            comment="Parameters for setting up recognizer instances.")
        microphone_parameters = Column(JSON, default={},
                                  comment="Parameters for setting up microphone instances.")
        loop_pause = Column(Float,
                            comment="Forced pause between recording loops in seconds.")

        created = Column(DateTime, server_default=func.now(),
                         comment="Timestamp of creation.")
        updated = Column(DateTime, server_default=func.now(), server_onupdate=func.now(),
                         comment="Timestamp of last update.")
        inactive = Column(Boolean, nullable=False, default=False,
                          comment="Inactivity flag.")
        
    class ChatModel(base):
        """
        Config class, representing a chat model instance.
        """
        __tablename__ = f"{schema}chat_model"
        __table_args__ = {
            "comment": "Chat model instance table.", "extend_existing": True}

        id = Column(Integer, primary_key=True, unique=True, nullable=False, autoincrement=True,
                    comment="ID of the chat model instance.")
        language_model_config = Column(JSON, 
                            comment="Language model config.")
        chat_parameters = Column(JSON, default={},
                            comment="Chat parameters.")
        system_prompt = Column(String,
                               comment="System prompt.")
        system_prompt = Column(Boolean, default=True,
                               comment="Flag for using history.")
        history = Column(JSON, default={"history": []},
                         comment="Chat parameters.")

        created = Column(DateTime, server_default=func.now(),
                         comment="Timestamp of creation.")
        updated = Column(DateTime, server_default=func.now(), server_onupdate=func.now(),
                         comment="Timestamp of last update.")
        inactive = Column(Boolean, nullable=False, default=False,
                          comment="Inactivity flag.")
        
    class RemoteChatModel(base):
        """
        Config class, representing a remote chat model instance.
        """
        __tablename__ = f"{schema}chat_model"
        __table_args__ = {
            "comment": "Remote chat model instance table.", "extend_existing": True}

        id = Column(Integer, primary_key=True, unique=True, nullable=False, autoincrement=True,
                    comment="ID of the chat model instance.")
        api_base = Column(String,
                         comment="API base.")
        api_token = Column(String,
                         comment="API token.")
        chat_parameters = Column(JSON, default={},
                            comment="Chat parameters.")
        system_prompt = Column(String,
                               comment="System prompt.")
        system_prompt = Column(Boolean, default=True,
                               comment="Flag for using history.")
        history = Column(JSON, default={"history": []},
                         comment="Chat parameters.")

        created = Column(DateTime, server_default=func.now(),
                         comment="Timestamp of creation.")
        updated = Column(DateTime, server_default=func.now(), server_onupdate=func.now(),
                         comment="Timestamp of last update.")
        inactive = Column(Boolean, nullable=False, default=False,
                          comment="Inactivity flag.")
        

    for dataclass in [Log, Transcriber, Synthesizer, SpeechRecorder, ChatModel, RemoteChatModel]:
        model[dataclass.__tablename__.replace(schema, "")] = dataclass

    base.metadata.create_all(bind=engine)


def get_default_entries() -> dict:
    """
    Returns default entries.
    :return: Default entries.
    """
    transcriber_config = {
        "id": 1,
        "backend": "faster-whisper",
        "model_path": os.path.join(cfg.PATHS.MODEL_PATH, 
                                "sound_generation/models/speech_to_text/faster_whisper_models/Systran_faster-whisper-tiny"),
        "model_parameters": {
            "device": "cuda",
            "compute_type": "float32",
            "local_files_only": True
        }
    }

    fallback_synthesizer_model = os.path.join(cfg.PATHS.MODEL_PATH, 
                                              "sound_generation/models/text_to_speech/coqui_models/tts_models-multilingual-multi-dataset-xtts_v2")
    fallback_speaker_wav = os.path.join(cfg.PATHS.MODEL_PATH, "sound_generation/models/text_to_speech/coqui_xtts/examples/female.wav")
    synthesizer_config = {
        "id": 1,
        "backend": "coqui-tts",
        "model_path": fallback_synthesizer_model,
        "model_parameters": {
            "config_path": f"{fallback_synthesizer_model}/config.json",
            "gpu": True
        },
        "synthesis_parameters": {
            "speaker_wav": fallback_speaker_wav,
            "language": "en"
        }
    }

    chat_model_config = {
        "id": 1,
        "model_path": os.path.join(cfg.PATHS.MODEL_PATH, 
                                        "text_generation/models/mradermacher_Llama-3.1-Storm-8B-i1-GGUF"),
        "model_file": "Llama-3.1-Storm-8B.i1-Q4_K_M.gguf",
        "model_parameters": {
            "n_ctx": 4096, 
            "temperature": 0.8, 
            "repetition_penalty": 1.6,
            "n_gpu_layers": 33
        },
        "generating_parameters": {
            "max_tokens": 256
        }
    }

    return {"transcriber": [transcriber_config],
            "synthesizer": [synthesizer_config], 
            "chat_model": [chat_model_config]}