# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import os
from sqlalchemy.orm import relationship, mapped_column, declarative_base
from sqlalchemy import Engine, Column, String, JSON, ForeignKey, Integer, DateTime, func, Uuid, Text, event, Boolean
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
    if not schema.endswith("."):
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
        
    class Config(base):
        """
        Config class, representing a configuration.
        """
        __tablename__ = f"{schema}config"
        __table_args__ = {
            "comment": "Config table.", "extend_existing": True}

        id = Column(Integer, autoincrement=True, primary_key=True, unique=True, nullable=False,
                    comment="ID of an instance.")
        config_type = Column(String, nullable=False,
                    comment="Target object / type of the configuration.")
        config = Column(JSON, nullable=False,
                        comment="Configuration content.")
        
        created = Column(DateTime, server_default=func.now(),
                        comment="Timestamp of creation.")
        updated = Column(DateTime, server_default=func.now(), onupdate=func.now(),
                        comment="Timestamp of last update.")
        inactive = Column(Boolean, nullable=False, default=False,
                        comment="Inactivity flag.")

    for dataclass in [Log, Config]:
        model[dataclass.__tablename__.replace(schema, "")] = dataclass

    base.metadata.create_all(bind=engine)


def get_default_entries() -> dict:
    """
    Returns default entries.
    :return: Default entries.
    """
    transcriber_config = {
        "id": 1,
        "config_type": "transcriber",
        "config": {
            "backend": "faster-whisper",
            "model_path": os.path.join(cfg.PATHS.MODEL_PATH, 
                                    "sound_generation_models/speech_to_text/faster_whisper_models/Systran_faster-whisper-tiny"),
            "model_parameters": {
                "device": "cuda",
                "compute_type": "float32",
                "local_files_only": True
            }
        }
    }

    fallback_synthesizer_model = os.path.join(cfg.PATHS.MODEL_PATH, 
                                              "sound_generation_models/text_to_speech/coqui_models/tts_models-multilingual-multi-dataset-xtts_v2")
    fallback_speaker_wav = os.path.join(cfg.PATHS.MODEL_PATH, "sound_generation_models//text_to_speech/coqui_xtts/examples/female.wav")
    synthesizer_config = {
        "id": 2,
        "config_type": "synthesizer",
        "config": {
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
    }

    chat_model_config = {
        "id": 3,
        "config_type": "chat_model",
        "config": {
            "model_path": os.path.join(cfg.PATHS.MODEL_PATH, 
                                        "text_generation_models/mradermacher_Llama-3.1-Storm-8B-i1-GGUF"),
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
    }

    return {"Config": [transcriber_config, synthesizer_config, chat_model_config]}