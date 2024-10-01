# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from sqlalchemy.orm import relationship, mapped_column, declarative_base
from sqlalchemy import Engine, Column, String, JSON, ForeignKey, Integer, DateTime, func, Uuid, Text, event, Boolean, Float
from uuid import uuid4, UUID
from typing import Any


def populate_data_instrastructure(engine: Engine, schema: str, model: dict) -> None:
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
                           comment="Timestamp of request recieval.")
        responded = Column(DateTime, server_default=func.now(), server_onupdate=func.now(),
                           comment="Timestamp of reponse transmission.")
        
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
        model_parameters = Column(JSON,
                                  comment="Parameters for the model instantiation.")
        transcription_parameters = Column(JSON,
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
        model_parameters = Column(JSON,
                                  comment="Parameters for the model instantiation.")
        synthesis_parameters = Column(JSON,
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
        recognizer_parameters = Column(JSON,
                            comment="Parameters for setting up recognizer instances.")
        microphone_parameters = Column(JSON,
                                  comment="Parameters for setting up microphone instances.")
        loop_pause = Column(Float,
                            comment="Forced pause between recording loops in seconds.")

        created = Column(DateTime, server_default=func.now(),
                         comment="Timestamp of creation.")
        updated = Column(DateTime, server_default=func.now(), server_onupdate=func.now(),
                         comment="Timestamp of last update.")
        inactive = Column(Boolean, nullable=False, default=False,
                          comment="Inactivity flag.")
        

    for dataclass in [Log, Transcriber, Synthesizer, SpeechRecorder]:
        model[dataclass.__tablename__.replace(schema, "")] = dataclass

    base.metadata.create_all(bind=engine)