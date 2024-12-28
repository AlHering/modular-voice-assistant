# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import os
from sqlalchemy.orm import relationship, mapped_column, declarative_base
from sqlalchemy import Engine, Column, String, JSON, Float, ForeignKey, Integer, DateTime, func, Uuid, Text, event, Boolean, UUID
from uuid import uuid4
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
        
    class ModuleConfig(base):
        """
        Config class, representing a VA module config.
        """
        __tablename__ = f"{schema}module_config"
        __table_args__ = {
            "comment": "VA module config table.", "extend_existing": True}

        id = Column(UUID(as_uuid=True), primary_key=True, unique=True, nullable=False, default=uuid4,
                    comment="ID of an instance.")
        module = Column(String,
                         comment="Target module type.")
        config = Column(JSON,
                         comment="Module config.")
        validated = Column(Boolean, default=False,
                         comment="Validation flag.")

        created = Column(DateTime, server_default=func.now(),
                         comment="Timestamp of creation.")
        updated = Column(DateTime, server_default=func.now(), server_onupdate=func.now(),
                         comment="Timestamp of last update.")
        inactive = Column(Boolean, nullable=False, default=False,
                          comment="Inactivity flag.")
        

    for dataclass in [Log, ModuleConfig]:
        model[dataclass.__tablename__.replace(schema, "")] = dataclass

    base.metadata.create_all(bind=engine)


def get_default_entries() -> dict:
    """
    Returns default entries.
    :return: Default entries.
    """
    return {
        "module_config": [
            {
                "module": "speech_recorder",
                "config": {}
            },
            {
                "module": "transcriber",
                "config": cfg.DEFAULT_TRANSCRIBER
            },
            {
                "module": "local_chat",
                "config": cfg.DEFAULT_LOCAL_CHAT
            },
            {
                "module": "remote_chat",
                "config": cfg.DEFAULT_REMOTE_CHAT
            },
            {
                "module": "synthesizer",
                "config": cfg.DEFAULT_SYNTHESIZER
            },
            {
                "module": "wave_output",
                "config": cfg.DEFAULT_WAVE_OUTPUT
            },
            {
                "module": "voice_assistant",
                "config": cfg.DEFAULT_VOICE_ASSISTANT
            },
        ]
    }