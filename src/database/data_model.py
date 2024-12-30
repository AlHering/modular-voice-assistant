# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from sqlalchemy.orm import declarative_base
from sqlalchemy import Engine, Column, String, JSON, Integer, DateTime, func, Boolean, UUID
from uuid import uuid4
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
        
    class ServiceConfig(base):
        """
        Config class, representing a pipeline service config.
        """
        __tablename__ = f"{schema}service_config"
        __table_args__ = {
            "comment": "Pipeline service config table.", "extend_existing": True}

        id = Column(UUID(as_uuid=True), primary_key=True, unique=True, nullable=False, default=uuid4,
                    comment="ID of an instance.")
        module_type = Column(String,
                         comment="Service type.")
        config = Column(JSON,
                         comment="Service config.")
        validated = Column(Boolean, default=False,
                         comment="Validation flag.")

        created = Column(DateTime, server_default=func.now(),
                         comment="Timestamp of creation.")
        updated = Column(DateTime, server_default=func.now(), server_onupdate=func.now(),
                         comment="Timestamp of last update.")
        inactive = Column(Boolean, nullable=False, default=False,
                          comment="Inactivity flag.")
        
    class Model(base):
        """
        Model class, representing a machine learning model.
        """
        __tablename__ = f"{schema}model"
        __table_args__ = {
            "comment": "Pipeline module config table.", "extend_existing": True}

        id = Column(Integer, autoincrement=True, primary_key=True, unique=True, nullable=False, 
                    comment="ID of a model.")
        module_type = Column(String,
                         comment="Target module type.")
        backend = Column(String,
                      comment="Model backend.")
        name = Column(String,
                      comment="Model name.")
        info = Column(String,
                      comment="Info link.")
        size = Column(String,
                      comment="Model size.")
        path = Column(String,
                      comment="Model path.")

        created = Column(DateTime, server_default=func.now(),
                         comment="Timestamp of creation.")
        updated = Column(DateTime, server_default=func.now(), server_onupdate=func.now(),
                         comment="Timestamp of last update.")
        inactive = Column(Boolean, nullable=False, default=False,
                          comment="Inactivity flag.")
        

    for dataclass in [Log, ServiceConfig, Model]:
        model[dataclass.__tablename__.replace(schema, "")] = dataclass

    base.metadata.create_all(bind=engine)


def get_default_entries() -> dict:
    """
    Returns default entries.
    :return: Default entries.
    """
    return {
        "service_config": [
            {
                "module_type": "speech_recorder",
                "config": {}
            },
            {
                "module_type": "transcriber",
                "config": cfg.DEFAULT_TRANSCRIBER
            },
            {
                "module_type": "local_chat",
                "config": cfg.DEFAULT_LOCAL_CHAT
            },
            {
                "module_type": "remote_chat",
                "config": cfg.DEFAULT_REMOTE_CHAT
            },
            {
                "module_type": "synthesizer",
                "config": cfg.DEFAULT_SYNTHESIZER
            },
            {
                "module_type": "audio_player",
                "config": cfg.DEFAULT_AUDIO_PLAYER
            },
            {
                "module_type": "voice_assistant",
                "config": cfg.DEFAULT_VOICE_ASSISTANT
            },
        ]
    }