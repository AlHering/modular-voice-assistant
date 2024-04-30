# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
from sqlalchemy.orm import relationship, mapped_column, declarative_base
from sqlalchemy import Engine, Column, String, JSON, ForeignKey, Integer, DateTime, func, Uuid, Text, event, Boolean
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
    if schema and not schema.endswith("."):
        schema += "."
    base = declarative_base()

    class LMInstance(base):
        """
        Config class, representing a language model instance.
        """
        __tablename__ = f"{schema}lm_instance"
        __table_args__ = {
            "comment": "LM instance table.", "extend_existing": True}

        id = Column(Integer, primary_key=True, unique=True, nullable=False, autoincrement=True,
                    comment="ID of the model instance.")
        backend = Column(String, nullable=False,
                         comment="Backend of the model instance.")
        model_path = Column(String, nullable=False,
                            comment="Path of the model instance.")

        model_file = Column(String,
                            comment="File of the model instance.")
        model_parameters = Column(JSON,
                                  comment="Parameters for the model instantiation.")
        tokenizer_path = Column(String,
                                comment="Path of the tokenizer.")
        tokenizer_parameters = Column(JSON,
                                      comment="Parameters for the tokenizer instantiation.")
        config_path = Column(String,
                             comment="Path of the config.")
        config_parameters = Column(JSON,
                                   comment="Parameters for the config.")

        default_system_prompt = Column(String,
                                       comment="Default system prompt of the model instance.")
        use_history = Column(Boolean, default=True,
                             comment="Flag for declaring whether to use a history.")
        encoding_parameters = Column(JSON,
                                     comment="Parameters for prompt encoding.")
        generating_parameters = Column(JSON,
                                       comment="Parameters for the response generation.")
        decoding_parameters = Column(JSON,
                                     comment="Parameters for response decoding.")

        resource_requirements = Column(JSON,
                                       comment="Resource profile for validating requirements.")

        created = Column(DateTime, server_default=func.now(),
                         comment="Timestamp of creation.")
        updated = Column(DateTime, server_default=func.now(), server_onupdate=func.now(),
                         comment="Timestamp of last update.")
        inactive = Column(Boolean, nullable=False, default=False,
                          comment="Inactivity flag.")
        
    
    class KBInstance(base):
        """
        Config class, representing a knowledgebase instance.
        """
        __tablename__ = f"{schema}kb_instance"
        __table_args__ = {
            "comment": "KB instance table.", "extend_existing": True}

        id = Column(Integer, primary_key=True, unique=True, nullable=False, autoincrement=True,
                    comment="ID of the model instance.")
        backend = Column(String, nullable=False,
                         comment="Backend of the knowledgebase instance.")
        knowledgebase_path = Column(String, nullable=False,
                            comment="Path of the knowledgebase.")

        knowledgebase_parameters = Column(JSON,
                                  comment="Parameters for knowledgebase instantiation.")
        preprocessing_parameters = Column(JSON,
                                      comment="Parameters for preprocessing instantiation.")
        embedding_parameters = Column(JSON,
                                   comment="Parameters for embedding.")

        default_retrieval_method = Column(String,
                                       comment="Default retrieval method of the knowledgebase instance.")
        retrieval_parameters = Column(JSON,
                                     comment="Parameters for retrieval.")

        created = Column(DateTime, server_default=func.now(),
                         comment="Timestamp of creation.")
        updated = Column(DateTime, server_default=func.now(), server_onupdate=func.now(),
                         comment="Timestamp of last update.")
        inactive = Column(Boolean, nullable=False, default=False,
                          comment="Inactivity flag.")

    class ToolArgument(base):
        """
        Config class, representing a Tool Argument.
        """
        __tablename__ = f"{schema}tool_argument"
        __table_args__ = {
            "comment": "Tool Argument table.", "extend_existing": True}

        id = Column(Integer, primary_key=True, unique=True, nullable=False, autoincrement=True,
                    comment="ID of the Tool Argument.")
        name = Column(String, nullable=False,
                      comment="Name of the Tool Argument.")
        type = Column(String, nullable=False,
                      comment="Type of the Tool Argument.")
        description = Column(String,
                             comment="Description of the Tool Argument.")
        value = Column(String,
                       comment="Value of the Tool Argument.")

        created = Column(DateTime, server_default=func.now(),
                         comment="Timestamp of creation.")
        updated = Column(DateTime, server_default=func.now(), server_onupdate=func.now(),
                         comment="Timestamp of last update.")
        inactive = Column(Boolean, nullable=False, default=False,
                          comment="Inactivity flag.")

        agent_tool_id = mapped_column(
            Integer, ForeignKey(f"{schema}agent_tool.id"))
        agent_tool = relationship(
            "AgentTool", back_populates="tool_arguments")

    class AgentTool(base):
        """
        Config class, representing an Agent Tool.
        """
        __tablename__ = f"{schema}agent_tool"
        __table_args__ = {
            "comment": "Agent Tool table.", "extend_existing": True}

        id = Column(Integer, primary_key=True, unique=True, nullable=False, autoincrement=True,
                    comment="ID of the Agent Tool.")
        name = Column(String, nullable=False,
                      comment="Name of the Agent Tool.")
        description = Column(String, nullable=False,
                             comment="Description of the Agent Tool.")
        func = Column(String, nullable=False,
                      comment="Function of the Agent Tool.")
        return_type = Column(String, nullable=False,
                             comment="Return type of the Agent Tool.")

        created = Column(DateTime, server_default=func.now(),
                         comment="Timestamp of creation.")
        updated = Column(DateTime, server_default=func.now(), server_onupdate=func.now(),
                         comment="Timestamp of last update.")
        inactive = Column(Boolean, nullable=False, default=False,
                          comment="Inactivity flag.")

        tool_arguments = relationship(
            "ToolArgument", back_populates="agent_tool")

    class AgentMemory(base):
        """
        Config class, representing an Agent Memory.
        """
        __tablename__ = f"{schema}agent_memory"
        __table_args__ = {
            "comment": "Agent Memory table.", "extend_existing": True}

        id = Column(Integer, primary_key=True, unique=True, nullable=False, autoincrement=True,
                    comment="ID of the Agent Memory.")
        backend = Column(String, nullable=False,
                         comment="Cache of the Agent Memory.")
        path = Column(String, nullable=False,
                      comment="Path of the Agent Memory.")
        paramters = Column(JSON,
                           comment="Parameters for the Agent Memory instantiation.")

        created = Column(DateTime, server_default=func.now(),
                         comment="Timestamp of creation.")
        updated = Column(DateTime, server_default=func.now(), server_onupdate=func.now(),
                         comment="Timestamp of last update.")
        inactive = Column(Boolean, nullable=False, default=False,
                          comment="Inactivity flag.")

    class Agent(base):
        """
        Config class, representing an Agent.
        """
        __tablename__ = f"{schema}agent"
        __table_args__ = {
            "comment": "Agent table.", "extend_existing": True}

        id = Column(Integer, primary_key=True, unique=True, nullable=False, autoincrement=True,
                    comment="ID of the Agent.")

        name = Column(String, nullable=False,
                      comment="Name of the Agent.")
        description = Column(String, nullable=False,
                             comment="Description of the Agent.")

        memory_id = mapped_column(
            Integer, ForeignKey(f"{schema}agent_memory.id"))
        memory = relationship("AgentMemory", foreign_keys=[memory_id])

        general_lm_id = mapped_column(
            Integer, ForeignKey(f"{schema}lm_instance.id"), nullable=False)
        general_lm = relationship(
            "LMInstance", foreign_keys=[general_lm_id])

        dedicated_planner_id = mapped_column(
            Integer, ForeignKey(f"{schema}lm_instance.id"))
        dedicated_planner_lm = relationship(
            "LMInstance", foreign_keys=[general_lm_id])
        dedicated_actor_id = mapped_column(
            Integer, ForeignKey(f"{schema}lm_instance.id"))
        dedicated_actor_lm = relationship(
            "LMInstance", foreign_keys=[dedicated_actor_id])
        dedicated_oberserver_lm_id = mapped_column(
            Integer, ForeignKey(f"{schema}lm_instance.id"))
        dedicated_oberserver_lm = relationship(
            "LMInstance", foreign_keys=[dedicated_oberserver_lm_id])

        created = Column(DateTime, server_default=func.now(),
                         comment="Timestamp of creation.")
        updated = Column(DateTime, server_default=func.now(), server_onupdate=func.now(),
                         comment="Timestamp of last update.")
        inactive = Column(Boolean, nullable=False, default=False,
                          comment="Inactivity flag.")
        # TODO: Include AgentTools

    for dataclass in [LMInstance, KBInstance, ToolArgument, AgentTool, AgentMemory, Agent]:
        model[dataclass.__tablename__.replace(schema, "")] = dataclass

    base.metadata.create_all(bind=engine)
