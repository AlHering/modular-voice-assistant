# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
from sqlalchemy.orm import relationship, mapped_column, declarative_base
from sqlalchemy import Engine, Column, String, JSON, ForeignKey, Integer, DateTime, func, Uuid, Text, event, Boolean
from uuid import uuid4
from typing import Any


def populate_data_instrastructure(engine: Engine, schema: str, model: dict) -> None:
    """
    Function for populating data infrastructure.
    :param engine: Database engine.
    :param schema: Schema for tables.
    :param model: Model dictionary for holding data classes.
    """
    schema = str(schema)
    base = declarative_base()

    class Model(base):
        """
        Model class, representing a machine learning model.
        """
        __tablename__ = f"{schema}model"
        __table_args__ = {
            "comment": "Model table.", "extend_existing": True}

        id = Column(Integer, primary_key=True, unique=True, nullable=False, autoincrement=True,
                    comment="ID of the model.")
        path = Column(String,
                      comment="Path of the model folder.")
        name = Column(String,
                      comment="Name of the model.")
        task = Column(String,
                      comment="Task of the model.")
        subtask = Column(String,
                      comment="Subtask of the model.")
        architecture = Column(String,
                              comment="Architecture of the model.")
        url = Column(String, unique=True,
                     comment="URL for the model.")
        source = Column(String,
                        comment="Main metadata source for the model.")
        meta_data = Column(JSON,
                           comment="Metadata of the model.")
        
        created = Column(DateTime, server_default=func.now(),
                         comment="Timestamp of creation.")
        updated = Column(DateTime, server_default=func.now(), server_onupdate=func.now(),
                         comment="Timestamp of last update.")
        inactive = Column(Boolean, nullable=False, default=False,
                          comment="Inactivity flag.")

        modelversions = relationship(
            "Modelversion", back_populates="model", viewonly=True)
        instances = relationship(
            "Modelinstance", back_populates="model", viewonly=True)
        assets = relationship("Asset", back_populates="model", viewonly=True)

    class Modelversion(base):
        """
        Modelversion class, representing a machine learning model version.
        """
        __tablename__ = f"{schema}modelversion"
        __table_args__ = {
            "comment": "Model version table.", "extend_existing": True}

        id = Column(Integer, primary_key=True, unique=True, nullable=False, autoincrement=True,
                    comment="ID of the modelversion.")
        path = Column(String,
                      comment="Relative path of the modelversion.")
        name = Column(String,
                      comment="Name of the modelversion.")
        basemodel = Column(String,
                           comment="Basemodel of the modelversion.")
        type = Column(String,
                      comment="Type of the modelversion.")
        format = Column(String, nullable=False,
                        comment="Format of the modelversion.")
        url = Column(String, unique=True,
                     comment="URL for the modelversion.")
        source = Column(String,
                        comment="Main metadata source for the modelversion.")
        sha256 = Column(Text,
                        comment="SHA256 hash for the modelversion.")
        meta_data = Column(JSON,
                           comment="Metadata of the modelversion.")
        
        created = Column(DateTime, server_default=func.now(),
                         comment="Timestamp of creation.")
        updated = Column(DateTime, server_default=func.now(), server_onupdate=func.now(),
                         comment="Timestamp of last update.")
        inactive = Column(Boolean, nullable=False, default=False,
                          comment="Inactivity flag.")

        model_id = mapped_column(Integer, ForeignKey(f"{schema}model.id"))
        model = relationship(
            "Model", back_populates="modelversions")
        instances = relationship(
            "Modelinstance", back_populates="modelversion", viewonly=True)
        assets = relationship(
            "Asset", back_populates="modelversion", viewonly=True)

    class Asset(base):
        """
        Asset class, representing an asset, connected to a machine learning model or model version.
        """
        __tablename__ = f"{schema}asset"
        __table_args__ = {
            "comment": "Asset table.", "extend_existing": True}

        id = Column(Integer, primary_key=True, unique=True, nullable=False, autoincrement=True,
                    comment="ID of the asset.")
        path = Column(String, nullable=False,
                      comment="Path of the asset.")
        type = Column(String, nullable=False,
                      comment="Type of the asset.")
        url = Column(String,
                     comment="URL for the asset.")
        source = Column(String,
                        comment="Main metadata source for the asset.")
        sha256 = Column(Text,
                        comment="SHA256 hash for the asset.")
        meta_data = Column(JSON,
                           comment="Metadata of the asset.")
        
        created = Column(DateTime, server_default=func.now(),
                         comment="Timestamp of creation.")
        updated = Column(DateTime, server_default=func.now(), server_onupdate=func.now(),
                         comment="Timestamp of last update.")
        inactive = Column(Boolean, nullable=False, default=False,
                          comment="Inactivity flag.")

        model_id = mapped_column(Integer, ForeignKey(f"{schema}model.id"))
        model = relationship(
            "Model", back_populates="assets")
        modelversion_id = mapped_column(
            Integer, ForeignKey(f"{schema}modelversion.id"))
        modelversion = relationship(
            "Modelversion", back_populates="assets")

    class ScrapingFail(base):
        """
        ScrapingFail class, representing an scraping fail, connected to a machine learning model or model version.
        """
        __tablename__ = f"{schema}scraping_fail"
        __table_args__ = {
            "comment": "Scraping fail table.", "extend_existing": True}

        id = Column(Integer, primary_key=True, unique=True, nullable=False, autoincrement=True,
                    comment="ID of the scraping fail.")
        url = Column(String,
                     comment="URL for the scraping fail.")
        source = Column(String, nullable=False,
                        comment="Source, under which the scraping fail appeared.")
        fetched_data = Column(JSON,
                              comment="Fetched data.")
        normalized_data = Column(JSON,
                                 comment="Normalized data.")
        exception_data = Column(JSON,
                                comment="Exception data.")
        
        created = Column(DateTime, server_default=func.now(),
                         comment="Timestamp of creation.")
        inactive = Column(Boolean, nullable=False, default=False,
                          comment="Inactivity flag.")

    for dataclass in [Model, Modelversion, Asset, ScrapingFail]:
        model[dataclass.__tablename__.replace(schema, "")] = dataclass

    base.metadata.create_all(bind=engine)
