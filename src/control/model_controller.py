# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os
from typing import List, Any, Dict
from src.configuration import configuration as cfg
from src.model.model_control.model_handler import GenericModelHandler
from src.model.model_control.data_model import populate_data_instrastructure
from src.model.model_control.api_wrapper import CivitaiAPIWrapper, HuggingfaceAPIWrapper
from src.utility.gold.basic_sqlalchemy_interface import BasicSQLAlchemyInterface


class ModelController(BasicSQLAlchemyInterface):
    """
    Class, representing ML Model Controller objects.
    A Model Controller manages Model Handlers, which use API Wrappers to provide
    model services for collecting metadata and downloading assets.
    """
    def __init__(self, working_directory: str = None, database_uri: str = None, handlers: Dict[str, GenericModelHandler] = None) -> None:
        """
        Initiation method.
        :param working_directory: Working directory.
            Defaults to folder 'processes' folder under standard backend data path.
        :param database_uri: Database URI.
            Defaults to 'backend.db' file under default data path.
        :param handlers: Handler dictionary.
            Defaults to None in which case only default handlers are initialized.
        """
        self._logger = cfg.LOGGER
        self.working_directory = cfg.PATHS.BACKEND_PATH if working_directory is None else working_directory
        if not os.path.exists(self.working_directory):
            os.makedirs(self.working_directory)
        self.database_uri = f"sqlite:///{os.path.join(cfg.PATHS.DATA_PATH, 'backend.db')}" if database_uri is None else database_uri

        # Database infrastructure
        super().__init__(self.working_directory, self.database_uri,
                         populate_data_instrastructure, "model_control.", self._logger)
        self.base = None
        self.engine = None
        self.model = None
        self.schema = None
        self.session_factory = None
        self.primary_keys = None
        self._setup_database()

        self.model_folder = None
        self.cache_path = None
        self.handlers = {} if handlers is None else handlers
        self._initiate_base_handlers()

    def _initiate_base_handlers(self) -> None:
        """
        Method for initiating base handlers.
        """
        for init_kwargs in [
            {
                "database": self,
                "model_folder": self.model_folder,
                "cache_path": self.cache_path,
                "apis": {"civitai": CivitaiAPIWrapper()},
                "sorting_field": "type",
                "sorters": ["BLIP", "BSRGAN", "CHECKPOINTS", "CODEFORMER",
                            "CONTROL_NET", "DEEPBOORU", "EMBEDDINGS", "ESRGAN",
                            "GFPGAN", "HYPERNETWORKS", "KARLO", "LDSR", "LORA",
                            "LYCORIS", "POSES", "REAL_ESRGAN", "SCUNET",
                            "STABLE_DIFFUSION", "SWINIR", "TEXTUAL_INVERSION",
                            "TORCH_DEEPDANBOORU", "VAE", "WILDCARDS"]
            },
            {
                "database": self,
                "model_folder": self.model_folder,
                "cache_path": self.cache_path,
                "apis": {"huggingface": HuggingfaceAPIWrapper()},
                "sorting_field": "task",
                "sorters": ["TEXT_GENERATION", "EMBEDDING", "LORA", "TEXT_CLASSIFICATION"]
            }
        ]:
            self.handlers.append(GenericModelHandler(**init_kwargs))

    def add_handler(self, handler: GenericModelHandler) -> None:
        """
        Method for adding handler.
        :param handler_class: Handler object.
        """
        self.handlers.append(handler)
