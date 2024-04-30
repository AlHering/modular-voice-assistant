# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os
from time import sleep
from datetime import datetime as dt
from typing import Optional, Any, List, Dict, Union
from src.configuration import configuration as cfg
from src.utility.gold.basic_sqlalchemy_interface import BasicSQLAlchemyInterface
from src.model.backend_control.data_model import populate_data_instrastructure
from src.control.text_generation_controller import TextGenerationController



class BackendController(BasicSQLAlchemyInterface):
    """
    Controller class for handling backend interface requests.
    """
    
    def __init__(self, working_directory: str = None, database_uri: str = None) -> None:
        """
        Initiation method.
        :param working_directory: Working directory.
            Defaults to folder 'processes' folder under standard backend data path.
        :param database_uri: Database URI.
            Defaults to 'backend.db' file under configured backend folder.
        """
        # Main instance variables
        self._logger = cfg.LOGGER
        self.working_directory = cfg.PATHS.BACKEND_PATH if working_directory is None else working_directory
        if not os.path.exists(self.working_directory):
            os.makedirs(self.working_directory)
        self.database_uri = f"sqlite:///{os.path.join(cfg.PATHS.BACKEND_PATH, 'backend.db')}" if database_uri is None else database_uri

        # Database infrastructure
        super().__init__(self.working_directory, self.database_uri,
                         populate_data_instrastructure, "backend.", self._logger)
        
        # Orchestrated controllers
        self.controllers = {
            "text_generation": TextGenerationController(
                working_directory=self.working_directory,
                database_uri=self.database_uri
            )
        }

        
    """
    Setup and shutdown methods
    """
    def setup(self) -> None:
        """
        Method for running setup process.
        """
        for controller_instance in self.controllers.values():
            controller_instance.setup()


    def shutdown(self) -> None:
        """
        Method for running shutdown process.
        """
        for controller_instance in self.controllers.values():
            controller_instance.shutdown()


    """
    Base interaction
    """
    def log(self, log_data: dict) -> None:
        """
        Method for adding a log entry.
        :param log_data: Log entry data.
        """
        self.post_object("log", **log_data)

    """
    Orchestration interaction
    """


