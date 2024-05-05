# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os
from dotenv import dotenv_values
from . import paths as PATHS


"""
Environment file
"""
ENV_PATH = os.path.join(PATHS.PACKAGE_PATH, ".env")
ENV = dotenv_values(ENV_PATH) if os.path.exists(ENV_PATH) else {}


"""
Logger
"""


class LOGGER_REPLACEMENT(object):
    """
    Logger replacement class.
    """

    def debug(self, text: str) -> None:
        """
        Method replacement for logging.
        :param text: Text to log.
        """
        print(f"[DEBUG] {text}")

    def info(self, text: str) -> None:
        """
        Method replacement for logging.
        :param text: Text to log.
        """
        print(f"[INFO] {text}")

    def warning(self, text: str) -> None:
        """
        Method replacement for logging.
        :param text: Text to log.
        """
        print(f"[WARNING] {text}")

    def warn(self, text: str) -> None:
        """
        Method replacement for logging.
        :param text: Text to log.
        """
        print(f"[WARNING] {text}")


LOGGER = LOGGER_REPLACEMENT()


"""
Project information
"""
PROJECT_NAME = "Modular voice assistant"
PROJECT_DESCRIPTION = "A modular voice assistant for general purposes."
PROJECT_VERSION = "v0.0.1"


"""
Network addresses
"""
BACKEND_HOST = ENV.get("BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = ENV.get("BACKEND_PORT", "7861")
BACKEND_TITLE =  PROJECT_NAME
BACKEND_DESCRIPTION = PROJECT_DESCRIPTION
BACKEND_VERSION = PROJECT_VERSION
BACKEND_ENDPOINT_BASE = "/api/v1"
FRONTEND_HOST = ENV.get("FRONTEND_HOST", "127.0.0.1")
FRONTEND_PORT = ENV.get("FRONTEND_PORT", "8868")


"""
Others
"""
FILE_UPLOAD_CHUNK_SIZE = 1024*1024
FILE_OUTPUT_CHUNK_SIZE = 1024