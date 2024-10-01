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
TEXT_GENERATION_BACKEND_HOST = ENV.get("TEXT_GENERATION_BACKEND_HOST", "127.0.0.1")
TEXT_GENERATION_BACKEND_PORT = ENV.get("TEXT_GENERATION_BACKEND_PORT", "7861")
TEXT_GENERATION_BACKEND_TITLE =  "Text Generation Backend"
TEXT_GENERATION_BACKEND_DESCRIPTION = "Backend interface for text generation services."
TEXT_GENERATION_BACKEND_VERSION = PROJECT_VERSION
TEXT_GENERATION_BACKEND_ENDPOINT_BASE = "/api/v1"
VOICE_ASSISTANT_BACKEND_HOST = ENV.get("VOICE_ASSISTANT_BACKEND_HOST", "127.0.0.1")
VOICE_ASSISTANT_BACKEND_PORT = ENV.get("VOICE_ASSISTANT_BACKEND_PORT", "7862")
VOICE_ASSISTANT_BACKEND_TITLE =  "Voice Assistant Backend"
VOICE_ASSISTANT_BACKEND_DESCRIPTION = "Backend interface for voice assistant services."
VOICE_ASSISTANT_BACKEND_VERSION = PROJECT_VERSION
VOICE_ASSISTANT_BACKEND_ENDPOINT_BASE = "/api/v1"

FRONTEND_HOST = ENV.get("FRONTEND_HOST", "127.0.0.1")
FRONTEND_PORT = ENV.get("FRONTEND_PORT", "8868")


"""
Others
"""
FILE_UPLOAD_CHUNK_SIZE = 1024*1024
FILE_OUTPUT_CHUNK_SIZE = 1024