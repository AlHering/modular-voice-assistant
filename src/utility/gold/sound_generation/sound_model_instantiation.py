# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import os
import traceback
from typing import List, Tuple, Any, Callable, Optional, Union
from datetime import datetime as dt

import TTS.api


def load_whisper_model(model_path: str,
                       model_parameters: dict = {}) -> Any:
    """
    Function for loading whisper based model instance.
    :param model_path: Path to model files.
    :param model_parameters: Model loading kwargs as dictionary.
        Defaults to empty dictionary.
    :return: Model instance.
    """
    import whisper

    return whisper.load_model(
            name=model_path,
            **model_parameters)


def load_faster_whisper_model(model_path: str,
                              model_parameters: dict = {}) -> Any:
    """
    Function for loading faster whisper based model instance.
    :param model_path: Path to model files.
    :param model_parameters: Model loading kwargs as dictionary.
        Defaults to empty dictionary.
    :return: Model instance.
    """
    import faster_whisper

    return faster_whisper.WhisperModel(
        model_size_or_path=model_path,
        **model_parameters
    )


def load_coqui_tts_model(model_path: str,
                         model_parameters: dict = {}) -> Any:
    """
    Function for loading coqui TTS based model instance.
    :param model_path: Path to model files.
    :param model_parameters: Model loading kwargs as dictionary.
        Defaults to empty dictionary.
    :return: Model instance.
    """
    from TTS.api import TTS

    if os.path.exists(model_path):
        return TTS().load_tts_model_by_path(
            model_path=model_path,
            config_path=f"{model_path}/config.json",
            **model_parameters)
    else:
         return TTS(
              model_name=model_path,
              **model_parameters
         )