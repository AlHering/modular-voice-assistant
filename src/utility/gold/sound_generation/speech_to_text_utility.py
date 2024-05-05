# -*- coding: utf-8 -*-
"""
****************************************************
*                      Utility                 
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from typing import Any
import os
import pyaudio
import wave
import torch
import whisper
from faster_whisper import WhisperModel


TEMPORARY_DATA_FOLDER = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "data")
if not os.path.exists:
    os.makedirs(TEMPORARY_DATA_FOLDER)
TEMPORARY_INPUT_PATH = os.path.join(TEMPORARY_DATA_FOLDER, "in.wav")


def get_whisper_model(model_name_or_path: str, instantiation_kwargs: dict = None) -> whisper.Whisper:
    """
    Returns a whisper based model instance.
    :param model_name_or_path: Model name or path.
    :param instantiation_kwargs: Instatiation keyword arguments.
        Defaults to None in which case default values are used.
    :returns: Whisper model instance.
    """
    instantiation_kwargs = {} if instantiation_kwargs is None else instantiation_kwargs
    return whisper.load_model(
            name=model_name_or_path,
            **instantiation_kwargs)


