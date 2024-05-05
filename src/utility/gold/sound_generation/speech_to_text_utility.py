# -*- coding: utf-8 -*-
"""
****************************************************
*                      Utility                 
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from typing import Any, Union, Tuple, List
import os
import pyaudio
import wave
import numpy as np
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


def transcribe_with_whisper(audio_input: Union[str, np.ndarray, torch.Tensor] = None, model: whisper.Whisper = None, transcription_kwargs: dict = None) -> Tuple[str, List[dict]]:
    """
    Transcribes wave file or waveform with whisper.
    :param audio_input: Wave file path or waveform.
    :param output_path: Output path.
        Defaults to None in which case the file "out.wav" under the temporary data folder is used.
    :param model: Whisper model. 
        Defaults to None in which case a default model is instantiated and used.
        Not providing a model therefore increases processing time tremendously!
    :param transcription_kwargs: Transcription keyword arguments. 
        Defaults to None in which case default values are used.
    :returns: Tuple of transcribed text and a list of metadata entries for the transcribed segments.
    """
    model = get_whisper_model(model_name_or_path="large-v3") if model is None else model
    audio_input = TEMPORARY_INPUT_PATH if audio_input is None else audio_input
    transcription_kwargs = {} if transcription_kwargs is None else transcription_kwargs
    
    transcription = model.transcribe(
        audio=audio_input,
        **transcription_kwargs
    )
    segment_metadatas = transcription["segments"]
    fulltext = transcription["text"]
    fulltext, segment_metadatas


def get_faster_whisper_model(model_name_or_path: str, instantiation_kwargs: dict = None) -> WhisperModel:
    """
    Returns a faster whisper based model instance.
    :param model_name_or_path: Model name or path.
    :param instantiation_kwargs: Instatiation keyword arguments.
        Defaults to None in which case default values are used.
    :returns: Whisper model instance.
    """
    instantiation_kwargs = {} if instantiation_kwargs is None else instantiation_kwargs
    return WhisperModel(
        model_size_or_path=model_name_or_path,
        **instantiation_kwargs
    )
