# -*- coding: utf-8 -*-
"""
****************************************************
*                      Utility                 
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import os
from typing import Any, Union, Tuple, List
import numpy as np
import torch
import faster_whisper


def download_faster_whisper_model(model_id: str,
                                  output_folder: str) -> None:
    """
    Function for downloading faster whisper models.
    :param model_id: Target model ID.
    :param output_folder: Output folder path.
    """
    faster_whisper.download_model(model_id, output_dir=output_folder)


def load_faster_whisper_model(model_path: str,
                              model_parameters: dict = {}) -> faster_whisper.WhisperModel:
    """
    Function for loading faster whisper based model instance.
    :param model_path: Path to model files.
    :param model_parameters: Model loading kwargs as dictionary.
        Defaults to empty dictionary.
    :return: Model instance.
    """
    return faster_whisper.WhisperModel(
        model_size_or_path=model_path,
        **model_parameters
    )


def normalize_audio_for_whisper(audio_input: Union[str, np.ndarray, torch.Tensor]) -> Union[str, np.ndarray, torch.Tensor]:
    """
    Function for normalizing audio data before transcribing with whisper or faster-whisper.
    :param audio_input: Wave file path or waveform.
    :param return: Normalized audio data.
    """
    if isinstance(audio_input, np.ndarray) and str(audio_input.dtype) not in ["float16", "float32"]:
        return np.frombuffer(audio_input, audio_input.dtype).flatten().astype(np.float32) / {
                    "int8": 128.0,
                    "int16": 32768.0,
                    "int32": 2147483648.0,
                    "int64": 9223372036854775808.0
                    }[str(audio_input.dtype)] 
    else:
        return audio_input


def transcribe(audio_input: Union[str, np.ndarray, torch.Tensor], model: faster_whisper.WhisperModel = None, transcription_parameters: dict | None = None) -> Tuple[str, dict]:
    """
    Transcribes wave file or waveform with faster whisper.
    :param audio_input: Wave file path or waveform.
    :param model: Faster whisper model. 
        Defaults to None in which case a default model is instantiated and used.
        Not providing a model therefore increases processing time tremendously!
    :param transcription_parameters: Transcription keyword arguments. 
        Defaults to None in which case default values are used.
    :returns: Tuple of transcribed text and a list of metadata entries for the transcribed segments.
    """
    model = load_faster_whisper_model(model_name_or_path="large-v3") if model is None else model
    audio_input = normalize_audio_for_whisper(audio_input)
    transcription_parameters = {} if transcription_parameters is None else transcription_parameters
    
    transcription, metadata = model.transcribe(
        audio=audio_input,
        **transcription_parameters
    )
    metadata = metadata._asdict()
    segment_metadatas = [segment._asdict() for segment in transcription]
    for segment in segment_metadatas:
        segment.update(metadata)
    fulltext = " ".join([segment["text"].strip() for segment in segment_metadatas])
    
    return fulltext, {"segments": segment_metadatas}