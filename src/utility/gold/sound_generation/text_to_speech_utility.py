# -*- coding: utf-8 -*-
"""
****************************************************
*                      Utility                 
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from typing import Any, Union, List, Tuple
import os
import pyaudio
import numpy as np
import wave
import torch
from TTS.api import TTS
from .sound_model_instantiation import load_coqui_tts_model


def synthesize_with_coqui_tts(text: str, 
                              model: TTS = None, 
                              synthesis_parameters: dict = None) -> Tuple[np.ndarray, dict]:
    """
    Synthesizes text with Coqui TTS and saves results to a file.
    :param text: Output text.
    :param model: TTS model. 
        Defaults to None in which case a default model is instantiated and used.
        Not providing a model therefore increases processing time tremendously!
    :param synthesis_parameters: Synthesis keyword arguments. 
        Defaults to None in which case default values are used.
    :returns: Synthesized audio and audio metadata which can be used as stream keyword arguments for outputting.
    """
    model = load_coqui_tts_model(TTS().list_models()[0]) if model is None else model
    synthesis_parameters = {} if synthesis_parameters is None else synthesis_parameters
    snythesized = model.tts(
        text=text,
        **synthesis_parameters)
    
    # Conversion taken from 
    # https://github.com/coqui-ai/TTS/blob/dev/TTS/utils/synthesizer.py and
    # https://github.com/coqui-ai/TTS/blob/dev/TTS/utils/audio/numpy_transforms.py
    if torch.is_tensor(snythesized):
        snythesized = snythesized.cpu().numpy()
    if isinstance(snythesized, list):
        snythesized = np.array(snythesized)

    snythesized = snythesized * (32767 / max(0.01, np.max(np.abs(snythesized))))
    snythesized = snythesized.astype(np.int16)
    return snythesized, {
        "rate": model.synthesizer.output_sample_rate,
        "format": pyaudio.paInt16,
        "channels": 1
    }


def synthesize_with_coqui_tts_to_file(text: str, output_path: str, model: TTS = None, synthesis_parameters: dict = None) -> str:
    """
    Synthesizes text with Coqui TTS and saves results to a file.
    :param text: Output text.
    :param output_path: Output path.
    :param model: TTS model. 
        Defaults to None in which case a default model is instantiated and used.
        Not providing a model therefore increases processing time tremendously!
    :param synthesis_parameters: Synthesis keyword arguments. 
        Defaults to None in which case default values are used.
    :returns: Output file path.
    """
    model = load_coqui_tts_model(TTS.list_models()[0]) if model is None else model
    synthesis_parameters = {} if synthesis_parameters is None else synthesis_parameters
    return model.tts_to_file(
        text=text,
        file_path=output_path,
        **synthesis_parameters)