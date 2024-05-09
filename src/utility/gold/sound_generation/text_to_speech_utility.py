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


TEMPORARY_DATA_FOLDER = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "data")
if not os.path.exists:
    os.makedirs(TEMPORARY_DATA_FOLDER)
TEMPORARY_OUTPUT_PATH = os.path.join(TEMPORARY_DATA_FOLDER, "out.wav")


def play_wave_file(wave_file: str, chunk_size: int = 1024, stream_kwargs: dict = None) -> None:
    """
    Plays wave audio file.
    :param wave_file: Wave file path.
    :param chunk_size: Chunk size for file handling. 
        Defaults to 1024.
    :param stream_kwargs: Stream keyword arguments.
        Defaults to None in which case defaults are based on the wave file.
    """
    pya = pyaudio.PyAudio()
    output_file = wave.open(wave_file, "rb")
    stream_kwargs = {
        "rate": output_file.getnchannels(),
        "format": pya.get_format_from_width(output_file.getsampwidth()),
        "channels": output_file.getnchannels()
    } if stream_kwargs is None else stream_kwargs
    if "output" not in stream_kwargs:
        stream_kwargs["output"] = True
    stream = pya.open(
        **stream_kwargs
    )
    data = output_file.readframes(chunk_size)
    while data != "":
        stream.write(data)
        data = output_file.readframes(chunk_size)
    stream.stop_stream()
    stream.close()
    pya.terminate()


def play_wave(wave: np.ndarray,
              stream_kwargs: dict = None) -> None:
    """
    Plays wave audio file.
    :param wave: Waveform as numpy array.
    :param stream_kwargs: Stream keyword arguments.
        Defaults to None in which case defaults are used.
    """
    pya = pyaudio.PyAudio()
    
    stream_kwargs = {
        "rate": 16000,
        "format": pyaudio.paInt16,
        "channels": 1
    } if stream_kwargs is None else stream_kwargs
    if "output" not in stream_kwargs:
        stream_kwargs["output"] = True
    stream = pya.open(
        **stream_kwargs
    )
    stream.write(wave.tobytes())

    stream.close()
    pya.terminate()


def get_coqui_tts_model(model_name_or_path: str, instantiation_kwargs: dict = None) -> TTS:
    """
    Returns a Coqui TTS based model instance.
    :param model_name_or_path: Model name or path.
    :param instantiation_kwargs: Instatiation keyword arguments.
        Defaults to None in which case default values are used.
    :returns: TTS model instance.
    """
    instantiation_kwargs = {} if instantiation_kwargs is None else instantiation_kwargs
    if os.path.exists(model_name_or_path):
        return TTS().load_tts_model_by_path(
            model_path=model_name_or_path,
            config_path=f"{model_name_or_path}/config.json",
            **instantiation_kwargs)
    else:
         return TTS(
              model_name=model_name_or_path,
              **instantiation_kwargs
         )

def synthesize_with_coqui_tts(text: str, 
                              model: TTS = None, 
                              synthesis_kwargs: dict = None) -> Tuple[np.ndarray, dict]:
    """
    Synthesizes text with Coqui TTS and saves results to a file.
    :param text: Output text.
    :param model: TTS model. 
        Defaults to None in which case a default model is instantiated and used.
        Not providing a model therefore increases processing time tremendously!
    :param synthesis_kwargs: Synthesis keyword arguments. 
        Defaults to None in which case default values are used.
    :returns: Synthesized audio and audio metadata which can be used as stream keyword arguments for outputting.
    """
    model = get_coqui_tts_model(TTS.list_models()[0]) if model is None else model
    synthesis_kwargs = {} if synthesis_kwargs is None else synthesis_kwargs
    snythesized = model.tts(
        text=text,
        **synthesis_kwargs)
    
    # Conversion taken from 
    # https://github.com/coqui-ai/TTS/blob/dev/TTS/utils/synthesizer.py and
    # https://github.com/coqui-ai/TTS/blob/dev/TTS/utils/audio/numpy_transforms.py
    if torch.is_tensor(snythesized):
        snythesized = snythesized.cpu().numpy()
    if isinstance(snythesized, list):
        snythesized = np.array(snythesized)

    print(f"Sample Rate: {model.synthesizer.defa}")

    snythesized = snythesized * (32767 / max(0.01, np.max(np.abs(snythesized))))
    snythesized = snythesized.astype(np.int16)
    return snythesized, {
        "rate": model.synthesizer.output_sample_rate,
        "format": pyaudio.paInt16,
        "channels": 1
    }


def synthesize_with_coqui_tts_to_file(text: str, output_path: str = None, model: TTS = None, synthesis_kwargs: dict = None) -> str:
    """
    Synthesizes text with Coqui TTS and saves results to a file.
    :param text: Output text.
    :param output_path: Output path.
        Defaults to None in which case the file "out.wav" under the temporary data folder is used.
    :param model: TTS model. 
        Defaults to None in which case a default model is instantiated and used.
        Not providing a model therefore increases processing time tremendously!
    :param synthesis_kwargs: Synthesis keyword arguments. 
        Defaults to None in which case default values are used.
    :returns: Output file path.
    """
    model = get_coqui_tts_model(TTS.list_models()[0]) if model is None else model
    output_path = "" if TEMPORARY_OUTPUT_PATH is None else output_path
    synthesis_kwargs = {} if synthesis_kwargs is None else synthesis_kwargs
    return model.tts_to_file(
        text=text,
        file_path=output_path,
        **synthesis_kwargs)