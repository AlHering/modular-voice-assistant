# -*- coding: utf-8 -*-
"""
****************************************************
*                     Utility                      *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import pyaudio
import wave
import numpy as np


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
    input_file = wave.open(wave_file, "rb")
    stream_kwargs = {
        "rate": input_file.getnchannels(),
        "format": pya.get_format_from_width(input_file.getsampwidth()),
        "channels": input_file.getnchannels()
    } if stream_kwargs is None else stream_kwargs
    if "output" not in stream_kwargs:
        stream_kwargs["output"] = True
    stream = pya.open(
        **stream_kwargs
    )
    data = input_file.readframes(chunk_size)
    while data != "":
        stream.write(data)
        data = input_file.readframes(chunk_size)
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
