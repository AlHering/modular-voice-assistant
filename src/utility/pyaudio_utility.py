# -*- coding: utf-8 -*-
"""
****************************************************
*                     Utility                      *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from typing import Any, List, Union
import time
from enum import Enum
import pyaudio
import wave
import numpy as np
try:
    import keyboard
except ImportError:
    keyboard = None


def play_wave_file(wave_file: str, chunk_size: int = 1024, stream_kwargs: dict | None = None) -> None:
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
              stream_kwargs: dict | None = None) -> None:
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


class InterruptMethod(Enum):
    """
    Represents interrupt methods.
    """
    TIME_INTERVAL: int = 0
    KEYBOARD_INTERRUPT: int = 1
    PAUSE_INTERVAL: int = 2


def record_audio_with_pyaudio(interrupt_method: InterruptMethod = InterruptMethod.TIME_INTERVAL,
                              interrupt_threshold: Any = 5.0,
                              chunk_size: int = 2024,
                              stream_kwargs: dict | None = None) -> List[bytes]:
    """
    Records audio with pyaudio.
    :param interrupt_method: Interrupt method as either "TIME_INTERVAL", "KEYBOARD_INTERRUPT". 
        Defaults to "TIME_INTERVAL".
    :param interrupt_threshold: Interrupt threshold as time interval in seconds for "TIME_INTERVAL", key(s) as string for "KEYBOARD_INTERRUPT".
        Defaults to 5.0 in connection with the "TIME_INTERVAL" method. 
    :param chunk_size: Chunk size for file handling. 
        Defaults to 1024.
    :param stream_kwargs: Stream keyword arguments.
        Defaults to None in which case default values are used.
    :returns: Audio stream as list of bytes.
    """
    pya = pyaudio.PyAudio()
    
    stream_kwargs = {
        "format": pyaudio.paInt16,
        "channels": 1,
        "rate": 16000,
        "frames_per_buffer": chunk_size
    } if stream_kwargs is None else stream_kwargs
    if not "input" in stream_kwargs:
        stream_kwargs["input"] = True
    
    frames = []
    stream = pya.open(**stream_kwargs)
    stream.start_stream()
    
    try:
        start_time = time.time()
        while True:
            data = stream.read(chunk_size)
            frames.append(data)
            if interrupt_method == InterruptMethod.KEYBOARD_INTERRUPT and (keyboard is not None and keyboard.is_pressed(interrupt_threshold)):
                break
            elif interrupt_method == InterruptMethod.TIME_INTERVAL and (time.time() - start_time >= interrupt_threshold):
                break
    except KeyboardInterrupt as ex:
        if interrupt_method == InterruptMethod.KEYBOARD_INTERRUPT and (not isinstance(interrupt_threshold, str) or interrupt_threshold == "ctrl+c"):
            # Interrupt method is keyboard interrupt and keyboard package is not available or threshold is not correctly set
            pass
        else:
            raise ex
        
    stream.stop_stream()
    stream.close()
    pya.terminate()

    return frames


def record_audio_with_pyaudio_to_file(output_path: str, 
                                      interrupt_method: InterruptMethod = InterruptMethod.TIME_INTERVAL,
                                      interrupt_threshold: Any = 5.0,
                                      chunk_size: int = 2024,
                                      stream_kwargs: dict | None = None) -> None:
    """
    Records audio with pyaudio and saves it to wave file.
    :param output_path: Wave file output path.
    :param interrupt_method: Interrupt method as either "TIME_INTERVAL", "KEYBOARD_INTERRUPT". 
        Defaults to "TIME_INTERVAL".
    :param interrupt_threshold: Interrupt threshold as time interval in seconds for "TIME_INTERVAL", key(s) as string for "KEYBOARD_INTERRUPT".
        Defaults to 5.0 in connection with the "TIME_INTERVAL" method. 
    :param chunk_size: Chunk size for file handling. 
        Defaults to 1024.
    :param stream_kwargs: Stream keyword arguments.
        Defaults to None in which case default values are used.
    """
    frames = record_audio_with_pyaudio(
        interrupt_method=interrupt_method,
        interrupt_threshold=interrupt_threshold,
        chunk_size=chunk_size,
        stream_kwargs=stream_kwargs
    )

    pya = pyaudio.PyAudio()
    wave_output = wave.open(output_path, "wb")
    wave_output.setsampwidth(pya.get_sample_size(stream_kwargs.get("format", pyaudio.paInt16)))
    wave_output.setnchannels(stream_kwargs.get("channels", 1)) 
    wave_output.setframerate(stream_kwargs.get("rate", 16000))
    wave_output.writeframes(b"".join(frames))
    pya.terminate()


def record_audio_with_pyaudio_to_numpy_array(interrupt_method: InterruptMethod = InterruptMethod.TIME_INTERVAL,
                                             interrupt_threshold: Any = 5.0,
                                             chunk_size: int = 2024,
                                             stream_kwargs: dict | None = None) -> np.ndarray:
    """
    Records audio with pyaudio to a numpy array.
    :param interrupt_method: Interrupt method as either "TIME_INTERVAL", "KEYBOARD_INTERRUPT". 
        Defaults to "TIME_INTERVAL".
    :param interrupt_threshold: Interrupt threshold as time interval in seconds for "TIME_INTERVAL", key(s) as string for "KEYBOARD_INTERRUPT".
        Defaults to 5.0 in connection with the "TIME_INTERVAL" method. 
    :param chunk_size: Chunk size for file handling. 
        Defaults to 1024.
    :param stream_kwargs: Stream keyword arguments.
        Defaults to None in which case default values are used.
    """
    frames = record_audio_with_pyaudio(
        interrupt_method=interrupt_method,
        interrupt_threshold=interrupt_threshold,
        chunk_size=chunk_size,
        stream_kwargs=stream_kwargs
    )

    return np.fromstring(b"".join(frames), dtype={
        pyaudio.paInt8: np.int8,
        pyaudio.paInt16: np.int16,
        pyaudio.paInt32: np.int32,
    }[stream_kwargs.get("format", pyaudio.paInt16)])
