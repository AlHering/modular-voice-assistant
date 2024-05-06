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
from enum import Enum
import wave
import time
import numpy as np
import torch
import whisper
from faster_whisper import WhisperModel
import keyboard


TEMPORARY_DATA_FOLDER = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "data")
if not os.path.exists:
    os.makedirs(TEMPORARY_DATA_FOLDER)
TEMPORARY_INPUT_PATH = os.path.join(TEMPORARY_DATA_FOLDER, "in.wav")


class InterruptMethod(Enum):
    """
    Represents interrupt methods.
    """
    TIME_INTERVAL: int = 0
    KEYBOARD_INTERRUPT: int = 1


def record_audio_with_pyaudio(interrupt_method: InterruptMethod = InterruptMethod.TIME_INTERVAL,
                              interrupt_handle: Any = 5.0,
                              chunk_size: int = 2024,
                              stream_kwargs: dict = None) -> List[bytes]:
    """
    Records audio with pyaudio.
    :param interrupt_method: Interrupt method as either "TIME_INTERVAL", "KEYBOARD_INTERRUPT". 
        Defaults to "TIME_INTERVAL".
    :param interrupt_handle: Interrupt handle as time interval in seconds for "TIME_INTERVAL", key(s) as string for "KEYBOARD_INTERRUPT".
        Defaults to 5.0 in connection with the "TIME_INTERVAL" method. 
    :param chunk_size: Chunk size for file handling. 
        Defaults to 1024.
    :param stream_kwargs: Stream keyword arguments.
        Defaults to None in which case defaults are based on the wave file.
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
            if interrupt_method == InterruptMethod.KEYBOARD_INTERRUPT and keyboard.is_pressed(interrupt_handle):
                break
            elif interrupt_method == InterruptMethod.TIME_INTERVAL and (time.time() - start_time >= interrupt_handle):
                break
    except KeyboardInterrupt as ex:
        if interrupt_method == InterruptMethod.KEYBOARD_INTERRUPT and (not isinstance(interrupt_handle, str) or interrupt_handle == "ctrl+c"):
            # Interrupt method is keyboard interrupt and handle is not correctly set or set to keyboard interrupt
            pass
        else:
            raise ex
        
    stream.stop_stream()
    stream.close()
    pya.terminate()

    return frames


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


def transcribe_with_faster_whisper(audio_input: Union[str, np.ndarray, torch.Tensor] = None, model: whisper.Whisper = None, transcription_kwargs: dict = None) -> Tuple[str, List[dict]]:
    """
    Transcribes wave file or waveform with faster whisper.
    :param audio_input: Wave file path or waveform.
    :param output_path: Output path.
        Defaults to None in which case the file "out.wav" under the temporary data folder is used.
    :param model: Faster whisper model. 
        Defaults to None in which case a default model is instantiated and used.
        Not providing a model therefore increases processing time tremendously!
    :param transcription_kwargs: Transcription keyword arguments. 
        Defaults to None in which case default values are used.
    :returns: Tuple of transcribed text and a list of metadata entries for the transcribed segments.
    """
    model = get_faster_whisper_model(model_name_or_path="large-v3") if model is None else model
    audio_input = TEMPORARY_INPUT_PATH if audio_input is None else audio_input
    transcription_kwargs = {} if transcription_kwargs is None else transcription_kwargs
    
    transcription, metadata = model.transcribe(
        audio=audio_input,
        **transcription_kwargs
    )
    metadata = metadata._asdict()
    segment_metadatas = [segment._asdict() for segment in transcription]
    for segment in segment_metadatas:
        segment.update(metadata)
    fulltext = " ".join([segment["text"].strip() for segment in segment_metadatas])
    
    return fulltext, segment_metadatas
