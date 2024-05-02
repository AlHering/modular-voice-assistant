# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from typing import List
import pyaudio
import sounddevice


def get_audio_devices(self, include_metadata: bool = False) -> list:
    """
    Returns a list of available devices.
    :param include_metadata: Flag for including metadata.
    :return: List of audio device names or audio device metadata dictionaries.
    """
    return [device if include_metadata else device["name"] for device in sounddevice.query_devices()]

def get_input_devices(self, include_metadata: bool = False) -> list:
    """
    Returns a list of available input devices.
    :param include_metadata: Flag for including metadata.
    :return: List of audio device names or audio device metadata dictionaries limited to input devices.
    """
    return [device if include_metadata else device["name"] for device in sounddevice.query_devices() if device["max_input_channels"] > 0]

def get_output_devices(self, include_metadata: bool = False) -> list:
    """
    Returns a list of available input devices.
    :param include_metadata: Flag for including metadata.
    :return: List of audio device names or audio device metadata dictionaries limited to output devices.
    """
    return [device if include_metadata else device["name"] for device in sounddevice.query_devices() if device["max_output_channels"] > 0]
    

class AudioHandler(object):
    """
    Represents an audio handler for recording and playing audio data.
    """
    def __init__(self) -> None:
        """
        Initiation method.
        """
        self.pya = pyaudio.PyAudio()

   


    def record(self, engine_kwargs: dict = None) -> None:
        stream = self.pya.open(

        )   


class STTHandler(object):
    """
    Class for representing Speech To Text handlers.
    """
    def __init__(self) -> None:
        """
        Initiation method.
        """
        pass