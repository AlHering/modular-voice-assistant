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


class AudioHandler(object):
    """
    Represents an audio handler for recording and playing audio data.
    """
    def __init__(self) -> None:
        """
        Initiation method.
        """
        self.pya = pyaudio.PyAudio()

    def get_audio_devices(self) -> List[str]:
        """
        Returns a list of available devices.
        :return: List of audio device names.
        """
        return [device["name"] for device in sounddevice.query_devices()]
    
    def get_audio_devices_metadata(self) -> dict:
        """
        Returns a dictionary of available devices and their metadata.
        """
        return {
            device.pop("name"): device for device in sounddevice.query_devices()
        }


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
