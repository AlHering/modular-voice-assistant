# -*- coding: utf-8 -*-
"""
****************************************************
*                      Utility                 
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import sounddevice
import pyaudio


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

