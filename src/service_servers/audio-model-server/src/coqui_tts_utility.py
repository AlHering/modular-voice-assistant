# -*- coding: utf-8 -*-
"""
****************************************************
*                      Utility                 
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import os
from typing import Any, Tuple, List
import os
import pyaudio
from .pyaudio_utility import play_wave
import numpy as np
import torch
from TTS.api import TTS
from TTS.utils.manage import ModelManager


def load_coqui_tts_model(model_path: str,
                         model_parameters: dict = {}) -> TTS:
    """
    Function for downloading coqui TTS model.
    :param model_path: Path to model files.
    :param model_parameters: Model loading kwargs as dictionary.
        Defaults to empty dictionary.
    :return: Model instance.
    """
    if os.path.exists(model_path):
        default_config_path = f"{model_path}/config.json"
        if "config_path" not in model_parameters and os.path.exists(default_config_path):
            model_parameters["config_path"] = default_config_path
        return TTS(model_path=model_path,
            **model_parameters)
    else:
         return TTS(
              model_name=model_path,
              **model_parameters
         )


def download_coqui_tts_model(model_id: str,
                             output_folder: str) -> None:
    """
    Function for downloading faster whisper models.
    :param model_id: Target model ID.
    :param output_folder: Output folder path.
    """
    manager = ModelManager(output_prefix=output_folder, progress_bar=True)
    manager.download_model(model_id)


def synthesize(text: str, 
               model: TTS = None, 
               synthesis_parameters: dict | None = None) -> Tuple[np.ndarray, dict]:
    """
    Synthesizes text with Coqui TTS and returns the results.
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


def synthesize_to_file(text: str, 
                       output_path: str, 
                       model: TTS = None, 
                       synthesis_parameters: dict | None = None) -> str:
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


def synthesize_and_play(text: str, 
                        model: TTS = None, 
                        synthesis_parameters: dict | None = None) -> Tuple[np.ndarray, dict]:
    """
    Synthesizes text with Coqui TTS and outputs the resulting audio data.
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

    metadata = {
        "rate": model.synthesizer.output_sample_rate,
        "format": pyaudio.paInt16,
        "channels": 1
    }

    play_wave(wave=snythesized, stream_kwargs=metadata)
    return snythesized, metadata


def test_available_speakers(model: TTS, 
                            synthesis_parameters: dict | None = None,
                            text: str = "This is a very short test.",
                            play_results: bool = False) -> List[Tuple[str, Tuple[np.ndarray, dict]]]:
    """
    Function for testing available speakers.
    :param model: TTS model.
    :param synthesis_parameters: Synthesis keyword arguments. 
        Defaults to None in which case default values are used.
    :param text: Text to synthesize.
        Defaults to "This is a very short test.".
    :param play_results: Flag for declaring, whether to play the synthesized results.
        Defaults to False.
    :returns: Tuple of speaker name and a tuple of synthesized audio and audio metadata.
    """
    results = []
    synthesis_parameters = {} if synthesis_parameters is None else synthesis_parameters
    for speaker in model.speakers:
        synthesis_parameters["speaker"] = speaker
        if play_results:
            print(speaker)
            results.append((speaker, synthesize_and_play(text=text, model=model, synthesis_parameters=synthesis_parameters)))
        else:
            results.append((speaker, synthesize(text=text, model=model, synthesis_parameters=synthesis_parameters)))
    return results
         

def output_available_speakers_to_file(output_dir: str,
                                      model: TTS, 
                                      synthesis_parameters: dict | None = None,
                                      text: str = "This is a very short test.") -> List[Tuple[str, str]]:
    """
    Function for testing available speakers by writing there output to files.
    :param output_dir: Folder in which to store the wave files.
    :param model: TTS model.
    :param synthesis_parameters: Synthesis keyword arguments. 
        Defaults to None in which case default values are used.
    :param text: Text to synthesize.
        Defaults to "This is a very short test.".
    :returns: List of tuples of speaker name and output path.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    results = []
    synthesis_parameters = {} if synthesis_parameters is None else synthesis_parameters
    synthesis_string = "_".join(f"{elem}={str(synthesis_parameters[elem])}" for elem in synthesis_parameters)
    for speaker in model.speakers:
        synthesis_parameters["speaker"] = speaker
        results.append((speaker, synthesize_to_file(
            text=text,
            output_path=os.path.join(output_dir, f"{speaker}_{synthesis_string}.wav"),
            model=model,
            synthesis_parameters=synthesis_parameters
        )))
    return results