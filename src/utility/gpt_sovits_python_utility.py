# -*- coding: utf-8 -*-
"""
****************************************************
*                      Utility                 
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from typing import Tuple, List
import os
import numpy as np
from gpt_sovits_python import TTS


def get_available_models() -> dict:
    """
    Fetches available models.
    :return: Dictionary with models and their metadata.
    """
    raise NotImplementedError()


def load_gpt_sovits_models(models_config: dict) -> TTS:
    """
    Loads up models.
    :param models_config: Models config.
    :return: Models instance.
    """
    raise NotImplementedError()


def download_gpt_sovits_model(model_id: str,
                             output_folder: str) -> None:
    """
    Function for downloading models.
    :param model_id: Target model ID.
    :param output_folder: Output folder pat
    """
    raise NotImplementedError()


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
    raise NotImplementedError()


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
    raise NotImplementedError()


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
    raise NotImplementedError()


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
    raise NotImplementedError()
         

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
    raise NotImplementedError()


def dump_normalized_models(output_path: str) -> None:
    """
    Dumps normalized models as json file.
    :param output_path: JSON file path.
    """
    raise NotImplementedError()


NORMALIZED_MODELS = {}