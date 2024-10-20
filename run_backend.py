# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import os
import sys
import json
import click
from typing import Tuple, Generator, Union
from src.configuration import configuration as cfg
from src.utility import json_utility
from src.backend.voice_assistant.language_model_abstractions import LlamaCPPModelInstance, ChatModelInstance, RemoteChatModelInstance
from src.backend.voice_assistant.modular_voice_assistant_abstractions import BasicVoiceAssistant, SpeechRecorder, Transcriber, Synthesizer


def setup_default_voice_assistant(use_remote_llm: bool = True, 
                                  download_model_files: bool = False,
                                  llm_parameters: dict | None = None,
                                  speech_recorder_parameters: dict | None = None,
                                  transcriber_parameters: dict | None = None,
                                  synthesizer_parameters: dict | None = None,
                                  voice_assistant_parameters: dict | None = None) -> BasicVoiceAssistant:
    """
    Sets up a default voice assistant for reference.
    :param use_remote_llm: Whether to use a remote model instance.
    :param download_model_files: Whether to download model files.
    :param llm_parameters: LLM instantiation parameters.
    :param speech_recorder_parameters: Speech recorder instantiation parameters.
    :param transcriber_parameters: Transcriber instantiation parameters.
    :param synthesizer_parameters: Synthesizer instantiation parameters.
    :param voice_assistant_parameters: Voice assistant instantiation parameters.
        Not containing the other configurable instances.
    """
    if download_model_files:
        raise NotImplementedError("Downloading models is not yet implemented!")

    if use_remote_llm:
        llm_parameters = llm_parameters or {
            "api_base": "http://localhost:8123/v1",
            "chat_parameters": {"model": "llama-3.1-storm-8B-i1-Q4KM"},
            "system_prompt": "You are a helpful and sentient assistant. Your task is to help the user in an effective and concise manner.",
        }
        chat_model_instance = RemoteChatModelInstance(**llm_parameters)
    else:
        llm_parameters = llm_parameters or {
            "model_path": os.path.join(cfg.PATHS.MODEL_PATH, 
                                       "text_generation_models/mradermacher_Llama-3.1-Storm-8B-i1-GGUF"),
            "model_file": "Llama-3.1-Storm-8B.i1-Q4_K_M.gguf",
            "model_parameters": {
                "n_ctx": 4096, 
                "temperature": 0.8, 
                "repetition_penalty": 1.6,
                "n_gpu_layers": 33
            },
            "generating_parameters": {
                "max_tokens": 256
            }
        }
        llm = LlamaCPPModelInstance(**llm_parameters)
        chat_model_instance = ChatModelInstance(
            language_model=llm,
            system_prompt="You are a helpful and sentient assistant. Your task is to help the user in an effective and concise manner."
        )
    

    speech_recorder_parameters = speech_recorder_parameters or {}
    speech_recorder = SpeechRecorder(**speech_recorder_parameters)

    transcriber_parameters = transcriber_parameters or {
        "backend": "faster-whisper",
        "model_path": os.path.join(cfg.PATHS.MODEL_PATH, 
                                   "sound_generation_models/speech_to_text/faster_whisper_models/Systran_faster-whisper-tiny"),
        "model_parameters": {
            "device": "cuda",
            "compute_type": "float32",
            "local_files_only": True
        }
    }
    transcriber = Transcriber(**transcriber_parameters)

    fallback_synthesizer_model = os.path.join(cfg.PATHS.MODEL_PATH, 
                                              "sound_generation_models/text_to_speech/coqui_models/tts_models-multilingual-multi-dataset-xtts_v2")
    fallback_speaker_wav = os.path.join(cfg.PATHS.MODEL_PATH, "sound_generation_models//text_to_speech/coqui_xtts/examples/female.wav")
    synthesizer_parameters = synthesizer_parameters or {
        "backend": "coqui-tts",
        "model_path": fallback_synthesizer_model,
        "model_parameters": {
            "config_path": f"{fallback_synthesizer_model}/config.json",
            "gpu": True
        },
        "synthesis_parameters": {
            "speaker_wav": fallback_speaker_wav,
            "language": "en"
        }
    }
    synthesizer = Synthesizer(**synthesizer_parameters)

    voice_assistant_parameters = voice_assistant_parameters or {
        "stream": True,
        "report": False,
        "forward_logging": True
    }
    return BasicVoiceAssistant(
        working_directory=os.path.join(cfg.PATHS.DATA_PATH, "voice_assistant"),
        speech_recorder=speech_recorder,
        transcriber=transcriber,
        chat_model=chat_model_instance,
        synthesizer=synthesizer,
        **voice_assistant_parameters
    )


def get_valid_config_path(config_path: str | None) -> str | None:
    """
    Returns valid config path.
    :param config_path: Base config path.
    :return: Valid config path or None.
    """
    if config_path is not None:
        if not config_path.lower().endswith(".json"):
            config_path += ".json"
        if os.path.exists(config_path):
            return config_path
        else:
            rel_path = os.path.join(cfg.PATHS.CONFIG_PATH, config_path)
            if os.path.exists(rel_path):
                return rel_path


"""
Click-based entrypoint
"""
@click.command()
@click.option("--config", default=None, help="Path or name json configuration file for the voice assistant.")
@click.option("--mode", default=None, help="Interaction mode: (0) conversation, (1) single interaction, (2) terminal based interaction.")
def run_voice_assistant(config: str, mode:int) -> None:
    """Runner program for a voice assistant."""
    config_path = get_valid_config_path(config_path=config)
    if config_path is not None:
        print(f"\nValid config path given: {config_path}.")
        config_data = json_utility.load(config_path)

        voice_assistant_parameters = config_data.get("voice_assistant", {})
        speech_recorder_parameters = config_data.get("speech_recorder", {})
        transcriber_parameters = config_data.get("transcriber", {})
        synthesizer_parameters = config_data.get("synthesizer", {})
        
        voice_assistant = setup_default_voice_assistant(
            use_remote_llm=config_data.get("use_remote_llm", True),
            download_model_files=config_data.get("download_model_files", False),
            llm_parameters=config_data.get("llm"),
            speech_recorder_parameters=speech_recorder_parameters,
            transcriber_parameters=transcriber_parameters,
            synthesizer_parameters=synthesizer_parameters,
            voice_assistant_parameters=voice_assistant_parameters
        )
    else:
        print(f"\nNo valid config path given, using default configuration.")
        voice_assistant = setup_default_voice_assistant(use_remote_llm=True)
    if mode == 2:
        voice_assistant.run_interaction()
    elif mode == 3:
        voice_assistant.run_terminal_conversation()
    else:
        voice_assistant.run_conversation()


if __name__ == "__main__":
    run_voice_assistant()