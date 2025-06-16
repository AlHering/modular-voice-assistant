# -*- coding: utf-8 -*-
"""
****************************************************
*                      Utility                 
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from typing import Union, Tuple
import numpy as np
import torch
import whisper


def get_available_models() -> dict:
    """
    Fetches available models.
    :return: Dictionary with models and their metadata.
    """
    return whisper._MODELS


def download_whisper_model(model_id: str,
                           output_folder: str) -> None:
    """
    Function for downloading whisper models.
    :param model_id: Target model ID.
    :param output_folder: Output folder path.
    """    
    whisper.load_model(model_id, download_root=output_folder)


def load_whisper_model(model_path: str,
                       model_parameters: dict = {}) -> whisper.Whisper:
    """
    Function for loading whisper based model instance.
    :param model_path: Path to model files.
    :param model_parameters: Model loading kwargs as dictionary.
        Defaults to empty dictionary.
    :return: Model instance.
    """
    return whisper.load_model(
            name=model_path,
            **model_parameters)


def normalize_audio_for_whisper(audio_input: Union[str, np.ndarray, torch.Tensor]) -> Union[str, np.ndarray, torch.Tensor]:
    """
    Function for normalizing audio data before transcribing with whisper or faster-whisper.
    :param audio_input: Wave file path or waveform.
    :param return: Normalized audio data.
    """
    if isinstance(audio_input, np.ndarray) and str(audio_input.dtype) not in ["float16", "float32"]:
        return np.frombuffer(audio_input, audio_input.dtype).flatten().astype(np.float32) / {
                    "int8": 128.0,
                    "int16": 32768.0,
                    "int32": 2147483648.0,
                    "int64": 9223372036854775808.0
                    }[str(audio_input.dtype)] 
    else:
        return audio_input


def transcribe(audio_input: Union[str, np.ndarray, torch.Tensor], model: whisper.Whisper = None, transcription_parameters: dict | None = None) -> Tuple[str, dict]:
    """
    Transcribes wave file or waveform with whisper.
    :param audio_input: Wave file path or waveform.
    :param model: Whisper model. 
        Defaults to None in which case a default model is instantiated and used.
        Not providing a model therefore increases processing time tremendously!
    :param transcription_parameters: Transcription keyword arguments. 
        Defaults to None in which case default values are used.
    :returns: Tuple of transcribed text and a list of metadata entries for the transcribed segments.
    """
    model = load_whisper_model(model_name_or_path="large-v3") if model is None else model
    audio_input = normalize_audio_for_whisper(audio_input)
    transcription_parameters = {} if transcription_parameters is None else transcription_parameters
    
    transcription = model.transcribe(
        audio=audio_input,
        **transcription_parameters
    )
    transcription["text"], transcription


def dump_normalized_models(output_path: str) -> None:
    """
    Dumps normalized models as json file.
    :param output_path: JSON file path.
    """
    import json
    import requests
    from copy import deepcopy

    base_dict = deepcopy(get_available_models)
    base_dict.pop("turbo")
    base_dict["large"] = base_dict.pop("large-v1")
    normalized = {value: {"name": key} for key, value in base_dict.items()}
    for model_id in normalized:
        model_name = normalized[model_id]["name"]
        normalized[model_id]["model_type"] = "stt"
        normalized[model_id]["language"] = "en" if model_name.endswith(".en") else "multilingual"
        entry = requests.get(f"https://huggingface.co/api/models/?full=true&config=true&id=openai/whisper-{model_name}").json()[0]
        relevant_files = [file for file in entry["siblings"]
            if not any(part in file['rfilename'] for part in [".bin", ".msgpack", "-of-", ".h5"])]
        normalized[model_id]["download_urls"] = [
            f"https://huggingface.co/openai/whisper-{model_name}/resolve/main/{file['rfilename']}" for file in relevant_files]
    open(output_path, "w").write(json.dumps(normalized, ensure_ascii=False, indent=4))


NORMALIZED_MODELS = {
    "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt": {
        "name": "tiny.en",
        "model_type": "stt",
        "language": "en",
        "download_urls": [
            "https://huggingface.co/openai/whisper-tiny.en/resolve/main/.gitattributes",
            "https://huggingface.co/openai/whisper-tiny.en/resolve/main/README.md",
            "https://huggingface.co/openai/whisper-tiny.en/resolve/main/added_tokens.json",
            "https://huggingface.co/openai/whisper-tiny.en/resolve/main/config.json",
            "https://huggingface.co/openai/whisper-tiny.en/resolve/main/generation_config.json",
            "https://huggingface.co/openai/whisper-tiny.en/resolve/main/merges.txt",
            "https://huggingface.co/openai/whisper-tiny.en/resolve/main/model.safetensors",
            "https://huggingface.co/openai/whisper-tiny.en/resolve/main/normalizer.json",
            "https://huggingface.co/openai/whisper-tiny.en/resolve/main/preprocessor_config.json",
            "https://huggingface.co/openai/whisper-tiny.en/resolve/main/special_tokens_map.json",
            "https://huggingface.co/openai/whisper-tiny.en/resolve/main/tokenizer.json",
            "https://huggingface.co/openai/whisper-tiny.en/resolve/main/tokenizer_config.json",
            "https://huggingface.co/openai/whisper-tiny.en/resolve/main/vocab.json"
        ]
    },
    "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt": {
        "name": "tiny",
        "model_type": "stt",
        "language": "multilingual",
        "download_urls": [
            "https://huggingface.co/openai/whisper-tiny/resolve/main/.gitattributes",
            "https://huggingface.co/openai/whisper-tiny/resolve/main/README.md",
            "https://huggingface.co/openai/whisper-tiny/resolve/main/added_tokens.json",
            "https://huggingface.co/openai/whisper-tiny/resolve/main/config.json",
            "https://huggingface.co/openai/whisper-tiny/resolve/main/generation_config.json",
            "https://huggingface.co/openai/whisper-tiny/resolve/main/merges.txt",
            "https://huggingface.co/openai/whisper-tiny/resolve/main/model.safetensors",
            "https://huggingface.co/openai/whisper-tiny/resolve/main/normalizer.json",
            "https://huggingface.co/openai/whisper-tiny/resolve/main/preprocessor_config.json",
            "https://huggingface.co/openai/whisper-tiny/resolve/main/special_tokens_map.json",
            "https://huggingface.co/openai/whisper-tiny/resolve/main/tokenizer.json",
            "https://huggingface.co/openai/whisper-tiny/resolve/main/tokenizer_config.json",
            "https://huggingface.co/openai/whisper-tiny/resolve/main/vocab.json"
        ]
    },
    "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt": {
        "name": "base.en",
        "model_type": "stt",
        "language": "en",
        "download_urls": [
            "https://huggingface.co/openai/whisper-base.en/resolve/main/.gitattributes",
            "https://huggingface.co/openai/whisper-base.en/resolve/main/README.md",
            "https://huggingface.co/openai/whisper-base.en/resolve/main/added_tokens.json",
            "https://huggingface.co/openai/whisper-base.en/resolve/main/config.json",
            "https://huggingface.co/openai/whisper-base.en/resolve/main/generation_config.json",
            "https://huggingface.co/openai/whisper-base.en/resolve/main/merges.txt",
            "https://huggingface.co/openai/whisper-base.en/resolve/main/model.safetensors",
            "https://huggingface.co/openai/whisper-base.en/resolve/main/normalizer.json",
            "https://huggingface.co/openai/whisper-base.en/resolve/main/preprocessor_config.json",
            "https://huggingface.co/openai/whisper-base.en/resolve/main/special_tokens_map.json",
            "https://huggingface.co/openai/whisper-base.en/resolve/main/tokenizer.json",
            "https://huggingface.co/openai/whisper-base.en/resolve/main/tokenizer_config.json",
            "https://huggingface.co/openai/whisper-base.en/resolve/main/vocab.json"
        ]
    },
    "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt": {
        "name": "base",
        "model_type": "stt",
        "language": "multilingual",
        "download_urls": [
            "https://huggingface.co/openai/whisper-base/resolve/main/.gitattributes",
            "https://huggingface.co/openai/whisper-base/resolve/main/README.md",
            "https://huggingface.co/openai/whisper-base/resolve/main/added_tokens.json",
            "https://huggingface.co/openai/whisper-base/resolve/main/config.json",
            "https://huggingface.co/openai/whisper-base/resolve/main/generation_config.json",
            "https://huggingface.co/openai/whisper-base/resolve/main/merges.txt",
            "https://huggingface.co/openai/whisper-base/resolve/main/model.safetensors",
            "https://huggingface.co/openai/whisper-base/resolve/main/normalizer.json",
            "https://huggingface.co/openai/whisper-base/resolve/main/preprocessor_config.json",
            "https://huggingface.co/openai/whisper-base/resolve/main/special_tokens_map.json",
            "https://huggingface.co/openai/whisper-base/resolve/main/tokenizer.json",
            "https://huggingface.co/openai/whisper-base/resolve/main/tokenizer_config.json",
            "https://huggingface.co/openai/whisper-base/resolve/main/vocab.json"
        ]
    },
    "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt": {
        "name": "small.en",
        "model_type": "stt",
        "language": "en",
        "download_urls": [
            "https://huggingface.co/openai/whisper-small.en/resolve/main/.gitattributes",
            "https://huggingface.co/openai/whisper-small.en/resolve/main/README.md",
            "https://huggingface.co/openai/whisper-small.en/resolve/main/added_tokens.json",
            "https://huggingface.co/openai/whisper-small.en/resolve/main/config.json",
            "https://huggingface.co/openai/whisper-small.en/resolve/main/generation_config.json",
            "https://huggingface.co/openai/whisper-small.en/resolve/main/merges.txt",
            "https://huggingface.co/openai/whisper-small.en/resolve/main/model.safetensors",
            "https://huggingface.co/openai/whisper-small.en/resolve/main/normalizer.json",
            "https://huggingface.co/openai/whisper-small.en/resolve/main/preprocessor_config.json",
            "https://huggingface.co/openai/whisper-small.en/resolve/main/special_tokens_map.json",
            "https://huggingface.co/openai/whisper-small.en/resolve/main/tokenizer.json",
            "https://huggingface.co/openai/whisper-small.en/resolve/main/tokenizer_config.json",
            "https://huggingface.co/openai/whisper-small.en/resolve/main/vocab.json"
        ]
    },
    "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt": {
        "name": "small",
        "model_type": "stt",
        "language": "multilingual",
        "download_urls": [
            "https://huggingface.co/openai/whisper-small/resolve/main/.gitattributes",
            "https://huggingface.co/openai/whisper-small/resolve/main/README.md",
            "https://huggingface.co/openai/whisper-small/resolve/main/added_tokens.json",
            "https://huggingface.co/openai/whisper-small/resolve/main/config.json",
            "https://huggingface.co/openai/whisper-small/resolve/main/generation_config.json",
            "https://huggingface.co/openai/whisper-small/resolve/main/merges.txt",
            "https://huggingface.co/openai/whisper-small/resolve/main/model.safetensors",
            "https://huggingface.co/openai/whisper-small/resolve/main/normalizer.json",
            "https://huggingface.co/openai/whisper-small/resolve/main/preprocessor_config.json",
            "https://huggingface.co/openai/whisper-small/resolve/main/special_tokens_map.json",
            "https://huggingface.co/openai/whisper-small/resolve/main/tokenizer.json",
            "https://huggingface.co/openai/whisper-small/resolve/main/tokenizer_config.json",
            "https://huggingface.co/openai/whisper-small/resolve/main/vocab.json"
        ]
    },
    "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt": {
        "name": "medium.en",
        "model_type": "stt",
        "language": "en",
        "download_urls": [
            "https://huggingface.co/openai/whisper-medium.en/resolve/main/.gitattributes",
            "https://huggingface.co/openai/whisper-medium.en/resolve/main/README.md",
            "https://huggingface.co/openai/whisper-medium.en/resolve/main/added_tokens.json",
            "https://huggingface.co/openai/whisper-medium.en/resolve/main/config.json",
            "https://huggingface.co/openai/whisper-medium.en/resolve/main/generation_config.json",
            "https://huggingface.co/openai/whisper-medium.en/resolve/main/merges.txt",
            "https://huggingface.co/openai/whisper-medium.en/resolve/main/model.safetensors",
            "https://huggingface.co/openai/whisper-medium.en/resolve/main/normalizer.json",
            "https://huggingface.co/openai/whisper-medium.en/resolve/main/preprocessor_config.json",
            "https://huggingface.co/openai/whisper-medium.en/resolve/main/special_tokens_map.json",
            "https://huggingface.co/openai/whisper-medium.en/resolve/main/tokenizer.json",
            "https://huggingface.co/openai/whisper-medium.en/resolve/main/tokenizer_config.json",
            "https://huggingface.co/openai/whisper-medium.en/resolve/main/vocab.json"
        ]
    },
    "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt": {
        "name": "medium",
        "model_type": "stt",
        "language": "multilingual",
        "download_urls": [
            "https://huggingface.co/openai/whisper-medium/resolve/main/.gitattributes",
            "https://huggingface.co/openai/whisper-medium/resolve/main/README.md",
            "https://huggingface.co/openai/whisper-medium/resolve/main/added_tokens.json",
            "https://huggingface.co/openai/whisper-medium/resolve/main/config.json",
            "https://huggingface.co/openai/whisper-medium/resolve/main/generation_config.json",
            "https://huggingface.co/openai/whisper-medium/resolve/main/merges.txt",
            "https://huggingface.co/openai/whisper-medium/resolve/main/model.safetensors",
            "https://huggingface.co/openai/whisper-medium/resolve/main/normalizer.json",
            "https://huggingface.co/openai/whisper-medium/resolve/main/preprocessor_config.json",
            "https://huggingface.co/openai/whisper-medium/resolve/main/special_tokens_map.json",
            "https://huggingface.co/openai/whisper-medium/resolve/main/tokenizer.json",
            "https://huggingface.co/openai/whisper-medium/resolve/main/tokenizer_config.json",
            "https://huggingface.co/openai/whisper-medium/resolve/main/vocab.json"
        ]
    },
    "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt": {
        "name": "large-v2",
        "model_type": "stt",
        "language": "multilingual",
        "download_urls": [
            "https://huggingface.co/openai/whisper-large-v2/resolve/main/.gitattributes",
            "https://huggingface.co/openai/whisper-large-v2/resolve/main/README.md",
            "https://huggingface.co/openai/whisper-large-v2/resolve/main/added_tokens.json",
            "https://huggingface.co/openai/whisper-large-v2/resolve/main/config.json",
            "https://huggingface.co/openai/whisper-large-v2/resolve/main/generation_config.json",
            "https://huggingface.co/openai/whisper-large-v2/resolve/main/merges.txt",
            "https://huggingface.co/openai/whisper-large-v2/resolve/main/model.safetensors",
            "https://huggingface.co/openai/whisper-large-v2/resolve/main/normalizer.json",
            "https://huggingface.co/openai/whisper-large-v2/resolve/main/preprocessor_config.json",
            "https://huggingface.co/openai/whisper-large-v2/resolve/main/special_tokens_map.json",
            "https://huggingface.co/openai/whisper-large-v2/resolve/main/tokenizer.json",
            "https://huggingface.co/openai/whisper-large-v2/resolve/main/tokenizer_config.json",
            "https://huggingface.co/openai/whisper-large-v2/resolve/main/vocab.json"
        ]
    },
    "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt": {
        "name": "large-v3",
        "model_type": "stt",
        "language": "multilingual",
        "download_urls": [
            "https://huggingface.co/openai/whisper-large-v3/resolve/main/.gitattributes",
            "https://huggingface.co/openai/whisper-large-v3/resolve/main/README.md",
            "https://huggingface.co/openai/whisper-large-v3/resolve/main/added_tokens.json",
            "https://huggingface.co/openai/whisper-large-v3/resolve/main/config.json",
            "https://huggingface.co/openai/whisper-large-v3/resolve/main/generation_config.json",
            "https://huggingface.co/openai/whisper-large-v3/resolve/main/merges.txt",
            "https://huggingface.co/openai/whisper-large-v3/resolve/main/model.safetensors",
            "https://huggingface.co/openai/whisper-large-v3/resolve/main/model.safetensors.index.fp32.json",
            "https://huggingface.co/openai/whisper-large-v3/resolve/main/normalizer.json",
            "https://huggingface.co/openai/whisper-large-v3/resolve/main/preprocessor_config.json",
            "https://huggingface.co/openai/whisper-large-v3/resolve/main/special_tokens_map.json",
            "https://huggingface.co/openai/whisper-large-v3/resolve/main/tokenizer.json",
            "https://huggingface.co/openai/whisper-large-v3/resolve/main/tokenizer_config.json",
            "https://huggingface.co/openai/whisper-large-v3/resolve/main/vocab.json"
        ]
    },
    "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt": {
        "name": "large",
        "model_type": "stt",
        "language": "multilingual",
        "download_urls": [
            "https://huggingface.co/openai/whisper-large/resolve/main/.gitattributes",
            "https://huggingface.co/openai/whisper-large/resolve/main/README.md",
            "https://huggingface.co/openai/whisper-large/resolve/main/added_tokens.json",
            "https://huggingface.co/openai/whisper-large/resolve/main/config.json",
            "https://huggingface.co/openai/whisper-large/resolve/main/generation_config.json",
            "https://huggingface.co/openai/whisper-large/resolve/main/merges.txt",
            "https://huggingface.co/openai/whisper-large/resolve/main/model.safetensors",
            "https://huggingface.co/openai/whisper-large/resolve/main/normalizer.json",
            "https://huggingface.co/openai/whisper-large/resolve/main/preprocessor_config.json",
            "https://huggingface.co/openai/whisper-large/resolve/main/special_tokens_map.json",
            "https://huggingface.co/openai/whisper-large/resolve/main/tokenizer.json",
            "https://huggingface.co/openai/whisper-large/resolve/main/tokenizer_config.json",
            "https://huggingface.co/openai/whisper-large/resolve/main/vocab.json"
        ]
    },
    "https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt": {
        "name": "large-v3-turbo",
        "model_type": "stt",
        "language": "multilingual",
        "download_urls": [
            "https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/.gitattributes",
            "https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/README.md",
            "https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/added_tokens.json",
            "https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/config.json",
            "https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/generation_config.json",
            "https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/merges.txt",
            "https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/model.safetensors",
            "https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/normalizer.json",
            "https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/preprocessor_config.json",
            "https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/special_tokens_map.json",
            "https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/tokenizer.json",
            "https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/tokenizer_config.json",
            "https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/vocab.json"
        ]
    }
}