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
import faster_whisper


def get_available_models() -> dict:
    """
    Fetches available models.
    :return: Dictionary with models and their metadata.
    """
    return faster_whisper.utils._MODELS


def download_faster_whisper_model(model_id: str,
                                  output_folder: str) -> None:
    """
    Function for downloading faster whisper models.
    :param model_id: Target model ID.
    :param output_folder: Output folder path.
    """
    faster_whisper.download_model(model_id, output_dir=output_folder)


def load_faster_whisper_model(model_path: str,
                              model_parameters: dict = {}) -> faster_whisper.WhisperModel:
    """
    Function for loading faster whisper based model instance.
    :param model_path: Path to model files.
    :param model_parameters: Model loading kwargs as dictionary.
        Defaults to empty dictionary.
    :return: Model instance.
    """
    return faster_whisper.WhisperModel(
        model_size_or_path=model_path,
        **model_parameters
    )


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


def transcribe(audio_input: Union[str, np.ndarray, torch.Tensor], model: faster_whisper.WhisperModel = None, transcription_parameters: dict | None = None) -> Tuple[str, dict]:
    """
    Transcribes wave file or waveform with faster whisper.
    :param audio_input: Wave file path or waveform.
    :param model: Faster whisper model. 
        Defaults to None in which case a default model is instantiated and used.
        Not providing a model therefore increases processing time tremendously!
    :param transcription_parameters: Transcription keyword arguments. 
        Defaults to None in which case default values are used.
    :returns: Tuple of transcribed text and a list of metadata entries for the transcribed segments.
    """
    model = load_faster_whisper_model(model_name_or_path="large-v3") if model is None else model
    audio_input = normalize_audio_for_whisper(audio_input)
    transcription_parameters = {} if transcription_parameters is None else transcription_parameters
    
    transcription, metadata = model.transcribe(
        audio=audio_input,
        **transcription_parameters
    )
    metadata = metadata._asdict()
    segment_metadatas = [segment._asdict() for segment in transcription]
    for segment in segment_metadatas:
        segment.update(metadata)
    fulltext = " ".join([segment["text"].strip() for segment in segment_metadatas])
    
    return fulltext, {"segments": segment_metadatas}


def dump_normalized_models(output_path: str) -> None:
    """
    Dumps normalized models as json file.
    :param output_path: JSON file path.
    """
    import json
    import requests
    
    base_dict = get_available_models()
    normalized = {value: {"name": key} for key, value in base_dict.items()}
    for model_id in NORMALIZED_MODELS:
        normalized[model_id]["model_type"] = "stt"
        normalized[model_id]["language"] = "en" if model_id.endswith(".en") else "multilingual"
        entry = requests.get(f"https://huggingface.co/api/models/?full=true&config=true&id={model_id}").json()[0]
        normalized[model_id]["download_urls"] = [
            f"https://huggingface.co/{model_id}/resolve/main/{file['rfilename']}" for file in entry["siblings"]
        ]
    open(output_path, "w").write(json.dumps(normalized, ensure_ascii=False, indent=4))


NORMALIZED_MODELS = {
    "Systran/faster-whisper-tiny.en": {
        "name": "tiny.en",
        "model_type": "stt",
        "language": "en",
        "download_urls": [
            "https://huggingface.co/Systran/faster-whisper-tiny.en/resolve/main/.gitattributes",
            "https://huggingface.co/Systran/faster-whisper-tiny.en/resolve/main/README.md",
            "https://huggingface.co/Systran/faster-whisper-tiny.en/resolve/main/config.json",
            "https://huggingface.co/Systran/faster-whisper-tiny.en/resolve/main/model.bin",
            "https://huggingface.co/Systran/faster-whisper-tiny.en/resolve/main/tokenizer.json",
            "https://huggingface.co/Systran/faster-whisper-tiny.en/resolve/main/vocabulary.txt"
        ]
    },
    "Systran/faster-whisper-tiny": {
        "name": "tiny",
        "model_type": "stt",
        "language": "multilingual",
        "download_urls": [
            "https://huggingface.co/Systran/faster-whisper-tiny/resolve/main/.gitattributes",
            "https://huggingface.co/Systran/faster-whisper-tiny/resolve/main/README.md",
            "https://huggingface.co/Systran/faster-whisper-tiny/resolve/main/config.json",
            "https://huggingface.co/Systran/faster-whisper-tiny/resolve/main/model.bin",
            "https://huggingface.co/Systran/faster-whisper-tiny/resolve/main/tokenizer.json",
            "https://huggingface.co/Systran/faster-whisper-tiny/resolve/main/vocabulary.txt"
        ]
    },
    "Systran/faster-whisper-base.en": {
        "name": "base.en",
        "model_type": "stt",
        "language": "en",
        "download_urls": [
            "https://huggingface.co/Systran/faster-whisper-base.en/resolve/main/.gitattributes",
            "https://huggingface.co/Systran/faster-whisper-base.en/resolve/main/README.md",
            "https://huggingface.co/Systran/faster-whisper-base.en/resolve/main/config.json",
            "https://huggingface.co/Systran/faster-whisper-base.en/resolve/main/model.bin",
            "https://huggingface.co/Systran/faster-whisper-base.en/resolve/main/tokenizer.json",
            "https://huggingface.co/Systran/faster-whisper-base.en/resolve/main/vocabulary.txt"
        ]
    },
    "Systran/faster-whisper-base": {
        "name": "base",
        "model_type": "stt",
        "language": "multilingual",
        "download_urls": [
            "https://huggingface.co/Systran/faster-whisper-base/resolve/main/.gitattributes",
            "https://huggingface.co/Systran/faster-whisper-base/resolve/main/README.md",
            "https://huggingface.co/Systran/faster-whisper-base/resolve/main/config.json",
            "https://huggingface.co/Systran/faster-whisper-base/resolve/main/model.bin",
            "https://huggingface.co/Systran/faster-whisper-base/resolve/main/tokenizer.json",
            "https://huggingface.co/Systran/faster-whisper-base/resolve/main/vocabulary.txt"
        ]
    },
    "Systran/faster-whisper-small.en": {
        "name": "small.en",
        "model_type": "stt",
        "language": "en",
        "download_urls": [
            "https://huggingface.co/Systran/faster-whisper-small.en/resolve/main/.gitattributes",
            "https://huggingface.co/Systran/faster-whisper-small.en/resolve/main/README.md",
            "https://huggingface.co/Systran/faster-whisper-small.en/resolve/main/config.json",
            "https://huggingface.co/Systran/faster-whisper-small.en/resolve/main/model.bin",
            "https://huggingface.co/Systran/faster-whisper-small.en/resolve/main/tokenizer.json",
            "https://huggingface.co/Systran/faster-whisper-small.en/resolve/main/vocabulary.txt"
        ]
    },
    "Systran/faster-whisper-small": {
        "name": "small",
        "model_type": "stt",
        "language": "multilingual",
        "download_urls": [
            "https://huggingface.co/Systran/faster-whisper-small/resolve/main/.gitattributes",
            "https://huggingface.co/Systran/faster-whisper-small/resolve/main/README.md",
            "https://huggingface.co/Systran/faster-whisper-small/resolve/main/config.json",
            "https://huggingface.co/Systran/faster-whisper-small/resolve/main/model.bin",
            "https://huggingface.co/Systran/faster-whisper-small/resolve/main/tokenizer.json",
            "https://huggingface.co/Systran/faster-whisper-small/resolve/main/vocabulary.txt"
        ]
    },
    "Systran/faster-whisper-medium.en": {
        "name": "medium.en",
        "model_type": "stt",
        "language": "en",
        "download_urls": [
            "https://huggingface.co/Systran/faster-whisper-medium.en/resolve/main/.gitattributes",
            "https://huggingface.co/Systran/faster-whisper-medium.en/resolve/main/README.md",
            "https://huggingface.co/Systran/faster-whisper-medium.en/resolve/main/config.json",
            "https://huggingface.co/Systran/faster-whisper-medium.en/resolve/main/model.bin",
            "https://huggingface.co/Systran/faster-whisper-medium.en/resolve/main/tokenizer.json",
            "https://huggingface.co/Systran/faster-whisper-medium.en/resolve/main/vocabulary.txt"
        ]
    },
    "Systran/faster-whisper-medium": {
        "name": "medium",
        "model_type": "stt",
        "language": "multilingual",
        "download_urls": [
            "https://huggingface.co/Systran/faster-whisper-medium/resolve/main/.gitattributes",
            "https://huggingface.co/Systran/faster-whisper-medium/resolve/main/README.md",
            "https://huggingface.co/Systran/faster-whisper-medium/resolve/main/config.json",
            "https://huggingface.co/Systran/faster-whisper-medium/resolve/main/model.bin",
            "https://huggingface.co/Systran/faster-whisper-medium/resolve/main/tokenizer.json",
            "https://huggingface.co/Systran/faster-whisper-medium/resolve/main/vocabulary.txt"
        ]
    },
    "Systran/faster-whisper-large-v1": {
        "name": "large-v1",
        "model_type": "stt",
        "language": "multilingual",
        "download_urls": [
            "https://huggingface.co/Systran/faster-whisper-large-v1/resolve/main/.gitattributes",
            "https://huggingface.co/Systran/faster-whisper-large-v1/resolve/main/README.md",
            "https://huggingface.co/Systran/faster-whisper-large-v1/resolve/main/config.json",
            "https://huggingface.co/Systran/faster-whisper-large-v1/resolve/main/model.bin",
            "https://huggingface.co/Systran/faster-whisper-large-v1/resolve/main/tokenizer.json",
            "https://huggingface.co/Systran/faster-whisper-large-v1/resolve/main/vocabulary.txt"
        ]
    },
    "Systran/faster-whisper-large-v2": {
        "name": "large-v2",
        "model_type": "stt",
        "language": "multilingual",
        "download_urls": [
            "https://huggingface.co/Systran/faster-whisper-large-v2/resolve/main/.gitattributes",
            "https://huggingface.co/Systran/faster-whisper-large-v2/resolve/main/README.md",
            "https://huggingface.co/Systran/faster-whisper-large-v2/resolve/main/config.json",
            "https://huggingface.co/Systran/faster-whisper-large-v2/resolve/main/model.bin",
            "https://huggingface.co/Systran/faster-whisper-large-v2/resolve/main/tokenizer.json",
            "https://huggingface.co/Systran/faster-whisper-large-v2/resolve/main/vocabulary.txt"
        ]
    },
    "Systran/faster-whisper-large-v3": {
        "name": "large",
        "model_type": "stt",
        "language": "multilingual",
        "download_urls": [
            "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/.gitattributes",
            "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/README.md",
            "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/config.json",
            "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/model.bin",
            "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/preprocessor_config.json",
            "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/tokenizer.json",
            "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/vocabulary.json"
        ]
    },
    "Systran/faster-distil-whisper-large-v2": {
        "name": "distil-large-v2",
        "model_type": "stt",
        "language": "multilingual",
        "download_urls": [
            "https://huggingface.co/Systran/faster-distil-whisper-large-v2/resolve/main/.gitattributes",
            "https://huggingface.co/Systran/faster-distil-whisper-large-v2/resolve/main/README.md",
            "https://huggingface.co/Systran/faster-distil-whisper-large-v2/resolve/main/config.json",
            "https://huggingface.co/Systran/faster-distil-whisper-large-v2/resolve/main/model.bin",
            "https://huggingface.co/Systran/faster-distil-whisper-large-v2/resolve/main/preprocessor_config.json",
            "https://huggingface.co/Systran/faster-distil-whisper-large-v2/resolve/main/tokenizer.json",
            "https://huggingface.co/Systran/faster-distil-whisper-large-v2/resolve/main/vocabulary.json"
        ]
    },
    "Systran/faster-distil-whisper-medium.en": {
        "name": "distil-medium.en",
        "model_type": "stt",
        "language": "en",
        "download_urls": [
            "https://huggingface.co/Systran/faster-distil-whisper-medium.en/resolve/main/.gitattributes",
            "https://huggingface.co/Systran/faster-distil-whisper-medium.en/resolve/main/README.md",
            "https://huggingface.co/Systran/faster-distil-whisper-medium.en/resolve/main/config.json",
            "https://huggingface.co/Systran/faster-distil-whisper-medium.en/resolve/main/model.bin",
            "https://huggingface.co/Systran/faster-distil-whisper-medium.en/resolve/main/preprocessor_config.json",
            "https://huggingface.co/Systran/faster-distil-whisper-medium.en/resolve/main/tokenizer.json",
            "https://huggingface.co/Systran/faster-distil-whisper-medium.en/resolve/main/vocabulary.json"
        ]
    },
    "Systran/faster-distil-whisper-small.en": {
        "name": "distil-small.en",
        "model_type": "stt",
        "language": "en",
        "download_urls": [
            "https://huggingface.co/Systran/faster-distil-whisper-small.en/resolve/main/.gitattributes",
            "https://huggingface.co/Systran/faster-distil-whisper-small.en/resolve/main/README.md",
            "https://huggingface.co/Systran/faster-distil-whisper-small.en/resolve/main/config.json",
            "https://huggingface.co/Systran/faster-distil-whisper-small.en/resolve/main/model.bin",
            "https://huggingface.co/Systran/faster-distil-whisper-small.en/resolve/main/preprocessor_config.json",
            "https://huggingface.co/Systran/faster-distil-whisper-small.en/resolve/main/tokenizer.json",
            "https://huggingface.co/Systran/faster-distil-whisper-small.en/resolve/main/vocabulary.json"
        ]
    },
    "Systran/faster-distil-whisper-large-v3": {
        "name": "distil-large-v3",
        "model_type": "stt",
        "language": "multilingual",
        "download_urls": [
            "https://huggingface.co/Systran/faster-distil-whisper-large-v3/resolve/main/.gitattributes",
            "https://huggingface.co/Systran/faster-distil-whisper-large-v3/resolve/main/README.md",
            "https://huggingface.co/Systran/faster-distil-whisper-large-v3/resolve/main/config.json",
            "https://huggingface.co/Systran/faster-distil-whisper-large-v3/resolve/main/model.bin",
            "https://huggingface.co/Systran/faster-distil-whisper-large-v3/resolve/main/preprocessor_config.json",
            "https://huggingface.co/Systran/faster-distil-whisper-large-v3/resolve/main/tokenizer.json",
            "https://huggingface.co/Systran/faster-distil-whisper-large-v3/resolve/main/vocabulary.json"
        ]
    }
}

    