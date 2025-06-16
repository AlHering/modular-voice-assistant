# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os
import logging
from dotenv import dotenv_values
from . import paths as PATHS


"""
Environment file
"""
ENV_PATH = os.path.join(PATHS.PACKAGE_PATH, ".env")
ENV = dotenv_values(ENV_PATH) if os.path.exists(ENV_PATH) else {}


"""
Logger
"""
LOGGER = logging.Logger("[ModularVoiceAssistant]")


"""
Project information
"""
PROJECT_NAME = "Modular voice assistant"
PROJECT_DESCRIPTION = "A modular voice assistant for general purposes."
PROJECT_VERSION = "v0.0.1"


"""
Network addresses and interfaces
"""
BACKEND_HOST = ENV.get("TEXT_GENERATION_BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(ENV.get("TEXT_GENERATION_BACKEND_PORT", "7861"))
BACKEND_TITLE =  "Modular Voice Assistant Backend"
BACKEND_DESCRIPTION = "Backend interface for modular voice assistants."
BACKEND_VERSION = PROJECT_VERSION
BACKEND_ENDPOINT_BASE = "/api/v1"

FRONTEND_HOST = ENV.get("FRONTEND_HOST", "127.0.0.1")
FRONTEND_PORT = int(ENV.get("FRONTEND_PORT", "8868"))


"""
Components
"""
DEFAULT_SPEECH_RECORDER = {
    "recognizer_parameters": {
        "energy_threshold": 1000,
        "dynamic_energy_threshold": False,
        "pause_threshold": .8
    },
    "microphone_parameters": {
            "sample_rate": 16000,
            "chunk_size": 1024
    }
}
DEFAULT_TRANSCRIBER = {
    "backend": "faster-whisper",
    "model_path": os.path.join(PATHS.MODEL_PATH, 
                                "sound_generation/models/speech_to_text/faster_whisper_models/Systran_faster-whisper-tiny"),
    "model_parameters": {
        "device": "cuda",
        "compute_type": "float32",
        "local_files_only": True
    }
}
DEFAULT_CHAT_REMOTE = {
    "api_base": "http://localhost:8123/v1",
    "chat_parameters": {"model": "llama-3.1-storm-8B-i1-Q4KM"},
    "system_prompt": "You are a helpful and sentient assistant. Your task is to help the user in an effective and concise manner.",
    "stream": True
}
DEFAULT_CHAT_LARGE = {
    "language_model": {
        "backend": "llama-cpp",
        "model_path": os.path.join(PATHS.MODEL_PATH, 
                                        "text_generation/models/text_generation_models/mradermacher_Llama-3.1-Storm-8B-i1-GGUF"),
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
}
DEFAULT_CHAT_SMALL = {
    "language_model": {
        "backend": "llama-cpp",
        "model_path": os.path.join(PATHS.MODEL_PATH, 
                                        "text_generation/models/text_generation_models/bartowski_Qwen2.5-3B-Instruct-GGUF"),
        "model_file": "Qwen2.5-3B-Instruct-Q8_0.gguf",
        "model_parameters": {
            "n_ctx": 4096, 
            "temperature": 0.8, 
            "repetition_penalty": 1.6
        },
        "generating_parameters": {
            "max_tokens": 256
        }
    }
}
DEFAULT_SYNTHESIZER = {
    "backend": "coqui-tts",
    "model_path": os.path.join(PATHS.MODEL_PATH, 
                               "sound_generation/models/text_to_speech/coqui_models/tts_models-multilingual-multi-dataset-xtts_v2"),
    "model_parameters": {
        "config_path": os.path.join(PATHS.MODEL_PATH, 
                                    f"sound_generation/models/text_to_speech/coqui_models/tts_models-multilingual-multi-dataset-xtts_v2/config.json"),
        "gpu": True
    },
    "synthesis_parameters": {
        "speaker_wav": os.path.join(PATHS.MODEL_PATH, "sound_generation/models/text_to_speech/coqui_xtts/examples/female.wav"),
        "language": "en"
    },
    "replace_symbols": {
        "*": "",
        "#": ""
    }
}
DEFAULT_AUDIO_PLAYER = {
    "backend": "pyaudio"
}
DEFAULT_VOICE_ASSISTANT = {
    "stream": True,
    "report": False,
    "forward_logging": True
}
DEFAULT_COMPONENT_CONFIG = {
    "use_remote_llm": False,
    "download_model_files": False,
    "speech_recorder": {},
    "transcriber": DEFAULT_TRANSCRIBER,
    "chat": DEFAULT_CHAT_SMALL,
    "synthesizer": DEFAULT_SYNTHESIZER,
    "audio_player": DEFAULT_AUDIO_PLAYER,
    "voice_assistant": DEFAULT_VOICE_ASSISTANT
}


"""
Others
"""
FILE_UPLOAD_CHUNK_SIZE = 1024*1024
FILE_OUTPUT_CHUNK_SIZE = 1024