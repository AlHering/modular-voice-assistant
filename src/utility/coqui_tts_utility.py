# -*- coding: utf-8 -*-
"""
****************************************************
*                      Utility                 
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import os
from typing import Tuple, List
import os
import pyaudio
from .pyaudio_utility import play_wave
import numpy as np
import torch
from TTS.api import TTS
from TTS.utils.manage import ModelManager


def get_available_models() -> dict:
    """
    Fetches available models.
    Coqui-TTS organizes and identifies models under <model_type>/<language>/<dataset>/<model_name>.
    :return: Dictionary with models and their metadata.
    """
    return TTS().manager.models_dict


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


def dump_normalized_models(output_path: str) -> None:
    """
    Dumps normalized models as json file.
    :param output_path: JSON file path.
    """
    import json
    
    base_dict = get_available_models()
    normalized = {}
    for model_type in base_dict:
        for language in base_dict[model_type]:
            for dataset in base_dict[model_type][language]:
                for model_name in base_dict[model_type][language][dataset]:
                    model_id = f"{model_type}/{language}/{dataset}/{model_name}"
                    normalized[model_id] = {"metadata": base_dict[model_type][language][dataset][model_name]}
                    normalized[model_id]["metadata"]["dataset"] = dataset
                    normalized[model_id].update({
                        "model_type": {"tts_models": "tts", "vocoder_models": "vocoder", "voice_conversion_models": "vc"}[model_type],
                        "language": language,
                        "model_name": model_name,
                        "download_urls": base_dict[model_type][language][dataset][model_name].get(
                            "hf_url", base_dict[model_type][language][dataset][model_name].get("github_rls_url", [])
                        )
                    })
                    if not isinstance(normalized[model_id]["download_urls"], list):
                        normalized[model_id]["download_urls"] = [
                            normalized[model_id]["download_urls"]
                        ]
    open(output_path, "w").write(json.dumps(normalized, ensure_ascii=False, indent=4))


NORMALIZED_MODELS = {
    "tts_models/multilingual/multi-dataset/xtts_v2": {
        "metadata": {
            "description": "XTTS-v2.0.3 by Coqui with 17 languages.",
            "hf_url": [
                "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth",
                "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/config.json",
                "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json",
                "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/hash.md5",
                "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/speakers_xtts.pth"
            ],
            "model_hash": "10f92b55c512af7a8d39d650547a15a7",
            "default_vocoder": None,
            "commit": "480a6cdf7",
            "license": "CPML",
            "contact": "info@coqui.ai",
            "tos_required": True,
            "dataset": "multi-dataset"
        },
        "model_type": "tts",
        "language": "multilingual",
        "name": "xtts_v2",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth",
            "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/config.json",
            "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json",
            "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/hash.md5",
            "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/speakers_xtts.pth"
        ]
    },
    "tts_models/multilingual/multi-dataset/xtts_v1.1": {
        "metadata": {
            "description": "XTTS-v1.1 by Coqui with 14 languages, cross-language voice cloning and reference leak fixed.",
            "hf_url": [
                "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v1/v1.1.2/model.pth",
                "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v1/v1.1.2/config.json",
                "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v1/v1.1.2/vocab.json",
                "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v1/v1.1.2/hash.md5"
            ],
            "model_hash": "7c62beaf58d39b729de287330dc254e7b515677416839b649a50e7cf74c3df59",
            "default_vocoder": None,
            "commit": "82910a63",
            "license": "CPML",
            "contact": "info@coqui.ai",
            "tos_required": True,
            "dataset": "multi-dataset"
        },
        "model_type": "tts",
        "language": "multilingual",
        "name": "xtts_v1.1",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v1/v1.1.2/model.pth",
            "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v1/v1.1.2/config.json",
            "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v1/v1.1.2/vocab.json",
            "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v1/v1.1.2/hash.md5"
        ]
    },
    "tts_models/multilingual/multi-dataset/your_tts": {
        "metadata": {
            "description": "Your TTS model accompanying the paper https://arxiv.org/abs/2112.02418",
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.10.1_models/tts_models--multilingual--multi-dataset--your_tts.zip",
            "default_vocoder": None,
            "commit": "e9a1953e",
            "license": "CC BY-NC-ND 4.0",
            "contact": "egolge@coqui.ai",
            "dataset": "multi-dataset"
        },
        "model_type": "tts",
        "language": "multilingual",
        "name": "your_tts",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.10.1_models/tts_models--multilingual--multi-dataset--your_tts.zip"
        ]
    },
    "tts_models/multilingual/multi-dataset/bark": {
        "metadata": {
            "description": "🐶 Bark TTS model released by suno-ai. You can find the original implementation in https://github.com/suno-ai/bark.",
            "hf_url": [
                "https://coqui.gateway.scarf.sh/hf/bark/coarse_2.pt",
                "https://coqui.gateway.scarf.sh/hf/bark/fine_2.pt",
                "https://coqui.gateway.scarf.sh/hf/bark/text_2.pt",
                "https://coqui.gateway.scarf.sh/hf/bark/config.json",
                "https://coqui.gateway.scarf.sh/hf/bark/tokenizer.pth"
            ],
            "default_vocoder": None,
            "commit": "e9a1953e",
            "license": "MIT",
            "contact": "https://www.suno.ai/",
            "dataset": "multi-dataset"
        },
        "model_type": "tts",
        "language": "multilingual",
        "name": "bark",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/hf/bark/coarse_2.pt",
            "https://coqui.gateway.scarf.sh/hf/bark/fine_2.pt",
            "https://coqui.gateway.scarf.sh/hf/bark/text_2.pt",
            "https://coqui.gateway.scarf.sh/hf/bark/config.json",
            "https://coqui.gateway.scarf.sh/hf/bark/tokenizer.pth"
        ]
    },
    "tts_models/bg/cv/vits": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--bg--cv--vits.zip",
            "default_vocoder": None,
            "commit": None,
            "author": "@NeonGeckoCom",
            "license": "bsd-3-clause",
            "dataset": "cv"
        },
        "model_type": "tts",
        "language": "bg",
        "name": "vits",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--bg--cv--vits.zip"
        ]
    },
    "tts_models/cs/cv/vits": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--cs--cv--vits.zip",
            "default_vocoder": None,
            "commit": None,
            "author": "@NeonGeckoCom",
            "license": "bsd-3-clause",
            "dataset": "cv"
        },
        "model_type": "tts",
        "language": "cs",
        "name": "vits",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--cs--cv--vits.zip"
        ]
    },
    "tts_models/da/cv/vits": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--da--cv--vits.zip",
            "default_vocoder": None,
            "commit": None,
            "author": "@NeonGeckoCom",
            "license": "bsd-3-clause",
            "dataset": "cv"
        },
        "model_type": "tts",
        "language": "da",
        "name": "vits",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--da--cv--vits.zip"
        ]
    },
    "tts_models/et/cv/vits": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--et--cv--vits.zip",
            "default_vocoder": None,
            "commit": None,
            "author": "@NeonGeckoCom",
            "license": "bsd-3-clause",
            "dataset": "cv"
        },
        "model_type": "tts",
        "language": "et",
        "name": "vits",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--et--cv--vits.zip"
        ]
    },
    "tts_models/ga/cv/vits": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--ga--cv--vits.zip",
            "default_vocoder": None,
            "commit": None,
            "author": "@NeonGeckoCom",
            "license": "bsd-3-clause",
            "dataset": "cv"
        },
        "model_type": "tts",
        "language": "ga",
        "name": "vits",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--ga--cv--vits.zip"
        ]
    },
    "tts_models/en/ek1/tacotron2": {
        "metadata": {
            "description": "EK1 en-rp tacotron2 by NMStoker",
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--en--ek1--tacotron2.zip",
            "default_vocoder": "vocoder_models/en/ek1/wavegrad",
            "commit": "c802255",
            "license": "apache 2.0",
            "dataset": "ek1"
        },
        "model_type": "tts",
        "language": "en",
        "name": "tacotron2",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--en--ek1--tacotron2.zip"
        ]
    },
    "tts_models/en/ljspeech/tacotron2-DDC": {
        "metadata": {
            "description": "Tacotron2 with Double Decoder Consistency.",
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--en--ljspeech--tacotron2-DDC.zip",
            "default_vocoder": "vocoder_models/en/ljspeech/hifigan_v2",
            "commit": "bae2ad0f",
            "author": "Eren Gölge @erogol",
            "license": "apache 2.0",
            "contact": "egolge@coqui.com",
            "dataset": "ljspeech"
        },
        "model_type": "tts",
        "language": "en",
        "name": "tacotron2-DDC",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--en--ljspeech--tacotron2-DDC.zip"
        ]
    },
    "tts_models/en/ljspeech/tacotron2-DDC_ph": {
        "metadata": {
            "description": "Tacotron2 with Double Decoder Consistency with phonemes.",
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--en--ljspeech--tacotron2-DDC_ph.zip",
            "default_vocoder": "vocoder_models/en/ljspeech/univnet",
            "commit": "3900448",
            "author": "Eren Gölge @erogol",
            "license": "apache 2.0",
            "contact": "egolge@coqui.com",
            "dataset": "ljspeech"
        },
        "model_type": "tts",
        "language": "en",
        "name": "tacotron2-DDC_ph",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--en--ljspeech--tacotron2-DDC_ph.zip"
        ]
    },
    "tts_models/en/ljspeech/glow-tts": {
        "metadata": {
            "description": "",
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--en--ljspeech--glow-tts.zip",
            "stats_file": None,
            "default_vocoder": "vocoder_models/en/ljspeech/multiband-melgan",
            "commit": "",
            "author": "Eren Gölge @erogol",
            "license": "MPL",
            "contact": "egolge@coqui.com",
            "dataset": "ljspeech"
        },
        "model_type": "tts",
        "language": "en",
        "name": "glow-tts",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--en--ljspeech--glow-tts.zip"
        ]
    },
    "tts_models/en/ljspeech/speedy-speech": {
        "metadata": {
            "description": "Speedy Speech model trained on LJSpeech dataset using the Alignment Network for learning the durations.",
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--en--ljspeech--speedy-speech.zip",
            "stats_file": None,
            "default_vocoder": "vocoder_models/en/ljspeech/hifigan_v2",
            "commit": "4581e3d",
            "author": "Eren Gölge @erogol",
            "license": "apache 2.0",
            "contact": "egolge@coqui.com",
            "dataset": "ljspeech"
        },
        "model_type": "tts",
        "language": "en",
        "name": "speedy-speech",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--en--ljspeech--speedy-speech.zip"
        ]
    },
    "tts_models/en/ljspeech/tacotron2-DCA": {
        "metadata": {
            "description": "",
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--en--ljspeech--tacotron2-DCA.zip",
            "default_vocoder": "vocoder_models/en/ljspeech/multiband-melgan",
            "commit": "",
            "author": "Eren Gölge @erogol",
            "license": "MPL",
            "contact": "egolge@coqui.com",
            "dataset": "ljspeech"
        },
        "model_type": "tts",
        "language": "en",
        "name": "tacotron2-DCA",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--en--ljspeech--tacotron2-DCA.zip"
        ]
    },
    "tts_models/en/ljspeech/vits": {
        "metadata": {
            "description": "VITS is an End2End TTS model trained on LJSpeech dataset with phonemes.",
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--en--ljspeech--vits.zip",
            "default_vocoder": None,
            "commit": "3900448",
            "author": "Eren Gölge @erogol",
            "license": "apache 2.0",
            "contact": "egolge@coqui.com",
            "dataset": "ljspeech"
        },
        "model_type": "tts",
        "language": "en",
        "name": "vits",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--en--ljspeech--vits.zip"
        ]
    },
    "tts_models/en/ljspeech/vits--neon": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--en--ljspeech--vits.zip",
            "default_vocoder": None,
            "author": "@NeonGeckoCom",
            "license": "bsd-3-clause",
            "contact": None,
            "commit": None,
            "dataset": "ljspeech"
        },
        "model_type": "tts",
        "language": "en",
        "name": "vits--neon",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--en--ljspeech--vits.zip"
        ]
    },
    "tts_models/en/ljspeech/fast_pitch": {
        "metadata": {
            "description": "FastPitch model trained on LJSpeech using the Aligner Network",
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--en--ljspeech--fast_pitch.zip",
            "default_vocoder": "vocoder_models/en/ljspeech/hifigan_v2",
            "commit": "b27b3ba",
            "author": "Eren Gölge @erogol",
            "license": "apache 2.0",
            "contact": "egolge@coqui.com",
            "dataset": "ljspeech"
        },
        "model_type": "tts",
        "language": "en",
        "name": "fast_pitch",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--en--ljspeech--fast_pitch.zip"
        ]
    },
    "tts_models/en/ljspeech/overflow": {
        "metadata": {
            "description": "Overflow model trained on LJSpeech",
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.10.0_models/tts_models--en--ljspeech--overflow.zip",
            "default_vocoder": "vocoder_models/en/ljspeech/hifigan_v2",
            "commit": "3b1a28f",
            "author": "Eren Gölge @erogol",
            "license": "apache 2.0",
            "contact": "egolge@coqui.ai",
            "dataset": "ljspeech"
        },
        "model_type": "tts",
        "language": "en",
        "name": "overflow",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.10.0_models/tts_models--en--ljspeech--overflow.zip"
        ]
    },
    "tts_models/en/ljspeech/neural_hmm": {
        "metadata": {
            "description": "Neural HMM model trained on LJSpeech",
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.11.0_models/tts_models--en--ljspeech--neural_hmm.zip",
            "default_vocoder": "vocoder_models/en/ljspeech/hifigan_v2",
            "commit": "3b1a28f",
            "author": "Shivam Metha @shivammehta25",
            "license": "apache 2.0",
            "contact": "d83ee8fe45e3c0d776d4a865aca21d7c2ac324c4",
            "dataset": "ljspeech"
        },
        "model_type": "tts",
        "language": "en",
        "name": "neural_hmm",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.11.0_models/tts_models--en--ljspeech--neural_hmm.zip"
        ]
    },
    "tts_models/en/vctk/vits": {
        "metadata": {
            "description": "VITS End2End TTS model trained on VCTK dataset with 109 different speakers with EN accent.",
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--en--vctk--vits.zip",
            "default_vocoder": None,
            "commit": "3900448",
            "author": "Eren @erogol",
            "license": "apache 2.0",
            "contact": "egolge@coqui.ai",
            "dataset": "vctk"
        },
        "model_type": "tts",
        "language": "en",
        "name": "vits",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--en--vctk--vits.zip"
        ]
    },
    "tts_models/en/vctk/fast_pitch": {
        "metadata": {
            "description": "FastPitch model trained on VCTK dataseset.",
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--en--vctk--fast_pitch.zip",
            "default_vocoder": None,
            "commit": "bdab788d",
            "author": "Eren @erogol",
            "license": "CC BY-NC-ND 4.0",
            "contact": "egolge@coqui.ai",
            "dataset": "vctk"
        },
        "model_type": "tts",
        "language": "en",
        "name": "fast_pitch",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--en--vctk--fast_pitch.zip"
        ]
    },
    "tts_models/en/sam/tacotron-DDC": {
        "metadata": {
            "description": "Tacotron2 with Double Decoder Consistency trained with Aceenture's Sam dataset.",
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--en--sam--tacotron-DDC.zip",
            "default_vocoder": "vocoder_models/en/sam/hifigan_v2",
            "commit": "bae2ad0f",
            "author": "Eren Gölge @erogol",
            "license": "apache 2.0",
            "contact": "egolge@coqui.com",
            "dataset": "sam"
        },
        "model_type": "tts",
        "language": "en",
        "name": "tacotron-DDC",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--en--sam--tacotron-DDC.zip"
        ]
    },
    "tts_models/en/blizzard2013/capacitron-t2-c50": {
        "metadata": {
            "description": "Capacitron additions to Tacotron 2 with Capacity at 50 as in https://arxiv.org/pdf/1906.03402.pdf",
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.7.0_models/tts_models--en--blizzard2013--capacitron-t2-c50.zip",
            "commit": "d6284e7",
            "default_vocoder": "vocoder_models/en/blizzard2013/hifigan_v2",
            "author": "Adam Froghyar @a-froghyar",
            "license": "apache 2.0",
            "contact": "adamfroghyar@gmail.com",
            "dataset": "blizzard2013"
        },
        "model_type": "tts",
        "language": "en",
        "name": "capacitron-t2-c50",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.7.0_models/tts_models--en--blizzard2013--capacitron-t2-c50.zip"
        ]
    },
    "tts_models/en/blizzard2013/capacitron-t2-c150_v2": {
        "metadata": {
            "description": "Capacitron additions to Tacotron 2 with Capacity at 150 as in https://arxiv.org/pdf/1906.03402.pdf",
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.7.1_models/tts_models--en--blizzard2013--capacitron-t2-c150_v2.zip",
            "commit": "a67039d",
            "default_vocoder": "vocoder_models/en/blizzard2013/hifigan_v2",
            "author": "Adam Froghyar @a-froghyar",
            "license": "apache 2.0",
            "contact": "adamfroghyar@gmail.com",
            "dataset": "blizzard2013"
        },
        "model_type": "tts",
        "language": "en",
        "name": "capacitron-t2-c150_v2",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.7.1_models/tts_models--en--blizzard2013--capacitron-t2-c150_v2.zip"
        ]
    },
    "tts_models/en/multi-dataset/tortoise-v2": {
        "metadata": {
            "description": "Tortoise tts model https://github.com/neonbjb/tortoise-tts",
            "github_rls_url": [
                "https://coqui.gateway.scarf.sh/v0.14.1_models/autoregressive.pth",
                "https://coqui.gateway.scarf.sh/v0.14.1_models/clvp2.pth",
                "https://coqui.gateway.scarf.sh/v0.14.1_models/cvvp.pth",
                "https://coqui.gateway.scarf.sh/v0.14.1_models/diffusion_decoder.pth",
                "https://coqui.gateway.scarf.sh/v0.14.1_models/rlg_auto.pth",
                "https://coqui.gateway.scarf.sh/v0.14.1_models/rlg_diffuser.pth",
                "https://coqui.gateway.scarf.sh/v0.14.1_models/vocoder.pth",
                "https://coqui.gateway.scarf.sh/v0.14.1_models/mel_norms.pth",
                "https://coqui.gateway.scarf.sh/v0.14.1_models/config.json"
            ],
            "commit": "c1875f6",
            "default_vocoder": None,
            "author": "@neonbjb - James Betker, @manmay-nakhashi Manmay Nakhashi",
            "license": "apache 2.0",
            "dataset": "multi-dataset"
        },
        "model_type": "tts",
        "language": "en",
        "name": "tortoise-v2",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.14.1_models/autoregressive.pth",
            "https://coqui.gateway.scarf.sh/v0.14.1_models/clvp2.pth",
            "https://coqui.gateway.scarf.sh/v0.14.1_models/cvvp.pth",
            "https://coqui.gateway.scarf.sh/v0.14.1_models/diffusion_decoder.pth",
            "https://coqui.gateway.scarf.sh/v0.14.1_models/rlg_auto.pth",
            "https://coqui.gateway.scarf.sh/v0.14.1_models/rlg_diffuser.pth",
            "https://coqui.gateway.scarf.sh/v0.14.1_models/vocoder.pth",
            "https://coqui.gateway.scarf.sh/v0.14.1_models/mel_norms.pth",
            "https://coqui.gateway.scarf.sh/v0.14.1_models/config.json"
        ]
    },
    "tts_models/en/jenny/jenny": {
        "metadata": {
            "description": "VITS model trained with Jenny(Dioco) dataset. Named as Jenny as demanded by the license. Original URL for the model https://www.kaggle.com/datasets/noml4u/tts-models--en--jenny-dioco--vits",
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.14.0_models/tts_models--en--jenny--jenny.zip",
            "default_vocoder": None,
            "commit": "ba40a1c",
            "license": "custom - see https://github.com/dioco-group/jenny-tts-dataset#important",
            "author": "@noml4u",
            "dataset": "jenny"
        },
        "model_type": "tts",
        "language": "en",
        "name": "jenny",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.14.0_models/tts_models--en--jenny--jenny.zip"
        ]
    },
    "tts_models/es/mai/tacotron2-DDC": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--es--mai--tacotron2-DDC.zip",
            "default_vocoder": "vocoder_models/universal/libri-tts/fullband-melgan",
            "commit": "",
            "author": "Eren Gölge @erogol",
            "license": "MPL",
            "contact": "egolge@coqui.com",
            "dataset": "mai"
        },
        "model_type": "tts",
        "language": "es",
        "name": "tacotron2-DDC",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--es--mai--tacotron2-DDC.zip"
        ]
    },
    "tts_models/es/css10/vits": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--es--css10--vits.zip",
            "default_vocoder": None,
            "commit": None,
            "author": "@NeonGeckoCom",
            "license": "bsd-3-clause",
            "dataset": "css10"
        },
        "model_type": "tts",
        "language": "es",
        "name": "vits",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--es--css10--vits.zip"
        ]
    },
    "tts_models/fr/mai/tacotron2-DDC": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--fr--mai--tacotron2-DDC.zip",
            "default_vocoder": "vocoder_models/universal/libri-tts/fullband-melgan",
            "commit": None,
            "author": "Eren Gölge @erogol",
            "license": "MPL",
            "contact": "egolge@coqui.com",
            "dataset": "mai"
        },
        "model_type": "tts",
        "language": "fr",
        "name": "tacotron2-DDC",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--fr--mai--tacotron2-DDC.zip"
        ]
    },
    "tts_models/fr/css10/vits": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--fr--css10--vits.zip",
            "default_vocoder": None,
            "commit": None,
            "author": "@NeonGeckoCom",
            "license": "bsd-3-clause",
            "dataset": "css10"
        },
        "model_type": "tts",
        "language": "fr",
        "name": "vits",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--fr--css10--vits.zip"
        ]
    },
    "tts_models/uk/mai/glow-tts": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--uk--mai--glow-tts.zip",
            "author": "@robinhad",
            "commit": "bdab788d",
            "license": "MIT",
            "contact": "",
            "default_vocoder": "vocoder_models/uk/mai/multiband-melgan",
            "dataset": "mai"
        },
        "model_type": "tts",
        "language": "uk",
        "name": "glow-tts",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--uk--mai--glow-tts.zip"
        ]
    },
    "tts_models/uk/mai/vits": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--uk--mai--vits.zip",
            "default_vocoder": None,
            "commit": None,
            "author": "@NeonGeckoCom",
            "license": "bsd-3-clause",
            "dataset": "mai"
        },
        "model_type": "tts",
        "language": "uk",
        "name": "vits",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--uk--mai--vits.zip"
        ]
    },
    "tts_models/zh-CN/baker/tacotron2-DDC-GST": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--zh-CN--baker--tacotron2-DDC-GST.zip",
            "commit": "unknown",
            "author": "@kirianguiller",
            "license": "apache 2.0",
            "default_vocoder": None,
            "dataset": "baker"
        },
        "model_type": "tts",
        "language": "zh-CN",
        "name": "tacotron2-DDC-GST",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--zh-CN--baker--tacotron2-DDC-GST.zip"
        ]
    },
    "tts_models/nl/mai/tacotron2-DDC": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--nl--mai--tacotron2-DDC.zip",
            "author": "@r-dh",
            "license": "apache 2.0",
            "default_vocoder": "vocoder_models/nl/mai/parallel-wavegan",
            "stats_file": None,
            "commit": "540d811",
            "dataset": "mai"
        },
        "model_type": "tts",
        "language": "nl",
        "name": "tacotron2-DDC",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--nl--mai--tacotron2-DDC.zip"
        ]
    },
    "tts_models/nl/css10/vits": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--nl--css10--vits.zip",
            "default_vocoder": None,
            "commit": None,
            "author": "@NeonGeckoCom",
            "license": "bsd-3-clause",
            "dataset": "css10"
        },
        "model_type": "tts",
        "language": "nl",
        "name": "vits",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--nl--css10--vits.zip"
        ]
    },
    "tts_models/de/thorsten/tacotron2-DCA": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--de--thorsten--tacotron2-DCA.zip",
            "default_vocoder": "vocoder_models/de/thorsten/fullband-melgan",
            "author": "@thorstenMueller",
            "license": "apache 2.0",
            "commit": "unknown",
            "dataset": "thorsten"
        },
        "model_type": "tts",
        "language": "de",
        "name": "tacotron2-DCA",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--de--thorsten--tacotron2-DCA.zip"
        ]
    },
    "tts_models/de/thorsten/vits": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.7.0_models/tts_models--de--thorsten--vits.zip",
            "default_vocoder": None,
            "author": "@thorstenMueller",
            "license": "apache 2.0",
            "commit": "unknown",
            "dataset": "thorsten"
        },
        "model_type": "tts",
        "language": "de",
        "name": "vits",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.7.0_models/tts_models--de--thorsten--vits.zip"
        ]
    },
    "tts_models/de/thorsten/tacotron2-DDC": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--de--thorsten--tacotron2-DDC.zip",
            "default_vocoder": "vocoder_models/de/thorsten/hifigan_v1",
            "description": "Thorsten-Dec2021-22k-DDC",
            "author": "@thorstenMueller",
            "license": "apache 2.0",
            "commit": "unknown",
            "dataset": "thorsten"
        },
        "model_type": "tts",
        "language": "de",
        "name": "tacotron2-DDC",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--de--thorsten--tacotron2-DDC.zip"
        ]
    },
    "tts_models/de/css10/vits-neon": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--de--css10--vits.zip",
            "default_vocoder": None,
            "author": "@NeonGeckoCom",
            "license": "bsd-3-clause",
            "commit": None,
            "dataset": "css10"
        },
        "model_type": "tts",
        "language": "de",
        "name": "vits-neon",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--de--css10--vits.zip"
        ]
    },
    "tts_models/ja/kokoro/tacotron2-DDC": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--ja--kokoro--tacotron2-DDC.zip",
            "default_vocoder": "vocoder_models/ja/kokoro/hifigan_v1",
            "description": "Tacotron2 with Double Decoder Consistency trained with Kokoro Speech Dataset.",
            "author": "@kaiidams",
            "license": "apache 2.0",
            "commit": "401fbd89",
            "dataset": "kokoro"
        },
        "model_type": "tts",
        "language": "ja",
        "name": "tacotron2-DDC",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--ja--kokoro--tacotron2-DDC.zip"
        ]
    },
    "tts_models/tr/common-voice/glow-tts": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--tr--common-voice--glow-tts.zip",
            "default_vocoder": "vocoder_models/tr/common-voice/hifigan",
            "license": "MIT",
            "description": "Turkish GlowTTS model using an unknown speaker from the Common-Voice dataset.",
            "author": "Fatih Akademi",
            "commit": None,
            "dataset": "common-voice"
        },
        "model_type": "tts",
        "language": "tr",
        "name": "glow-tts",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--tr--common-voice--glow-tts.zip"
        ]
    },
    "tts_models/it/mai_female/glow-tts": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--it--mai_female--glow-tts.zip",
            "default_vocoder": None,
            "description": "GlowTTS model as explained on https://github.com/coqui-ai/TTS/issues/1148.",
            "author": "@nicolalandro",
            "license": "apache 2.0",
            "commit": None,
            "dataset": "mai_female"
        },
        "model_type": "tts",
        "language": "it",
        "name": "glow-tts",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--it--mai_female--glow-tts.zip"
        ]
    },
    "tts_models/it/mai_female/vits": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--it--mai_female--vits.zip",
            "default_vocoder": None,
            "description": "GlowTTS model as explained on https://github.com/coqui-ai/TTS/issues/1148.",
            "author": "@nicolalandro",
            "license": "apache 2.0",
            "commit": None,
            "dataset": "mai_female"
        },
        "model_type": "tts",
        "language": "it",
        "name": "vits",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--it--mai_female--vits.zip"
        ]
    },
    "tts_models/it/mai_male/glow-tts": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--it--mai_male--glow-tts.zip",
            "default_vocoder": None,
            "description": "GlowTTS model as explained on https://github.com/coqui-ai/TTS/issues/1148.",
            "author": "@nicolalandro",
            "license": "apache 2.0",
            "commit": None,
            "dataset": "mai_male"
        },
        "model_type": "tts",
        "language": "it",
        "name": "glow-tts",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--it--mai_male--glow-tts.zip"
        ]
    },
    "tts_models/it/mai_male/vits": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--it--mai_male--vits.zip",
            "default_vocoder": None,
            "description": "GlowTTS model as explained on https://github.com/coqui-ai/TTS/issues/1148.",
            "author": "@nicolalandro",
            "license": "apache 2.0",
            "commit": None,
            "dataset": "mai_male"
        },
        "model_type": "tts",
        "language": "it",
        "name": "vits",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--it--mai_male--vits.zip"
        ]
    },
    "tts_models/ewe/openbible/vits": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.2_models/tts_models--ewe--openbible--vits.zip",
            "default_vocoder": None,
            "license": "CC-BY-SA 4.0",
            "description": "Original work (audio and text) by Biblica available for free at www.biblica.com and open.bible.",
            "author": "@coqui_ai",
            "commit": "1b22f03",
            "dataset": "openbible"
        },
        "model_type": "tts",
        "language": "ewe",
        "name": "vits",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.2_models/tts_models--ewe--openbible--vits.zip"
        ]
    },
    "tts_models/hau/openbible/vits": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.2_models/tts_models--hau--openbible--vits.zip",
            "default_vocoder": None,
            "license": "CC-BY-SA 4.0",
            "description": "Original work (audio and text) by Biblica available for free at www.biblica.com and open.bible.",
            "author": "@coqui_ai",
            "commit": "1b22f03",
            "dataset": "openbible"
        },
        "model_type": "tts",
        "language": "hau",
        "name": "vits",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.2_models/tts_models--hau--openbible--vits.zip"
        ]
    },
    "tts_models/lin/openbible/vits": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.2_models/tts_models--lin--openbible--vits.zip",
            "default_vocoder": None,
            "license": "CC-BY-SA 4.0",
            "description": "Original work (audio and text) by Biblica available for free at www.biblica.com and open.bible.",
            "author": "@coqui_ai",
            "commit": "1b22f03",
            "dataset": "openbible"
        },
        "model_type": "tts",
        "language": "lin",
        "name": "vits",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.2_models/tts_models--lin--openbible--vits.zip"
        ]
    },
    "tts_models/tw_akuapem/openbible/vits": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.2_models/tts_models--tw_akuapem--openbible--vits.zip",
            "default_vocoder": None,
            "license": "CC-BY-SA 4.0",
            "description": "Original work (audio and text) by Biblica available for free at www.biblica.com and open.bible.",
            "author": "@coqui_ai",
            "commit": "1b22f03",
            "dataset": "openbible"
        },
        "model_type": "tts",
        "language": "tw_akuapem",
        "name": "vits",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.2_models/tts_models--tw_akuapem--openbible--vits.zip"
        ]
    },
    "tts_models/tw_asante/openbible/vits": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.2_models/tts_models--tw_asante--openbible--vits.zip",
            "default_vocoder": None,
            "license": "CC-BY-SA 4.0",
            "description": "Original work (audio and text) by Biblica available for free at www.biblica.com and open.bible.",
            "author": "@coqui_ai",
            "commit": "1b22f03",
            "dataset": "openbible"
        },
        "model_type": "tts",
        "language": "tw_asante",
        "name": "vits",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.2_models/tts_models--tw_asante--openbible--vits.zip"
        ]
    },
    "tts_models/yor/openbible/vits": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.2_models/tts_models--yor--openbible--vits.zip",
            "default_vocoder": None,
            "license": "CC-BY-SA 4.0",
            "description": "Original work (audio and text) by Biblica available for free at www.biblica.com and open.bible.",
            "author": "@coqui_ai",
            "commit": "1b22f03",
            "dataset": "openbible"
        },
        "model_type": "tts",
        "language": "yor",
        "name": "vits",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.2_models/tts_models--yor--openbible--vits.zip"
        ]
    },
    "tts_models/hu/css10/vits": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--hu--css10--vits.zip",
            "default_vocoder": None,
            "commit": None,
            "author": "@NeonGeckoCom",
            "license": "bsd-3-clause",
            "dataset": "css10"
        },
        "model_type": "tts",
        "language": "hu",
        "name": "vits",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--hu--css10--vits.zip"
        ]
    },
    "tts_models/el/cv/vits": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--el--cv--vits.zip",
            "default_vocoder": None,
            "commit": None,
            "author": "@NeonGeckoCom",
            "license": "bsd-3-clause",
            "dataset": "cv"
        },
        "model_type": "tts",
        "language": "el",
        "name": "vits",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--el--cv--vits.zip"
        ]
    },
    "tts_models/fi/css10/vits": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--fi--css10--vits.zip",
            "default_vocoder": None,
            "commit": None,
            "author": "@NeonGeckoCom",
            "license": "bsd-3-clause",
            "dataset": "css10"
        },
        "model_type": "tts",
        "language": "fi",
        "name": "vits",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--fi--css10--vits.zip"
        ]
    },
    "tts_models/hr/cv/vits": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--hr--cv--vits.zip",
            "default_vocoder": None,
            "commit": None,
            "author": "@NeonGeckoCom",
            "license": "bsd-3-clause",
            "dataset": "cv"
        },
        "model_type": "tts",
        "language": "hr",
        "name": "vits",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--hr--cv--vits.zip"
        ]
    },
    "tts_models/lt/cv/vits": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--lt--cv--vits.zip",
            "default_vocoder": None,
            "commit": None,
            "author": "@NeonGeckoCom",
            "license": "bsd-3-clause",
            "dataset": "cv"
        },
        "model_type": "tts",
        "language": "lt",
        "name": "vits",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--lt--cv--vits.zip"
        ]
    },
    "tts_models/lv/cv/vits": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--lv--cv--vits.zip",
            "default_vocoder": None,
            "commit": None,
            "author": "@NeonGeckoCom",
            "license": "bsd-3-clause",
            "dataset": "cv"
        },
        "model_type": "tts",
        "language": "lv",
        "name": "vits",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--lv--cv--vits.zip"
        ]
    },
    "tts_models/mt/cv/vits": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--mt--cv--vits.zip",
            "default_vocoder": None,
            "commit": None,
            "author": "@NeonGeckoCom",
            "license": "bsd-3-clause",
            "dataset": "cv"
        },
        "model_type": "tts",
        "language": "mt",
        "name": "vits",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--mt--cv--vits.zip"
        ]
    },
    "tts_models/pl/mai_female/vits": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--pl--mai_female--vits.zip",
            "default_vocoder": None,
            "commit": None,
            "author": "@NeonGeckoCom",
            "license": "bsd-3-clause",
            "dataset": "mai_female"
        },
        "model_type": "tts",
        "language": "pl",
        "name": "vits",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--pl--mai_female--vits.zip"
        ]
    },
    "tts_models/pt/cv/vits": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--pt--cv--vits.zip",
            "default_vocoder": None,
            "commit": None,
            "author": "@NeonGeckoCom",
            "license": "bsd-3-clause",
            "dataset": "cv"
        },
        "model_type": "tts",
        "language": "pt",
        "name": "vits",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--pt--cv--vits.zip"
        ]
    },
    "tts_models/ro/cv/vits": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--ro--cv--vits.zip",
            "default_vocoder": None,
            "commit": None,
            "author": "@NeonGeckoCom",
            "license": "bsd-3-clause",
            "dataset": "cv"
        },
        "model_type": "tts",
        "language": "ro",
        "name": "vits",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--ro--cv--vits.zip"
        ]
    },
    "tts_models/sk/cv/vits": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--sk--cv--vits.zip",
            "default_vocoder": None,
            "commit": None,
            "author": "@NeonGeckoCom",
            "license": "bsd-3-clause",
            "dataset": "cv"
        },
        "model_type": "tts",
        "language": "sk",
        "name": "vits",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--sk--cv--vits.zip"
        ]
    },
    "tts_models/sl/cv/vits": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--sl--cv--vits.zip",
            "default_vocoder": None,
            "commit": None,
            "author": "@NeonGeckoCom",
            "license": "bsd-3-clause",
            "dataset": "cv"
        },
        "model_type": "tts",
        "language": "sl",
        "name": "vits",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--sl--cv--vits.zip"
        ]
    },
    "tts_models/sv/cv/vits": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--sv--cv--vits.zip",
            "default_vocoder": None,
            "commit": None,
            "author": "@NeonGeckoCom",
            "license": "bsd-3-clause",
            "dataset": "cv"
        },
        "model_type": "tts",
        "language": "sv",
        "name": "vits",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.8.0_models/tts_models--sv--cv--vits.zip"
        ]
    },
    "tts_models/ca/custom/vits": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.10.1_models/tts_models--ca--custom--vits.zip",
            "default_vocoder": None,
            "commit": None,
            "description": " It is trained from zero with 101460 utterances consisting of 257 speakers, approx 138 hours of speech. We used three datasets;\nFestcat and Google Catalan TTS (both TTS datasets) and also a part of Common Voice 8. It is trained with TTS v0.8.0.\nhttps://github.com/coqui-ai/TTS/discussions/930#discussioncomment-4466345",
            "author": "@gullabi",
            "license": "CC-BY-4.0",
            "dataset": "custom"
        },
        "model_type": "tts",
        "language": "ca",
        "name": "vits",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.10.1_models/tts_models--ca--custom--vits.zip"
        ]
    },
    "tts_models/fa/custom/glow-tts": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.10.1_models/tts_models--fa--custom--glow-tts.zip",
            "default_vocoder": None,
            "commit": None,
            "description": "persian-tts-female-glow_tts model for text to speech purposes. Single-speaker female voice Trained on persian-tts-dataset-famale. \nThis model has no compatible vocoder thus the output quality is not very good. \nDataset: https://www.kaggle.com/datasets/magnoliasis/persian-tts-dataset-famale.",
            "author": "@karim23657",
            "license": "CC-BY-4.0",
            "dataset": "custom"
        },
        "model_type": "tts",
        "language": "fa",
        "name": "glow-tts",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.10.1_models/tts_models--fa--custom--glow-tts.zip"
        ]
    },
    "tts_models/bn/custom/vits-male": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.13.3_models/tts_models--bn--custom--vits_male.zip",
            "default_vocoder": None,
            "commit": None,
            "description": "Single speaker Bangla male model. For more information -> https://github.com/mobassir94/comprehensive-bangla-tts",
            "author": "@mobassir94",
            "license": "Apache 2.0",
            "dataset": "custom"
        },
        "model_type": "tts",
        "language": "bn",
        "name": "vits-male",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.13.3_models/tts_models--bn--custom--vits_male.zip"
        ]
    },
    "tts_models/bn/custom/vits-female": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.13.3_models/tts_models--bn--custom--vits_female.zip",
            "default_vocoder": None,
            "commit": None,
            "description": "Single speaker Bangla female model. For more information -> https://github.com/mobassir94/comprehensive-bangla-tts",
            "author": "@mobassir94",
            "license": "Apache 2.0",
            "dataset": "custom"
        },
        "model_type": "tts",
        "language": "bn",
        "name": "vits-female",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.13.3_models/tts_models--bn--custom--vits_female.zip"
        ]
    },
    "tts_models/be/common-voice/glow-tts": {
        "metadata": {
            "description": "Belarusian GlowTTS model created by @alex73 (Github).",
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.16.6/tts_models--be--common-voice--glow-tts.zip",
            "default_vocoder": "vocoder_models/be/common-voice/hifigan",
            "commit": "c0aabb85",
            "license": "CC-BY-SA 4.0",
            "contact": "alex73mail@gmail.com",
            "dataset": "common-voice"
        },
        "model_type": "tts",
        "language": "be",
        "name": "glow-tts",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.16.6/tts_models--be--common-voice--glow-tts.zip"
        ]
    },
    "vocoder_models/universal/libri-tts/wavegrad": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.1_models/vocoder_models--universal--libri-tts--wavegrad.zip",
            "commit": "ea976b0",
            "author": "Eren Gölge @erogol",
            "license": "MPL",
            "contact": "egolge@coqui.com",
            "dataset": "libri-tts"
        },
        "model_type": "vocoder",
        "language": "universal",
        "name": "wavegrad",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.1_models/vocoder_models--universal--libri-tts--wavegrad.zip"
        ]
    },
    "vocoder_models/universal/libri-tts/fullband-melgan": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.1_models/vocoder_models--universal--libri-tts--fullband-melgan.zip",
            "commit": "4132240",
            "author": "Eren Gölge @erogol",
            "license": "MPL",
            "contact": "egolge@coqui.com",
            "dataset": "libri-tts"
        },
        "model_type": "vocoder",
        "language": "universal",
        "name": "fullband-melgan",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.1_models/vocoder_models--universal--libri-tts--fullband-melgan.zip"
        ]
    },
    "vocoder_models/en/ek1/wavegrad": {
        "metadata": {
            "description": "EK1 en-rp wavegrad by NMStoker",
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.1_models/vocoder_models--en--ek1--wavegrad.zip",
            "commit": "c802255",
            "license": "apache 2.0",
            "dataset": "ek1"
        },
        "model_type": "vocoder",
        "language": "en",
        "name": "wavegrad",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.1_models/vocoder_models--en--ek1--wavegrad.zip"
        ]
    },
    "vocoder_models/en/ljspeech/multiband-melgan": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.1_models/vocoder_models--en--ljspeech--multiband-melgan.zip",
            "commit": "ea976b0",
            "author": "Eren Gölge @erogol",
            "license": "MPL",
            "contact": "egolge@coqui.com",
            "dataset": "ljspeech"
        },
        "model_type": "vocoder",
        "language": "en",
        "name": "multiband-melgan",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.1_models/vocoder_models--en--ljspeech--multiband-melgan.zip"
        ]
    },
    "vocoder_models/en/ljspeech/hifigan_v2": {
        "metadata": {
            "description": "HiFiGAN_v2 LJSpeech vocoder from https://arxiv.org/abs/2010.05646.",
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.1_models/vocoder_models--en--ljspeech--hifigan_v2.zip",
            "commit": "bae2ad0f",
            "author": "@erogol",
            "license": "apache 2.0",
            "contact": "egolge@coqui.ai",
            "dataset": "ljspeech"
        },
        "model_type": "vocoder",
        "language": "en",
        "name": "hifigan_v2",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.1_models/vocoder_models--en--ljspeech--hifigan_v2.zip"
        ]
    },
    "vocoder_models/en/ljspeech/univnet": {
        "metadata": {
            "description": "UnivNet model finetuned on TacotronDDC_ph spectrograms for better compatibility.",
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.1_models/vocoder_models--en--ljspeech--univnet_v2.zip",
            "commit": "4581e3d",
            "author": "Eren @erogol",
            "license": "apache 2.0",
            "contact": "egolge@coqui.ai",
            "dataset": "ljspeech"
        },
        "model_type": "vocoder",
        "language": "en",
        "name": "univnet",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.1_models/vocoder_models--en--ljspeech--univnet_v2.zip"
        ]
    },
    "vocoder_models/en/blizzard2013/hifigan_v2": {
        "metadata": {
            "description": "HiFiGAN_v2 LJSpeech vocoder from https://arxiv.org/abs/2010.05646.",
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.7.0_models/vocoder_models--en--blizzard2013--hifigan_v2.zip",
            "commit": "d6284e7",
            "author": "Adam Froghyar @a-froghyar",
            "license": "apache 2.0",
            "contact": "adamfroghyar@gmail.com",
            "dataset": "blizzard2013"
        },
        "model_type": "vocoder",
        "language": "en",
        "name": "hifigan_v2",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.7.0_models/vocoder_models--en--blizzard2013--hifigan_v2.zip"
        ]
    },
    "vocoder_models/en/vctk/hifigan_v2": {
        "metadata": {
            "description": "Finetuned and intended to be used with tts_models/en/vctk/sc-glow-tts",
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.1_models/vocoder_models--en--vctk--hifigan_v2.zip",
            "commit": "2f07160",
            "author": "Edresson Casanova",
            "license": "apache 2.0",
            "contact": "",
            "dataset": "vctk"
        },
        "model_type": "vocoder",
        "language": "en",
        "name": "hifigan_v2",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.1_models/vocoder_models--en--vctk--hifigan_v2.zip"
        ]
    },
    "vocoder_models/en/sam/hifigan_v2": {
        "metadata": {
            "description": "Finetuned and intended to be used with tts_models/en/sam/tacotron_DDC",
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.1_models/vocoder_models--en--sam--hifigan_v2.zip",
            "commit": "2f07160",
            "author": "Eren Gölge @erogol",
            "license": "apache 2.0",
            "contact": "egolge@coqui.ai",
            "dataset": "sam"
        },
        "model_type": "vocoder",
        "language": "en",
        "name": "hifigan_v2",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.1_models/vocoder_models--en--sam--hifigan_v2.zip"
        ]
    },
    "vocoder_models/nl/mai/parallel-wavegan": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.1_models/vocoder_models--nl--mai--parallel-wavegan.zip",
            "author": "@r-dh",
            "license": "apache 2.0",
            "commit": "unknown",
            "dataset": "mai"
        },
        "model_type": "vocoder",
        "language": "nl",
        "name": "parallel-wavegan",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.1_models/vocoder_models--nl--mai--parallel-wavegan.zip"
        ]
    },
    "vocoder_models/de/thorsten/wavegrad": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.1_models/vocoder_models--de--thorsten--wavegrad.zip",
            "author": "@thorstenMueller",
            "license": "apache 2.0",
            "commit": "unknown",
            "dataset": "thorsten"
        },
        "model_type": "vocoder",
        "language": "de",
        "name": "wavegrad",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.1_models/vocoder_models--de--thorsten--wavegrad.zip"
        ]
    },
    "vocoder_models/de/thorsten/fullband-melgan": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.1_models/vocoder_models--de--thorsten--fullband-melgan.zip",
            "author": "@thorstenMueller",
            "license": "apache 2.0",
            "commit": "unknown",
            "dataset": "thorsten"
        },
        "model_type": "vocoder",
        "language": "de",
        "name": "fullband-melgan",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.1_models/vocoder_models--de--thorsten--fullband-melgan.zip"
        ]
    },
    "vocoder_models/de/thorsten/hifigan_v1": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.8.0_models/vocoder_models--de--thorsten--hifigan_v1.zip",
            "description": "HifiGAN vocoder model for Thorsten Neutral Dec2021 22k Samplerate Tacotron2 DDC model",
            "author": "@thorstenMueller",
            "license": "apache 2.0",
            "commit": "unknown",
            "dataset": "thorsten"
        },
        "model_type": "vocoder",
        "language": "de",
        "name": "hifigan_v1",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.8.0_models/vocoder_models--de--thorsten--hifigan_v1.zip"
        ]
    },
    "vocoder_models/ja/kokoro/hifigan_v1": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.1_models/vocoder_models--ja--kokoro--hifigan_v1.zip",
            "description": "HifiGAN model trained for kokoro dataset by @kaiidams",
            "author": "@kaiidams",
            "license": "apache 2.0",
            "commit": "3900448",
            "dataset": "kokoro"
        },
        "model_type": "vocoder",
        "language": "ja",
        "name": "hifigan_v1",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.1_models/vocoder_models--ja--kokoro--hifigan_v1.zip"
        ]
    },
    "vocoder_models/uk/mai/multiband-melgan": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.1_models/vocoder_models--uk--mai--multiband-melgan.zip",
            "author": "@robinhad",
            "commit": "bdab788d",
            "license": "MIT",
            "contact": "",
            "dataset": "mai"
        },
        "model_type": "vocoder",
        "language": "uk",
        "name": "multiband-melgan",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.1_models/vocoder_models--uk--mai--multiband-melgan.zip"
        ]
    },
    "vocoder_models/tr/common-voice/hifigan": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.6.1_models/vocoder_models--tr--common-voice--hifigan.zip",
            "description": "HifiGAN model using an unknown speaker from the Common-Voice dataset.",
            "author": "Fatih Akademi",
            "license": "MIT",
            "commit": None,
            "dataset": "common-voice"
        },
        "model_type": "vocoder",
        "language": "tr",
        "name": "hifigan",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.6.1_models/vocoder_models--tr--common-voice--hifigan.zip"
        ]
    },
    "vocoder_models/be/common-voice/hifigan": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.16.6/vocoder_models--be--common-voice--hifigan.zip",
            "description": "Belarusian HiFiGAN model created by @alex73 (Github).",
            "author": "@alex73",
            "license": "CC-BY-SA 4.0",
            "commit": "c0aabb85",
            "dataset": "common-voice"
        },
        "model_type": "vocoder",
        "language": "be",
        "name": "hifigan",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.16.6/vocoder_models--be--common-voice--hifigan.zip"
        ]
    },
    "voice_conversion_models/multilingual/vctk/freevc24": {
        "metadata": {
            "github_rls_url": "https://coqui.gateway.scarf.sh/v0.13.0_models/voice_conversion_models--multilingual--vctk--freevc24.zip",
            "description": "FreeVC model trained on VCTK dataset from https://github.com/OlaWod/FreeVC",
            "author": "Jing-Yi Li @OlaWod",
            "license": "MIT",
            "commit": None,
            "dataset": "vctk"
        },
        "model_type": "vc",
        "language": "multilingual",
        "name": "freevc24",
        "download_urls": [
            "https://coqui.gateway.scarf.sh/v0.13.0_models/voice_conversion_models--multilingual--vctk--freevc24.zip"
        ]
    }
}