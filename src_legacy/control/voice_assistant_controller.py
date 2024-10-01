# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import os
from time import sleep
import gc
from datetime import datetime as dt
import numpy as np
from functools import wraps
from typing import Optional, Any, List, Dict, Union, Tuple
from src_legacy.configuration import configuration as cfg
from src.backend.basic_sqlalchemy_interface import BasicSQLAlchemyInterface
from src_legacy.model.voice_assistant_control.data_model import populate_data_instrastructure
from src.backend.sound_model_abstractions import Transcriber, Synthesizer, SpeechRecorder


def update_cached_workers(object_type: str) -> Optional[Any]:
    """
    Cached workers updating decorator.
    :return: Decorator wrapper.
    """

    def wrapper(func: Any) -> Optional[Any]:
        """
        Function wrapper.
        :param func: Wrapped function.
        :return: Process data, containing error message if process failed, else function return.
        """
        @wraps(func)
        async def inner(*args: Optional[Any], **kwargs: Optional[Any]):
            """
            Inner function wrapper.
            :param args: Arguments.
            :param kwargs: Keyword arguments.
            """
            controller: VoiceAssistantController = args[0]
            id_keyword = f"{object_type}_id"
            if kwargs and id_keyword in kwargs:
                target_id = kwargs[id_keyword]
            else:
                target_id = args[1]

            if str(target_id) not in controller.workers[object_type]:
                entry = controller.get_object_by_id(object_type, target_id)
                controller.workers[object_type][str(target_id)] = controller.entry_to_obj(object_type, entry)
            
            return func(*args, **kwargs)
        return inner
    return wrapper



class VoiceAssistantController(BasicSQLAlchemyInterface):
    """
    Controller class for handling voice assistant interface requests.
    """
    
    def __init__(self, working_directory: Optional[str] = None, database_uri: Optional[str] = None) -> None:
        """
        Initiation method.
        :param working_directory: Working directory.
            Defaults to folder 'processes' folder under standard backend data path.
        :param database_uri: Database URI.
            Defaults to 'backend.db' file under configured backend folder.
        """
        # Main instance variables
        self._logger = cfg.LOGGER
        self.working_directory = cfg.PATHS.BACKEND_PATH if working_directory is None else working_directory
        if not os.path.exists(self.working_directory):
            os.makedirs(self.working_directory)
        self.database_uri = f"sqlite:///{os.path.join(cfg.PATHS.BACKEND_PATH, 'voice_assistant.db')}" if database_uri is None else database_uri

        # Database infrastructure
        super().__init__(self.working_directory, self.database_uri,
                         populate_data_instrastructure, "voice_assistant.", self._logger)
        
        # Orchestrated workers
        self.workers = {
            "transcriber": {},
            "synthesizer": {},
            "speech_recorder": {}
        }

        
    """
    Setup and shutdown methods
    """
    def setup(self) -> None:
        """
        Method for running setup process.
        """
        for object_type in ["transcriber", "synthesizer", "speech_recorder"]:
            if self.get_object_count_by_type(object_type) == 0:
                self._create_base_configs(object_type)

    def shutdown(self) -> None:
        """
        Method for running shutdown process.
        """
        self.workers = {
            "transcribers": {},
            "synthesizers": {},
            "speech_recorders": {}
        }
        gc.collect()

    def _create_base_configs(self, object_type: str) -> None:
        """
        Internal method for creating basic configurations.
        :param object_type: Target object type.
        """
        obj_kwargs = {
            "transcriber": {
                "backend": "faster-whisper",
                "model_path": f"{cfg.PATHS.SOUND_GENERATION_MODEL_PATH}/speech_to_text/faster_whisper_models/Systran_faster-whisper-tiny",
                "model_parameters": {
                    "device": "cuda",
                    "compute_type": "float32",
                    "local_files_only": True
                }
            },
            "synthesizer": {
                "backend": "coqui-tts",
                "model_path": f"{cfg.PATHS.SOUND_GENERATION_MODEL_PATH}/text_to_speech/coqui_models/tts_models--multilingual--multi-dataset--xtts_v2",
                "model_parameters": {
                    "config_path": f"{cfg.PATHS.SOUND_GENERATION_MODEL_PATH}/text_to_speech/coqui_models/tts_models--multilingual--multi-dataset--xtts_v2/config.json",
                    "gpu": True
                },
                "synthesis_parameters": {
                    "speaker_wav": f"{cfg.PATHS.SOUND_GENERATION_MODEL_PATH}/text_to_speech/coqui_xtts/examples/female.wav",
                    "language": "en"
                }
            },
            "speech_recorder": {
                "input_device_index": None,
                "recognizer_parameters": {
                    "energy_threshold": 1000,
                    "dynamic_energy_threshold": False,
                    "pause_threshold": 0.8
                },
                "microphone_parameters": {
                    "sample_rate": 16000,
                    "chunk_size": 1024
                },
                "loop_pause": 0.1
            }
        }[object_type]
        self.post_object(object_type, **obj_kwargs)


    """
    Base interaction
    """
    def log(self, data: dict) -> None:
        """
        Method for adding a log entry.
        :param data: Log entry data.
        """
        self.post_object("log", **data)

    def entry_to_obj(self, object_type: str, entry: Any) -> Optional[Any]:
        """
        Transform a database entry to an object.
        :param object_type: Object type.
        :param entry: Database entry.
        :return: Object if initiation was successful.
        """
        if object_type == "transcriber":
            return Transcriber(
                backend=entry.backend,
                model_path=entry.model_path,
                model_parameters=entry.model_parameters,
                transcription_parameters=entry.transcription_parameters
            )
        elif object_type == "synthesizer":
            return Synthesizer(
                backend=entry.backend,
                model_path=entry.model_path,
                model_parameters=entry.model_parameters,
                synthesis_parameters=entry.synthesis_parameters
            )
        elif object_type == "speech_recorders":
            return SpeechRecorder(
                input_device_index=entry.input_device_index,
                recognizer_parameters=entry.recognizer_parameters,
                microphone_parameters=entry.microphone_parameters,
                loop_pause=entry.loop_pause
            )
    
    """
    Extended interaction
    """
    @update_cached_workers("transcriber")
    def transcribe(self, transcriber_id: int, audio_input: np.ndarray, transcription_parameters: Optional[dict] = None) -> Tuple[str, dict]:
        """
        Method for transcribing audio data with specific transriber.
        :param transcriber_id: Transcriber ID.
        :param audio_input: Audio input data.
        :param transcription_parameters: Transcription parameters as dictionary.
            Defaults to None.
        :return: Tuple of transcription and metadata.
        """
        return self.workers["transcriber"][str(transcriber_id)].transcribe(audio_input=audio_input,
                                                                            transcription_parameters=transcription_parameters)

    @update_cached_workers("synthesizer")
    def synthesize(self, synthesizer_id: int, text: str, synthesis_parameters: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Method for synthesizing audio data from text.
        :param synthesizer_id: Synthesizer ID.
        :param text: Text to synthesize audio for.
        :param synthesis_parameters: Synthesis parameters as dictionary.
            Defaults to None.
        :return: Tuple of synthesis and metadata.
        """
        return self.workers["synthesizer"][str(synthesizer_id)].synthesize(text=text,
                                                                            synthesis_parameters=synthesis_parameters)
    
    @update_cached_workers("speech_recorder")
    def record(self, speech_recorder_id: int, recognizer_parameters: Optional[dict] = None, microphone_parameters: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Method for recording audio.
        :param speech_recorder_id: SpeechRecorder ID.
        :param recognizer_parameters: Keyword arguments for setting up recognizer instances.
            Defaults to None in which case default values are used.
        :param microphone_parameters: Keyword arguments for setting up microphone instances.
            Defaults to None in which case default values are used.
        :return: Tuple of recorded audio and metadata.
        """
        return self.workers["speech_recorder"][str(speech_recorder_id)].record_single_input(recognizer_parameters=recognizer_parameters,
                                                                                             microphone_parameters=microphone_parameters)

