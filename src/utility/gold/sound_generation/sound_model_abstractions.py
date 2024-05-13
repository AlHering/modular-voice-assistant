# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import os
import traceback
from queue import Queue
import speech_recognition
from typing import List, Tuple, Any, Callable, Optional, Union
from datetime import datetime as dt
from enum import Enum
import pyaudio
import numpy as np
import time
from ...bronze.audio_utility import get_input_devices, get_output_devices
from ...bronze.time_utility import get_timestamp
from .sound_model_instantiation import load_whisper_model, load_faster_whisper_model, load_coqui_tts_model
from .speech_to_text_utility import transcribe_with_faster_whisper, transcribe_with_whisper
from .text_to_speech_utility import synthesize_with_coqui_tts, synthesize_with_coqui_tts_to_file


class Transcriber(object):
    """
    Represents transcriber.
    """

    supported_backends: List[str] = ["faster-whisper", "whisper"]

    def __init__(self,
                 backend: str,
                 model_path: str,
                 model_parameters: dict = None,
                 transcription_parameters: dict = None) -> None:
        """
        Initiation method.
        :param backend: Backend for model loading.
            Check Transcriber.supported_backends for supported backends.
        :param model_path: Path to model files.
        :param model_parameters: Model loading kwargs as dictionary.
            Defaults to None.
        :param transcription_parameters: Transcription kwargs as dictionary.
            Defaults to None.
        """
        self.backend = backend

        self.model_parameters = {} if model_parameters is None else model_parameters
        self.transcription_parameters = {} if transcription_parameters is None else transcription_parameters

        self.model = {
            "faster-whisper": load_faster_whisper_model,
            "whisper": load_whisper_model
        }[self.backend](
            model_path=model_path,
            model_parameters=model_parameters
        )

        self.transcription_function = {
            "faster-whisper": transcribe_with_faster_whisper,
            "whisper": transcribe_with_whisper
        }[self.backend]

    def transcribe(self, audio_input: str, transcription_parameters: dict = None) -> Tuple[str, dict]:
        """
        Transcribes audio to text.
        :param audio_input: Wave file path or waveform.
        :param transcription_parameters: Transcription kwargs as dictionary.
            Defaults to None.
        """
        return self.transcription_function(
            audio_input=audio_input,
            model=self.model,
            transcription_kwargs=self.transcription_parameters if transcription_parameters is None else transcription_parameters
        )


class Synthesizer(object):
    """
    Represents synthesizer.
    """

    supported_backends: List[str] = ["coqui-tts"]

    def __init__(self,
                 backend: str,
                 model_path: str,
                 model_parameters: dict = None,
                 synthesis_parameters: dict = None) -> None:
        """
        Initiation method.
        :param backend: Backend for model loading.
            Check Transcriber.supported_backends for supported backends.
        :param model_path: Path to model files.
        :param model_parameters: Model loading kwargs as dictionary.
            Defaults to None.
        :param synthesis_parameters: Synthesis kwargs as dictionary.
            Defaults to None.
        """
        self.backend = backend

        self.model_parameters = {} if model_parameters is None else model_parameters
        self.synthesis_parameters = {} if synthesis_parameters is None else synthesis_parameters

        self.model = {
            "coqui-tts": load_coqui_tts_model,
        }[self.backend](
            model_path=model_path,
            model_parameters=model_parameters
        )

        self.sound_out_snythesis_functions = {
            "coqui-tts": synthesize_with_coqui_tts
        }[self.backend]
        self.file_out_snythesis_functions = {
            "coqui-tts": synthesize_with_coqui_tts_to_file
        }[self.backend]

    def synthesize(self, text: str, synthesis_parameters: dict = None) -> Tuple[np.ndarray, dict]:
        """
        Synthesize text to audio.
        :param text: Text to synthesize to audio.
        :param synthesis_parameters: Synthesis kwargs as dictionary.
            Defaults to None.
        :return: File path and metadata.
        """
        return self.sound_out_snythesis_functions(
            text=text, 
            model=self.model,
            synthesis_kwargs=self.synthesis_parameters if synthesis_parameters is None else synthesis_parameters)

    def synthesize_to_file(self, text: str, output_path: str, synthesis_parameters: dict = None) -> Tuple[np.ndarray, dict]:
        """
        Synthesize text to audio.
        :param text: Text to synthesize to audio.
        :param output_path: Path for output file.
        :param synthesis_parameters: Synthesis kwargs as dictionary.
            Defaults to None.
        :return: Output file path and metadata.
        """
        return self.file_out_snythesis_functions(
            text=text, 
            output_path=output_path,
            model=self.model,
            synthesis_kwargs=self.synthesis_parameters if synthesis_parameters is None else synthesis_parameters)
        

class SpeechRecorder(object):
    """
    Represents a speech recorder.
    """

    def __init__(self,
                 transcriber: Transcriber = None,
                 input_device_index: int = None,
                 recognizer_kwargs: dict = None,
                 microphone_kwargs: dict = None,
                 loop_pause = .1) -> None:
        """
        Initiation method.
        :param transcriber: Optional transcriber instance.
            If provided, recordings are returned as transcribed text instead of audio data.
        :param input_device_index: Input device index.
            Defaults to None in which case the default input device index is fetched.
        :param recognizer_kwargs: Keyword arguments for setting up recognizer instances.
            Defaults to None in which case default values are used.
        :param microphone_kwargs: Keyword arguments for setting up microphone instances.
            Defaults to None in which case default values are used.
        :param loop_pause: Forced pause between loops in seconds.
            Defaults to 0.1.
        """
        self.transcriber = transcriber
        if input_device_index is None:
            pya = pyaudio.PyAudio()
            input_device_index = pya.get_default_input_device_info().get("index")
            pya.terminate()
        self.input_device_index = input_device_index
        self.loop_pause = loop_pause
        self.interrupt_flag = False

        self.recognizer_kwargs = {
            "energy_threshold": 1000,
            "dynamic_energy_threshold": False,
            "pause_threshold": .8
        } if recognizer_kwargs is None else recognizer_kwargs
        self.microphone_kwargs = {
            "device_index": self.input_device_index,
            "sample_rate": 16000,
            "chunk_size": 1024
        } if microphone_kwargs is None else microphone_kwargs

    def _optionally_transcribe(self, audio_input: np.ndarray) -> Tuple[Union[str, np.ndarray], dict]:
        """
        Transcribes audio input.
        :param audio_input: Audio input.
        :return: Recorded input as audio data or text, if a transcriber is available and recording metadata.
        """
        return (audio_input, {}) if self.transcriber is None else self.transcriber.transcribe(
            audio_input=audio_input
        )

    def record_single_input(self,
                            recognizer_kwargs: dict = None,
                            microphone_kwargs: dict = None,
                            interrupt_threshold: float = None) -> Tuple[Union[str, np.ndarray], dict]:
        """
        Records continuesly and puts results into output_queue.
        :param output_queue: Queue to put tuples of recordings and metadata into.
            Recordings are either audio data (as numpy arrays) or texts, if a transriber is available.
        :param recognizer_kwargs: Keyword arguments for setting up recognizer instances.
            Defaults to None in which case default values are used.
        :param microphone_kwargs: Keyword arguments for setting up microphone instances.
            Defaults to None in which case default values are used.
        :param interrupt_threshold: Interrupt threshold of silence after which the recording loop stops.
            Defaults to None in which case the loop runs indefinitely
        :return: Recorded input as audio data or text, if a transcriber is available and recording metadata.
        """
        recognizer_kwargs = self.recognizer_kwargs if recognizer_kwargs is None else recognizer_kwargs
        microphone_kwargs = self.microphone_kwargs if microphone_kwargs is None else microphone_kwargs

        recognizer = speech_recognition.Recognizer()
        for key in recognizer_kwargs:
            setattr(recognizer, key, recognizer_kwargs[key])
        microphone = speech_recognition.Microphone(**microphone_kwargs)
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
        
        audio = None
        with microphone as source:
            audio = recognizer.listen(
                source=source
            )
        audio_as_numpy_array = np.frombuffer(audio.get_wav_data(), dtype=np.int16).astype(np.float32) / 32768.0
        
        text, metadata_entries = self._optionally_transcribe(
            audio_input=audio_as_numpy_array
        )
        return text, {"timestamp": get_timestamp(), "input_method": "speech_to_text", "transcription_metadata": metadata_entries}

    def record(self, 
               output_queue: Queue,
               recognizer_kwargs: dict = None,
               microphone_kwargs: dict = None,
               interrupt_threshold: float = None) -> None:
        """
        Records continuesly and puts results into output_queue.
        :param output_queue: Queue to put tuples of recordings and metadata into.
            Recordings are either audio data (as numpy arrays) or texts, if a transriber is available.
        :param recognizer_kwargs: Keyword arguments for setting up recognizer instances.
            Defaults to None in which case default values are used.
        :param microphone_kwargs: Keyword arguments for setting up microphone instances.
            Defaults to None in which case default values are used.
        :param interrupt_threshold: Interrupt threshold of silence after which the recording loop stops.
            Defaults to None in which case the loop runs indefinitely
        """
        recognizer_kwargs = self.recognizer_kwargs if recognizer_kwargs is None else recognizer_kwargs
        microphone_kwargs = self.microphone_kwargs if microphone_kwargs is None else microphone_kwargs

        audio_queue = Queue()
        recognizer = speech_recognition.Recognizer()
        for key in recognizer_kwargs:
            setattr(recognizer, key, recognizer_kwargs[key])
        microphone = speech_recognition.Microphone(**microphone_kwargs)
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
        
        def speech_recognition_callback(_, audio: speech_recognition.AudioData) -> None:
            """
            Collects audio data.
            :param audio: Audio data.
            """
            data = audio.get_raw_data()
            audio_queue.put(data)

        recognizer.listen_in_background(
            source=microphone,
            callback=speech_recognition_callback,
        )
        
        # Starting transcription loop
        recording_started = False
        interrupt_flag = False
        last_empty_transcription = None
        while not interrupt_flag:
            try:
                if not audio_queue.empty():
                    recording_started = True
                    last_empty_transcription = None
                    audio = b"".join(audio_queue.queue)
                    audio_queue.queue.clear()

                    # Convert and transcribe audio input
                    audio_as_numpy_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
                    fulltext, segment_metadatas =  self._optionally_transcribe(
                        audio_input=audio_as_numpy_array
                    )
                    output_queue.put((fulltext, segment_metadatas))
                elif interrupt_threshold is not None and recording_started: 
                    if last_empty_transcription is None:
                        last_empty_transcription = time.time()
                    elif time.time() - last_empty_transcription >= interrupt_threshold:
                        interrupt_flag = True
                time.sleep(self.loop_pause)
            except KeyboardInterrupt:
                interrupt_flag = True