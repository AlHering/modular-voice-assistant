# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from __future__ import annotations
from multiprocessing import Queue, Event as MEvent
from threading import Thread, Event
from queue import Empty
import speech_recognition
from typing import List, Tuple, Union, Dict
import pyaudio
import copy
from src.utility.commandline_utility import silence_stderr
import numpy as np
import time
from src.utility.sounddevice_utility import get_input_devices, get_output_devices
from src.utility.pyaudio_utility import play_wave
from src.utility.time_utility import get_timestamp
from src.utility.whisper_utility import load_whisper_model, transcribe as transcribe_with_whisper
from src.utility.faster_whisper_utility import load_faster_whisper_model, transcribe as transcribe_with_faster_whisper
from src.utility.coqui_tts_utility import load_coqui_tts_model, synthesize as synthesize_with_coqui_tts, synthesize_to_file as synthesize_with_coqui_tts_to_file


class Transcriber(object):
    """
    Represents transcriber.
    """
    supported_backends: List[str] = ["faster-whisper", "whisper"]
    default_models: Dict[str, List[str]] = {
        "faster-whisper": ["large-v3"], 
        "whisper": ["large-v3"]
    }

    def __init__(self,
                 backend: str,
                 model_path: str | None = None,
                 model_parameters: dict | None = None,
                 transcription_parameters: dict | None = None) -> None:
        """
        Initiation method.
        :param backend: Backend for model loading.
            Check Transcriber.supported_backends for supported backends.
        :param model_path: Path to model files.
            Defaults to None in which case a default model is used.
            The latter will most likely result in it being downloaded.
        :param model_parameters: Model loading parameters as dictionary.
            Defaults to None.
        :param transcription_parameters: Transcription parameters as dictionary.
            Defaults to None.
        """
        self.backend = backend

        self.model_path = self.default_models[self.backend][0] if model_path is None else model_path
        self.model_parameters = {} if model_parameters is None else model_parameters
        self.transcription_parameters = {} if transcription_parameters is None else transcription_parameters

        self.model = {
            "faster-whisper": load_faster_whisper_model,
            "whisper": load_whisper_model
        }[self.backend](
            model_path=self.model_path,
            model_parameters=self.model_parameters
        )

        self.transcription_function = {
            "faster-whisper": transcribe_with_faster_whisper,
            "whisper": transcribe_with_whisper
        }[self.backend]

    def transcribe(self, audio_input: Union[str, list, np.ndarray], transcription_parameters: dict | None = None) -> Tuple[str, dict]:
        """
        Transcribes audio to text.
        :param audio_input: Wave file path or waveform.
        :param transcription_parameters: Transcription parameters as dictionary.
            Defaults to None.
        """
        if transcription_parameters and "dtype" in transcription_parameters and isinstance(audio_input, list):
            audio_input = np.array(audio_input, dtype=transcription_parameters["dtype"])
        return self.transcription_function(
            audio_input=audio_input,
            model=self.model,
            transcription_parameters=copy.deepcopy(self.transcription_parameters) if transcription_parameters is None else transcription_parameters
        )


class Synthesizer(object):
    """
    Represents synthesizer.
    """
    supported_backends: List[str] = ["coqui-tts"]
    default_models: Dict[str, List[str]] = {
        "coqui-tts": ["tts_models/multilingual/multi-dataset/xtts_v2"]
    }

    def __init__(self,
                 backend: str,
                 model_path: str | None = None,
                 model_parameters: dict | None = None,
                 synthesis_parameters: dict | None = None) -> None:
        """
        Initiation method.
        :param backend: Backend for model loading.
            Check Transcriber.supported_backends for supported backends.
        :param model_path: Path to model files.
            Defaults to None in which case a default model is used.
            The latter will most likely result in it being downloaded.
        :param model_parameters: Model loading parameters as dictionary.
            Defaults to None.
        :param synthesis_parameters: Synthesis parameters as dictionary.
            Defaults to None.
        """
        self.backend = backend

        self.model_path = self.default_models[self.backend][0] if model_path is None else model_path
        self.model_parameters = {} if model_parameters is None else model_parameters
        self.synthesis_parameters = {} if synthesis_parameters is None else synthesis_parameters

        self.model = {
            "coqui-tts": load_coqui_tts_model,
        }[self.backend](
            model_path=self.model_path,
            model_parameters=self.model_parameters
        )

        self.sound_out_synthesis_functions = {
            "coqui-tts": synthesize_with_coqui_tts
        }[self.backend]
        self.file_out_synthesis_functions = {
            "coqui-tts": synthesize_with_coqui_tts_to_file
        }[self.backend]

    def synthesize(self, text: str, synthesis_parameters: dict | None = None) -> Tuple[np.ndarray, dict]:
        """
        Synthesize text to audio.
        :param text: Text to synthesize to audio.
        :param synthesis_parameters: Synthesis parameters as dictionary.
            Defaults to None.
        :return: Audio data and metadata.
        """
        return self.sound_out_synthesis_functions(
            text=text, 
            model=self.model,
            synthesis_parameters=copy.deepcopy(self.synthesis_parameters) if synthesis_parameters is None else synthesis_parameters)

    def synthesize_to_file(self, text: str, output_path: str, synthesis_parameters: dict | None = None) -> Tuple[str, dict]:
        """
        Synthesize text to audio.
        :param text: Text to synthesize to audio.
        :param output_path: Path for output file.
        :param synthesis_parameters: Synthesis parameters as dictionary.
            Defaults to None.
        :return: Output file path and metadata.
        """
        return self.file_out_synthesis_functions(
            text=text, 
            output_path=output_path,
            model=self.model,
            synthesis_parameters=copy.deepcopy(self.synthesis_parameters) if synthesis_parameters is None else synthesis_parameters), {}
        

class SpeechRecorder(object):
    """
    Represents a speech recorder.
    """
    supported_input_devices = get_input_devices(include_metadata=True)

    def __init__(self,
                 input_device_index: int | None = None,
                 recognizer_parameters: dict | None = None,
                 microphone_parameters: dict | None = None,
                 loop_pause: float = .1) -> None:
        """
        Initiation method.
        :param input_device_index: Input device index.
            Check SpeechRecorder.supported_input_devices for available input device profiles.
            Defaults to None in which case the default input device index is fetched.
        :param recognizer_parameters: Keyword arguments for setting up recognizer instances.
            Defaults to None in which case default values are used.
        :param microphone_parameters: Keyword arguments for setting up microphone instances.
            Defaults to None in which case default values are used.
        :param loop_pause: Forced pause between loops in seconds.
            Defaults to 0.1.
        """
        if input_device_index is None:
            pya = pyaudio.PyAudio()
            input_device_index = pya.get_default_input_device_info().get("index")
            pya.terminate()
        self.input_device_index = input_device_index
        self.loop_pause = loop_pause
        self.interrupt_flag = False

        self.recognizer_parameters = {
            "energy_threshold": 1000,
            "dynamic_energy_threshold": False,
            "pause_threshold": .8
        } if recognizer_parameters is None else recognizer_parameters
        self.microphone_parameters = {
            "sample_rate": 16000,
            "chunk_size": 1024
        } if microphone_parameters is None else microphone_parameters
        if "device_index" not in self.microphone_parameters:
            self.microphone_parameters["device_index"] = self.input_device_index

    def record_single_input(self,
                            recognizer_parameters: dict | None = None,
                            microphone_parameters: dict | None = None) -> Tuple[np.ndarray, dict]:
        """
        Records continuously and puts results into output_queue.
        :param output_queue: Queue to put tuples of recordings and metadata into.
            Recordings are either audio data (as numpy arrays) or texts, if a transcriber is available.
        :param recognizer_parameters: Keyword arguments for setting up recognizer instances.
            Defaults to None in which case default values are used.
        :param microphone_parameters: Keyword arguments for setting up microphone instances.
            Defaults to None in which case default values are used.
        :return: Recorded input as numpy array and recording metadata.
        """
        recognizer_parameters = copy.deepcopy(self.recognizer_parameters) if recognizer_parameters is None else recognizer_parameters
        microphone_parameters = copy.deepcopy(self.microphone_parameters) if microphone_parameters is None else microphone_parameters

        recognizer = speech_recognition.Recognizer()
        for key in recognizer_parameters:
            setattr(recognizer, key, recognizer_parameters[key])
        microphone = speech_recognition.Microphone(**microphone_parameters)
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
        
        audio = None
        with microphone as source:
            audio = recognizer.listen(source=source)
        audio_as_numpy_array = np.frombuffer(audio.get_wav_data(), dtype=np.int16).astype(np.float32) / 32768.0
        return audio_as_numpy_array, {"timestamp": get_timestamp()}

    def record(self, 
               output_queue: Queue,
               recognizer_parameters: dict | None = None,
               microphone_parameters: dict | None = None,
               interrupt_threshold: float | None = None) -> None:
        """
        Records continuously and puts results into output_queue.
        :param output_queue: Queue to put tuples of recorded audio data (as numpy array) and recording metadata.
        :param recognizer_parameters: Keyword arguments for setting up recognizer instances.
            Defaults to None in which case default values are used.
        :param microphone_parameters: Keyword arguments for setting up microphone instances.
            Defaults to None in which case default values are used.
        :param interrupt_threshold: Interrupt threshold of silence after which the recording loop stops.
            Defaults to None in which case the loop runs indefinitely.
        """
        recognizer_parameters = copy.deepcopy(self.recognizer_parameters) if recognizer_parameters is None else recognizer_parameters
        microphone_parameters = copy.deepcopy(self.microphone_parameters) if microphone_parameters is None else microphone_parameters

        audio_queue = Queue()
        recognizer = speech_recognition.Recognizer()
        for key in recognizer_parameters:
            setattr(recognizer, key, recognizer_parameters[key])
        microphone = speech_recognition.Microphone(**microphone_parameters)
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
                    output_queue.put((audio_as_numpy_array, {"timestamp": get_timestamp()}))
                elif interrupt_threshold is not None and recording_started: 
                    if last_empty_transcription is None:
                        last_empty_transcription = time.time()
                    elif time.time() - last_empty_transcription >= interrupt_threshold:
                        interrupt_flag = True
                time.sleep(self.loop_pause)
            except KeyboardInterrupt:
                interrupt_flag = True


class AudioPlayer(object):
    """
    Represents a audio player.
    """
    supported_backends: List[str] = ["pyaudio"]
    supported_output_devices = get_output_devices(include_metadata=True)

    def __init__(self,
                 backend: str,
                 output_device_index: int | None = None,
                 playback_parameters: dict | None = None) -> None:
        """
        Initiation method.
        :param backend: Backend for audio handling.
            Check AudioPlayer.supported_backends for supported backends.
        :param output_device_index: Output device index.
            Check SpeechRecorder.supported_input_devices for available input device profiles.
            Defaults to None in which case the default input device index is fetched.
        :param playback_parameters: Keyword arguments for configuring playback.
        """
        self.backend = backend
        if output_device_index is None:
            pya = pyaudio.PyAudio()
            output_device_index = pya.get_default_output_device_info().get("index")
            pya.terminate()
        self.output_device_index = output_device_index
        self.playback_parameters = {} if playback_parameters is None else playback_parameters

    def play(self, 
             audio_input: Union[str, list, np.ndarray], 
             playback_parameters: dict | None = None) -> None:
        """
        Plays audio.
        :param audio_input: Wave file path or waveform.
        :param playback_parameters: Playback parameters as dictionary.
            Defaults to None.
        """
        playback_parameters = copy.deepcopy(self.playback_parameters) if playback_parameters is None else playback_parameters
        if "dtype" in playback_parameters and isinstance(audio_input, list):
            audio_input = np.array(audio_input, dtype=playback_parameters.pop("dtype"))
        if self.backend == "pyaudio":
            play_wave(wave=audio_input,
                      stream_kwargs=playback_parameters)
            
    def spawn_output_thread(self, 
                            input_queue: Queue,
                            stop_event = Event,
                            loop_pause: float = .1) -> Thread:
        """
        Plays audio.
        :param input_queue: Input queue.
        :param stop_event: Stop event.
        :param audio_input: Wave file path or waveform.
        :param loop_pause: Pause between queue checks.
        :return: Output thread.
        """
        def handle_audio_queue() -> None:
            pause_event = MEvent()
            while not stop_event.is_set():
                if not pause_event.is_set():
                    try:
                        input_package = input_queue.get(timeout=loop_pause)
                        pause_event.set()
                        if isinstance(input_package, Tuple):
                            audio_input, playback_parameters = input_package
                        else:
                            audio_input = input_package
                            playback_parameters = None
                        forwarded_playback_parameters = copy.deepcopy(self.playback_parameters) if playback_parameters is None else playback_parameters
                        with silence_stderr():
                            self.play(audio_input=audio_input, playback_parameters=forwarded_playback_parameters)
                        pause_event.clear()
                    except Empty:
                        time.sleep(loop_pause)

        thread = Thread(target=handle_audio_queue)
        thread.daemon = True
        thread.start()
        return thread
        
        
