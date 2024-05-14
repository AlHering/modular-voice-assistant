# -*- coding: utf-8 -*-
"""
****************************************************
*                      Utility                 
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from enum import Enum
from inspect import getfullargspec
from typing import Any, Union, Tuple, List, Optional, Callable, Dict
import os
import gc
import time
import numpy as np
import pyaudio
import speech_recognition
from datetime import datetime as dt
from src.configuration import configuration as cfg
from threading import Thread, Event as TEvent
from queue import Empty, Queue as TQueue
from ..text_generation.language_model_abstractions import LanguageModelInstance
from ..text_generation.agent_abstractions import Agent
from ...bronze.concurrency_utility import PipelineComponentThread
from ...bronze.time_utility import get_timestamp
from . import speech_to_text_utility, text_to_speech_utility
from .sound_model_abstractions import Transcriber, Synthesizer, SpeechRecorder


class ConversationHandler(object):
    """
    Represents a conversation handler for handling audio based interaction.
    A conversation handler manages the following components:
        - speech_recorder: A recorder for spoken input.
        - transcriber: A transcriber to transcribe spoken input into text.
        - worker: A worker to compute an output for the given input.
        - synthesizer: A synthesizer to convert output texts to sound.
    """

    def __init__(self, 
                 working_directory: str,
                 speech_recorder: SpeechRecorder,
                 transcriber: Transcriber,
                 synthesizer: Synthesizer,
                 worker_function: Callable,
                 history: List[dict] = None,
                 loop_pause: float = 0.1) -> None:
        """
        Initiation method.
        :param working_directory: Directory for productive files.
        :param speech_recorder: Speech recorder for STT processes.
        :param transcriber: Transcriber for STT processes.
        :param synthesizer: Synthesizer for TTS processes.
        :param worker_function: Worker function for handling cleaned input.
        :param history: History as list of dictionaries of the structure
            {"process": <"tts"/"stt">, "text": <text content>, "metadata": {...}}
        :param input_method: Input method.
            Defaults to SPEECH (STT).
        :param output_method: Output method.
            Defaults to SPEECH (TTS).
        :param loop_pause: Pause in seconds between processing loops.
            Defaults to 0.1.
        """
        cfg.LOGGER.info("Initiating Conversation Handler...")
        if not os.path.exists(working_directory):
            os.makedirs(working_directory)
        self.working_directory = working_directory
        self.input_path = os.path.join(self.working_directory, "input.wav") 
        self.output_path = os.path.join(self.working_directory, "output.wav") 

        pya = pyaudio.PyAudio()
        self.input_device_index = pya.get_default_input_device_info().get("index")
        self.output_device_index = pya.get_default_output_device_info().get("index")
        pya.terminate()

        self.speech_recorder = speech_recorder
        self.transcriber = transcriber
        self.worker_function = worker_function
        self.synthesizer = synthesizer
        self.component_functions = {
            "transcriber": self.transcriber.transcribe,
            "worker": self.worker_function,
            "synthesizer": self.synthesizer.synthesize,
        }

        self.interrupt = None
        self.interrupts = {}
        self.queues = {}
        self.threads = {}

        self.history = history
        self.loop_pause = loop_pause
        self._reset()

    def _stop(self) -> None:
        """
        Method for stopping processes.
        """
        if self.interrupt is not None:
            self.interrupt.set()
        for component in self.interrupts:
            self.interrupts[component].set()
        for component in self.threads:
            self.threads[component].join(1) 

    def _setup_components(self) -> None:
        """
        Method for setting up components.
        """
        self.interrupt = TEvent()
        self.interrupts = {component: TEvent() for component in self.component_functions}

        self.queues = {f"{component}_in": TQueue() for component in self.component_functions}
        self.queues.update({f"{component}_out": TQueue() for component in self.component_functions})

        self.threads = {}
        for component in self.component_functions:
            self.threads[component] = PipelineComponentThread(
                pipeline_function=self.component_functions[component],
                input_queue=self.queues.get(f"{component}_in"),
                output_queue=self.queues.get(f"{component}_out"),
                interrupt=self.interrupts[component],
                loop_pause=self.loop_pause
            )
            self.threads[component].daemon = True

        for component in self.threads:
            self.threads[component].start()

    def _reset(self, delete_history: bool = False) -> None:
        """
        Method for setting up and resetting handler. 
        :param delete_history: Flag for declaring whether to delete history.    
            Defaults to None.
        """
        cfg.LOGGER.info("(Re)setting Conversation Handler...")
        self._stop()
        self.interrupt = None
        self.interrupts = None
        self.queues = None
        self.threads = None
        gc.collect()

        self._setup_components()
        self.history = [] if self.history is None or delete_history else self.history
        cfg.LOGGER.info("Setup is done.")

    def handle_input(self) -> None:
        """
        Acquires and reroutes input based on configured input method.
        """
        result = self.speech_recorder.record_single_input()
        if result[0].any():
            cfg.LOGGER.info(f"Forwarding audio data to transcriber...")
            self.queues["transcriber_in"].put(result[0])
    
    def input_to_worker_gateway(self) -> bool:
        """
        Collects and refines input before passing on to LLM.
        :return: True, if data was successfully forwarded, else False.
        """
        try:
            cfg.LOGGER.info(f"Trying to fetch 'transcriber_out' data ...")
            raw_input, raw_input_metadata = self.queues["transcriber_out"].get(self.loop_pause)
            # TODO: Refine input
            cfg.LOGGER.info(f"Forwarding {raw_input} to worker...")
            self.queues["worker_in"].put(raw_input)
            return True
        except Empty:
            cfg.LOGGER.info(f"'transcriber_out'-queue is empty.")
            return False
        
    def worker_to_output_gateway(self) -> bool:
        """
        Collects and refines LLM output.
        :return: True, if data was successfully forwarded, else False.
        """
        try:
            cfg.LOGGER.info(f"Trying to fetch 'worker_out' data ...")
            raw_output, raw_output_metadata = self.queues["worker_out"].get(self.loop_pause)
            # TODO: Refine output
            cfg.LOGGER.info(f"Forwarding {raw_output} to synthesizer...")
            self.queues["synthesizer_in"].put(raw_output)
            return True
        except Empty:
            cfg.LOGGER.info(f"'worker_out'-queue is empty.")
            return False

    def handle_output(self) -> bool:
        """
        Acquires and reroutes generated output based on configured output method.
        :return: True, if data was successfully outputted, else False.
        """  
        try:
            cfg.LOGGER.info(f"Trying to fetch 'synthesizer_out' data ...")
            output, output_metadata = self.queues["synthesizer_out"].get(self.loop_pause)
            if output:
                cfg.LOGGER.info(f"Ouputting synthesized response...")
                text_to_speech_utility.play_wave(output, output_metadata)
            return True
        except Empty:
            cfg.LOGGER.info(f"'synthesizer_out'-queue is empty.")
            return False

    def run_conversation_loop(self, blocking: bool = True) -> None:
        """
        Runs conversation loop.
        :param blocking: Flag which declares whether or not to wait for each step.
            Defaults to True.
        """
        cfg.LOGGER.info(f"Starting conversation loop...")
        while not self.interrupt.is_set():
            try:
                cfg.LOGGER.info(f"[1/4] Handling input...")
                self.handle_input()
                cfg.LOGGER.info(f"[2/4] Preparing worker input...")
                while not self.input_to_worker_gateway() and blocking:
                    time.sleep(self.loop_pause)
                cfg.LOGGER.info(f"[3/4] Preparing worker output...")
                while not self.worker_to_output_gateway() and blocking:
                    time.sleep(self.loop_pause)
                cfg.LOGGER.info(f"[4/4] Handling output...")
                while not self.handle_output() and blocking:
                    time.sleep(self.loop_pause)
               
                time.sleep(self.loop_pause)
            except KeyboardInterrupt:
                cfg.LOGGER.info(f"Recieved keyboard interrupt, shutting down handler ...")
                self._stop()
        

if __name__ == "__main__":
    faster_whisper_model = f"{cfg.PATHS.SOUND_GENERATION_MODEL_PATH}/speech_to_text/faster_whisper_models/Systran_faster-whisper-tiny"
    coqui_tts_model = f"{cfg.PATHS.SOUND_GENERATION_MODEL_PATH}/text_to_speech/coqui_models/tts_models--multilingual--multi-dataset--xtts_v2"
    coqui_tts_speaker_wav = f"{cfg.PATHS.SOUND_GENERATION_MODEL_PATH}/text_to_speech/coqui_xtts/examples/female.wav"
    phi_3_path = f"{cfg.PATHS.TEXT_GENERATION_MODEL_PATH}/microsoft_Phi-3-mini-4k-instruct-gguf"
    phi_3_file = f"Phi-3-mini-4k-instruct-q4.gguf"
    llama_3_norefusal_path = f"{cfg.PATHS.TEXT_GENERATION_MODEL_PATH}/text_generation_models/mradermacher_Llama-3-8B-Instruct-norefusal-i1-GGUF"
    llama_3_norefusal_file = f"Llama-3-8B-Instruct-norefusal.i1-Q4_K_M.gguf"

    transcriber_kwargs = {
        "backend": "faster-whisper",
        "model_path": faster_whisper_model,
        "model_parameters": {
            "device": "cuda",
            "compute_type": "float32",
            "local_files_only": True
        }
    }
    synthesizer_kwargs = {
        "backend": "coqui-tts",
        "model_path": coqui_tts_model,
        "model_parameters": {
            "config_path": f"{coqui_tts_model}/config.json",
            "gpu": True
        },
        "synthesis_parameters": {
            "speaker_wav": coqui_tts_speaker_wav,
            "language": "en"
        }
    }
    phi_3_llm_kwargs = {
        "backend": "transformers",
        "model_path": phi_3_path,
        "model_file": phi_3_file,
        "model_parameters": {"context_length": 4096, "max_new_tokens": 1024},
        "prompt_maker": lambda history: "\n".join(f"<|{entry[0]}|>\n{entry[1]}{'<|end|>' if entry[0] == 'user' else ''}\n<|assistant|>" for entry in history),
    }

    transcriber = Transcriber(
        **transcriber_kwargs
    )

    synthesizer = Synthesizer(
        **synthesizer_kwargs
    )

    llm = LanguageModelInstance(
        **phi_3_llm_kwargs
    )

    handler = ConversationHandler(
        working_directory=os.path.join(cfg.PATHS.DATA_PATH, "voice_assistant"),
        transcriber=transcriber,
        synthesizer=synthesizer,
        worker_function=llm.generate
    )

    handler.run_conversation_loop()