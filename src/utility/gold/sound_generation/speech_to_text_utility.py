# -*- coding: utf-8 -*-
"""
****************************************************
*                      Utility                 
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from typing import Any, List, Callable
import time
import numpy as np
import speech_recognition
from queue import Queue
import keyboard
from ...bronze.pyaudio_utility import InterruptMethod


def record_and_transcribe_speech_with_speech_recognition(transcription_callback: Callable,
                                                         transcription_model: Any = None,
                                                         interrupt_method: InterruptMethod = InterruptMethod.PAUSE_INTERVAL,
                                                         interrupt_threshold: Any = 5.0,
                                                         pause_threshold: float = 1.0,
                                                         chunk_size: int = 2024,
                                                         recognizer_kwargs: dict = None,
                                                         microphone_kwargs: dict = None,
                                                         logger: Any = None) -> List[bytes]:
    """
    Records and transcribes speech with the speech recognition framework.
    :param transcription_callback: Transcription callback function.
    :param transcription_model: Transcription model.
        Defaults to None in which case the callback function spawns its default model.
        Not providing a model therefore increases processing time tremendously!
    :param interrupt_method: Interrupt method as either "TIME_INTERVAL", "KEYBOARD_INTERRUPT", "PAUSE_INTERVAL". 
        Defaults to "PAUSE_INTERVAL".
    :param interrupt_threshold: Interrupt threshold as time interval in seconds for "TIME_INTERVAL", key(s) as string for "KEYBOARD_INTERRUPT",
        maximum pausing time in seconds for "PAUSE_INTERVAL".
        Defaults to 5.0 in connection with the "PAUSE_INTERVAL" method. 
    :param pause_threshold: The pausing time in seconds after which the next input is considered a new phrase.
        Defaults to 3.0.
    :param chunk_size: Chunk size for file handling. 
        Defaults to 1024.
    :param input_device_index: Input device index.
        Defaults to None in which case the default device is used.
    :param recognizer_kwargs: Recognizer keyword arguments.
        Defaults to None in which case default values are used.
    :param microphone_kwargs: Microphone keyword arguments.
        Defaults to None in which case default values are used.
    :param logger: Logger for logging process data.
        Defaults to None in which no additional log is outputted.
    :returns: Audio stream as list of bytes.
    """
    # Setting up speech recognition components
    audio_queue = Queue()
    recognizer = speech_recognition.Recognizer()
    recognizer_kwargs = {
        "energy_threshold": 1000,
        "dynamic_energy_threshold": False,
        "pause_threshold": pause_threshold
    } if recognizer_kwargs is None else recognizer_kwargs
    for key in recognizer_kwargs:
        setattr(recognizer, key, recognizer_kwargs[key])
    
    microphone_kwargs = {
        "device_index": None,
        "sample_rate": 16000,
        "chunk_size": chunk_size
    } if microphone_kwargs is None else microphone_kwargs
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
    transcriptions = []
    interrupt_flag = False
    transcription_start = time.time()
    last_empty_transcription = None

    if logger is not None:
        logger.info("Starting speech recognition loop...")

    while not interrupt_flag:
        try:
            if not audio_queue.empty():
                if logger is not None:   
                    logger.info("Recieved audio input.")
                audio = b"".join(audio_queue.queue)
                audio_queue.queue.clear()

                # Convert and transcribe audio input
                audio_as_numpy_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
                fulltext, segment_metadatas = transcription_callback(
                    audio_input=audio_as_numpy_array,
                    model=transcription_model,
                    transcription_parameters=None
                )
                transcriptions.append((fulltext, segment_metadatas))
                if logger is not None:
                    logger.info(f"Transcribed: {fulltext}")

                # Handle empty transcription and potential pause interrupt
                if fulltext.strip() == "":
                    if last_empty_transcription is None:
                        last_empty_transcription = time.time()
                    elif interrupt_method == InterruptMethod.PAUSE_INTERVAL and time.time() - last_empty_transcription >= interrupt_threshold:
                        interrupt_flag = True
                elif last_empty_transcription is not None:
                    last_empty_transcription = None
                
                # Handle other potential interrupts
                if interrupt_method == InterruptMethod.TIME_INTERVAL and time.time() - transcription_start >= interrupt_threshold:
                    if logger is not None:
                        logger.info("Time interval threshold was surpassed, interrupting...")
                    interrupt_flag = True
                elif interrupt_method == InterruptMethod.KEYBOARD_INTERRUPT and keyboard.is_pressed(interrupt_threshold):
                    if logger is not None:
                        logger.info("Keyboard interrupt detected, interrupting...")
                    interrupt_flag = True
            else:
                if logger is not None:
                    logger.info("Recieved no audio input.")
                # Handle empty input and potential pause interrupt
                if last_empty_transcription is None:
                        last_empty_transcription = time.time()
                elif interrupt_method == InterruptMethod.PAUSE_INTERVAL and time.time() - last_empty_transcription >= interrupt_threshold:
                    if logger is not None:
                        logger.info("Pause interval threshold was surpassed, interrupting...")
                    interrupt_flag = True
                time.sleep(.2)
        except KeyboardInterrupt as ex:
            if interrupt_method == InterruptMethod.KEYBOARD_INTERRUPT and (not isinstance(interrupt_threshold, str) or interrupt_threshold == "ctrl+c"):
                # Interrupt method is keyboard interrupt and handle is not correctly set or set to keyboard interrupt
                interrupt_flag = True
            else:
                raise ex
            
    for entry in transcriptions:
        print(entry[0])
