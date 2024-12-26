# -*- coding: utf-8 -*-
"""
****************************************************
*                      Utility                 
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from typing import Any, Tuple, List, Callable, Generator
import numpy as np
from queue import Empty
from src.utility.commandline_utility import silence_stderr
from src.utility.pyaudio_utility import play_wave
from src.sound_model_abstractions import Transcriber, Synthesizer, SpeechRecorder
from src.modules.module_abstractions import VAModule, VAPackage


class SpeechRecorderModule(VAModule):
    """
    Speech recorder module.
    Records a speech snipped from the user and forwards it as VAPackage.
    """
    def __init__(self, 
                 speech_recorder: SpeechRecorder, 
                 *args: Any | None, 
                 **kwargs: Any | None) -> None:
        """
        Initiates an instance.
        :param speech_recorder: Speech recorder instance.
        :param args: Arbitrary arguments.
        :param kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.speech_recorder = speech_recorder

    def process(self) -> VAPackage | Generator[VAPackage, None, None] | None:
        """
        Module processing method.
        :returns: Voice assistant package, containing the recorded audio data and a metadata stack with the
            recording metadata as first element.
        """
        if not self.pause.is_set():
            recorder_output, recorder_metadata = self.speech_recorder.record_single_input()
            self.log_info(f"Got voice input.")
            yield VAPackage(content=recorder_output, metadata_stack=[recorder_metadata])
        

class WaveOutputModule(VAModule):
    """
    Wave output module.
    Takes in VAPackages with wave audio data and outputs it. 
    Note, that the last metadata stack element must be stream parameters for outputting.
    """
    def process(self) -> VAPackage | Generator[VAPackage, None, None] | None:
        """
        Module processing method.
        Note, that the last metadata stack element must be stream parameters for outputting.
        Follow the "play_wave"-function for more information.
        """
        if not self.pause.is_set():
            try:
                input_package: VAPackage = self.input_queue.get(block=True, timeout=self.input_timeout)
                self.pause.set()
                self.add_uuid(self.received, input_package.uuid)
                self.log_info(f"Received input:\n'{input_package.content}'")
                with silence_stderr():
                    play_wave(input_package.content, input_package.metadata_stack[-1])
                self.pause.clear()
            except Empty:
                pass
        

class BasicHandlerModule(VAModule):
    """
    Basic handler module.
    """
    def __init__(self, 
                 handler_method: Callable, 
                 *args: Any | None, 
                 **kwargs: Any | None) -> None:
        """
        Initiates an instance.
        :param handler_method: Handler method.
        :param args: Arbitrary arguments.
        :param kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.handler_method = handler_method

    def process(self) -> VAPackage | Generator[VAPackage, None, None] | None:
        """
        Module processing method.
        :returns: Voice assistant package, package generator or None.
        """
        if not self.pause.is_set():
            try:
                input_package: VAPackage = self.input_queue.get(block=True, timeout=self.input_timeout)
                self.add_uuid(self.received, input_package.uuid)
                self.log_info(f"Received input:\n'{input_package.content}'")
                valid_input = (isinstance(input_package.content, np.ndarray) and input_package.content.size > 0) or input_package.content

                if valid_input:
                    result = self.handler_method(input_package.content)
                    if isinstance(result, Generator):
                        for response_tuple in result:
                            self.log_info(f"Received response shard\n'{response_tuple[0]}'.")   
                            yield VAPackage(uuid=input_package.uuid, content=response_tuple[0], metadata_stack=input_package.metadata_stack + [response_tuple[1]])
                    else:
                        self.log_info(f"Received response\n'{result[0]}'.")             
                        yield VAPackage(uuid=input_package.uuid, content=result[0], metadata_stack=input_package.metadata_stack + [result[1]])
            except Empty:
                pass


class TranscriberModule(BasicHandlerModule):
    """
    Transcriber module.
    Transcribes audio data.
    """
    def __init__(self, 
                 transcriber: Transcriber, 
                 *args: Any | None, 
                 **kwargs: Any | None) -> None:
        """
        Initiates an instance.
        :param transcriber: Transcriber.
        :param args: Arbitrary arguments.
        :param kwargs: Arbitrary keyword arguments.
        """
        super().__init__(handler_method=transcriber.transcribe, *args, **kwargs)


class SynthesizerModule(BasicHandlerModule):
    """
    Synthesizer module.
    Synthesizes and forwards audio data.
    """
    def __init__(self, 
                 synthesizer: Synthesizer, 
                 *args: Any | None, 
                 **kwargs: Any | None) -> None:
        """
        Initiates an instance.
        :param synthesizer: Synthesizer.
        :param args: Arbitrary arguments.
        :param kwargs: Arbitrary keyword arguments.
        """
        super().__init__(handler_method=synthesizer.synthesize, *args, **kwargs)