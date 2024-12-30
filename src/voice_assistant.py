# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from typing import Tuple
from prompt_toolkit import PromptSession, HTML, print_formatted_text
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding.key_bindings import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style as PTStyle
from src.configuration import configuration as cfg
from threading import Event as TEvent
from src.utility.string_utility import separate_pattern_from_text, extract_matches_between_bounds, remove_multiple_spaces, EMOJI_PATTERN
from src.modules.abstractions import PipelinePackage, ModularPipeline
from src.modules.input_modules import SpeechRecorderModule, TranscriberModule
from src.modules.worker_modules import BasicHandlerModule, LocalChatModule, RemoteChatModule
from src.modules.output_modules import SynthesizerModule, WaveOutputModule


AVAILABLE_MODULES = {
    "speech_recorder": SpeechRecorderModule,
    "transcriber": TranscriberModule,
    "local_chat": LocalChatModule,
    "remote_chat": RemoteChatModule,
    "synthesizer": SynthesizerModule,
    "wave_output": WaveOutputModule
}


def setup_prompt_session(bindings: KeyBindings = None) -> PromptSession:
    """
    Function for setting up a command line prompt session.
    :param bindings: Key bindings.
        Defaults to None.
    :return: Prompt session.
    """
    return PromptSession(
        bottom_toolbar=[
        ("class:bottom-toolbar",
         "Prompt 'STOP' to exit.",)
    ],
        style=PTStyle.from_dict({
        "bottom-toolbar": "#333333 bg:#ffcc00"
    }),
        auto_suggest=AutoSuggestFromHistory(),
        key_bindings=bindings
    )
        

def clean_worker_output(text: str) -> Tuple[str, dict]:
    """
    Cleanse worker output from emojis and emotional hints.
    :param text: Worker output.
    :return: Cleaned text and metadata.
    """
    metadata = {"full_text": text}
    metadata["text_without_emojis"], metadata["emojis"] = separate_pattern_from_text(text=text, pattern=EMOJI_PATTERN)
    metadata["emotional_hints"] = [f"*{hint}*" for hint in extract_matches_between_bounds(start_bound=r"*", end_bound=r"*", text=metadata["text_without_emojis"])]
    metadata["text_without_emotional_hints"] = metadata["text_without_emojis"]
    if metadata["emotional_hints"]:
        for hint in metadata["emotional_hints"]:
            metadata["text_without_emotional_hints"] = metadata["text_without_emotional_hints"].replace(hint, "")
    return remove_multiple_spaces(text=metadata["text_without_emotional_hints"]), metadata


class BasicVoiceAssistant(object):
    """
    Represents a basic voice assistant.
    """

    def __init__(self,
                 speech_recorder: SpeechRecorderModule,
                 transcriber: TranscriberModule,
                 worker: BasicHandlerModule,
                 synthesizer: SynthesizerModule,
                 wave_output: WaveOutputModule,
                 stream: bool = True,
                 forward_logging: bool = False,
                 report: bool = False) -> None:
        """
        Initiation method.
        :param speech_recorder: Speech Recorder module.
        :param transcriber: Transcriber module.
        :param worker: Worker module, e.g. LocalChatModule or RemoteChatModule.
        :param synthesizer: Synthesizer module.
        :param wave_output: Wave output module.
        :param stream: Declares, whether chat model should stream its response.
        :param forward_logging: Flag for forwarding logger to modules.
        :param report: Flag for running report thread.
        """
        self.stream = stream

        forward_logging = cfg.LOGGER if forward_logging else None
        for va_module in [speech_recorder, transcriber, worker, synthesizer, wave_output]:
            va_module.logger = forward_logging

        self.pipeline = ModularPipeline(
            input_modules=[speech_recorder, transcriber],
            worker_modules=[worker, 
                            BasicHandlerModule(handler_method=clean_worker_output,
                               logger=forward_logging,
                               name="Cleaner")],
            output_modules=[synthesizer, wave_output]
        )
        self.pipeline.reroute_pipeline_queues()

        self.pipeline_kwargs = {}
        if isinstance(worker, LocalChatModule) or isinstance(worker, RemoteChatModule):
            if worker.chat_model.history[-1]["role"] == "assistant":
                self.pipeline_kwargs["greeting"] = worker.chat_model.history[-1]["content"]
        self.pipeline_kwargs["report"] = report

    def reset(self) -> None:
        """
        Method for resetting the conversation pipeline.
        """
        self.pipeline.reset()

    def stop(self) -> None:
        """
        Method for stopping conversation pipeline.
        """
        self.pipeline.stop_modules()

    def run_conversation(self, blocking: bool = True) -> None:
        """
        Method for running a looping conversation.
        :param blocking: Flag which declares whether or not to wait for each conversation step.
            Defaults to True.
        """
        self.pipeline.run_pipeline(blocking=blocking, **self.pipeline_kwargs)

    def run_interaction(self, blocking: bool = True) -> None:
        """
        Method for running an conversational interaction.
        :param blocking: Flag which declares whether or not to wait for each conversation step.
            Defaults to True.
        """
        self.pipeline.run_pipeline(blocking=blocking, loop=False, **self.pipeline_kwargs)

    def inject_prompt(self, prompt: str) -> None:
        """
        Injects a prompt into a running conversation.
        :param prompt: Prompt to inject.
        """
        self.pipeline.input_modules[-1].output_queue.put(PipelinePackage(content=prompt))

    def run_terminal_conversation(self) -> None:
        """
        Runs conversation loop with terminal input.
        """
        run_terminal_conversation(pipeline=self.pipeline, pipeline_kwargs=self.pipeline_kwargs)


def run_terminal_conversation(pipeline: ModularPipeline, pipeline_kwargs: dict = None) -> None:
    """
    Runs conversation loop with terminal input.
    :param pipeline: Pipeline.
    :param pipeline_kwargs: Pipeline keyword arguments, such as a 'greeting'.
    """
    pipeline_kwargs = {} if pipeline_kwargs is None else pipeline_kwargs
    stop = TEvent()

    pipeline.input_modules[0].pause.set()
    pipeline.run_pipeline(blocking=False, loop=True, **pipeline_kwargs)
    
    while not stop.is_set():
        with patch_stdout():
            user_input = input(
                "User: ")
            if user_input == "STOP":
                pipeline.reset()
                stop.set()
                print_formatted_text(HTML("<b>Bye...</b>"))
            elif user_input is not None:
                pipeline.input_modules[-1].output_queue.put(PipelinePackage(content=user_input))

def setup_default_voice_assistant(config: dict | None = None) -> BasicVoiceAssistant:
    """
    Sets up a default voice assistant for reference.
    :param config: Config to overwrite voice assistant default config with.
    :return: Basic voice assistant.
    """
    config = cfg.DEFAULT_COMPONENT_CONFIG if config is None else config
    if config.get("download_model_files"):
        raise NotImplementedError("Downloading models is not yet implemented!")

    return BasicVoiceAssistant(
        speech_recorder=SpeechRecorderModule(**config.get("speech_recorder", cfg.DEFAULT_SPEECH_RECORDER)),
        transcriber=TranscriberModule(**config.get("transcriber", cfg.DEFAULT_TRANSCRIBER)),
        worker=RemoteChatModule(**config.get("remote_chat", cfg.DEFAULT_REMOTE_CHAT)) if config.get("use_remote_llm") else LocalChatModule(**config.get("local_chat", cfg.DEFAULT_LOCAL_CHAT)),
        synthesizer=SynthesizerModule(**config.get("synthesizer", cfg.DEFAULT_SYNTHESIZER)),
        wave_output=WaveOutputModule(**config.get("wave_output", cfg.DEFAULT_WAVE_OUTPUT)),
        **config.get("voice_assistant", cfg.DEFAULT_VOICE_ASSISTANT)
    )