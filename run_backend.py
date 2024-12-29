# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import os
import click
from src.configuration import configuration as cfg
from src.utility import json_utility
from src.voice_assistant import BasicVoiceAssistant, setup_default_voice_assistant
from src.interface import run


def get_valid_config_path(config_path: str | None) -> str | None:
    """
    Returns valid config path.
    :param config_path: Base config path.
    :return: Valid config path or None.
    """
    if config_path is not None:
        if not config_path.lower().endswith(".json"):
            config_path += ".json"
        if os.path.exists(config_path):
            return config_path
        else:
            rel_path = os.path.join(cfg.PATHS.CONFIG_PATH, config_path)
            if os.path.exists(rel_path):
                return rel_path


"""
Click-based entrypoint
"""
@click.command()
@click.option("--config", default=None, help="Path or name json configuration file for the voice assistant.")
@click.option("--mode", default=None, help="Interaction mode: (0) conversation, (1) single interaction, (2) terminal based interaction.")
def run_voice_assistant(config: str, mode:int) -> None:
    """Runner program for a voice assistant."""
    config_path = get_valid_config_path(config_path=config)
    if config_path is not None:
        print(f"\nValid config path given: {config_path}.")
        config_data = json_utility.load(config_path)
        voice_assistant: BasicVoiceAssistant = setup_default_voice_assistant(**config_data)
    else:
        print(f"\nNo valid config path given, using default configuration.")
        voice_assistant: BasicVoiceAssistant = setup_default_voice_assistant(use_remote_llm=True)
    if mode == 2:
        voice_assistant.run_interaction()
    elif mode == 3:
        voice_assistant.run_terminal_conversation()
    else:
        voice_assistant.run_conversation()


if __name__ == "__main__":
    run()
