# -*- coding: utf-8 -*-
"""
****************************************************
*                      Utility                 
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import os
import sys
import time
import traceback
from typing import Any, Tuple, Union
from .llama_cpp_python_utility import load_llamacpp_server_subprocess


def initiate_llama_cpp_server_based_chat(server_config: Union[str, dict],
                                         initiation_message: str,
                                         assistant_llm_config: dict = None,
                                         user_proxy_llm_config: dict = None) -> None:
    """
    Function for initiating a llama cpp server based chat.
    :param server_config: Path to llama cpp server config file or config dictionary.
    :param initiation_message: Chat initiation message.
    :param start_up_time: Startup time to wait before connecting to server.
        Defaults to 5 seconds.
    :param assistant_llm_config: LLM config for assistant.
    :param user_proxy_llm_config: LLM config for user proxy.
    """
    process = load_llamacpp_server_subprocess(
        config=server_config,
        wait_for_startup=True
    )
    try:
        config_list = [{
            "model": "llama-3",
            "base_url": "http://localhost:8080/v1", 
            "api_key": "NULL"}]
        llm_config = {"timeout": 800, "config_list": config_list}

        from autogen.agentchat import AssistantAgent, UserProxyAgent

        assistant = AssistantAgent(
            "assistant",
            llm_config=assistant_llm_config
        )
        user_proxy = UserProxyAgent(
            "user_proxy",
            is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="NEVER",
            code_execution_config={
                "work_dir": "sandbox"
                "use_docker": False,
            },
            llm_config=user_proxy_llm_config
        )

        user_proxy.initiate_chat(
            recipient=assistant,
            message=initiation_message
        )
    except Exception as ex:
        traceback.print_exc()
    process.terminate()