# -*- coding: utf-8 -*-
"""
****************************************************
*                      Utility                 
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import os
import signal
import sys
import psutil
import time
import traceback
from autogen.agentchat import AssistantAgent, UserProxyAgent
from typing import Any, Tuple, Union
from .llama_cpp_python_utility import load_llamacpp_server_subprocess, terminate_llamacpp_server_subprocess


def run_llama_cpp_server_based_chat(server_config: Union[str, dict],
                                         initiation_message: str,
                                         assistant_llm_config: dict = None,
                                         user_proxy_llm_config: dict = None) -> None:
    """
    Function for running a llama cpp server based chat.
    :param server_config: Path to llama cpp server config file or config dictionary.
    :param initiation_message: Chat initiation message.
    :param assistant_llm_config: LLM config for assistant.
    :param user_proxy_llm_config: LLM config for user proxy.
    """
    process = load_llamacpp_server_subprocess(
        config=server_config,
        wait_for_startup=True
    )
    try:
        assistant = AssistantAgent(
            "assistant",
            llm_config=assistant_llm_config
        )
        user_proxy = UserProxyAgent(
            "user_proxy",
            is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),            code_execution_config={
                "work_dir": "sandbox",
                "use_docker": False
            },
            llm_config=user_proxy_llm_config
        )

        user_proxy.initiate_chat(
            recipient=assistant,
            message=initiation_message
        )
    except Exception:
        traceback.print_exc()
        
    terminate_llamacpp_server_subprocess(process=process)