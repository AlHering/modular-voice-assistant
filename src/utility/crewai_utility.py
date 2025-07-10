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
from crewai_workbench import Agent, Task, Crew, Process
from crewai_tools import Tool
from typing import Any, Tuple, Union, List, Dict, Optional
from .llama_cpp_python_utility import load_llamacpp_server_subprocess, terminate_llamacpp_server_subprocess


def run_llama_cpp_server_based_crew(server_config: Union[str, dict],
                                    agent_configs: Dict[str, dict],
                                    task_configs: Dict[str, dict],
                                    crew_config: dict) -> Optional[str]:
    """
    Function for running a llama cpp server based crew.
    :param server_config: Path to llama cpp server config file or config dictionary.
    :param agent_configs: Agent configs under the appropriate agent name as dictionary.
    :param task_configs: Task configs under the appropriate task name as dictionary.
    :param crew_config: Crew config.
    :return: Crew response.
    """
    response = None
    process = load_llamacpp_server_subprocess(
        config=server_config,
        wait_for_startup=True
    )

    try:
        os.environ["OPENAI_API_BASE"] = f"http://{server_config['host']}:{server_config['port']}/v1"
        os.environ["OPENAI_MODEL_NAME"] = server_config["models"][0].get("model_alias", server_config["models"][0]["model"])
        os.environ["OPENAI_API_KEY"] = "NA"

        agents = {
            agent: Agent(**agent_configs[agent]) for agent in agent_configs
        }
        for task in [task for task in task_configs if "agent" in task_configs[task]]:
            task_configs[task]["agent"] = agents[task_configs[task]["agent"]]
        tasks = {
            task: Task(**task_configs[task]) for task in task_configs
        }
        crew_config["agents"] = [agents[agent] for agent in crew_config["agents"]]
        crew_config["tasks"] = [tasks[task] for task in crew_config["tasks"]]

        crew = Crew(**crew_config)

        response = crew.kickoff()
    except:
        traceback.print_exc()
    terminate_llamacpp_server_subprocess(process=process)

    return response


def run_llama_cpp_server_research_task(server_config: Union[str, dict],
                                       goal: str) -> Optional[str]:
    """
    Function for running a llama cpp server based crew.
    :param server_config: Path to llama cpp server config file or config dictionary.
    :param goal: Goal of the researcher.
    :return: Crew response.
    """
    from langchain_community.tools import DuckDuckGoSearchRun
    search_tool = DuckDuckGoSearchRun()
    search_tool.config_schema

    return run_llama_cpp_server_based_crew(
        server_config=server_config,
        agent_configs={
            "researcher": {
                "role": "Researcher",
                "goal": goal,
                "backstory": "Your specialty is finding and condensing large amounts of information.",
                "verbose": True,
                "allow_delegation": False,
                "tools": [search_tool]
            },
            "advisor": {
                "role": "Advisor",
                "goal": "Create advice from given information.",
                "backstory": "You take information and transfer it into easy to understand advice.",
                "verbose": True,
                "allow_delegation": True
            }
        },
        task_configs={
            "task_a": {
                "description": "Collect and condense information on the given topic.",
                "expected_output": "Full research report in bullet points.",
                "agent": "researcher"
            },
            "task_b": {
                "description": "Take the provided information and create advice from it.",
                "expected_output": "Advice in bullet points.",
                "agent": "advisor"
            }
        },
        crew_config={
            "agents": ["researcher", "advisor"],
            "tasks": ["task_a", "task_b"],
            "verbose": 2,
        }
    )