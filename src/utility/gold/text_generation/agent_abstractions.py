
# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import traceback
from typing import List, Any, Callable, Optional, Type, Union, Tuple
from uuid import uuid4
from datetime import datetime as dt
from .language_model_abstractions import LanguageModelInstance, ChatModelInstance, RemoteChatModelInstance
from .knowledgebase_abstractions import MemoryMetadata, MemoryEntry, Memory
from ..filter_mask import FilterMask


"""
Abstractions
"""


class ToolArgument(object):
    """
    Class, representing a tool argument.
    """

    def __init__(self,
                 name: str,
                 type: Type,
                 description: str,
                 value: Any) -> None:
        """
        Initiation method.
        :param name: Name of the argument.
        :param type: Type of the argument.
        :param description: Description of the argument.
        :param value: Value of the argument.
        """
        self.name = name
        self.type = type
        self.description = description
        self.value = value

    def extract(self, input: str) -> bool:
        """
        Method for extracting argument from input.
        :param input: Input to extract argument from.
        :return: True, if extraction was successful, else False.
        """
        try:
            self.value = self.type(input)
            return True
        except TypeError:
            return False

    def __call__(self) -> Any:
        """
        Call method for returning value.
        :return: Stored value.
        """
        return self.value


class AgentTool(object):
    """
    Class, representing a tool.
    """

    def __init__(self,
                 name: str,
                 description: str,
                 func: Callable,
                 arguments: List[ToolArgument],
                 return_type: Type) -> None:
        """
        Initiation method.
        :param name: Name of the tool.
        :param description: Description of the tool.
        :param func: Function of the tool.
        :param arguments: Arguments of the tool.
        :param return_type: Return type of the tool.
        """
        self.name = name
        self.description = description
        self.func = func
        self.arguments = arguments
        self.return_type = return_type

    def get_guide(self) -> str:
        """
        Method for acquiring the tool guide.
        :return: Tool guide as string.
        """
        arguments = ", ".join(
            f"{arg.name}: {arg.type}" for arg in self.arguments)
        return f"{self.name}: {self.func.__name__}({arguments}) -> {self.return_type} - {self.description}"

    def get_openai_function_representation(self) -> dict:
        """
        Method for acquiring openai function representation.
        :return: Tool as openai function dictionary.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    argument.name: {
                        "type": str(argument.type).lower(),
                        "description": argument.description
                    } for argument in self.arguments
                }
            }
        }


    def __call__(self) -> Any:
        """
        Call method for running tool function with arguments.
        """
        return self.func(**{arg.name: arg.value for arg in self.arguments})


class AgentMemoryEntry(object):
    """
    Class, representing an agent's memory entry.
    """
    def __init__(
            self,
            id: Union[int, str],
            timestamp: dt,
            content: str,
            embedding: List[float],
            importance: int = -1,
            layer: int = 0,
            metadata: dict = {}) -> None:
        """
        Initiation method.
        :param id: The ID of the entry.
        :param timestamp: Timestamp of creation.
        :param content: Textual content of the memory.
        :param embedding: Embedded content.
        :param importance: Memory importance. The higher the number, the more important the memory.
        :param layer: Memorization layer. Level one means an atomic memory. This number is incremented with
            every iteration of memory condensation (concatenation and summarization of multiple entries).
        :param metadata: Metadata for additional storage. Examples would be agents in a multi-agent-system or
            entity relationships.
        """
        self.id = id
        self.timestamp = timestamp
        self.content = content
        self.embedding = embedding
        self.importance = importance
        self.layer = layer
        self.metadata = metadata


class Agent(object):
    """
    Class, representing an agent.
    """

    def __init__(self,
                 llm: ChatModelInstance,
                 tools: List[AgentTool] = None,
                 memory: Memory = None) -> None:
        """
        Initiation method.
        :param llm: ChatModelInstance for handling text generation tasks.
        :param tools: List of tools to be used by the agent.
            Defaults to None in which case no tools are used.
        :param memory: Memory to use.
            Defaults to None.
        """
        self.llm = llm
        self.tools = tools
        self.tool_guide = self.create_tool_guide()
        self.memory = memory
        self.cache = []

        self.planner_format = f"""Task: {{prompt}} 
            
            Create a plan for solving the task. Respond in the following format:
            THOUGHT: Formulate precisely, what you want to do.
            TOOL: The name of the tool to use. Should be one of [{', '.join(tool.name for tool in self.tools)}]. Only add this line if you want to use a tool in this step.
            INPUTS: The inputs for the tool, separated by a comma. Only add arguments, if the tool needs them. Only add the arguments appropriate for the tool. Only add this line if you want to use a tool in this step."""

    def create_tool_guide(self) -> Optional[str]:
        """
        Method for creating a tool guide.
        """
        if self.tools is None:
            return None
        return "\n\n" + "\n\n".join(tool.get_guide() for tool in self.tools) + "\n\n"
    
    def parse_tool_call(self, content: str, metdata: dict) -> Optional[Any]:
        """
        Method for parsing a tool call.
        :param content: The generated text content.
        :param metadata: Planning step metadata.
        :return: Return value of a tool, if a tool call was detected.
        """
        pass

    def interact(self, prompt: str) -> Optional[List[dict]]:
        """
        Method for a single interaction.
        :param prompt: User prompt.
        :return: Agent history.
        """
        current_history = []
        try:
            self.plan(prompt)
            current_history.append(self.cache[-1])
            self.act()
            current_history.append(self.cache[-1])
            self.observe()
            current_history.append(self.cache[-1])
        except Exception as ex:
            current_history.append[{
                "exception": str(ex),
                "trace": traceback.format_exc()
            }]

    def loop(self, start_prompt: str, stop_on_error: bool = True) -> Any:
        """
        Method for starting handler loop.
        :param start_prompt: Start prompt.
        :param stop_on_error: Whether to stop when encountering exceptions. 
            Defaults to True.
        :return: Answer.
        """
        current_history = []
        while not current_history or "FINISHED" in current_history[-1].get("content"):
            try:
                if current_history and "ITERATE" in current_history[-1].get("content"):
                    if "traceback" in current_history[-1]:
                        self.plan(prompt=f"The task is not solved. An error appeared:\n```{current_history[-1]['traceback']}```\nRework your initial plan accordingly.")
                    else:
                        self.plan(prompt=f"The task is not solved. Rework your initial plan accordingly.")
                else:
                    self.plan(prompt=start_prompt)
                current_history.append(self.cache[-1])
                self.act()
                current_history.append(self.cache[-1])
                self.observe()
                current_history.append(self.cache[-1])
            except Exception as ex:
                current_history.append[{
                    "exception": str(ex),
                    "traceback": traceback.format_exc(),
                    "content": "FINISHED" if stop_on_error else "ITERATE"
                }]
        

    def plan(self, prompt: str, wrap: bool = True) -> None:
        """
        Method for handling an planning step.
        :param prompt: Prompt to wrap into planning prompt.
        :param wrap: Wrap prompt into planning prompt before forwarding.
            Defaults to True.
        """
        answer, metadata = self.llm.chat(
            self.planner_format.format(prompt=prompt) if wrap else prompt
        )
        self.cache.append({
            "step": "plan",
            "content": answer,
            "metadata": metadata
        }) 

    def act(self) -> None:
        """
        Method for handling an acting step.
        """
        entry = self.cache[-1]
        if entry["step"] == "plan":
            tool_response = self.parse_tool_call(entry["content"], entry["metadata"])
            if tool_response is not None:
                self.cache.append({
                    "step": "act",
                    "content": tool_response,
                    "metadata": {
                        "type": "tool_response"
                    }
                })
            else:
                answer, metadata = self.llm.chat("Solve the task, that is described in your previous THOUGHT.")
                self.cache.append({
                    "step": "act",
                    "content": answer,
                    "metadata": metadata
                }) 

    def observe(self) -> None:
        """
        Method for handling an observation step.
        """
        answer, metadata = self.llm.chat("Did your last response solve the task? If yes, please answer with 'FINISHED' if not, answer with 'ITERATE'.")
        self.cache.append({
            "step": "act",
            "content": answer,
            "metadata": metadata
        }) 

