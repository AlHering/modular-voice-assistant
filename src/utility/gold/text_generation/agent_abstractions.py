
# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
from typing import List, Any, Callable, Optional, Type, Union
from uuid import uuid4
from datetime import datetime as dt
from .language_model_abstractions import LanguageModelInstance
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

    def __call__(self) -> str:
        """
        Call method for returning value as string.
        :return: Stored value as string.
        """
        return str(self.value)


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
        :return tool guide as string.
        """
        arguments = ", ".join(
            f"{arg.name}: {arg.type}" for arg in self.arguments)
        return f"{self.name}: {self.func.__name__}({arguments}) -> {self.return_type} - {self.description}"

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


class AgentMemory(object):
    """
    Class, representing a longterm memory.
    """
    supported_backends: List[str] = ["cache"]

    def __init__(self, uuid: str = None, backend: str = "cache", memories: List[AgentMemoryEntry] = None, path: str = None) -> None:
        """
        Initiation method.
        :param uuid: UUID for identifying memory object.
            Defaults to None, in which case a new UUID is generated.
        :param backend: Memory backend. Defaults to "cache".
            Check AgentMemory.supported_backends for supported backends.
        :param memories: List of memory entries to initialize memory with.
            Defaults to None.
        :param path: Path for reading and writing memory, if the backend supports it.
            Defaults to None.
        """
        self.uuid = str(uuid4()) if uuid is None else uuid
        self.backend = backend
        self.path = path
        self._initiate_memory(memories)

    def _initiate_memory(self, memories: List[AgentMemoryEntry] = None) -> None:
        """
        Method for initiating memories.
        :param memories: List of memory entries for initialization.
            Defaults to None.
        """
        pass

    def memorize(self, content: str, metadata: dict) -> None:
        """
        Method for memorizing something.
        This method should be used for memory model agnostic usage.
        :param content: Memory content.
        :param metadata: Metadata for memory.
        """
        pass

    def remember(self, reference: str, metadata: dict) -> Optional[List[str]]:
        """
        Method for remembering something.
        This method should be used for memory model agnostic usage.
        :param reference: Recall reference.
        :param metadata: Metadata.
        :return: Memory contents as list of strings.
        """
        pass

    def add_memory(self, memory: AgentMemoryEntry) -> None:
        """
        Method to add a memory.
        :param memory: Memory to add.
        """
        pass

    def retrieve_memories(self, retrieval_method: str = "similarity", *args: Optional[Any], **kwargs: Optional[Any]) -> List[AgentMemoryEntry]:
        """
        Method to add a memory.
        :param args: Arbitrary initiation arguments.
        :param kwargs: Arbitrary initiation keyword arguments.
        :return: List of memories.
        """
        pass

    def retrieve_memories_by_filtermask(self, filtermasks: List[FilterMask]) -> List[AgentMemoryEntry]:
        """
        Method for retrieving memory by filtermasks.
        :param filtermasks: List of filtermasks.
        """
        pass

    def retrieve_memories_by_ids(self, ids: List[Union[int, str]]) -> List[AgentMemoryEntry]:
        """
        Method for retrieving memories by IDs.
        :param ids: IDs of the memories to retrieve.
        """
        pass

    def retrieve_memories_by_similarity(self, reference: str, filtermasks: List[FilterMask] = None, retrieval_parameters: dict = None) -> List[AgentMemoryEntry]:
        """
        Method for retrieving memories by similarity.
        :param reference: Reference for similarity search.
        :param filtermasks: List of filtermasks for additional filering.
            Defaults to None.
        :param retrieval_parameters: Keyword arguments for retrieval.
            Defaults to None.
        """
        pass



class Agent(object):
    """
    Class, representing an agent.
    """

    def __init__(self,
                 general_llm: LanguageModelInstance,
                 tools: List[AgentTool] = None,
                 memory: AgentMemory = None,
                 dedicated_planner_llm: LanguageModelInstance = None,
                 dedicated_actor_llm: LanguageModelInstance = None,
                 dedicated_oberserver_llm: LanguageModelInstance = None) -> None:
        """
        Initiation method.
        :param general_llm: LanguageModelInstance for general tasks.
        :param tools: List of tools to be used by the agent.
            Defaults to None in which case no tools are used.
        :param memory: Memory to use.
            Defaults to None.
        :param dedicated_planner_llm: LanguageModelInstance for planning.
            Defaults to None in which case the general LLM is used for this task.
        :param dedicated_actor_llm: LanguageModelInstance for acting.
            Defaults to None in which case the general LLM is used for this task.
        :param dedicated_oberserver_llm: LanguageModelInstance for observing.
            Defaults to None in which case the general LLM is used for this task.
        """
        self.general_llm = general_llm
        self.tools = tools
        self.tool_guide = self._create_tool_guide()
        self.cache = None
        self.memory = memory
        self.planner_llm = self.general_llm if dedicated_planner_llm is None else dedicated_planner_llm
        self.actor_llm = self.general_llm if dedicated_actor_llm is None else dedicated_actor_llm
        self.observer_llm = self.general_llm if dedicated_oberserver_llm is None else dedicated_oberserver_llm

        self.system_prompt = f"""You are a helpful assistant. You have access to the following tools: {self.tool_guide} Your goal is to help the user as best as you can."""

        self.general_llm.use_history = False
        self.general_llm.system_prompt = self.system_prompt

        self.planner_answer_format = f"""Answer in the following format:
            THOUGHT: Formulate precisely what you want to do.
            TOOL: The name of the tool to use. Should be one of [{', '.join(tool.name for tool in self.tools)}]. Only add this line if you want to use a tool in this step.
            INPUTS: The inputs for the tool, separated by a comma. Only add arguments if the tool needs them. Only add the arguments appropriate for the tool. Only add this line if you want to use a tool in this step."""

    def _create_tool_guide(self) -> Optional[str]:
        """
        Method for creating a tool guide.
        """
        if self.tools is None:
            return None
        return "\n\n" + "\n\n".join(tool.get_guide() for tool in self.tools) + "\n\n"

    def loop(self, start_prompt: str) -> Any:
        """
        Method for starting handler loop.
        :param start_prompt: Start prompt.
        :return: Answer.
        """
        self.cache.add(("user", start_prompt, {"timestamp": dt.now()}))
        kickoff_prompt = self.base_prompt + """Which steps need to be taken?
        Answer in the following format:

        STEP 1: Describe the first step. If you want to use a tools, describe how to use it. Use only one tool per step.
        STEP 2: ...
        """
        self.cache.add(("system", kickoff_prompt, {"timestamp": dt.now()}))
        self.cache.add(
            ("general", *self.general_llm.generate(kickoff_prompt)))

        self.system_prompt += f"\n\n The plan is as follows:\n{self.cache.get(-1)[1]}"
        for llm in [self.planner_llm, self.observer_llm]:
            llm.use_history = False
            llm.system_prompt = self.system_prompt
        while not self.cache.get(-1)[1] == "FINISHED":
            for step in [self.plan, self.act, self.observe]:
                step()
                self.report()

    def plan(self) -> Any:
        """
        Method for handling an planning step.
        :return: Answer.
        """
        if self.cache.get(-1)[0] == "general":
            answer, metadata = self.planner_llm.generate(
                f"Plan out STEP 1. {self.planner_answer_format}"
            )
        else:
            answer, metadata = self.planner_llm.generate(
                f"""The current step is {self.cache.get(-1)[1]}
                Plan out this step. {self.planner_answer_format}
                """
            )
        # TODO: Add validation
        self.cache.add("planner", answer, metadata)

    def act(self) -> Any:
        """
        Method for handling an acting step.
        :return: Answer.
        """
        thought = self.cache.get(-1)[1].split("THOUGHT: ")[1].split("\n")[0]
        if "TOOL: " in self.cache.get(-1)[1] and "INPUTS: " in self.cache.get(-1)[1]:
            tool = self.cache.get(-1)[1].split("TOOL: ")[1].split("\n")[0]
            inputs = self.cache.get(-1)[1].split(
                "TOOL: ")[1].split("\n")[0].strip()
            for part in [tool, inputs]:
                if part.endswith("."):
                    part = part[:-1]
                part = part.strip()
            inputs = [inp.strip() for inp in inputs.split(",")]
            # TODO: Catch tool and input failures and repeat previous step.
            tool_to_use = [
                tool_option for tool_option in self.tools if tool.name == tool][0]
            result = tool_to_use.func(
                *[arg.type(inputs[index]) for index, arg in enumerate(tool_to_use.arguments)]
            )
            self.cache.add("actor", f"THOUGHT: {thought}\nRESULT:{result}", {
                "timestamp": dt.now(), "tool_used": tool.name, "arguments_used": inputs})
        else:
            self.cache.add("actor", *self.actor_llm.generate(
                f"Solve the following task: {thought}.\n Answer in following format:\nTHOUGHT: Describe your thoughts on the task.\nRESULT: State your result for the task."
            ))

    def observe(self) -> Any:
        """
        Method for handling an oberservation step.
        :return: Answer.
        """
        current_step = "STEP 1" if self.cache.get(
            -3)[0] == "general" else self.cache.get(-3)[1]
        planner_answer = self.cache.get(-2)[1]
        actor_answer = self.cache.get(-1)[1]
        self.cache.add("observer", *self.observer_llm.generate(
            f"""The current step is {current_step}.
            
            An assistant created the following plan:
            {planner_answer}

            Another assistant implemented this plan as follows:
            {actor_answer}

            Validate, wether the current step is solved. Answer in only one word:
            If the solution is correct and this was the last step, answer 'FINISHED'.
            If the solution is correct but there are more steps, answer 'NEXT'.
            If the solution is not correct, answer the current step in the format 'CURRENT'.
            Your answer should be one of ['FINISHED', 'NEXT', 'CURRENT']
            """
        ))
        # TODO: Add validation and error handling.
        self.cache.add("system", {"FINISHED": "FINISHED", "NEXT": "NEXT", "CURRENT": "CURRENT"}[
            self.cache.get(-1)[1].replace("'", "")], {"timestamp": dt.now()})

    def report(self) -> None:
        """
        Method for printing an report.
        """
        pass
