
# -*- coding: utf-8 -*-
"""

WARNING: LEGACY CODE - just for reference

****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import traceback
from pydantic import BaseModel, Field
from typing import List, Any, Callable, Optional, Type, Union, Tuple
from uuid import uuid4
from datetime import datetime as dt
from src.model.abstractions.language_model_abstractions import ChatModelInstance, RemoteChatModelInstance
from .knowledgebase_abstractions import Knowledgebase, Document
from src.utility.filter_mask_utility import FilterMask


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


def create_memory_timestamp() -> str:
    """
    Returns memory timestamp.
    :return: Ctime formatted timestamp.
    """
    return dt.ctime(dt.now())


class MemoryMetadata(BaseModel):
    """
    Represents a memory entry metadata.
    """
    timestamp: str = Field(default_factory=create_memory_timestamp)
    importance: int = -1
    layer: int = 0
    sources: List[Union[int, str]] = []


class MemoryEntry(BaseModel):
    """
    Represents a memory entry.
    """
    id: Union[int, str]
    content: str
    embedding: List[float] | None = None
    metadata: MemoryMetadata = MemoryMetadata()


class Memory(object):
    """
    Represents a knowledgebase based memory.
    """
    def __init__(self, knowledgebase: Knowledgebase, memories: List[MemoryEntry] | None = None) -> None:
        """
        Initiation method.
        :param knowledgebase: Knowledgebase for managing memories.
        :param memories: List of memory entries for initialization.
            Defaults to None.
        """
        self.knowledgebase = knowledgebase
        self._initiate_memory(memories=memories)

    @classmethod
    def from_knowledgebase(cls, knowledgebase: Knowledgebase) -> Any:
        """
        Returns session instance from a json file.
        :param file_path: Path to json file.
        :returns: VoiceAssistantSession instance.
        """
        return cls(knowledgebase=knowledgebase)

    def _initiate_memory(self, memories: List[MemoryEntry] | None = None) -> None:
        """
        Method for initiating memories.
        :param memories: List of memory entries for initialization.
            Defaults to None.
        """
        if memories is not None:
            for memory in memories:
                self.add_memory(memory)

    """
    Conversion functionality
    """

    def memory_to_document(self, memory: MemoryEntry) -> Document:
        """
        Method for converting memory entries to documents.
        :param memory: Memory entry.
        :return: Document.
        """
        return Document(
            id=memory.id,
            content=memory.content,
            metadata=memory.metadata.model_dump(),
            embedding=memory.embedding
        )
    
    def document_to_memory(self, document: Document, importance: int = -1, layer: int = 0) -> MemoryEntry:
        """
        Method for converting documents to memory entries. Importance and layer can be used to organize memories.
        It is recommended, to use negative integers for custom memory types (like factual knowledge from books).
        It is further recommended, to consolidate similar lower layer memories into higher layer memories every now and then, 
        and afterwards respectively retrieve memories from higher to lower layer.
        :param document: Document.
        :param importance: Document importance.
            Defaults to -1.
        :param layer: Memory layer.
            Defaults to 0. 
        :return: Memory entry.
        """
        return MemoryEntry(
            id=document.id,
            content=document.content,
            metadata=MemoryMetadata(
                importance=document.metadata.get("importance", importance),
                layer=document.metadata.get("layer", layer)
            ),
            embedding=document.embedding
        )
    
    """
    Insertion functionality
    """

    def memorize(self, content: str) -> None:
        """
        Method for memorizing something.
        This method should be used for memory model agnostic usage.
        :param content: Memory content.
        """
        self.add_memory(MemoryEntry(
            id=str(uuid4()),
            content=content,
            metadata=MemoryMetadata()
        ))

    def remember(self, 
                 reference: str, 
                 max_retrievals: int = 4,
                 min_importance: int | None = None, 
                 min_layer: int | None = None,
                 max_importance: int | None = None, 
                 max_layer: int | None = None) -> Optional[List[Tuple[str, dict]]]:
        """
        Method for remembering something.
        This method should be used for memory model agnostic usage.
        Importance and layer can be used to filter for specific memories.
        It is recommended, to use negative integers for custom memory types (like factual knowledge from books).
        It is further recommended, to consolidate similar lower layer memories into higher layer memories every now and then, 
        and afterwards respectively retrieve memories from higher to lower layer.
        :param reference: Recall reference.
        :param max_retrievals: Maximum number of retrieved memories.
        :param min_importance: Minimum importance of the memory.
            Defaults to None.
        :param min_layer: Minimum layer of the memory.
            Defaults to None.
        :param max_importance: Maximum importance of the memory.
            Defaults to None.
        :param max_layer: Maximum layer of the memory.
            Defaults to None.
        :return: Memories as list of memory texts and metadata entries.
        """
        filtermask = []
        
        if min_importance is not None:
            filtermask.append(["importance", ">=", min_importance])
        if min_layer is not None:
            filtermask.append(["layer", ">=", min_layer])
        if max_importance is not None:
            filtermask.append(["importance", "<=", max_importance])
        if max_layer is not None:
            filtermask.append(["layer", "<=", max_layer])

        if filtermask:
            memories = self.retrieve_memories_by_similarity(
                    reference=reference,
                    filtermasks=[filtermask],
                    retrieval_parameters={"n_results": max_retrievals,
                                          "include": ["embeddings", "metadatas", "documents", "distances"]})
        else:
            memories = self.retrieve_memories_by_similarity(
                    reference=reference,
                    retrieval_parameters={"n_results": max_retrievals,
                                          "include": ["embeddings", "metadatas", "documents", "distances"]})
        return [(memory.content, memory.metadata.model_dump()) for memory in memories]

    def add_memory(self, memory: MemoryEntry) -> None:
        """
        Method to add a memory.
        :param memory: Memory to add.
        """
        self.knowledgebase.embed_documents(
            documents=[Document(
                id=memory.id,
                content=memory.content,
                metadata=memory.metadata.model_dump(),
                embedding=memory.embedding
            )]
        )

    """
    Retrieval functionality
    """

    def retrieve_all_memories(self) -> List[MemoryEntry]:
        """
        Method to retrieve all memories.
        :return: List of memories.
        """
        return [self.document_to_memory(doc) for doc in self.knowledgebase.get_all_documents()]

    def retrieve_memories_by_similarity(self, reference: str, filtermasks: List[FilterMask] | None = None, retrieval_parameters: dict | None = None) -> List[MemoryEntry]:
        """
        Method for retrieving memories by similarity.
        :param reference: Reference for similarity search.
        :param filtermasks: List of filtermasks for additional filtering.
            Defaults to None.
        :param retrieval_parameters: Keyword arguments for retrieval.
            Defaults to None.
        """
        return [self.document_to_memory(doc) for doc in
            self.knowledgebase.retrieve_documents(query=reference, filtermasks=filtermasks, retrieval_parameters=retrieval_parameters)]

    """
    Consolidation functionality
    """

    def consolidate_by_time(self) -> None:
        """
        Method for consolidating memory by time deltas.
        """
        pass

    def consolidate_by_similarity(self) -> None:
        """
        Method for consolidating memory by similarity.
        """
        pass


class Agent(object):
    """
    Class, representing an agent.
    """

    def __init__(self,
                 llm: ChatModelInstance | RemoteChatModelInstance,
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
    
    def parse_tool_call(self, content: str, metadata: dict) -> Optional[Any]:
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



