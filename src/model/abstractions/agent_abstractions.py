# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2025 Alexander Hering             *
****************************************************
"""
from typing import Callable, Any, List, Tuple
from pydantic import BaseModel
from uuid import UUID, uuid4
from enum import Enum
import json
from datetime import datetime as dt
from src.model.abstractions.language_model_abstractions import ChatModelInstance, RemoteChatModelInstance
from src.model.abstractions.knowledgebase_abstractions import Entry, ChromaKnowledgeBase, Neo4jKnowledgebase, Knowledgebase
from src.utility.filter_mask_utility import FilterMask
from src.utility.foreign.json_schema_to_grammar import SchemaConverter


def convert_pydantic_model_to_grammar(model: BaseModel) -> str:
    """
    Converts pydantic BaseModel class to GBNF grammar.
    :param model: Pydantic BaseModel class.
    :return: GBNF grammar as string.
    """
    conv = SchemaConverter(prop_order={},
                           allow_fetch=False,
                           dotall=False,
                           raw_pattern=False)
    schema = conv.resolve_refs(model.model_json_schema(), "")
    conv.visit(schema, "")
    return conv.format_grammar()


class MemoryType(str, Enum):
    chroma = "chroma"
    neo4j = "neo4j"


class AgentMemory(object):
    """
    Class, representing an agent's memory.
    """
    def __init__(self, 
                 memory_type: MemoryType = MemoryType.chroma, 
                 memory_parameters: dict | None = None,
                 retrieval_parameters: dict | None = None) -> None:
        """
        Initiation method.
        :param memory_type: Memory type.
        :param memory_parameters: Memory instantiation parameters.
        :param retrieval_parameters: Retrieval parameters. 
        """
        self.memory_parameters = {} if memory_parameters is None else memory_parameters
        self.retrieval_parameters = {} if retrieval_parameters is None else retrieval_parameters

        self.memory_type = {
            "chroma": ChromaKnowledgeBase,
            "neo4j": Neo4jKnowledgebase
        }[memory_type.value()]
        self.memory: Knowledgebase = self.memory_type(**self.memory_parameters)

    def store(self, content: str, metadata: dict) -> Entry:
        """
        Stores a memory.
        :param content: Textual content.
        :param metadata: Memory metadata.
        """
        entry = Entry(id=uuid4(),
                      content=content,
                      metadata=metadata)
        self.memory.store_entry(entry=entry)
        return entry
    
    def prepare_prompt_addition(self, one_or_multiple_entries: Entry | List[Entry]) -> str:
        """
        Prepares prompt addition for incorporating memory entries.
        :param one_or_multiple_entries: Memory entry or list of memory entries.
        :return: Prompt memory addition as string.
        """
        if isinstance(one_or_multiple_entries, Entry):
            one_or_multiple_entries = [one_or_multiple_entries]
        return "\nHere is a list of memories:\n" + "\n".join(
            f"Memory '{entry.id}':\nContent: '{entry.content}'\nMetadata{entry.metadata}" 
            for entry in one_or_multiple_entries) * "\n\n"

    def retrieve_by_id(self, entry_id: int | str | UUID) -> str:
        """
        Retrieves a memory by its ID.
        :param entry_id: Entry ID.
        :return: Prompt memory addition as string.
        """
        return self.prepare_prompt_addition(self.memory.get_entry_by_id(entry_id=entry_id))

    def retrieve_by_metadata(self, filtermasks: List[FilterMask]) -> str:
        """
        Retrieves a memories by metadata filters.
        :param filtermasks: List of filter masks.
        :return: Prompt memory addition as string.
        """
        return self.prepare_prompt_addition(self.memory.retrieve_entries(filtermasks=filtermasks))

    def retrieve_by_similarity(self, reference: str, filtermasks: List[FilterMask] | None = None) -> str:
        """
        Retrieves a memories by similarity.
        :param reference: Textual reference.
        :param filtermasks: List of filter masks.
        :param metadata: Prompt memory addition as string.
        """
        return self.prepare_prompt_addition(self.memory.retrieve_entries(query=reference, filtermasks=filtermasks))


class AgentTool(object):
    """
    Class, representing an agent tool.
    """
    def __init__(self,
                 name: str,
                 description: str,
                 func: Callable,
                 input_declaration: BaseModel | None = None,
                 output_declaration: BaseModel | None = None) -> None:
        """
        Initiation method.
        :param name: Name of the tool.
        :param description: Description of the tool.
        :param func: Function of the tool.
        :param input_declaration: Pydantic based input declaration.
        :param output_declaration: Pydantic based output declaration.
        """
        self.name = name
        self.description = description
        self.func = func
        self.input_declaration = input_declaration
        self.output_declaration = output_declaration

    @classmethod
    def from_function(cls, func: Callable) -> Any:
        """
        Returns a tool instance from a function.
        :param func: Target function.
        :return: AgentTool instance.
        """
        raise NotImplementedError("Function to tool conversion not yet implemented.")
        parameters = {}
        return cls(**parameters)

    def get_openai_function_representation(self) -> dict:
        """
        Method for acquiring openai function representation.
        :return: Tool as openai function dictionary.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_declaration.model_json_schema()
        }
    
    def get_input_grammar(self) -> str:
        """
        Returns input grammar.
        """
        if self.input_declaration is None:
            return ""
        else:
            return convert_pydantic_model_to_grammar(model=self.input_declaration)

    def get_output_grammar(self) -> str:
        """
        Returns output grammar.
        """
        if self.output_declaration is None:
            return ""
        else:
            return convert_pydantic_model_to_grammar(model=self.input_declaration)
    
    def __call__(self, *args: Any | None, **kwargs: Any | None) -> Any:
        """
        Call method for running tool function with arguments.
        :param args: Arbitrary function arguments.
        :param kwargs: Arbitrary function keyword arguments.
        :return: Function output.
        """
        return self.func(*args, **kwargs)
    

class AgentStep(BaseModel):
    """
    Representation of agent step.
    """
    task: str # the task to solve in this step
    goal: str # the goal of solving the task
    tool: str | None = None # the name of the tool if a tool needs to be used, else None


class AgentPlan(BaseModel):
    """
    Representation of agent plan.
    """
    steps: List[AgentStep]


class GenerationIntent(str, Enum):
    """
    Representation of generation intent.
    """
    chatting = "chatting"  # request for casual chatting
    planning = "planning"  # request for planning a solution
    solving = "solving"  # request for solving a task
    validating = "validating"  # request for validating the solution of a task


class ChattingMetadata(BaseModel):
    """
    Metadata for chatting intent.
    """
    topics: List[str]  # list of keywords on the requested content topics
    memory: List[str]  # list of textual memories related to the conversation
    knowledge: List[str]  # list of potentially relevant world knowledge entries
    tone: str  # formulation tone such as serious, concise, snarky, funny, etc.


class PlanningMetadata(BaseModel):
    """
    Metadata for planning intent.
    """
    task: str  # overall task to create a plan for
    memory: List[str]  # list of potentially relevant memories
    knowledge: List[str]  # list of potentially relevant world knowledge entries

class SolvingMetadata(BaseModel):
    """
    Metadata for solving intent.
    """
    task: str  # the task to solve
    goal: str  # the goal of solving the task
    tool: str | None = None  # the name of the tool if needed, else None
    tool_input: dict | None = None  # input for the tool if required, else None


class ValidatingMetadata(BaseModel):
    """
    Metadata for validating intent.
    """
    task: str  # the task to validate
    goal: str  # the goal of validation
    tool: str | None = None  # the name of the tool if needed, else None
    tool_input: dict | None = None  # input for the tool if required, else None
    tool_output: dict | None = None  # output from the tool if applicable, else None
    response: str  # response or evaluation of the validation


class GenerationRequest(BaseModel):
    """
    Represents a generation request.
    """
    content: str  # request content
    metadata: dict  # request metadata
    intent: GenerationIntent | None = None  # request intent
    intent_metadata: ChattingMetadata | PlanningMetadata | SolvingMetadata | ValidatingMetadata = None  # metadata on the intent


class GenerationRequest(BaseModel):
    """
    Represents a generation request.
    """
    content: str #request prompt
    metadata: dict  #request metadata
    intent: GenerationIntent | None = None #request intent
    intended_tool: str | None = None #intended tool, only in case of intent being 'using_tool' 


class Agent(object):
    """
    Class, representing an agent.
    """
    def __init__(self,
                 chat_model_instance: ChatModelInstance | RemoteChatModelInstance,
                 tools: List[AgentTool],
                 memory: ChromaKnowledgeBase,
                 world_knowledge: Neo4jKnowledgebase) -> None:
        """
        Initiation method.
        :param chat_model_instance: Chat model instance for handling generative tasks.
        :param tools: List of available tools.
        :param memory: Conversation memory.
        :param world_knowledge: World knowledge memory.
        :param grammar_field: Grammar field in chat method metadata for constrained decoding.
        """
        self.chat_model_instance = chat_model_instance
        self.tools = {tool.name.lower(): tool for tool in tools}
        # TODO: Store previous sessions, feedback and relevant entities in memory.
        self.memory = memory
        # TODO: Enrich world knowledge for faster and offline retrieval
        self.world_knowledge = world_knowledge

        self.system_prompt = "\n".join([
            "You are a helpful AI Agent, assisting the user with his tasks. You work involves planning and executing planned steps to solve tasks."
            "You can use the following tools:"
        ])
        tool_explanations = ["\n".join([
            f"Tool: {tool.name}",
            f"Description: {tool.description}",
            f"Input JSON: {'No input needed.' if tool.input_declaration is None else tool.input_declaration.model_json_schema()}",
            f"Output JSON: {'No output.' if tool.output_declaration is None else tool.output_declaration.model_json_schema()}",
        ]) for tool in tools]
        tool_explanations.append("")
        self.system_prompt += "\n\n".join(tool_explanations)

    def handle_request(self, request: GenerationRequest) -> Any:
        """
        Handles a generation request.
        :param request: Generation Request.
        :return: Generation response.
        """
        if request.intent is None:
            # Replace with more efficient intent prediction algorithm later on
            intention_generation_prompt = "\n".join([
                f"Your task is to choose the intent of the incoming request. Here is the structure of a request:",
                "",
                "```python",
                "class GenerationIntent(str, Enum):",
                "   chatting = \"chatting\" #request for casual chatting",
                "   tutoring = \"tutoring\" #request for tutoring",
                "   planning = \"planning\" #request for planning a solution",
                "   solving = \"solving\" #request for solving a task",
                "   validating = \"validating\" #request for validating the solution of a task",
                "   using_tool = \"using_tool\" #request for using a specific tool"
                "",
                "",
                "class GenerationRequest(BaseModel):",
                    "content: str #request prompt",
                    "metadata: dict  #request metadata",
                    "intent: GenerationIntent | None = None #request intent",
                "```",
                "",
                "Here is the incoming request without intent:",
                json.dumps(request.model_dump_json()),
                "",
                "Choose the correct intent for the given request and respond with the resulting GenerationRequest in JSON format."
            ])
            response, _ = self.chat_model_instance.chat(
                prompt=intention_generation_prompt,
                chat_parameters={"grammar": convert_pydantic_model_to_grammar(GenerationRequest)}
            )
            request.intent = json.loads(response)["intent"]
        if request.intent == GenerationIntent.chatting.value():
            response = self.chat_model_instance.chat(
                prompt=request.content,
                chat_parameters=request.metadata.get("chat_parameters")
            )
            return response[0]
        elif request.intent == GenerationIntent.planning.value():
            return self.plan(task=request.content)
        elif request.intent == GenerationIntent.solving.value():
            pass
        elif request.intent == GenerationIntent.validating.value():
            pass
        elif request.intent == GenerationIntent.using_tool.value():
            if request.intended_tool.lower() in self.tools:
                tool = self.tools[request.intended_tool.lower()]
                tool_input = {}
                if tool.input_declaration is not None:
                    tool_input_prompt = "\n".join([
                        f"Your task is to generate the input data for the tool '{request.intended_tool}'.",
                        "Here is the schema for the tool input:",
                        json.dump(tool.input_declaration.model_json_schema(), indent=4, ensure_ascii=False),
                        "",
                        "Here is the user request:",
                        json.dump(request.model_dump_json(), indent=4, ensure_ascii=False),
                        "",
                        "Generate the tool input from the user request in JSON format. Respond with the generated JSON."
                    ])
                    tool_input = json.loads(self.chat_model_instance.chat(
                        prompt=tool_input_prompt,
                        chat_parameters={"grammar": convert_pydantic_model_to_grammar(tool.input_declaration)}
                    )[0])
                return self.use_tool(tool_name=request.intended_tool.lower(), tool_input=tool_input)

    def use_tool(self, tool_name: str, tool_input: dict | BaseModel) -> Any:
        """
        Utilizes tool to fetch result.
        :param tool_name: Tool name.
        :param tool_input: Tool input.
        :return: Tool result.
        """
        if tool_name in self.tools:
            if isinstance(tool_input, dict):
                return self.tools[tool_name](**tool_input)
            else:
                return self.tools[tool_name](**tool_input.model_dump_json())
        
    def plan(self, task: str) -> AgentPlan:
        """
        Creates a plan, consisting of agent steps.
        :param task: Task description.
        :result: Plan.
        """
        self.chat_model_instance.history = [{
            "role": "system", 
            "content": self.system_prompt,
            "metadata": {"initiated": dt.now()}
        }]

        prompt = "\n".join([
            f"Please create an AgentPlan in JSON format to handle given task which consist of solution steps. Such a solution step is called AgentStep.",
            f"Here is the structure of an AgentPlan:",
            "```python",
            "class AgentStep:",
            "   task: str #Description of the task in this step",
            "   use_tool: bool #Whether to use a tool in this step",
            "   tool: str #Name of the tool, if a tool is to be used in this step",
            "",
            "class AgentPlan:",
            "steps: List[AgentStep] #The solution steps that build up the plan",
            "return_tool_output: bool #Whether the final result is a tool output",
            "```",
            "",
            "TASK:",
            f"{task}",
            "",
            "Return only the JSON representation for an AgentPlan object."
        ])
        plan_params, _ = self.chat_model_instance.chat(
            prompt=prompt,
            chat_parameters={"grammar": convert_pydantic_model_to_grammar(AgentPlan)}
        ) 
        result_params = json.loads(plan_params)
        # TODO: Remove / adjust after test run
        if "AgentPlan" in result_params:
            result_params = result_params["AgentPlan"]
        return AgentPlan(**result_params)
    
    def act(self, step: AgentStep, step_index: int | str) -> Tuple[bool, Any]:
        """
        Handles an agent step.
        :param step: Step to handle.
        :param step_index: Step index.
        :return: True and result if successful else False and reason as string.
        """
        if step.use_tool:
            tool = self.tools.get(step.tool.lower())
            if tool is None:
                return False, f"The tool '{step.tool}' is not available."
            else:
                tool_input = {}
                if tool.input_declaration is not None:
                    response, _ = self.chat_model_instance.chat(
                        prompt="\n".join([
                            f"Your goal is to solve task {step_index}: {step.task}"
                            f"Please respond with the input parameters for the tool '{step.tool} as JSON structure. Follow the tool's Input JSON structure."
                        ]),
                        chat_parameters={"grammar": convert_pydantic_model_to_grammar(tool.input_declaration)}
                    ) 
                    try:
                        tool_input = json.loads(response)
                    except json.JSONDecodeError:
                        return False, "Tool input is no valid JSON structure."
                return True, self.use_tool(tool_name=tool.name.lower(), tool_input=tool_input)
        else:
            return True, self.chat_model_instance.chat(
                prompt=f"Your goal is to solve task {step_index}: {step.task}. Please respond with the solution."
            )[0]
    
    def run(self, task: str, max_step_retries: int = 3) -> Any:
        """
        Runs agent cycle.
        :param task: User task.
        :param max_step_retries: Maximum step retries.
        :return: Task result.
        """
        plan = self.plan(task=task)
        for step_index, step in enumerate(plan.steps):
            retries = 0
            success, result_or_reason = self.act(step=step, step_index=step_index+1)
            while not success and retries < max_step_retries:
                new_step, _ = self.chat_model_instance.chat(
                    prompt=f"Task {step_index+1}: {step.task} failed: {result_or_reason}.\nRespond with an additional AgentStep JSON object for solving the issue.",
                    chat_parameters={"grammar": convert_pydantic_model_to_grammar(AgentStep)}
                ) 
                try:
                    new_step = json.loads(new_step)
                except json.JSONDecodeError:
                    new_step, _ = self.chat_model_instance.chat(
                        prompt=f"The AgentStep object from the previous response is not a valid JSON structure. Please respond with a valid structure.",
                        chat_parameters={"grammar": convert_pydantic_model_to_grammar(AgentStep)}
                    ) 
                success, result_or_reason = self.act(step=AgentStep(**new_step), step_index=f"{step_index+1}.{retries+1}")
                max_step_retries += 1
            if not success:
                return None
        if success:
            return result_or_reason
        # TODO: Save session, result, potentially feedback in memory


# TODO: Implement hybrid assistant/agent with intention prediction (chat - based on multiple templates, plan, act, observe, tool use)