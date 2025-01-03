
# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import copy
from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict
from typing import List, Tuple, Any, Callable, Optional, Union, Dict, Generator
from datetime import datetime as dt
from src.utility.string_utility import SENTENCE_CHUNK_STOPS
import requests
import json

"""
Abstractions
"""
class LanguageModelConfig(BaseModel):
    """
    Language model config.
    """
    model_config: ConfigDict = ConfigDict(protected_namespaces=())

    backend: str
    model_path: str | None = None
    model_file: str | None = None
    model_parameters: dict | None = None
    tokenizer_path: str | None = None
    tokenizer_parameters: dict | None = None
    embeddings_path: str | None = None
    embeddings_parameters: dict | None = None
    config_path: str | None = None
    config_parameters: dict | None = None
    encoding_parameters: dict | None = None
    embedding_parameters: dict | None = None
    generating_parameters: dict | None = None
    decoding_parameters: dict | None = None


class LanguageModelInstance(ABC):
    """
    Abstract language model class.
    """
    supported_backends: List[str] = ["llama-cpp"]
    default_models: Dict[str, List[str]] = {
        "llama-cpp": ["bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"]
    }

    @classmethod
    def from_configuration(cls, config: LanguageModelConfig) -> Any:
        """
        Returns a language model instance from configuration.
        :param config: Language model configuration.
        :return: Language model instance.
        """
        if config.backend not in cls.supported_backends:
            raise ValueError(f"Backend '{config.backend}' is not in supported backends: {cls.supported_backends}")
        if config.model_path is None and config.backend in cls.default_models and cls.default_models[config.backend]:
            config.model_path = cls.default_models[config.backend][0]
        if config.backend == "llama-cpp":
            return LlamaCPPModelInstance(**config.model_dump())
        
    @classmethod
    def from_dict(cls, config: dict) -> Any:
        """
        Returns a language model instance from dictionary.
        :param config: Language model dict.
        :return: Language model instance.
        """
        if config["backend"] not in cls.supported_backends:
            raise ValueError(f"Backend '{config['backend']}' is not in supported backends: {cls.supported_backends}")
        if config["model_path"] is None and config["backend"] in cls.default_models and cls.default_models[config["backend"]]:
            config["model_path"] = cls.default_models[config["backend"]][0]
        if config["backend"] == "llama-cpp":
            return LlamaCPPModelInstance(**config)

    """
    Generation methods
    """
    @abstractmethod
    def tokenize(self,
                input: str,
                encoding_parameters: dict | None = None
                ) -> List[float]:
        """
        Method for embedding an input.
        :param input: Input to embed.
        :param encoding_parameters: Kwargs for encoding as dictionary.
            Defaults to None.
        """
        pass

    @abstractmethod
    def embed(self,
              input: str,
              encoding_parameters: dict | None = None,
              embedding_parameters: dict | None = None,
              ) -> List[float]:
        """
        Method for embedding an input.
        :param input: Input to embed.
        :param encoding_parameters: Kwargs for encoding as dictionary.
            Defaults to None.
        :param embedding_parameters: Kwargs for embedding as dictionary.
            Defaults to None.
        """
        pass

    @abstractmethod
    def generate(self,
                 prompt: str,
                 encoding_parameters: dict | None = None,
                 generating_parameters: dict | None = None,
                 decoding_parameters: dict | None = None) -> Tuple[str, Optional[dict]]:
        """
        Method for generating a response to a given prompt and conversation history.
        :param prompt: Prompt.
        :param encoding_parameters: Kwargs for encoding as dictionary.
            Defaults to None.
        :param generating_parameters: Kwargs for generating as dictionary.
            Defaults to None.
        :param decoding_parameters: Kwargs for decoding as dictionary.
            Defaults to None.
        :return: Tuple of textual answer and metadata.
        """
        pass


class LlamaCPPModelInstance(LanguageModelInstance):
    """
    Llama CPP based model instance.
    """
    def __init__(self,
                 model_path: str,
                 model_file: str | None = None,
                 model_parameters: dict | None = None,
                 encoding_parameters: dict | None = None,
                 embedding_parameters: dict | None = None,
                 generating_parameters: dict | None = None,
                 decoding_parameters: dict | None = None,
                 *args: Any | None,
                 **kwargs: Any | None
                 ) -> None:
        """
        Initiation method.
        :param model_path: Path to model files.
        :param model_file: Model file to load.
            Defaults to None.
        :param model_parameters: Model loading kwargs as dictionary.
            Defaults to None.
        :param encoding_parameters: Kwargs for encoding in the generation process as dictionary.
            Defaults to None in which case an empty dictionary is created and can be filled depending on the backend in the 
            different initiation methods.
        :param embedding_parameters: Kwargs for embedding as dictionary.
            Defaults to None in which case an empty dictionary is created and can be filled depending on the backend in the 
            different initiation methods.
        :param generating_parameters: Kwargs for generating in the generation process as dictionary.
            Defaults to None in which case an empty dictionary is created and can be filled depending on the backend in the 
            different initiation methods.
        :param decoding_parameters: Kwargs for decoding in the generation process as dictionary.
            Defaults to None in which case an empty dictionary is created and can be filled depending on the backend in the 
            different initiation methods.
        :param args: Arbitrary arguments.
        :param kwargs: Arbitrary keyword arguments.
        """
        from src.utility.llama_cpp_python_utility import load_llamacpp_model
        self.model = load_llamacpp_model(
            model_path=model_path,
            model_file=model_file,
            model_parameters=model_parameters)
        self.encoding_parameters = encoding_parameters or {}
        self.embedding_parameters = embedding_parameters or {}
        self.generating_parameters = generating_parameters or {}
        self.decoding_parameters = decoding_parameters or {}

    """
    Generation methods
    """
    def tokenize(self,
                input: str,
                encoding_parameters: dict | None = None
                ) -> List[float]:
        """
        Method for embedding an input.
        :param input: Input to embed.
        :param encoding_parameters: Kwargs for encoding as dictionary.
            Defaults to None.
        """
        parameters = copy.deepcopy(self.encoding_parameters)
        if encoding_parameters:
            parameters.update(encoding_parameters)
        return self.model.tokenize(input, **parameters)


    def embed(self,
              input: str,
              encoding_parameters: dict | None = None,
              embedding_parameters: dict | None = None,
              ) -> List[float]:
        """
        Method for embedding an input.
        :param input: Input to embed.
        :param encoding_parameters: Kwargs for encoding as dictionary.
            Defaults to None.
        :param embedding_parameters: Kwargs for embedding as dictionary.
            Defaults to None.
        """
        parameters = copy.deepcopy(self.embedding_parameters)
        parameters.update(self.encoding_parameters)
        for additional_parameters in [encoding_parameters, embedding_parameters]:
            if additional_parameters:
                parameters.update(additional_parameters)
        return self.model.embed(input, **parameters)

    def generate(self,
                 prompt: str,
                 encoding_parameters: dict | None = None,
                 generating_parameters: dict | None = None,
                 decoding_parameters: dict | None = None) -> Tuple[str, Optional[dict]]:
        """
        Method for generating a response to a given prompt and conversation history.
        :param prompt: Prompt.
        :param encoding_parameters: Kwargs for encoding as dictionary.
            Defaults to None.
        :param generating_parameters: Kwargs for generating as dictionary.
            Defaults to None.
        :param decoding_parameters: Kwargs for decoding as dictionary.
            Defaults to None.
        :return: Tuple of textual answer and metadata.
        """
        parameters = copy.deepcopy(self.encoding_parameters)
        parameters.update(self.generating_parameters)
        parameters.update(self.decoding_parameters)
        for additional_parameters in [encoding_parameters, generating_parameters, decoding_parameters]:
            if additional_parameters:
                parameters.update(additional_parameters)
        metadata = self.model(prompt, **generating_parameters)
        answer = metadata["choices"][0]["text"]
        return answer, metadata
    

class ChatModelConfig(BaseModel):
    """
    Chat model configuration class.
    """
    language_model_config: LanguageModelConfig
    chat_parameters: dict | None = None
    system_prompt: str | None = None
    prompt_maker: Callable | None = None
    use_history: bool = True
    history: List[Dict[str, Union[str, dict]]] | None = None


class ChatModelInstance(object):
    """
    Chat model class.
    """

    def __init__(self,
                 language_model: dict | LanguageModelConfig | LanguageModelInstance ,
                 chat_parameters: dict | None = None,
                 system_prompt: str | None = None,
                 prompt_maker: Callable | None = None,
                 use_history: bool = True,
                 history: List[Dict[str, Union[str, dict]]] | None = None) -> None:
        """
        Initiation method.
        :param language_model: Language model config (as class or dict) or instance.
        :param chat_parameters: Kwargs for chatting in the chatting process as dictionary.
            Defaults to None in which case an empty dictionary is created and can be filled depending on the language instance's
            model backend.
        :param system_prompt: Default system prompt.
            Defaults to a None in which case no system prompt is used.
        :param prompt_maker: Function which takes the prompt history as a list of {"role": <role>, "content": <message>, "metadata": <metadata>}-dictionaries 
            (already including the new user prompt) as argument and calculates the final prompt. 
            Only necessary, if the backend does not support chat interaction via message list.
        :param use_history: Flag, declaring whether to use the history.
            Defaults to True.
        :param history: Interaction history as list of {"role": <role>, "content": <message>, "metadata": <metadata>}-dictionaries.
            Defaults to None.
        """
        if isinstance(language_model, LanguageModelInstance):
            self.language_model_instance = language_model
        elif isinstance(language_model, LanguageModelConfig):
            self.language_model_instance = LanguageModelInstance.from_configuration(language_model)
        elif isinstance(language_model, dict):
            self.language_model_instance = LanguageModelInstance.from_dict(language_model)
            
        self.chat_parameters = self.language_model_instance.generating_parameters if chat_parameters is None else chat_parameters
        self.system_prompt = system_prompt
        if prompt_maker is None:
            def prompt_maker(history: List[Dict[str, Union[str, dict]]]) -> str:
                """
                Default Prompt maker function.
                :param history: History.
                """
                return "\n".join(f"<s>{entry['role']}:\n{entry['content']}</s>" for entry in history) + "\n"
        self.prompt_maker = prompt_maker

        self.use_history = use_history

        if history is None:
            self.history = [{
                "role": "system", 
                "content": "You are a helpful AI assistant. Please help users with their tasks." if system_prompt is None else system_prompt, 
                "metadata": {"initiated": dt.now()}
            }]
        else:
            self.history = history

    @classmethod
    def from_configuration(cls, config: ChatModelConfig) -> Any:
        """
        Returns a chat model instance from configuration.
        :param config: Chat model configuration.
        :return: Chat model instance.
        """
        return cls(**config.model_dump())

    @classmethod
    def from_dict(cls, config: dict) -> Any:
        """
        Returns a chat model instance from dictionary.
        :param config: Chat model dictionary.
        :return: Chat model instance.
        """
        return cls(**config)

    """
    Generation methods
    """    
    def chat(self, prompt: str, chat_parameters: dict | None = None) -> Tuple[str, dict]:
        """
        Method for chatting with language model instance.
        :param prompt: User input.
        :param chat_parameters: Kwargs for chatting in the chatting process as dictionary.
            Defaults to None in which case an empty dictionary is created and can be filled depending on the language instance's
            model backend.
        :return: Response and metadata.
        """
        acc_parameters = copy.deepcopy(self.chat_parameters)
        if chat_parameters:
            acc_parameters.update(chat_parameters)
        if not self.use_history:
            self.history = [self.history[0]]
        self.history.append({"role": "user", "content": prompt})
        full_prompt = self.prompt_maker(self.history)

        metadata = {}
        answer = ""
        
        if isinstance(self.language_model_instance, LlamaCPPModelInstance):
            metadata = self.language_model_instance.model.create_chat_completion(
                messages=self.history,
                **acc_parameters
            )
            answer = metadata["choices"][0]["message"].get("content", "")
        
        if self.use_history:
            self.history.append({
                "role": "assistant",
                "content": answer,
                "metadata": metadata
            })
        return answer, metadata
    
    def chat_stream(self, prompt: str, chat_parameters: dict | None = None, minium_yielded_characters: int = 10) -> Generator[Tuple[str, dict], None, None]:
        """
        Method for chatting with language model instance via stream.
        :param prompt: User input.
        :param chat_parameters: Kwargs for chatting in the chatting process as dictionary.
            Defaults to None in which case an empty dictionary is created and can be filled depending on the language instance's
            model backend.
        :param minium_yielded_characters: Minimum yielded alphabetic characters, defaults to 10.
        :return: Response and metadata stream.
        """
        acc_parameters = copy.deepcopy(self.chat_parameters)
        if chat_parameters:
            acc_parameters.update(chat_parameters)
        acc_parameters["stream"] = True
        if not self.use_history:
            self.history = [self.history[0]]
        self.history.append({"role": "user", "content": prompt})

        #full prompt for the usage of raw generation instead of chat completion
        full_prompt = self.prompt_maker(self.history)

        metadata = {}
        answer = ""

        if isinstance(self.language_model_instance, LlamaCPPModelInstance):
            stream = self.language_model_instance.model.create_chat_completion(
                messages=self.history,
                **acc_parameters
            )
            chunks = []
            sentence = ""
            for chunk in stream:
                chunks.append(chunk)
                delta = chunk["choices"][0]["delta"]
                if "content" in delta:
                    sentence += delta["content"]
                    if delta["content"][-1] in SENTENCE_CHUNK_STOPS:
                        answer += sentence
                        if len([elem for elem in sentence if elem.isalpha()]) >= minium_yielded_characters:
                            yield sentence, chunk
                            sentence = ""
            answer += sentence
            metadata = {"chunks": chunks}
            yield sentence, chunk
            
        if self.use_history:
            self.history.append({
                "role": "assistant",
                "content": answer,
                "metadata": metadata
            })
        return answer, metadata

    def legacy_chat(self, prompt: str, chat_parameters: dict | None = None) -> Tuple[str, dict]:
        """
        Method for chatting with language model instance.
        :param prompt: User input.
        :param chat_parameters: Kwargs for chatting in the chatting process as dictionary.
            Defaults to None in which case an empty dictionary is created and can be filled depending on the language instance's
            model backend.
        :return: Response and metadata.
        """
        acc_parameters = copy.deepcopy(self.chat_parameters)
        if chat_parameters:
            acc_parameters.update(chat_parameters)
        if not self.use_history:
            self.history = [self.history[0]]
        self.history.append({"role": "user", "content": prompt})
        full_prompt = self.prompt_maker(self.history)

        metadata = {}
        answer = ""

        if self.language_model_instance.backend == "ctransformers":
            answer, metadata = self.language_model_instance.generate(full_prompt)
        elif self.language_model_instance.backend == "langchain_llamacpp":
            answer, metadata = self.language_model_instance.generate(full_prompt)
        elif self.language_model_instance.backend == "transformers":
            chat_encoding_kwargs = copy.deepcopy(self.language_model_instance.encoding_parameters)
            chat_encoding_kwargs["add_generation_prompt"] = True
            input_tokens = self.language_model_instance.tokenizer.apply_chat_template(
                    self.history, 
                    **chat_encoding_kwargs).to(self.language_model_instance.model.device)
            output_tokens = self.language_model_instance.model.generate(
                **input_tokens, **acc_parameters)[0]
            metadata = self.language_model_instance.tokenizer.decode(
                output_tokens, 
                **self.language_model_instance.decoding_parameters)
            answer, metadata = answer, metadata
        elif self.language_model_instance.backend == "autogptq":
            answer, metadata = self.language_model_instance.generate(full_prompt)
        elif self.language_model_instance.backend == "llamacpp":
            metadata = self.language_model_instance.model.create_chat_completion(
                messages=self.history,
                **acc_parameters
            )
            answer = metadata["choices"][0]["message"].get("content", "")
        elif self.language_model_instance.backend == "exllamav2":
            answer, metadata = self.language_model_instance.generate(full_prompt)
        
        if self.use_history:
            self.history.append({
                "role": "assistant",
                "content": answer,
                "metadata": metadata
            })
        return answer, metadata
    
    def legacy_chat_stream(self, prompt: str, chat_parameters: dict | None = None, minium_yielded_characters: int = 10) -> Generator[Tuple[str, dict], None, None]:
        """
        Method for chatting with language model instance via stream.
        :param prompt: User input.
        :param chat_parameters: Kwargs for chatting in the chatting process as dictionary.
            Defaults to None in which case an empty dictionary is created and can be filled depending on the language instance's
            model backend.
        :param minium_yielded_characters: Minimum yielded alphabetic characters, defaults to 10.
        :return: Response and metadata stream.
        """
        acc_parameters = copy.deepcopy(self.chat_parameters)
        if chat_parameters:
            acc_parameters.update(chat_parameters)
        acc_parameters["stream"] = True
        if not self.use_history:
            self.history = [self.history[0]]
        self.history.append({"role": "user", "content": prompt})
        full_prompt = self.prompt_maker(self.history)

        metadata = {}
        answer = ""

        if self.language_model_instance.backend == "ctransformers":
            answer, metadata = self.language_model_instance.generate(full_prompt)
        elif self.language_model_instance.backend == "langchain_llamacpp":
            answer, metadata = self.language_model_instance.generate(full_prompt)
        elif self.language_model_instance.backend == "transformers":
            chat_encoding_kwargs = copy.deepcopy(self.language_model_instance.encoding_parameters)
            chat_encoding_kwargs["add_generation_prompt"] = True
            input_tokens = self.language_model_instance.tokenizer.apply_chat_template(
                    self.history, 
                    **chat_encoding_kwargs).to(self.language_model_instance.model.device)
            output_tokens = self.language_model_instance.model.generate(
                **input_tokens, **acc_parameters)[0]
            metadata = self.language_model_instance.tokenizer.decode(
                output_tokens, 
                **self.language_model_instance.decoding_parameters)
            answer, metadata = answer, metadata
        elif self.language_model_instance.backend == "autogptq":
            answer, metadata = self.language_model_instance.generate(full_prompt)
        elif self.language_model_instance.backend == "llamacpp":
            stream = self.language_model_instance.model.create_chat_completion(
                messages=self.history,
                **acc_parameters
            )
            chunks = []
            sentence = ""
            for chunk in stream:
                chunks.append(chunk)
                delta = chunk["choices"][0]["delta"]
                if "content" in delta:
                    sentence += delta["content"]
                    if delta["content"][-1] in SENTENCE_CHUNK_STOPS:
                        answer += sentence
                        if len([elem for elem in sentence if elem.isalpha()]) >= minium_yielded_characters:
                            yield sentence, chunk
                            sentence = ""
            answer += sentence
            metadata = {"chunks": chunks}
            yield sentence, chunk
        elif self.language_model_instance.backend == "exllamav2":
            answer, metadata = self.language_model_instance.generate(full_prompt)
        
        if self.use_history:
            self.history.append({
                "role": "assistant",
                "content": answer,
                "metadata": metadata
            })
        return answer, metadata
    
    def __del__(self) -> None:
        """
        Deconstructs instance.
        """
        if hasattr(self, "language_model_instance"):
            for attr in ["model", "tokenizer"]:
                if hasattr(self.language_model_instance, attr):
                    artifact = getattr(self.language_model_instance, attr)
                    del artifact


class RemoteChatModelConfig(BaseModel):
    """
    Remote chat model configuration class.
    """
    api_base: str
    api_token: str | None = None
    chat_parameters: dict | None = None
    system_prompt: str | None = None
    prompt_maker: Callable | None = None
    use_history: bool = True
    history: List[Dict[str, Union[str, dict]]] | None = None


class RemoteChatModelInstance(ChatModelInstance):
    """
    Remote chat model class.
    """

    def __init__(self,
                 api_base: str,
                 api_token: str | None = None,
                 chat_parameters: dict | None = None,
                 system_prompt: str | None = None,
                 prompt_maker: Callable | None = None,
                 use_history: bool = True,
                 history: List[Dict[str, Union[str, dict]]] | None = None) -> None:
        """
        Initiation method.
        :param api_base: API base URL in the format http://<host>:<port>/v1.
        :param api_token: API token, if necessary.
        :param chat_parameters: Kwargs for chatting in the chatting process as dictionary.
            Defaults to None in which case an empty dictionary is created and can be filled depending on the language instance's
            model backend.
        :param system_prompt: Default system prompt.
            Defaults to a None in which case no system prompt is used.
        :param prompt_maker: Function which takes the prompt history as a list of {"role": <role>, "content": <message>, "metadata": <metadata>}-dictionaries 
            (already including the new user prompt) as argument and calculates the final prompt. 
            Only necessary, if the API does not support chat interaction.
        :param use_history: Flag, declaring whether to use the history.
            Defaults to True.
        :param history: Interaction history as list of {"role": <role>, "content": <message>, "metadata": <metadata>}-dictionaries.
            Defaults to None.
        """
        self.api_base = api_base
        self.api_token = api_token
        self.request_headers = {} if self.api_token is None else {
            "Authorization": f"Bearer {self.api_token}",
        }

        self.chat_parameters = chat_parameters or {}
        self.system_prompt = system_prompt
        if prompt_maker is None:
            def prompt_maker(history: List[Dict[str, Union[str, dict]]]) -> str:
                """
                Default Prompt maker function.
                :param history: History.
                """
                return "\n".join(f"<s>{entry['role']}:\n{entry['content']}</s>" for entry in history) + "\n"
        self.prompt_maker = prompt_maker

        self.use_history = use_history

        self.completions_endpoint = f"{api_base}/completions"
        self.embeddings_endpoint = f"{api_base}/embeddings"
        self.chat_completions_endpoint = f"{api_base}/chat/completions"
        self.models_endpoint = f"{api_base}/models"

        self.tokenize_endpoint = f"{api_base}/extras/tokenize"
        self.tokenize_count_endpoint = f"{api_base}/extras/tokenize/count"
        self.detokenize_endpoint = f"{api_base}/extras/detokenize"

        if history is None:
            self.history = [{
                "role": "system", 
                "content": "You are a helpful AI assistant. Please help users with their tasks." if system_prompt is None else system_prompt, 
                "metadata": {"initiated": dt.ctime(dt.now())}
            }]
        else:
            self.history = history

    @classmethod
    def from_configuration(cls, config: RemoteChatModelConfig) -> Any:
        """
        Returns a remote chat model instance from configuration.
        :param config: Remote chat configuration.
        :return: Remote chat instance.
        """
        return cls(**config.model_dump())

    @classmethod
    def from_dict(cls, config: dict) -> Any:
        """
        Returns a chat model instance from dictionary.
        :param config: Chat model dictionary.
        :return: Chat model instance.
        """
        return cls(**config)

    """
    Generation methods
    """

    def chat(self, prompt: str, chat_parameters: dict | None = None) -> Tuple[str, dict]:
        """
        Method for chatting with the remote language model instance.
        :param prompt: User input.
        :param chat_parameters: Kwargs for chatting in the chatting process as dictionary.
            Defaults to None in which case an empty dictionary is created and can be filled depending on the language instance's
            model backend.
        :return: Response and metadata.
        """
        chat_parameters = chat_parameters or {}
        if not self.use_history:
            self.history = [self.history[0]]
        self.history.append({"role": "user", "content": prompt})
        full_prompt = self.prompt_maker(self.history)

        metadata = {}
        answer = ""

        json_payload = copy.deepcopy(self.chat_parameters)
        json_payload.update(chat_parameters)
        json_payload["messages"] = self.history

        response = requests.post(self.chat_completions_endpoint, headers=self.request_headers, json=json_payload)
        if response.status_code == 200:
            metadata = response.json()
            answer = metadata["choices"][0]["message"]["content"]
        else:
            json_payload.pop("messages")
            json_payload["prompt"] = full_prompt
            response = requests.post(self.completions_endpoint, headers=self.request_headers, json=json_payload)
            if response.status_code == 200:
                metadata = response.json()
                answer = metadata["choices"][0]["text"]
        if self.use_history:
            self.history.append({"role": "assistant", "content": answer, "metadata": metadata})
        return answer, metadata
    
    def chat_stream(self, prompt: str, chat_parameters: dict | None = None, minium_yielded_characters: int = 10) -> Generator[Tuple[str, dict], None, None]:
        """
        Method for chatting with the remote language model instance via stream.
        :param prompt: User input.
        :param chat_parameters: Kwargs for chatting in the chatting process as dictionary.
            Defaults to None in which case an empty dictionary is created.
        :param minium_yielded_characters: Minimum yielded alphabetic characters, defaults to 10.
        :return: Response and metadata stream.
        """
        chat_parameters = chat_parameters or {}
        chat_parameters["stream"] = True
        if not self.use_history:
            self.history = [self.history[0]]
        self.history.append({"role": "user", "content": prompt})
        full_prompt = self.prompt_maker(self.history)

        metadata = {}
        answer = ""

        json_payload = copy.deepcopy(self.chat_parameters)
        json_payload.update(chat_parameters)
        json_payload["messages"] = self.history

        response = requests.post(self.chat_completions_endpoint, headers=self.request_headers, json=json_payload, stream=True)
        if response.status_code != 200:
            json_payload.pop("messages")
            json_payload["prompt"] = full_prompt
            response = requests.post(self.completions_endpoint, headers=self.request_headers, json=json_payload, stream=True)
        if response.status_code != 200:
            return answer, metadata
        
        chunks = []
        sentence = ""

        for encoded_chunk in response.iter_lines():
            if encoded_chunk:
                decoded_chunk = encoded_chunk.decode("utf-8")
                if decoded_chunk.startswith("data: "):
                    decoded_chunk = decoded_chunk[6:]
                if not (decoded_chunk.startswith(" ping -") or decoded_chunk.startswith(": ping -")):
                    if decoded_chunk != "[DONE]":
                        try:
                            chunk = json.loads(decoded_chunk)
                            chunks.append(chunk)
                            delta = chunk["choices"][0]["delta"]
                        except json.decoder.JSONDecodeError:
                            delta = {}
                    else:
                        delta = {}
                    if "content" in delta:
                        sentence += delta["content"]
                        if delta["content"] and delta["content"][-1] in SENTENCE_CHUNK_STOPS:
                            if len([elem for elem in sentence if elem.isalpha()]) >= minium_yielded_characters:
                                yield sentence, chunk
                                answer += sentence
                                sentence = ""
        answer += sentence
        metadata = {"chunks": chunks}
        yield sentence, chunk
        if self.use_history:
            self.history.append({"role": "assistant", "content": answer, "metadata": metadata})

    """
    Additional endpoint wrappers
    """
    def get_models(self) -> List[dict] | None:
        """
        Method for retrieving a list of available models.
        :return: List of model dictionaries, if fetching was successful.
        """        
        response = requests.get(self.models_endpoint, headers=self.request_headers)
        if response.status_code == 200:
            metadata = response.json()
            return metadata["data"]
    
    def embed(self, input: str, embeddings_parameters: dict | None = None) -> List[List[float]] | List[float] | None:
        """
        Method for retrieving embeddings.
        :param input: Input to generate embeddings from.
        :param embeddings_parameters: Embeddings parameters. 
            Defaults to None in which case an empty dictionary is created.
        :return: Embeddings.
        """        
        if embeddings_parameters is None:
            embeddings_parameters = {"input": input}
        else:
            embeddings_parameters["input"] = input
        response = requests.post(self.embeddings_endpoint, headers=self.request_headers, json=embeddings_parameters)
        if response.status_code == 200:
            metadata = response.json()
            return metadata["data"][0]["embedding"]
        
    def tokenize(self, input: str, tokenization_parameters: dict | None = None) -> List[List[float]] | None:
        """
        Method for tokenizing an input.
        :param input: Input to tokenize.
        :param tokenization_parameters: Tokenization parameters. 
            Defaults to None in which case an empty dictionary is created.
        :return: Input tokens.
        """        
        if tokenization_parameters is None:
            tokenization_parameters = {"input": input}
        else:
            tokenization_parameters["input"] = input
        response = requests.post(self.tokenize_endpoint, headers=self.request_headers, json=tokenization_parameters)
        if response.status_code == 200:
            metadata = response.json()
            return metadata["tokens"]
    
    def detokenize(self, tokens: List[int], detokenization_parameters: dict | None = None) -> List[List[float]] | None:
        """
        Method for detokenizing an input.
        :param tokens: Tokens to detokenize.
        :param detokenization_parameters: Detokenization parameters. 
            Defaults to None in which case an empty dictionary is created.
        :return: Input tokens.
        """   
        raise NotImplementedError("Currently not working with Llama CPP Server.")     
        if detokenization_parameters is None:
            detokenization_parameters = {"tokens": input}
        else:
            detokenization_parameters["input"] = input
        response = requests.post(self.detokenize_endpoint, headers=self.request_headers, json=detokenization_parameters)
        if response.status_code == 200:
            metadata = response.json()
            return metadata["data"]

