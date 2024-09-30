
# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import traceback
import copy
from pydantic import BaseModel, ConfigDict
from typing import List, Tuple, Any, Callable, Optional, Union, Dict, Generator
from datetime import datetime as dt
from ...bronze.string_utility import SENTENCE_CHUNK_STOPS
from .language_model_instantiation import load_ctransformers_model, load_transformers_model, load_llamacpp_model, load_autogptq_model, load_exllamav2_model, load_langchain_llamacpp_model
import requests
import json

"""
Abstractions
"""


class LanguageModelConfig(BaseModel):
    """
    Language model config class.
    """
    model_config: ConfigDict = ConfigDict(protected_namespaces=())

    backend: str
    model_path: str
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


class LanguageModelInstance(object):
    """
    Language model class.
    """
    supported_backends: List[str] = ["ctransformers", "transformers",
                                     "llamacpp", "autogptq", "exllamav2", "langchain_llamacpp"]

    def __init__(self,
                 backend: str,
                 model_path: str,
                 model_file: str | None = None,
                 model_parameters: dict | None = None,
                 tokenizer_path: str | None = None,
                 tokenizer_parameters: dict | None = None,
                 embeddings_path: str | None = None,
                 embeddings_parameters: dict | None = None,
                 config_path: str | None = None,
                 config_parameters: dict | None = None,
                 encoding_parameters: dict | None = None,
                 embedding_parameters: dict | None = None,
                 generating_parameters: dict | None = None,
                 decoding_parameters: dict | None = None
                 ) -> None:
        """
        Initiation method.
        :param backend: Backend for model loading.
            Check LanguageModelInstance.supported_backends for supported backends.
        :param model_path: Path to model files.
        :param model_file: Model file to load.
            Defaults to None.
        :param model_parameters: Model loading kwargs as dictionary.
            Defaults to None.
        :param tokenizer_path: Tokenizer path.
            Defaults to None.
        :param tokenizer_parameters: Tokenizer loading kwargs as dictionary.
            Defaults to None.
        :param embeddings_path: Embeddings path.
            Defaults to None.
        :param embeddings_parameters: Embeddings loading kwargs as dictionary.
            Defaults to None.
        :param config_path: Config path.
            Defaults to None.
        :param config_parameters: Config loading kwargs as dictionary.
            Defaults to None.
        :param encoding_parameters: Kwargs for encoding in the generation process as dictionary.
            Defaults to None in which case an empty dictionary is created and can be filled depending on the backend in the 
            different initation methods.
        :param embedding_parameters: Kwargs for embedding as dictionary.
            Defaults to None in which case an empty dictionary is created and can be filled depending on the backend in the 
            different initation methods.
        :param generating_parameters: Kwargs for generating in the generation process as dictionary.
            Defaults to None in which case an empty dictionary is created and can be filled depending on the backend in the 
            different initation methods.
        :param decoding_parameters: Kwargs for decoding in the generation process as dictionary.
            Defaults to None in which case an empty dictionary is created and can be filled depending on the backend in the 
            different initation methods.
        """
        self.backend = backend

        self.encoding_parameters = {} if encoding_parameters is None else encoding_parameters
        self.embedding_parameters = {} if embedding_parameters is None else embedding_parameters
        self.generating_parameters = {} if generating_parameters is None else generating_parameters
        self.decoding_parameters = {} if decoding_parameters is None else decoding_parameters

        self.config, self.tokenizer, self.embeddings, self.model, self.generator = {
            "ctransformers": load_ctransformers_model,
            "transformers": load_transformers_model,
            "llamacpp": load_llamacpp_model,
            "autogptq": load_autogptq_model,
            "exllamav2": load_exllamav2_model,
            "langchain_llamacpp": load_langchain_llamacpp_model
        }[backend](
            model_path=model_path,
            model_file=model_file,
            model_parameters=model_parameters,
            tokenizer_path=tokenizer_path,
            tokenizer_parameters=tokenizer_parameters,
            embeddings_path=embeddings_path,
            embeddings_parameters=embeddings_parameters,
            config_path=config_path,
            config_parameters=config_parameters
        )

    @classmethod
    def from_configuration(cls, config: LanguageModelConfig) -> Any:
        """
        Returns a language model instance from configuration.
        :param config: Language model configuration.
        :return: Language model instance.
        """
        return cls(**config.model_dump())

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
        # TODO: Implement
        encoding_parameters = self.encoding_parameters if encoding_parameters is None else encoding_parameters

        if self.backend == "ctransformers":
            raise NotImplemented("Not yet implemented.")
        elif self.backend == "langchain_llamacpp":
            raise NotImplemented("Not yet implemented.")
        elif self.backend == "transformers":
            return self.tokenizer.encode(input, **encoding_parameters)
        elif self.backend == "autogptq":
            return self.tokenizer.encode(input, **encoding_parameters)
        elif self.backend == "llamacpp":
            return self.model.tokenize(input)
        elif self.backend == "exllamav2":
            raise NotImplemented("Not yet implemented.")


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
        encoding_parameters = self.encoding_parameters if encoding_parameters is None else encoding_parameters
        embedding_parameters = self.embedding_parameters if embedding_parameters is None else embedding_parameters

        if self.backend == "ctransformers":
            return self.model.embed(input, **embedding_parameters)
        elif self.backend == "langchain_llamacpp":
            return self.embeddings.embed_query(input)
        elif self.backend == "transformers":
            input_ids = self.tokenizer.encode(input, **encoding_parameters)
            return self.model.model.embed_tokens(input_ids)
        elif self.backend == "autogptq":
            input_ids = self.tokenizer.encode(input, **encoding_parameters)
            return self.model.model.embed_tokens(input_ids)
        elif self.backend == "llamacpp":
            return self.model.embed(input)
        elif self.backend == "exllamav2":
            input_ids = self.tokenizer.encode(input, **encoding_parameters)
            return self.model.model.embed_tokens(input_ids)

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
        encoding_parameters = self.encoding_parameters if encoding_parameters is None else encoding_parameters
        generating_parameters = self.generating_parameters if generating_parameters is None else generating_parameters
        decoding_parameters = self.decoding_parameters if decoding_parameters is None else decoding_parameters

        metadata = {}
        answer = ""

        if self.backend == "ctransformers" or self.backend == "langchain_llamacpp":
            metadata = self.model(prompt, **generating_parameters)
        elif self.backend == "transformers" or self.backend == "autogptq":
            input_tokens = self.tokenizer(
                prompt, **encoding_parameters).to(self.model.device)
            output_tokens = self.model.generate(
                **input_tokens, **generating_parameters)[0]
            metadata = self.tokenizer.decode(
                output_tokens, **decoding_parameters)
        elif self.backend == "llamacpp":
            metadata = self.model(prompt, **generating_parameters)
            answer = metadata["choices"][0]["text"]
        elif self.backend == "exllamav2":
            metadata = self.generator.generate_simple(
                prompt, **generating_parameters)

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
                 language_model: LanguageModelConfig | LanguageModelInstance,
                 chat_parameters: dict | None = None,
                 system_prompt: str | None = None,
                 prompt_maker: Callable | None = None,
                 use_history: bool = True,
                 history: List[Dict[str, Union[str, dict]]] | None = None) -> None:
        """
        Initiation method.
        :param language_model: Language model config or instance.
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
        self.language_model_instance = language_model if isinstance(language_model, LanguageModelInstance) else LanguageModelInstance.from_configuration(language_model)
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
                "metadata": {"intitated": dt.now()}
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
                elif not delta:
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
                "metadata": {"intitated": dt.ctime(dt.now())}
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
                if not decoded_chunk.startswith(" ping -"):
                    if decoded_chunk != "[DONE]":
                        chunk = json.loads(decoded_chunk)
                        chunks.append(chunk)
                        delta = chunk["choices"][0]["delta"]
                    else:
                        delta = {}
                    if "content" in delta:
                        sentence += delta["content"]
                        if delta["content"] and delta["content"][-1] in SENTENCE_CHUNK_STOPS:
                            if len([elem for elem in sentence if elem.isalpha()]) >= minium_yielded_characters:
                                yield sentence, chunk
                                answer += sentence
                                sentence = ""
                    elif not delta:
                        if not answer.endswith(sentence):
                            answer += sentence
                        metadata = {"chunks": chunks}
                        yield sentence, chunk
        if self.use_history:
            self.history.append({"role": "assistant", "content": answer, "metadata": metadata})

    """
    Addtional endpoint wrappers
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
    
    def embed(self, input: str, embeddings_parameters: dict | None = None) -> List[List[float]] | None:
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


"""
Templates
"""
TEMPLATES = {

}


"""
Interfacing
"""
def spawn_language_model_instance(template: str) -> Union[LanguageModelInstance, dict]:
    """
    Function for spawning language model instance based on configuration templates.
    :param template: Instance template.
    :return: Language model instance if configuration was successful else an error report.
    """
    try:
        return LanguageModelInstance(**TEMPLATES[template])
    except Exception as ex:
        return {"exception": ex, "trace": traceback.format_exc()}
