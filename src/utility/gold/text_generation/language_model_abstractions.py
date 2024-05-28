
# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import traceback
import copy
from typing import List, Tuple, Any, Callable, Optional, Union, Dict
from datetime import datetime as dt
from .language_model_instantiation import load_ctransformers_model, load_transformers_model, load_llamacpp_model, load_autogptq_model, load_exllamav2_model, load_langchain_llamacpp_model


"""
Abstractions
"""
class LanguageModelInstance(object):
    """
    Language model class.
    """
    supported_backends: List[str] = ["ctransformers", "transformers",
                                     "llamacpp", "autogptq", "exllamav2", "langchain_llamacpp"]

    def __init__(self,
                 backend: str,
                 model_path: str,
                 model_file: str = None,
                 model_parameters: dict = None,
                 tokenizer_path: str = None,
                 tokenizer_parameters: dict = None,
                 embeddings_path: str = None,
                 embeddings_parameters: dict = None,
                 config_path: str = None,
                 config_parameters: dict = None,
                 encoding_parameters: dict = None,
                 embedding_parameters: dict = None,
                 generating_parameters: dict = None,
                 decoding_parameters: dict = None
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
        :param embedding_paramters: Kwargs for embedding as dictionary.
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

    """
    Generation methods
    """

    def embed(self,
              input: str,
              encoding_parameters: dict = None,
              embedding_parameters: dict = None,
              ) -> List[float]:
        """
        Method for embedding an input.
        :param input: Input to embed.
        :param encoding_parameters: Kwargs for encoding as dictionary.
            Defaults to None.
        :param embedding_paramters: Kwargs for embedding as dictionary.
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
                 encoding_parameters: dict = None,
                 generating_parameters: dict = None,
                 decoding_parameters: dict = None) -> Tuple[str, Optional[dict]]:
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
    

class ChatModelInstance(object):
    """
    Chat model class.
    """

    def __init__(self,
                 language_model_instance: LanguageModelInstance,
                 chat_parameters: dict = None,
                 default_system_prompt: str = None,
                 prompt_maker: Callable = None,
                 use_history: bool = True,
                 history: List[Dict[str, Union[str, dict]]] = None,
                 ) -> None:
        """
        Initiation method.
        :param LanguageModelInstance: Language model instance.
        :param chat_parameters: Kwargs for chatting in the chatting process as dictionary.
            Defaults to None in which case an empty dictionary is created and can be filled depending on the language instance's
            model backend.
        :param default_system_prompt: Default system prompt.
            Defaults to a None in which case no system prompt is used.
        :param prompt_maker: Function which takes the prompt history as a list of {"role": <role>, "content": <message>, "metadata": <metadata>}-dictionaries 
            (already including the new user prompt) as argument and calculates the final prompt. Only necessary, if the backend does not support chat interaction.
        :param use_history: Flag, declaring whether to use the history.
            Defaults to True.
        :param history: Interaction history as list of {"role": <role>, "content": <message>, "metadata": <metadata>}-dictionaries.
            Defaults to None.
        """
        self.language_model_instance = language_model_instance
        self.chat_parameters = self.language_model_instance.generating_parameters if chat_parameters is None else chat_parameters
        self.system_prompt = default_system_prompt
        if prompt_maker is None:
            def prompt_maker(history: List[Dict[str, Union[str, dict]]]) -> str:
                """
                Default Prompt maker function.
                :param history: History.
                """
                return "\n".join(f"<s>{entry['role']}:\n{entry['content']}</s>" for entry in history) + "\n"
        self.prompt_maker = prompt_maker

        self.use_history = use_history
        self.history = [] if history is None and default_system_prompt is None else [
            ("system", self.system_prompt, {
            "intitated": dt.now()})
            ] if history is None else history

    """
    Generation methods
    """

    def chat(self, prompt: str, chat_parameters: dict = None) -> Tuple[str, dict]:
        """
        Method for chatting with language model isntance.
        :param prompt: User input.
        :param chat_parameters: Kwargs for chatting in the chatting process as dictionary.
            Defaults to None in which case an empty dictionary is created and can be filled depending on the language instance's
            model backend.
        :return: Response and metadata.
        """
        chat_parameters = self.chat_parameters if chat_parameters is None else chat_parameters
        self.history.append({"role": "user", "content": prompt})
        full_prompt = self.prompt_maker(self.history)

        metadata = {}
        answer = ""

        if self.language_model_instance.backend == "ctransformers":
            return self.language_model_instance.generate(full_prompt)
        elif self.language_model_instance.backend == "langchain_llamacpp":
            return self.language_model_instance.generate(full_prompt)
        elif self.language_model_instance.backend == "transformers":
            chat_encoding_kwargs = copy.deepcopy(self.language_model_instance.encoding_parameters)
            chat_encoding_kwargs["add_generation_prompt"] = True
            input_tokens = self.language_model_instance.tokenizer.apply_chat_template(
                    self.history, 
                    **chat_encoding_kwargs).to(self.language_model_instance.model.device)

            output_tokens = self.language_model_instance.model.generate(
                **input_tokens, **chat_parameters)[0]
            metadata = self.language_model_instance.tokenizer.decode(
                output_tokens, 
                **self.language_model_instance.decoding_parameters)
            return answer, metadata
        elif self.language_model_instance.backend == "autogptq":
            return self.language_model_instance.generate(full_prompt)
        elif self.language_model_instance.backend == "llamacpp":
            metadata = self.language_model_instance.model.create_chat_completion(
                messages=self.history,
                **chat_parameters
            )
            answer = metadata["choices"][0]["message"].get("content", "")
            return answer, metadata
        elif self.language_model_instance.backend == "exllamav2":
            return self.language_model_instance.generate(full_prompt)


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
