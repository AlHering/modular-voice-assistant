# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2025 Alexander Hering             *
****************************************************
"""
import os
from typing import Generator, Tuple
import numpy as np
import requests
from copy import deepcopy
from src.configuration import configuration as cfg
from src.services.service_abstractions import Service, ServicePackage, FinalPackage
from src.model.abstractions.sound_model_abstractions import Transcriber, Synthesizer
from src.model.abstractions.language_model_abstractions import LanguageModelInstance, ChatModelInstance, RemoteChatModelInstance


class TranscriberService(Service):
    """
    Transcriber service.
    """
    def __init__(self):
        """
        Initiates an instance.
        """
        super().__init__(
            name="transcriber", 
            description="Transcribes audio data.", 
            config=cfg.DEFAULT_TRANSCRIBER, 
            logger=cfg.LOGGER)

    @classmethod
    def validate_configuration(cls, process_config: dict) -> Tuple[bool | None, str]:
        """
        Validates an configuration.
        :param config: Module configuration.
        :return: True or False and validation report depending on validation success. 
            None and validation report in case of warnings. 
        """
        if not process_config["backend"] in Transcriber.supported_backends:
            return False, f"Transcriber backend '{process_config['backend']}' is not supported."
        model_path = Transcriber.default_models[process_config["backend"]][0] if process_config.get("model_path") is None else process_config.get("model_path")
        if not os.path.exists(model_path):
            return None, f"Model path '{model_path}' is not in local filesystem.\nModel access must be handled remotely by the chosen backend."
        return True, "Validation succeeded."
    
    def setup(self) -> bool:
        """
        Sets up service.
        :returns: True, if successful else False.
        """
        self.cache = {
            "transcriber": Transcriber(**self.config)
        }
        return True

    def process(self, input_package: ServicePackage) -> ServicePackage | Generator[ServicePackage, None, None] | None:
        """
        Processes an input package.
        :param input_package: Input package.
        :returns: Service package, a service package generator or None.
        """
        if isinstance(input_package, ServicePackage):
            self.add_uuid(self.received, input_package.uuid)
            if not isinstance(input_package.content, np.ndarray):
                input_content = np.array(input_package.content, input_package.metadata_stack[-1].get("dtype"))
                self.log_info(f"Received input:\n'Numpy Array of shape {input_content.shape}'")
            else:
                input_content = input_package.content
                self.log_info(f"Received input:\n'{input_content}'")
            self.log_info(f"Received metadata:\n'{input_package.metadata_stack[-1]}'")
                
            result = self.cache["transcriber"].transcribe(
                audio_input=input_content,
                transcription_parameters=input_package.metadata_stack[-1].get("transcription_parameters"))
            self.log_info(f"Received response\n'{result[0]}'.")             
            yield FinalPackage(uuid=input_package.uuid, content=result[0], metadata_stack=input_package.metadata_stack + [result[1]])


class ChatService(Service):
    """
    Chat service.
    """
    def __init__(self):
        """
        Initiates an instance.
        """
        super().__init__(
            name="chat", 
            description="Generates chat responses via a language model.", 
            config=cfg.DEFAULT_CHAT_SMALL, 
            logger=cfg.LOGGER)
        
    @classmethod
    def validate_configuration(cls, process_config: dict) -> Tuple[bool | None, str]:
        """
        Validates an configuration.
        :param config: Module configuration.
        :return: True or False and validation report depending on validation success. 
            None and validation report in case of warnings. 
        """
        local = "api_base" not in process_config
        
        if local:
            if process_config["backend"] not in LanguageModelInstance.supported_backends:
                return False, f"Language model backend '{process_config['backend']}' is not supported."
            model_path = process_config["model_path"]
            if not os.path.exists(model_path):
                return None, f"Model path '{model_path}' is not in local filesystem.\nModel access must be handled remotely by the chosen backend."
            else:
                model_file = process_config.get("model_file")
                if model_file:
                    full_model_path = os.path.join(model_path, model_file)
                    if not os.path.exists(full_model_path):
                        return None, f"Model file '{full_model_path}' is not in local filesystem.\nModel file access must be handled by chosen backend."
            return True, "Validation succeeded."
        else:
            models_endpoint = f"{process_config['api_base']}/models"
            token = process_config.get("api_token")
            headers = {} if token is None else {
                "Authorization": f"Bearer {token}",
            }
            try:
                response = requests.get(models_endpoint, headers=headers)
                if response.status_code == 200:
                    return True, "Validation succeeded."
                else:
                    return None, f"Models endpoint {models_endpoint} resulted in response status code {response.status_code}."
            except requests.ConnectionError:
                return False, f"Connection to models endpoint {models_endpoint} did not succeed."
    
    def setup(self) -> bool:
        """
        Sets up service.
        :returns: True, if successful else False.
        """
        local = "api_base" not in self.config 
        model_instance = ChatModelInstance.from_dict(self.config 
            ) if local else RemoteChatModelInstance.from_dict(self.config )
        self.cache = {
            "local": local,
            "worker_config": self.config,
            "chat_model": model_instance,
            "chat_method": model_instance.chat,
            "streamed_chat_method": model_instance.chat_stream
        }
        return True

    def process(self, input_package: ServicePackage) -> ServicePackage | Generator[ServicePackage, None, None] | None:
        """
        Processes an input package.
        :param input_package: Input package.
        :returns: Service package, a service package generator or None.
        """
        if isinstance(input_package, ServicePackage):
            self.add_uuid(self.received, input_package.uuid)
            self.log_info(f"Received input:\n'{input_package.content}'")
            self.log_info(f"Received metadata:\n'{input_package.metadata_stack[-1]}'")

            streamed = input_package.metadata_stack[-1].get("chat_parameters", {}).get("stream", False)
            if streamed:
                result = self.cache["streamed_chat_method"](
                        prompt=input_package.content,
                        chat_parameters=input_package.metadata_stack[-1].get("chat_parameters"))
            else:
                result = self.cache["chat_method"](
                        prompt=input_package.content,
                        chat_parameters=input_package.metadata_stack[-1].get("chat_parameters"))
            if isinstance(result, Generator):
                for response_tuple in result:
                    self.log_info(f"Received response shard\n'{response_tuple[0]}'.")   
                    yield ServicePackage(uuid=input_package.uuid, content=response_tuple[0], metadata_stack=input_package.metadata_stack + [response_tuple[1]])
                yield FinalPackage(uuid=input_package.uuid, content="", metadata_stack=input_package.metadata_stack + [response_tuple[1]])
            else: 
                self.log_info(f"Received response\n'{result[0]}'.") 
                yield FinalPackage(uuid=input_package.uuid, content=result[0], metadata_stack=input_package.metadata_stack + [result[1]])
           

class SynthesizerService(Service):
    """
    Synthesizer service.
    """
    def __init__(self):
        """
        Initiates an instance.
        """
        super().__init__(
            name="synthesizer", 
            description="Synthesizes audio from text.", 
            config=cfg.DEFAULT_SYNTHESIZER, 
            logger=cfg.LOGGER)
        
    @classmethod
    def validate_configuration(cls, process_config: dict) -> Tuple[bool | None, str]:
        """
        Validates an configuration.
        :param config: Module configuration.
        :return: True or False and validation report depending on validation success. 
            None and validation report in case of warnings. 
        """
        if not process_config["backend"] in Synthesizer.supported_backends:
            return False, f"Synthesizer backend '{process_config['backend']}' is not supported."
        model_path = Synthesizer.default_models[process_config["backend"]][0] if process_config.get("model_path") is None else process_config.get("model_path")
        if not os.path.exists(model_path):
            return None, f"Model path '{model_path}' is not in local filesystem.\nModel access must be handled remotely by the chosen backend."
        return True, "Validation succeeded."
    
    def setup(self) -> bool:
        """
        Sets up service.
        :returns: True, if successful else False.
        """
        config = deepcopy(self.config)
        replace_symbols = config.pop("replace_symbols") if "replace_symbols" in config else []

        self.cache = {
            "replace_symbols": replace_symbols,
            "synthesizer": Synthesizer(**self.config)
        }
        return True

    def process(self, input_package: ServicePackage) -> ServicePackage | Generator[ServicePackage, None, None] | None:
        """
        Processes an input package.
        :param input_package: Input package.
        :returns: Service package, a service package generator or None.
        """
        if isinstance(input_package, ServicePackage):
            self.add_uuid(self.received, input_package.uuid)
            self.log_info(f"Received input:\n'{input_package.content}'")
            self.log_info(f"Received metadata:\n'{input_package.metadata_stack[-1]}'")

            input_package_content = input_package.content
            if self.cache["replace_symbols"]:
                for to_remove in self.cache["replace_symbols"]:
                    input_package_content.replace(to_remove, self.cache["replace_symbols"][to_remove])
                self.log_info(f"Cleaned input:\n'{input_package_content}'")

            result = self.cache["synthesizer"].synthesize(
                    text=input_package_content,
                    synthesis_parameters=input_package.metadata_stack[-1].get("synthesis_parameters"))
            self.log_info(f"Received response\n'{result[0]}'.") 
            synthesis_metadata = result[1]
            synthesis_metadata["dtype"] = str(result[0].dtype)
            yield FinalPackage(uuid=input_package.uuid, content=result[0].tolist(), metadata_stack=input_package.metadata_stack + [synthesis_metadata])