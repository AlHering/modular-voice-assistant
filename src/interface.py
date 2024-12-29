# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import os
from typing import List, Generator, Any
import logging
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from uuid import UUID
import traceback
from datetime import datetime as dt
from gc import collect
from functools import wraps
from src.configuration import configuration as cfg
from src.database.basic_sqlalchemy_interface import BasicSQLAlchemyInterface, FilterMask
from src.database.data_model import populate_data_infrastructure, get_default_entries
from src.voice_assistant import AVAILABLE_MODULES, BasicVoiceAssistant, TranscriberModule, SynthesizerModule


def interaction_log() -> Any | None:
    """
    Interaction logging decorator.
    :return: Error report if operation failed, else function return.
    """
    def wrapper(func: Any) -> Any | None:
        """
        Function wrapper.
        :param func: Wrapped function.
        :return: Error report if operation failed, else function return.
        """
        @wraps(func)
        async def inner(*args: Any | None, **kwargs: Any | None):
            """
            Inner function wrapper.
            :param args: Arbitrary arguments.
            :param kwargs: Arbitrary keyword arguments.
            """
            requested = dt.now()
            try:
                response = func(*args, **kwargs)
            except Exception as ex:
                response = {
                    "status": "error",
                    "exception": str(ex),
                    "trace": traceback.format_exc()
                }
            responded = dt.now()
            log_data = {
                "request": {
                    "function": func.__name__,
                    "args": str(args),
                    "kwargs": str(kwargs)
                },
                "response": {
                    key: str(response[key]) for key in response
                },
                "requested": requested,
                "responded": responded
            }
            args[0].database.post_object(
                object_type="log",
                **log_data
            )
            logging_message = f"Interaction with {args[0]}: {log_data}"
            logging.info(logging_message) if log_data["response"].get("status", "success") == "error" else logging.warning(
                logging_message)
            return response
        return inner
    return wrapper



class VoiceAssistantInterface(object):
    """
    Voice assistant interface.
    """
    def __init__(self, working_directory: str = None) -> None:
        """
        Initiation method.
        :param working_directory: Working directory.
        """
        self.working_directory = os.path.join(cfg.PATHS.DATA_PATH, "voice_assistant_interface") if working_directory is None else working_directory
        self.database = BasicSQLAlchemyInterface(
            working_directory=os.path.join(self.working_directory, "database"),
            population_function=populate_data_infrastructure,
            default_entries=get_default_entries()
        )
        
        self.modules = {key: None for key in AVAILABLE_MODULES}
        self.module_uuids = {key: None for key in AVAILABLE_MODULES}
        self.module_titles = {key: " ".join(key.split("_")).title() for key in AVAILABLE_MODULES}
        self.assistant: BasicVoiceAssistant | None = None
        self.router: APIRouter | None = None

    def setup_router(self) -> None:
        """
        Sets up API router.
        """
        self.router = APIRouter(prefix=cfg.BACKEND_ENDPOINT_BASE)
        self.router.add_api_route(path="/", endpoint=self.check_connection, methods=["GET"])

        # Config and module handling
        self.router.add_api_route(path="/configs/add", endpoint=self.add_config, methods=["POST"])
        self.router.add_api_route(path="/configs/patch", endpoint=self.overwrite_config, methods=["POST"])
        self.router.add_api_route(path="/configs/get", endpoint=self.get_configs, methods=["GET"])
        self.router.add_api_route(path="/modules/load", endpoint=self.load_module, methods=["POST"])
        self.router.add_api_route(path="/modules/unload", endpoint=self.unload_module, methods=["POST"])

        # Assistant Interaction
        self.router.add_api_route(path="/assistant/setup", endpoint=self.setup_assistant, methods=["POST"])
        self.router.add_api_route(path="/assistant/reset", endpoint=self.assistant.setup, methods=["POST"])
        self.router.add_api_route(path="/assistant/stop", endpoint=self.assistant.stop, methods=["POST"])
        self.router.add_api_route(path="/assistant/inject-prompt", endpoint=self.assistant.inject_prompt, methods=["POST"])
        self.router.add_api_route(path="/assistant/interaction", endpoint=self.assistant.run_interaction, methods=["POST"])
        self.router.add_api_route(path="/assistant/conversation", endpoint=self.assistant.run_conversation, methods=["POST"])

        # Underlying Services
        self.router.add_api_route(path="/services/transcribe", endpoint=self.transcribe, methods=["POST"])
        self.router.add_api_route(path="/services/synthesize", endpoint=self.synthesize, methods=["POST"])
        self.router.add_api_route(path="/services/chat", endpoint=self.chat, methods=["POST"])
        self.router.add_api_route(path="/services/chat-stream", endpoint=self.chat_stream, methods=["POST"])

    def check_connection(self) -> dict:
        """
        Checks connection.
        :return: Connection response.
        """
        return {"success": f"Backend is available!"}

    """
    Config handling
    """
    @interaction_log
    def add_config(self,
                   module_type: str,
                   config: dict) -> dict:
        """
        Adds a config to the database.
        :param module_type: Target module type.
        :param config: Config.
        :return: Response.
        """
        return self.database.obj_as_dict(self.database.put_object(object_type="module_config", module_type=module_type, **config))
    
    @interaction_log
    def overwrite_config(self,
                   module_type: str,
                   config: dict) -> dict:
        """
        Overwrites a config in the database.
        :param module_type: Target module type.
        :param config: Config.
        :return: Response.
        """
        return self.database.obj_as_dict(self.database.patch_object(object_type="module_config", object_id=config.pop("id"), module_type=module_type, **config))
    
    @interaction_log
    def get_configs(self,
                    module_type: str = None) -> List[dict]:
        """
        Adds a config to the database.
        :param module_type: Target module type.
            Defaults to None in which case all configs are returned.
        :return: Response.
        """
        if module_type is None:
            return [self.database.obj_as_dict(entry) for entry in self.database.get_objects_by_type(object_type="module_config")]
        else:
            return [self.database.obj_as_dict(entry) for entry in self.database.get_objects_by_filtermasks(object_type="module_config", filtermasks=[FilterMask([["module_type", "==", module_type]])])]
        
    """
    Module handling
    """

    @interaction_log
    def load_module(self,
                    module_type: str,
                    config_uuid: str) -> dict:
        """
        Loads a module from the given config UUID.
        :param module_type: Target module type.
        :param config_uuid: Config UUID.
        :return: Response.
        """
        config_uuid = UUID(config_uuid)
        if self.module_uuids[module_type] != config_uuid:
            self.unload_module(module_type=module_type, config_uuid=self.module_uuids[module_type])
            entry = self.database.obj_as_dict(self.database.get_object_by_id("module_config", object_id=config_uuid))
            if entry:
                self.modules[module_type] = AVAILABLE_MODULES[module_type](**entry["config"])
                self.module_uuids[module_type] = UUID(entry["id"])
                return {"success": f"Set active {self.module_titles[module_type]} to UUID '{config_uuid}':\n{entry}"}
            else:
                return {"error": f"No {self.module_titles[module_type]} with UUID '{config_uuid}'"}
        else:
            return {"success": f"Active {self.module_titles[module_type]} is already set to UUID '{config_uuid}'"}
            
    @interaction_log
    def unload_module(self,
                      module_type: str,
                      config_uuid: str) -> dict:
        """
        Unloads a module from the given config UUID.
        :param module_type: Target module type.
        :param config_uuid: Config UUID.
        :return: Response.
        """
        config_uuid = UUID(config_uuid)
        if self.module_uuids[module_type] != config_uuid:
            return {"error": f"Active {self.module_titles[module_type]} has UUID '{self.module_uuids[module_type]}', not '{config_uuid}'"}
        elif self.module_uuids[module_type] is not None:
            module_obj = self.modules.pop(module_type) 
            del module_obj
            self.module_uuids[module_type] = None
            collect()
            return {"success": f"Unloaded {self.module_titles[module_type]} with config '{config_uuid}'"}
        else:
            return {"error": f"No active {self.module_titles[module_type]}"}

    """
    Assistant handling
    """

    @interaction_log
    def setup_assistant(self,
                        speech_recorder_uuid: str,
                        transcriber_uuid: str,
                        worker_uuid: str,
                        synthesizer_uuid: str,
                        wave_output_uuid: str,
                        stream: bool = True,
                        forward_logging: bool = False,
                        report: bool = False) -> dict:
        """
        Sets up a voice assistant from currently configured modules.
        :param speech_recorder_uuid: Speech Recorder config UUID.
        :param transcriber_uuid: Transcriber  config UUID.
        :param worker_uuid: Worker config UUID, e.g. for LocalChatModule or RemoteChatModule.
        :param synthesizer_uuid: Synthesizer config UUID.
        :param wave_output_uuid: Wave output config UUID.
        :param stream: Declares, whether chat model should stream its response.
        :param forward_logging: Flag for forwarding logger to modules.
        :param report: Flag for running report thread.
        """
        res = self.load_module(module_type="speech_recorder", config_uuid=UUID(speech_recorder_uuid))
        if "error" in res:
            return res
        res = self.load_module(module_type="transcriber", config_uuid=UUID(transcriber_uuid))
        if "error" in res:
            return res
        entry = self.database.obj_as_dict(self.database.get_object_by_id(UUID(worker_uuid)))
        if entry is not None:
            res = self.load_module(module_type=entry[entry["module_type"]], config_uuid=UUID(entry["id"]))
            if "error" in res:
                return res
        else:
            return {"error": f"No Worker Module config with UUID '{worker_uuid}'"}
        res = self.load_module(module_type="synthesizer", config_uuid=UUID(synthesizer_uuid))
        if "error" in res:
            return res
        res = self.load_module(module_type="wave_output", config_uuid=UUID(wave_output_uuid))
        if "error" in res:
            return res

        if self.assistant is not None:
            self.assistant.stop()
            self.assistant = None
            collect()
        self.assistant = BasicVoiceAssistant(
            working_directory=os.path.join(cfg.PATHS.DATA_PATH, "voice_assistant"),
            speech_recorder=self.modules["speech_recorder"],
            transcriber=self.modules["transcriber"],
            worker=self.modules[entry["module_type"]],
            synthesizer=self.modules["synthesizer"],
            wave_output=self.modules["wave_output"],
            stream=stream,
            forward_logging=forward_logging,
            report=report
        )
        self.assistant.setup()


    """
    Direct module access
    """

    @interaction_log
    def transcribe(self, 
                   audio_input: List[int | float] | str, 
                   dtype: str | None = None, 
                   transcription_parameters: dict | None = None) -> dict:
        """
        Transcribes audio to text.
        :param audio_input: Audio data or wave file path.
        :param dtype: Dtype in case of audio data input.
        :param transcription_parameters: Transcription parameters as dictionary.
            Defaults to None.
        :return: Transcript and metadata if successful, else error report.
        """
        if isinstance(self.module_uuids["transcriber"], TranscriberModule):
            response = self.modules["transcriber"].transcriber.transcribe(
                audio_input=audio_input if isinstance(audio_input, str) else np.asanyarray(audio_input, dtype=dtype), 
                transcription_parameters=transcription_parameters)
            return {"transcript": response[0], "metadata": response[1]}
        else:
            return {"error": f"No active Transcriber set."}

    def synthesize(self, text: str, synthesis_parameters: dict | None = None) -> dict:
        """
        Synthesizes audio from input text.
        :param text: Text to synthesize to audio.
        :param synthesis_parameters: Synthesis parameters as dictionary.
            Defaults to None.
        :return: Audio data, dtype and metadata if successful, else error report.
        """
        if isinstance(self.module_uuids["synthesizer"], SynthesizerModule):
            response = self.modules["synthesizer"].synthesizer.synthesize(
                text=text, 
                synthesis_parameters=synthesis_parameters)
            return {"synthesis": response[0].tolist(), "dtype": str(response[0].dtype), "metadata": response[1]}
        else:
            return {"error": f"No active Synthesizer set."}
        
    @interaction_log
    def chat(self, 
             prompt: str, 
             chat_parameters: dict | None = None,
             local: bool = True) -> dict:
        """
        Generates a chat response.
        :param prompt: User input.
        :param chat_parameters: Kwargs for chatting in the chatting process as dictionary.
            Defaults to None in which case an empty dictionary is created.
        :return: Generated response and metadata if successful, else error report.
        """
        target_worker = "local_chat" if local else "remote_chat"
        target_worker_class = LocalChatModule if local else RemoteChatModule
        if isinstance(target_worker, target_worker_class):
            response = self.modules[target_worker].chat_model.chat(
                prompt=prompt, 
                chat_parameters=chat_parameters)
            return {"response": response[0], "metadata": response[1]}
        else:
            return {"error": f"No active {self.module_titles[target_worker]} set."}
        
    def _wrapped_streamed_chat(self,
                               target_worker: str,
                               prompt: str, 
                               chat_parameters: dict | None = None,) -> Generator[dict, None, None]:
        """
        Wraps a streamed chat response.
        :param target_worker: Target worker module name.
        :param prompt: User input.
        :param chat_parameters: Kwargs for chatting in the chatting process as dictionary.
            Defaults to None in which case an empty dictionary is created.
        :return: Generated response and metadata if successful, else error report.
        """
        for response in self.modules[target_worker].chat_model.chat_stream(
                prompt=prompt, 
                chat_parameters=chat_parameters):
            yield {"response": response[0], "metadata": response[1]}
        
    @interaction_log
    def chat_stream(self, 
             prompt: str, 
             chat_parameters: dict | None = None,
             local: bool = True) -> StreamingResponse | dict:
        """
        Generates a streamed chat response.
        :param prompt: User input.
        :param chat_parameters: Kwargs for chatting in the chatting process as dictionary.
            Defaults to None in which case an empty dictionary is created.
        :return: Generated response and metadata if successful, else error report.
        """
        target_worker = "local_chat" if local else "remote_chat"
        target_worker_class = LocalChatModule if local else RemoteChatModule
        if isinstance(target_worker, target_worker_class):
            return StreamingResponse(self._wrapped_streamed_chat(
                target_worker=target_worker,
                prompt=prompt,
                chat_parameters=chat_parameters
            ),
            media_type="application/x-ndjson")
        else:
            return {"error": f"No active {self.module_titles[target_worker]} set."}
        
