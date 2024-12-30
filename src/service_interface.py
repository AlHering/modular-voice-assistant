# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from __future__ import annotations
import os
from typing import Any
import logging
from fastapi import FastAPI, APIRouter
from uuid import UUID
import traceback
from datetime import datetime as dt
import gc
from fastapi.responses import RedirectResponse
import uvicorn
from functools import wraps
from src.configuration import configuration as cfg
from src.database.basic_sqlalchemy_interface import BasicSQLAlchemyInterface, FilterMask
from src.database.data_model import populate_data_infrastructure, get_default_entries
from src.utility.sound_model_abstractions import Transcriber, Synthesizer, SpeechRecorder, AudioPlayer
from src.utility.language_model_abstractions import ChatModelInstance, RemoteChatModelInstance


AVAILABLE_SERVICES = {
    "speech_recorder": SpeechRecorder,
    "transcriber": Transcriber,
    "local_chat": ChatModelInstance,
    "remote_chat": RemoteChatModelInstance,
    "synthesizer": Synthesizer,
    "audio_player": AudioPlayer
}
APP = FastAPI(title=cfg.PROJECT_NAME, version=cfg.PROJECT_VERSION,
              description=cfg.PROJECT_DESCRIPTION)
INTERFACE: VoiceAssistantInterface | None = None
cfg.LOGGER = logging.getLogger("uvicorn.error")
cfg.LOGGER.setLevel(logging.DEBUG)


@APP.get("/", include_in_schema=False)
async def root() -> dict:
    """
    Redirects to Swagger UI docs.
    :return: Redirect to Swagger UI docs.
    """
    return RedirectResponse(url="/docs")


def interaction_log(func: Any) -> Any | None:
    """
    Interaction logging decorator.
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
            response = await func(*args, **kwargs)
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
            "response": str(response),
            "requested": requested,
            "responded": responded
        }
        args[0].database.post_object(
            object_type="log",
            **log_data
        )
        logging_message = f"Interaction with {args[0]}: {log_data}"
        logging.info(logging_message)
        return response
    return inner


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
            working_directory=self.working_directory,
            population_function=populate_data_infrastructure,
            default_entries=get_default_entries()
        )
        
        self.services = {key: None for key in AVAILABLE_SERVICES}
        self.service_uuids = {key: None for key in AVAILABLE_SERVICES}
        self.service_titles = {key: " ".join(key.split("_")).title() for key in AVAILABLE_SERVICES}
        self.router: APIRouter | None = None

    def setup_router(self) -> APIRouter:
        """
        Sets up API router.
        """
        self.router = APIRouter(prefix=cfg.BACKEND_ENDPOINT_BASE)
        self.router.add_api_route(path="/check", endpoint=self.check_connection, methods=["GET"])

        # Config and service handling
        self.router.add_api_route(path="/configs/add", endpoint=self.add_config, methods=["POST"])
        self.router.add_api_route(path="/configs/patch", endpoint=self.overwrite_config, methods=["POST"])
        self.router.add_api_route(path="/configs/get", endpoint=self.get_configs, methods=["POST"])
        self.router.add_api_route(path="/services/load", endpoint=self.load_service, methods=["POST"])
        self.router.add_api_route(path="/services/unload", endpoint=self.unload_service, methods=["POST"])

        # Underlying Services
        self.router.add_api_route(path="/services/record", endpoint=self.record_speech, methods=["POST"])
        self.router.add_api_route(path="/services/play-audio", endpoint=self.play_audio, methods=["POST"])
        self.router.add_api_route(path="/services/transcribe", endpoint=self.transcribe, methods=["POST"])
        self.router.add_api_route(path="/services/synthesize", endpoint=self.synthesize, methods=["POST"])
        self.router.add_api_route(path="/services/local-chat", endpoint=self.local_chat, methods=["POST"])
        self.router.add_api_route(path="/services/local-chat-stream", endpoint=self.local_chat_streamed, methods=["POST"])
        self.router.add_api_route(path="/services/remote-chat", endpoint=self.remote_chat, methods=["POST"])
        self.router.add_api_route(path="/services/remote-chat-stream", endpoint=self.remote_chat_streamed, methods=["POST"])

        return self.router

    def check_connection(self) -> dict:
        """
        Checks connection.
        :return: Connection response.
        """
        return {"status": "success", "message": f"Backend is available!"}

    """
    Config handling
    """
    @interaction_log
    async def add_config(self,
                         service_type: str,
                         config: dict) -> dict:
        """
        Adds a config to the database.
        :param service_type: Target service type.
        :param config: Config.
        :return: Response.
        """
        if "id" in config:
            config["id"] = UUID(config["id"])
        result = self.database.obj_as_dict(self.database.put_object(object_type="service_config", service_type=service_type, **config))
        return {"status": "success", "result": result}
    
    @interaction_log
    async def overwrite_config(self,
                   payload: dict) -> dict:
        """
        Overwrites a config in the database.
        :param service_type: Target service type.
        :param config: Config.
        :return: Response.
        """
        service_type = payload["service_type"]
        config = payload["config"]
        result = self.database.obj_as_dict(self.database.patch_object(object_type="service_config", object_id=UUID(config.pop("id")), service_type=service_type, **config))
        return {"status": "success", "result": result}
    
    @interaction_log
    async def get_configs(self,
                          service_type: str | None = None) -> dict:
        """
        Adds a config to the database.
        :param service_type: Target service type.
            Defaults to None in which case all configs are returned.
        :return: Response.
        """
        if service_type is None:
            result = [self.database.obj_as_dict(entry) for entry in self.database.get_objects_by_type(object_type="service_config")]
        else:
            result = [self.database.obj_as_dict(entry) for entry in self.database.get_objects_by_filtermasks(object_type="service_config", filtermasks=[FilterMask([["service_type", "==", service_type]])])]
        return {"status": "success", "result": result}

    """
    Service handling
    """

    @interaction_log
    async def load_service(self,
                    service_type: str,
                    config_uuid: str | UUID) -> dict:
        """
        Loads a service from the given config UUID.
        :param service_type: Target service type.
        :param config_uuid: Config UUID.
        :return: Response.
        """
        if isinstance(config_uuid, str):
            config_uuid = UUID(config_uuid)
        if self.service_uuids[service_type] == config_uuid:
            return {"success": f"Active {self.service_titles[service_type]} is already set to UUID '{config_uuid}'"}
        elif self.service_uuids[service_type] is not None:
            await self.unload_service(service_type=service_type, config_uuid=self.service_uuids[service_type])
        entry = self.database.obj_as_dict(self.database.get_object_by_id("service_config", object_id=config_uuid))
        if entry:
            self.services[service_type] = AVAILABLE_SERVICES[service_type](**entry["config"])
            self.service_uuids[service_type] = UUID(entry["id"])
            return {"success": f"Set active {self.service_titles[service_type]} to UUID '{config_uuid}':\n{entry}"}
        else:
            return {"error": f"No {self.service_titles[service_type]} with UUID '{config_uuid}'"}
            
    @interaction_log
    async def unload_service(self,
                      payload: dict) -> dict:
        """
        Unloads a service from the given config UUID.
        :param service_type: Target service type.
        :param config_uuid: Config UUID.
        :return: Response.
        """
        service_type = payload["service_type"]
        config_uuid = payload["config_uuid"]
        config_uuid = UUID(config_uuid)
        if self.service_uuids[service_type] != config_uuid:
            return {"error": f"Active {self.service_titles[service_type]} has UUID '{self.service_uuids[service_type]}', not '{config_uuid}'"}
        elif self.service_uuids[service_type] is not None:
            service_obj = self.services.pop(service_type) 
            del service_obj
            self.service_uuids[service_type] = None
            gc.collect()
            return {"success": f"Unloaded {self.service_titles[service_type]} with config '{config_uuid}'"}
        else:
            return {"error": f"No active {self.service_titles[service_type]}"}

    """
    Assistant handling
    """

    """
    Direct service access
    """
    @interaction_log
    async def record_speech(self, 
                      config_uuid: str | UUID,
                      recognizer_parameters: dict | None = None,
                      microphone_parameters: dict | None = None) -> dict:
        if isinstance(config_uuid, str):
            config_uuid = UUID(config_uuid)
        await self.load_service(service_type="speech_recorder", config_uuid=config_uuid)
        result = self.services["speech_recorder"].record_single_input(
            recognizer_parameters=recognizer_parameters,
            microphone_parameters=microphone_parameters
        )

    @interaction_log
    async def play_audio(self, 
                    config_uuid: str | UUID,
                    audio_input: str | list, 
                    playback_parameters: dict | None = None) -> dict:
        if isinstance(config_uuid, str):
            config_uuid = UUID(config_uuid)
        await self.load_service(service_type="audio_player", config_uuid=config_uuid)
        result = self.services["audio_player"].play(
            audio_input=audio_input,
            playback_parameters=playback_parameters
        )

    @interaction_log
    async def transcribe(self, 
                   config_uuid: str | UUID,
                   audio_input: str | list, 
                   transcription_parameters: dict | None = None) -> dict:
        if isinstance(config_uuid, str):
            config_uuid = UUID(config_uuid)
        await self.load_service(service_type="transcriber", config_uuid=config_uuid)

        result = self.services["transcriber"].transcribe(
            audio_input=audio_input,
            transcription_parameters=self.transcription_parameters if transcription_parameters is None else transcription_parameters
        )

    @interaction_log
    async def synthesize(self, 
                   config_uuid: str | UUID,
                   text: str,
                   synthesis_parameters: dict | None = None) -> dict:
        if isinstance(config_uuid, str):
            config_uuid = UUID(config_uuid)
        await self.load_service(service_type="synthesizer", config_uuid=config_uuid)
        result = self.services["synthesizer"].synthesize(
            text=text, 
            synthesis_parameters=synthesis_parameters)
        
    @interaction_log
    async def local_chat(self, 
                   config_uuid: str | UUID,
                   prompt: str, 
                   chat_parameters: dict | None = None) -> dict:
        if isinstance(config_uuid, str):
            config_uuid = UUID(config_uuid)
        await self.load_service(service_type="local_chat", config_uuid=config_uuid)
        result = self.services["local_chat"].chat(prompt=prompt, chat_parameters=chat_parameters)
        
    @interaction_log
    async def remote_chat(self, 
                   config_uuid: str | UUID,
                   prompt: str, 
                   chat_parameters: dict | None = None) -> dict:
        if isinstance(config_uuid, str):
            config_uuid = UUID(config_uuid)
        await self.load_service(service_type="remote_chat", config_uuid=config_uuid)
        result = self.services["remote_chat"].chat(prompt=prompt, chat_parameters=chat_parameters)
        
    @interaction_log
    async def local_chat_streamed(self, 
                   config_uuid: str | UUID,
                   prompt: str, 
                   chat_parameters: dict | None = None,
                   minium_yielded_characters: int = 10) -> dict:
        if isinstance(config_uuid, str):
            config_uuid = UUID(config_uuid)
        await self.load_service(service_type="local_chat", config_uuid=config_uuid)
        result = self.services["local_chat"].chat_streamed(
            prompt=prompt, 
            chat_parameters=chat_parameters,
            minium_yielded_characters=minium_yielded_characters)
        
    @interaction_log
    async def remote_chat_streamed(self, 
                   config_uuid: str | UUID,
                   prompt: str, 
                   chat_parameters: dict | None = None,
                   minium_yielded_characters: int = 10) -> dict:
        if isinstance(config_uuid, str):
            config_uuid = UUID(config_uuid)
        await self.load_service(service_type="remote_chat", config_uuid=config_uuid)
        result = self.services["remote_chat"].chat_streamed(
            prompt=prompt, 
            chat_parameters=chat_parameters,
            minium_yielded_characters=minium_yielded_characters)
        
        """return StreamingResponse(
            self._wrapped_streamed_chat(
                prompt=prompt,
                chat_parameters=chat_parameters,
                local=local),
            media_type="application/x-ndjson")"""
        
        
"""
Backend server
"""
def run() -> None:
    """
    Runs backend server.
    """
    global APP, INTERFACE
    INTERFACE = VoiceAssistantInterface()
    APP.include_router(INTERFACE.setup_router())
    uvicorn.run("src.service_interface:APP",
                host=cfg.BACKEND_HOST,
                port=cfg.BACKEND_PORT,
                log_level="debug")


if __name__ == "__main__":
    run()