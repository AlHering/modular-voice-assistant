# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from __future__ import annotations
import os
from typing import Any, List
import logging
from fastapi import FastAPI, APIRouter
from uuid import UUID
import traceback
from datetime import datetime as dt
from pydantic import BaseModel
import gc
from fastapi.responses import RedirectResponse, StreamingResponse
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


class TextRequest(BaseModel):
    """
    Text request data class.
    """
    config_uuid: str | UUID
    text: str
    parameters: dict | None = None


class AudioRequest(BaseModel):
    """
    Audio request data class.
    """
    config_uuid: str | UUID
    audio: str | List[int | float]
    dtype: str | None = None
    parameters: dict | None = None


class RecorderRequest(BaseModel):
    """
    Recorder request data class.
    """
    config_uuid: str | UUID
    recognizer_parameters: dict | None = None
    microphone_parameters: dict | None = None


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

        # Assistant functionality
        self.router.add_api_route(path="/assistant/run_input_pipeline", endpoint=self.run_input_pipeline, methods=["POST"])
        self.router.add_api_route(path="/assistant/run_output_pipeline", endpoint=self.run_output_pipeline, methods=["POST"])

        # Underlying services
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
                    config_uuid: str | UUID | None = None) -> dict:
        """
        Loads a service from the given config UUID.
        :param service_type: Target service type.
        :param config_uuid: Config UUID.
        :return: Response.
        """
        # No config declared, loading all.
        if config_uuid is None:
            responses = []
            for service_type in self.service_uuids:
                if self.service_uuids[service_type] is None:
                    available_configs = (await self.get_configs(service_type=service_type)).get("result")
                    if available_configs:
                        responses.append(await self.load_service(service_type=service_type, config_uuid=available_configs[0]))
                error_responses = [response["error"] for response in responses if "error" in response]
                if error_responses:
                    return {"error": "\n".join(error_responses)}
                else:
                    return {"success": f"Unloaded multiple services."}
                    
        # Loading declared config.
        if isinstance(config_uuid, str):
            config_uuid = UUID(config_uuid)
        if self.service_uuids[service_type] == config_uuid:
            return {"success": f"Active {self.service_titles[service_type]} is already set to UUID '{config_uuid}'"}
        elif self.service_uuids[service_type] is not None:
            unloading_response = await self.unload_service(service_type=service_type, config_uuid=self.service_uuids[service_type])
            if "error" in unloading_response:
                return {"error": f"Running {self.service_titles[service_type]} with UUID '{config_uuid}' could not be unloaded", "report": unloading_response}
        entry = self.database.obj_as_dict(self.database.get_object_by_id("service_config", object_id=config_uuid))
        if entry:
            self.services[service_type] = AVAILABLE_SERVICES[service_type](**entry["config"])
            self.service_uuids[service_type] = entry["id"]
            return {"success": f"Set active {self.service_titles[service_type]} to UUID '{config_uuid}':\n{entry}"}
        else:
            return {"error": f"No {self.service_titles[service_type]} with UUID '{config_uuid}'"}
            
    @interaction_log
    async def unload_service(self,
                    service_type: str,
                    config_uuid: str | UUID | None = None) -> dict:
        """
        Unloads a service from the given config UUID.
        :param service_type: Target service type.
        :param config_uuid: Config UUID.
        :return: Response.
        """
        # No config declared, unloading all.
        if config_uuid is None:
            responses = []
            for service_type in self.service_uuids:
                if self.service_uuids[service_type] is not None:
                    responses.append(await self.unload_service(service_type=service_type, config_uuid=self.service_uuids[service_type]))
                error_responses = [response["error"] for response in responses if "error" in response]
                if error_responses:
                    return {"error": "\n".join(error_responses)}
                else:
                    return {"success": f"Unloaded multiple services."}
        
        # Unloading declared config.
        if isinstance(config_uuid, str):
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
    @interaction_log
    async def run_input_pipeline(self,
                                 speech_recorder_uuid: str | UUID,
                                 transcriber_uuid: str | UUID) -> dict:
        if isinstance(speech_recorder_uuid, str):
            speech_recorder_uuid = UUID(speech_recorder_uuid)
        if isinstance(transcriber_uuid, str):
            transcriber_uuid = UUID(transcriber_uuid)
        speech_recorder_loading_response = await self.load_service(service_type="speech_recorder", config_uuid=speech_recorder_uuid)
        transcriber_loading_response = await self.load_service(service_type="transcriber", config_uuid=transcriber_uuid)
        for intermediate_response in [speech_recorder_loading_response, transcriber_loading_response]:
            if "error" in intermediate_response:
                return {"error": f"Loading service failed", "report": intermediate_response}

        recording_result = self.services["speech_recorder"].record_single_input()
        transcription_result = self.services["transcriber"].transcribe(audio_input=recording_result[0])
        return {"transcript": transcription_result[0], "metadata": transcription_result[1]}
    

    @interaction_log
    async def run_output_pipeline(self,
                                  synthesizer_uuid: str | UUID,
                                  audio_player_uuid: str | UUID,
                                  text: str) -> dict:
        if isinstance(synthesizer_uuid, str):
            synthesizer_uuid = UUID(synthesizer_uuid)
        if isinstance(audio_player_uuid, str):
            audio_player_uuid = UUID(audio_player_uuid)
        synthesizer_loading_response = await self.load_service(service_type="synthesizer", config_uuid=synthesizer_uuid)
        audio_player_loading_response = await self.load_service(service_type="audio_player", config_uuid=audio_player_uuid)
        for intermediate_response in [synthesizer_loading_response, audio_player_loading_response]:
            if "error" in intermediate_response:
                return {"error": f"Loading service failed", "report": intermediate_response}

        synthesis_result = self.services["synthesizer"].synthesize(text)
        self.services["audio_player"].play(audio_input=synthesis_result[0], playback_parameters=synthesis_result[1])
        return {"success": "Playback finished."}


    """
    Direct service access
    """
    @interaction_log
    async def record_speech(self, payload: RecorderRequest) -> dict:
        if isinstance(payload.config_uuid, str):
            payload.config_uuid = UUID(payload.config_uuid)
        loading_response = await self.load_service(service_type="speech_recorder", config_uuid=payload.config_uuid)
        if "error" in loading_response:
            return {"error": f"Loading service failed", "report": loading_response}
        result = self.services["speech_recorder"].record_single_input(
            recognizer_parameters=payload.recognizer_parameters,
            microphone_parameters=payload.microphone_parameters
        )
        return {"recording": result[0].tolist(), "dtype": str(result[0].dtype), "metadata": result[1]}

    @interaction_log
    async def transcribe(self, payload: AudioRequest) -> dict:
        if isinstance(payload.config_uuid, str):
            payload.config_uuid = UUID(payload.config_uuid)
        loading_response = await self.load_service(service_type="transcriber", config_uuid=payload.config_uuid)
        if "error" in loading_response:
            return {"error": f"Loading service failed", "report": loading_response}

        result = self.services["transcriber"].transcribe(
            audio_input=payload.audio,
            transcription_parameters=payload.parameters
        )
        return {"transcript": result[0], "metadata": result[1]}

    @interaction_log
    async def synthesize(self, payload: TextRequest) -> dict:
        if isinstance(payload.config_uuid, str):
            payload.config_uuid = UUID(payload.config_uuid)
        loading_response = await self.load_service(service_type="synthesizer", config_uuid=payload.config_uuid)
        if "error" in loading_response:
            return {"error": f"Loading service failed", "report": loading_response}
        result = self.services["synthesizer"].synthesize(
            text=payload.text, 
            synthesis_parameters=payload.parameters)
        return {"synthesis": result[0].tolist(), "dtype": str(result[0].dtype), "metadata": result[1]}

    @interaction_log
    async def play_audio(self, payload: AudioRequest) -> dict:
        if isinstance(payload.config_uuid, str):
            payload.config_uuid = UUID(payload.config_uuid)
        loading_response = await self.load_service(service_type="audio_player", config_uuid=payload.config_uuid)
        if "error" in loading_response:
            return {"error": f"Loading service failed", "report": loading_response}
        result = self.services["audio_player"].play(
            audio_input=payload.audio,
            playback_parameters=payload.parameters
        )
        return {"success": "Playback finished."}
        
    @interaction_log
    async def local_chat(self, payload: TextRequest) -> dict:
        if isinstance(payload.config_uuid, str):
            payload.config_uuid = UUID(payload.config_uuid)
        loading_response = await self.load_service(service_type="local_chat", config_uuid=payload.config_uuid)
        if "error" in loading_response:
            return {"error": f"Loading service failed", "report": loading_response}
        result = self.services["local_chat"].chat(prompt=payload.text, chat_parameters=payload.parameters)
        return {"response": result[0], "metadata": result[1]}
        
    @interaction_log
    async def remote_chat(self, payload: TextRequest) -> dict:
        if isinstance(payload.config_uuid, str):
            payload.config_uuid = UUID(payload.config_uuid)
        loading_response = await self.load_service(service_type="remote_chat", config_uuid=payload.config_uuid)
        if "error" in loading_response:
            return {"error": f"Loading service failed", "report": loading_response}
        result = self.services["remote_chat"].chat(prompt=payload.text, chat_parameters=payload.parameters)
        return {"response": result[0], "metadata": result[1]}

    @interaction_log
    async def local_chat_streamed(self, payload: TextRequest) -> StreamingResponse:
        if isinstance(payload.config_uuid, str):
            payload.config_uuid = UUID(payload.config_uuid)
        loading_response = await self.load_service(service_type="local_chat", config_uuid=payload.config_uuid)
        if "error" in loading_response:
            return {"error": f"Loading service failed", "report": loading_response}
        def wrap_response(**kwargs):
            for result in self.services["local_chat"].chat_streamed(**kwargs):
                yield {"response": result[0], "metadata": result[1]}
        return StreamingResponse(wrap_response(
            prompt=payload.text, 
            chat_parameters=payload.parameters),
            media_type="application/x-ndjson")

    @interaction_log
    async def remote_chat_streamed(self, payload: TextRequest) -> StreamingResponse:
        if isinstance(payload.config_uuid, str):
            payload.config_uuid = UUID(payload.config_uuid)
        loading_response = await self.load_service(service_type="remote_chat", config_uuid=payload.config_uuid)
        if "error" in loading_response:
            return {"error": f"Loading service failed", "report": loading_response}
        def wrap_response(**kwargs):
            for result in self.services["local_chat"].chat_streamed(**kwargs):
                yield {"response": result[0], "metadata": result[1]}
        return StreamingResponse(wrap_response(
            prompt=payload.text, 
            chat_parameters=payload.parameters),
            media_type="application/x-ndjson")
        
        
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