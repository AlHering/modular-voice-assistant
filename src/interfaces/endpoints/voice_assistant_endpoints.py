# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from typing import Callable, Optional, Union
from fastapi import FastAPI, Header
from pydantic import BaseModel
import numpy as np
import json
from src.control.voice_assistant_controller import VoiceAssistantController
from src.utility.bronze.pyaudio_utility import play_wave


class Transcriber(BaseModel):
    """
    Transcriber dataclass.
    """
    backend: str
    model_path: str
    model_path: Optional[str] = None
    model_parameters: Optional[dict] = None
    transcription_parameters: Optional[dict] = None


class Synthesizer(BaseModel):
    """
    Synthesizer dataclass.
    """
    backend: str
    model_path: str
    model_path: Optional[str] = None
    model_parameters: Optional[dict] = None
    synthesis_parameters: Optional[dict] = None


class SpeechRecorder(BaseModel):
    """
    SpeechRecorder dataclass.
    """
    input_device_index: Optional[int] = None
    recognizer_parameters: Optional[dict] = None
    microphone_parameters: Optional[dict] = None
    loop_pause: Optional[float] = 0.1


def register_endpoints(backend: FastAPI,
                       interaction_decorator: Callable,
                       controller: VoiceAssistantController,
                       endpoint_base: str) -> None:
    """
    Function for registering endpoints to given FastAPI based backend.
    :param backend: backend to register endpoints under. 
    :param interaction_decorator: Decorator function for wrapping endpoint functions.
    :param controller: Backend controller to handle endpoint accesses.
    :param endpoint_base: Endpoint base.
    """
    if endpoint_base[-1] == "/":
        endpoint_base = endpoint_base[:-1]

    target_classes = {
        "transcriber": Transcriber,
        "synthesizer": Synthesizer,
        "speech_recorder": SpeechRecorder
    }

    for target in list(target_classes.keys()):
        constructed_endpoint = endpoint_base + f"/{target}"

        @backend.get(f"{constructed_endpoint}")
        @interaction_decorator()
        async def get_all(object_type: str = Header(target, include_in_schema=False)) -> dict:
            """
            Endpoint for getting all entries.
            :return: Response.
            """
            return {object_type: [controller.return_obj_as_dict(obj) for obj in controller.get_objects_by_type(object_type)]}

        @backend.post(f"{constructed_endpoint}")
        @interaction_decorator()
        async def post(data: Union[Transcriber, Synthesizer, SpeechRecorder],
                       object_type: str = Header(target, include_in_schema=False)) -> dict:
            """
            Endpoint for posting entries.
            :param data: Instance data.
            :return: Response.
            """
            return {object_type: controller.return_obj_as_dict(controller.post_object(object_type, **dict(data)))}

        @backend.get(f"{constructed_endpoint}/{{id}}")
        @interaction_decorator()
        async def get(id: int,
                      object_type: str = Header(target, include_in_schema=False)) -> dict:
            """
            Endpoint for getting entries.
            :param id: Instance ID.
            :return: Response.
            """
            return {object_type: controller.return_obj_as_dict(controller.get_object_by_id(object_type, id))}

        @backend.delete(f"{constructed_endpoint}/{{id}}")
        @interaction_decorator()
        async def delete(id: int,
                         object_type: str = Header(target, include_in_schema=False)) -> dict:
            """
            Endpoint for deleting entries.
            :param id: Instance ID.
            :return: Response.
            """
            return {object_type: controller.return_obj_as_dict(controller.delete_object(object_type, id))}

        @backend.patch(f"{constructed_endpoint}/{{id}}")
        @interaction_decorator()
        async def patch(id: int, patch: dict,
                        object_type: str = Header(target, include_in_schema=False)) -> dict:
            """
            Endpoint for patching entries.
            :param id: Instance ID.
            :param patch: Patch payload.
            :return: Response.
            """
            return {object_type: controller.return_obj_as_dict(controller.patch_object(object_type, id, **patch))}

        @backend.put(f"{constructed_endpoint}")
        @interaction_decorator()
        async def put(data: Union[Transcriber, Synthesizer, SpeechRecorder],
                      object_type: str = Header(target, include_in_schema=False)) -> dict:
            """
            Endpoint for posting or updating entries.
            :param data: Instance data.
            :return: Response.
            """
            return {object_type: controller.put_object(object_type, **dict(data))}
        
    """
    Extended interaction
    """
    @backend.post(f"{endpoint_base}/transcriber/{{id}}/transcribe")
    async def transcribe(id: int, audio_input: list, transcription_parameters: Optional[dict] = None) -> dict:
        """
        Endpoint for transcribing.
        :param id: Transcriber ID.
        :param audio_input: Numpy compatible audio data to transcribe.
        :param transcription_parameters: Transcription parameters as dictionary.
            Defaults to None.
        :return: Response.
        """
        transcription_parameters = transcription_parameters if transcription_parameters else None
        transcript, metadata = controller.transcribe(transcriber_id=id,
                                                     audio_input=np.ndarray(audio_input),
                                                     transcription_parameters=transcription_parameters)
        return {"transcript": transcript, "metadata": metadata}
    
    @backend.post(f"{endpoint_base}/synthesizer/{{id}}/synthesize")
    async def synthesize(id: int, text: str, synthesis_parameters: Optional[dict] = None) -> dict:
        """
        Endpoint for synthesis.
        :param id: Synthesizer ID.
        :param text: Text to synthesize audio for.
        :param synthesis_parameters: Synthesis parameters as dictionary.
            Defaults to None.
        :return: Response.
        """
        synthesis_parameters = synthesis_parameters if synthesis_parameters else None
        result = await controller.synthesize(synthesizer_id=id,
                                       text=text,
                                       synthesis_parameters=synthesis_parameters)
        result[1]["dtype"] = str(result[0].dtype)
        return {"synthesis": result[0].tolist(), "metadata": result[1]}
    
    @backend.post(f"{endpoint_base}/speech_recorder/{{id}}/record")
    async def record(id: int, recognizer_parameters: Optional[dict] = None, microphone_parameters: Optional[dict] = None) -> dict:
        """
        Endpoint for recording.
        :param id: SpeechRecorder ID.
        :param recognizer_parameters: Keyword arguments for setting up recognizer instances.
            Defaults to None in which case default values are used.
        :param microphone_parameters: Keyword arguments for setting up microphone instances.
            Defaults to None in which case default values are used.
        :return: Response.
        """
        recognizer_parameters = recognizer_parameters if recognizer_parameters else None
        microphone_parameters = microphone_parameters if microphone_parameters else None
        audio, metadata = controller.record(speech_recorder_id=id,
                                            recognizer_parameters=recognizer_parameters,
                                            microphone_parameters=microphone_parameters)
        return {"audio": audio.tolist(), "metadata": metadata}
        
    descriptions = {
        "get": "Endpoint for getting entries.",
        "post": "Endpoint for posting entries.",
        "patch": "Endpoint for patching entries.",
        "put": "Endpoint for putting entries.",
        "delete": "Endpoint for deleting entries."
    }
        
    scheme = backend.openapi()
    for path in [path for path in scheme["paths"] 
                 if not any(path.endswith(interaction_end) 
                            for interaction_end in ["transcribe", "synthesize", "record"])]:
        target = path.replace(f"{endpoint_base}/", "").split("/")[0]
        if target in target_classes:
            for method in ["post", "put"]:
                # 'requestBody': {'content': {'application/json': {'schema': {'$ref': '#/components/schemas/Transcriber'}}}, 'required': True}
                if "requestBody" in scheme["paths"][path].get(method, {}):
                    scheme["paths"][path][method]["requestBody"]["content"]["application/json"]["schema"] = {"$ref": f"#/components/schemas/{target_classes[target].__name__}"}
                    scheme["paths"][path][method]["summary"] += f" {target_classes[target].__name__}"
            for method in scheme["paths"][path]:
                scheme["paths"][path][method]["description"] = descriptions[method]
    backend.openapi_schema = scheme

