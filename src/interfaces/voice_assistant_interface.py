# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import os
import uvicorn
import traceback
import logging
from typing import Optional, Any, Union
from datetime import datetime as dt
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from fastapi.responses import RedirectResponse, JSONResponse, StreamingResponse
from functools import wraps
from src.control.voice_assistant_controller import VoiceAssistantController
from src.configuration import configuration as cfg
from src.utility.silver.file_system_utility import safely_create_path
from src.interfaces.endpoints.voice_assistant_endpoints import register_endpoints


"""
Backend control
"""
BACKEND: FastAPI = None
CONTROLLER: VoiceAssistantController = None


def interface_function() -> Optional[Any]:
    """
    Validation decorator.
    :return: Decorator wrapper.
    """
    global CONTROLLER

    def wrapper(func: Any) -> Optional[Any]:
        """
        Function wrapper.
        :param func: Wrapped function.
        :return: Process data, containing error message if process failed, else function return.
        """
        @wraps(func)
        async def inner(*args: Optional[Any], **kwargs: Optional[Any]):
            """
            Inner function wrapper.
            :param args: Arguments.
            :param kwargs: Keyword arguments.
            """
            requested = dt.now()
            try:
                response = await func(*args, **kwargs)
                response["success"] = True
            except Exception as ex:
                response = {
                    "success": False,
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
            CONTROLLER.log(
                data=log_data
            )
            logging_message = f"Backend interaction: {log_data}"
            logging.info(logging_message) if log_data["response"]["success"] else logging.warn(
                logging_message)
            return response
        return inner
    return wrapper


"""
Endpoints
"""
def setup_endpoints() -> None:
    """
    Function for adding in router.
    """
    global BACKEND, CONTROLLER

    # Default endpoints
    @BACKEND.get("/", include_in_schema=False)
    async def root() -> dict:
        """
        Root endpoint.
        :return: Redirect to SwaggerUI Docs.
        """
        return RedirectResponse(url="/docs")


    @BACKEND.post(f"{cfg.VOICE_ASSISTANT_BACKEND_ENDPOINT_BASE}/upload")
    @interface_function()
    async def upload_file(file_name: str, file_data: UploadFile = File(...)) -> dict:
        """
        Endpoint for uplaoding a file.
        :param file_name: File name.
        :param file_data: File data.
        :return: Response.
        """
        upload_path = os.path.join(cfg.PATHS.FILE_PATH, file_name)
        with open(upload_path, "wb") as output_file:
            while contents := file_data.file.read(cfg.FILE_UPLOAD_CHUNK_SIZE):
                output_file.write(contents)
        file_data.file.close()
        return {"file_path": upload_path}

    # CUSTOM endpoints
    register_endpoints(backend=BACKEND,
                    interaction_decorator=interface_function,
                    controller=CONTROLLER,
                    endpoint_base=cfg.VOICE_ASSISTANT_BACKEND_ENDPOINT_BASE)


"""
Backend runner
"""
def run_backend(host: str = None, port: int = None, reload: bool = True) -> None:
    """
    Function for running backend server.
    :param host: Server host. Defaults to None in which case "127.0.0.1" is set.
    :param port: Server port. Defaults to None in which case either environment variable "BACKEND_PORT" is set or 7861.
    :param reload: Reload flag for server. Defaults to True.
    """
    global BACKEND, CONTROLLER
    if host is not None:
        cfg.VOICE_ASSISTANT_BACKEND_HOST = host
    if port is not None:
        cfg.VOICE_ASSISTANT_BACKEND_PORT = port
        
    BACKEND = FastAPI(title=cfg.VOICE_ASSISTANT_BACKEND_TITLE, version=cfg.VOICE_ASSISTANT_BACKEND_VERSION,
                  description=cfg.VOICE_ASSISTANT_BACKEND_DESCRIPTION)
    CONTROLLER = VoiceAssistantController()
    CONTROLLER.setup()
    for path in [cfg.PATHS.FILE_PATH]:
        safely_create_path(path)

    setup_endpoints()

    uvicorn.run("src.interfaces.voice_assistant_interface:BACKEND",
                host=cfg.VOICE_ASSISTANT_BACKEND_HOST,
                port=int(cfg.VOICE_ASSISTANT_BACKEND_PORT),
                reload=reload)


if __name__ == "__main__":
    run_backend()
