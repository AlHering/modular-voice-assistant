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
from functools import partial
from typing import Optional, Any, Union
from datetime import datetime as dt
from fastapi import FastAPI, APIRouter, File, UploadFile
from pydantic import BaseModel, create_model
from fastapi.responses import RedirectResponse, JSONResponse, StreamingResponse
from functools import wraps
from src_legacy.control.voice_assistant_controller import VoiceAssistantController
from src_legacy.configuration import configuration as cfg
from src_legacy.utility.silver.file_system_utility import safely_create_path
from src_legacy.utility.bronze.sqlalchemy_utility import SQLALCHEMY_TYPING_FROM_COLUMN_DICTIONARY as TYPE_CONVERSION
from src_legacy.interfaces.endpoints.voice_assistant_endpoints import register_endpoints


"""
Backend control
"""
BACKEND: FastAPI = None
CONTROLLER: VoiceAssistantController = None
OBJECT_REPR: dict | None = None
ROUTER: APIRouter = None

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
    global BACKEND, CONTROLLER, ROUTER

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
        global CONTROLLER
        upload_path = os.path.join(cfg.PATHS.FILE_PATH, file_name)
        with open(upload_path, "wb") as output_file:
            while contents := file_data.file.read(cfg.FILE_UPLOAD_CHUNK_SIZE):
                output_file.write(contents)
        file_data.file.close()
        return {"file_path": upload_path}

    # Router
    ROUTER = APIRouter(prefix=cfg.VOICE_ASSISTANT_BACKEND_ENDPOINT_BASE)
    OBJECT_REPR = CONTROLLER.get_model_representation(
        ignore_columns=["created", "updated", "inactive"],
        ignore_object_types=["log"],
        types_as_strings=False
    )
    
    for object_type in OBJECT_REPR:
        OBJECT_REPR[object_type]["dataclass"] = create_model(
            "".join(object_type.split("_")).title(), 
            **{
                param["name"]: (Optional[param["type"]] if param["nullable"] else param["type"], param["default"])
            for param in OBJECT_REPR[object_type]["parameters"]}
        )
        ROUTER.add_api_route(
            path=f"/{object_type}", 
            endpoint=partial(CONTROLLER.get_objects_by_type,
                                   object_type=object_type, as_dict=True), methods=["GET"]
        )
        ROUTER.add_api_route(
            path=f"/{object_type}/{{id}}", 
            endpoint=partial(CONTROLLER.get_object_by_id,
                                   object_type=object_type, as_dict=True), methods=["GET"]
        )
    
    BACKEND.include_router(ROUTER)
    


"""
Backend runner
"""
def run_backend(host: str | None = None, port: int | None = None, reload: bool = True) -> None:
    """
    Function for running backend server.
    :param host: Server host. Defaults to None in which case "127.0.0.1" is set.
    :param port: Server port. Defaults to None in which case either environment variable "BACKEND_PORT" is set or 7861.
    :param reload: Reload flag for server. Defaults to True.
    """
    global BACKEND, CONTROLLER, ROUTER
    if host is not None:
        cfg.VOICE_ASSISTANT_BACKEND_HOST = host
    if port is not None:
        cfg.VOICE_ASSISTANT_BACKEND_PORT = port
        
    BACKEND = FastAPI(title=cfg.VOICE_ASSISTANT_BACKEND_TITLE, version=cfg.VOICE_ASSISTANT_BACKEND_VERSION,
                  description=cfg.VOICE_ASSISTANT_BACKEND_DESCRIPTION)
    CONTROLLER = VoiceAssistantController()
    CONTROLLER.setup()
    setup_endpoints()
    for path in [cfg.PATHS.FILE_PATH]:
        safely_create_path(path)

    uvicorn.run("src.interfaces.experimental_voice_assistant_interface:BACKEND",
                host=cfg.VOICE_ASSISTANT_BACKEND_HOST,
                port=int(cfg.VOICE_ASSISTANT_BACKEND_PORT),
                reload=reload)


if __name__ == "__main__":
    run_backend()
