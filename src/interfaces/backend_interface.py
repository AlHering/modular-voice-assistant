# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
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
from fastapi.responses import RedirectResponse
from functools import wraps
from src.configuration import configuration as cfg
from src.interfaces.endpoints.lm_instances import register_endpoints as register_lm_instance_endpoints
from src.interfaces.endpoints.tooling import register_endpoints as register_tooling_endpoints
from src.interfaces.endpoints.agent_memory import register_endpoints as register_agent_memory_endpoints
from src.interfaces.endpoints.agent import register_endpoints as register_agent_endpoints
from src.control.backend_controller import BackendController
from src.utility.silver.file_system_utility import safely_create_path


"""
Backend control
"""
BACKEND = FastAPI(title=cfg.BACKEND_TITLE, version=cfg.BACKEND_VERSION,
                  description=cfg.BACKEND_DESCRIPTION)
CONTROLLER: BackendController = BackendController()
CONTROLLER.setup()
for path in [cfg.PATHS.FILE_PATH]:
    safely_create_path(path)


def interface_function() -> Optional[Any]:
    """
    Validation decorator.
    :param func: Decorated function.
    :return: Error message if status is incorrect, else function return.
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
for registering_function in [register_lm_instance_endpoints,
                             register_tooling_endpoints,
                             register_agent_memory_endpoints,
                             register_agent_endpoints]:
    registering_function(backend=BACKEND,
                         interaction_decorator=interface_function,
                         controller=CONTROLLER,
                         endpoint_base=cfg.ENDPOINT_BASE)


@BACKEND.get("/", include_in_schema=False)
async def root() -> dict:
    """
    Root endpoint.
    :return: Redirect to SwaggerUI Docs.
    """
    return RedirectResponse(url="/docs")


@BACKEND.post(f"{cfg.BACKEND_ENDPOINT_BASE}/upload")
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
    if host is not None:
        cfg.BACKEND_HOST = host
    if port is not None:
        cfg.BACKEND_PORT = port
    uvicorn.run("src.interface.backend_interface:BACKEND",
                host=cfg.BACKEND_HOST,
                port=int(cfg.BACKEND_PORT),
                reload=reload)


if __name__ == "__main__":
    run_backend()
