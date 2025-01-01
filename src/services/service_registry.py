# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2025 Alexander Hering             *
****************************************************
"""
from __future__ import annotations
import os
from typing import Any
from time import time
from pydantic import BaseModel
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi import FastAPI, APIRouter
from queue import Empty
import traceback
from typing import List, AsyncGenerator
import uvicorn
import traceback
from datetime import datetime as dt
from uuid import UUID
from functools import wraps
import logging
from src.services.service_abstractions import Service, ServicePackage
from src.services.services import TranscriberService, ChatService, SynthesizerService
from src.database.basic_sqlalchemy_interface import BasicSQLAlchemyInterface, FilterMask
from src.database.data_model import populate_data_infrastructure, get_default_entries
from src.configuration import configuration as cfg


APP = FastAPI(title=cfg.PROJECT_NAME, version=cfg.PROJECT_VERSION,
              description=cfg.PROJECT_DESCRIPTION)
INTERFACE: ServiceRegistry | None = None
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


class ServiceRequest(BaseModel):
    """Config payload class."""
    service: str
    input_package: ServicePackage
    timeout: float | None = None


class ConfigPayload(BaseModel):
    """Config payload class."""
    service: str
    config: dict 


class BaseResponse(BaseModel):
    """Config payload class."""
    status: str
    results: List[dict] 
    metadata: dict | None = None


class ServiceRegistry(object):
    """
    Service registry.
    """

    def __init__(self, services: List[Service]) -> None:
        """
        Initiation method.
        :param services: Services.
        """
        self.services = {service.name: service for service in services}
        self.service_uuids = {key: None for key in self.services}
        self.service_titles = {key: " ".join(key.split("_")).title() for key in self.services}
        self.working_directory = os.path.join(cfg.PATHS.DATA_PATH, "service_registry")
        self.database = BasicSQLAlchemyInterface(
            working_directory=self.working_directory,
            population_function=populate_data_infrastructure,
            default_entries=get_default_entries()
        )
        self.router: APIRouter | None = None

    def setup_router(self) -> APIRouter:
        """
        Sets up an API router.
        :return: API router.
        """
        self.router = APIRouter(prefix=cfg.BACKEND_ENDPOINT_BASE)
        self.router.add_api_route(path=f"/service/get", endpoint=self.get_services, methods=["GET"])
        self.router.add_api_route(path=f"/service/process", endpoint=self.process, methods=["POST"])
        self.router.add_api_route(path=f"/service/stream", endpoint=self.process_as_stream, methods=["POST"])
        self.router.add_api_route(path=f"/service/run", endpoint=self.setup_and_run_service, methods=["POST"])
        self.router.add_api_route(path=f"/service/reset", endpoint=self.reset_service, methods=["POST"])
        self.router.add_api_route(path=f"/service/stop", endpoint=self.stop_service, methods=["POST"])
        self.router.add_api_route(path="/configs/get", endpoint=self.get_configs, methods=["POST"])
        self.router.add_api_route(path="/configs/add", endpoint=self.add_config, methods=["POST"])
        self.router.add_api_route(path="/configs/patch", endpoint=self.patch_config, methods=["POST"])
        return self.router

    @interaction_log
    async def get_services(self) -> BaseResponse:
        """
        Responds available services.
        """
        return BaseResponse(status="success", results=list(self.services.keys()))

    @interaction_log
    async def setup_and_run_service(self, service: str, config_uuid: str | UUID) -> bool:
        """
        Sets up and runs a service.
        :param service: Target service name.
        :param config_uuid: Config UUID.
        :return: True, if setup was successful else False.
        """
        service = self.services[service]
        if isinstance(config_uuid, str):
            config_uuid = UUID(config_uuid)
        try:
            entry = self.database.obj_as_dict(self.database.get_objects_by_filtermasks(object_type="service_config", filtermasks=[FilterMask([["service_type", "==", service], "id", "==", config_uuid])]))
            service.config = entry.config
            if service.thread is not None and service.thread.is_alive():
                service.reset(restart_thread=True)
            else:
                thread = service.to_thread()
                thread.start()
            return BaseResponse(status="success", results=[{"service": service.name, "config_uuid": config_uuid}])
        except Exception as ex:
            return BaseResponse(status="error", results=[{"service": service.name, "config_uuid": config_uuid}], metadata={
                "error": str(ex), "trace": traceback.format_exc()
            })
    
    @interaction_log
    async def reset_service(self, service: str, config_uuid: str | UUID) -> BaseResponse:
        """
        Resets a service.
        :param service: Target service name.
        :param config_uuid: Config UUID.
        :return: True, if reset was successful else False.
        """
        service = self.services[service]
        if isinstance(config_uuid, str):
            config_uuid = UUID(config_uuid)
        try:
            entry = self.database.obj_as_dict(self.database.get_objects_by_filtermasks(object_type="service_config", filtermasks=[FilterMask([["service_type", "==", service], "id", "==", config_uuid])]))
            service.config = entry.config
            service.reset(restart_thread=True)
            return BaseResponse(status="success", results=[{"service": service, "config_uuid": config_uuid}])
        except Exception as ex:
            return BaseResponse(status="error", results=[{"service": service, "config_uuid": config_uuid}], metadata={
                "error": str(ex), "trace": traceback.format_exc()
            })
    
    @interaction_log
    async def stop_service(self, service: str) -> BaseResponse:
        """
        Stops a service.
        :param service: Target service name.
        :return: True, if stopping was successful else False.
        """
        service = self.services[service]
        try:
            service.reset()
            return BaseResponse(status="success", results=[{"service": service}])
        except Exception as ex:
            return BaseResponse(status="error", results=[{"service": service}], metadata={
                "error": str(ex), "trace": traceback.format_exc()
            })
    
    @interaction_log
    async def process(self, service_request: ServiceRequest) -> ServicePackage | None:
        """
        Runs a service process.
        :param service_request: Service request.
        :return: Service package response.
        """
        service = self.services[service_request.service]
        service.input_queue.put(service_request.input_package)
        try:
            return service.output_queue.get(timeout=service_request.timeout)
        except Empty:
            return None

    async def stream(self, service_request: ServiceRequest) -> AsyncGenerator[ServicePackage, None]:
        """
        Runs a service process.
        :param service_request: Service request.
        :return: Service package generator.
        """
        service = self.services[service_request.service]
        input_uuid = service_request.input_package.uuid
        service.input_queue.put(service_request.input_package)
        start = time()
        response = service.output_queue.get()
        duration = time() - start

        wrongly_fetched = []
        while response:
            yield response
            try:
                response = service.output_queue.get(timeout=duration*1.5)
                if response.uuid != input_uuid:
                    wrongly_fetched.append(response)
            except Empty:
                response = None
        for response in wrongly_fetched:
            service.output_queue.put(response)

    @interaction_log
    async def process_as_stream(self, service_request: ServiceRequest) -> StreamingResponse:
        """
        Runs a service process in streamed mode.
        :param service_request: Service request.
        :return: Service package response.
        """
        return StreamingResponse(self.stream(service_request=service_request))

    """
    Config handling
    """
    @interaction_log
    async def add_config(self, payload: ConfigPayload) -> BaseResponse:
        """
        Adds a config to the database.
        :param service: Target service.
        :param config: Config.
        :return: Response.
        """
        if "id" in payload.config:
            payload.config["id"] = UUID(payload.config["id"])
        result = self.database.obj_as_dict(self.database.put_object(object_type="service_config", service_type=payload.service, **payload.config))
        return BaseResponse(status="success", results=[result])
    
    @interaction_log
    async def patch_config(self, payload: ConfigPayload) -> BaseResponse:
        """
        Overwrites a config in the database.
        :param service: Target service type.
        :param config: Config.
        :return: Response.
        """
        if "id" in payload.config:
            payload.config["id"] = UUID(payload.config["id"])
        result = self.database.obj_as_dict(self.database.patch_object(object_type="service_config", object_id=payload.config["id"], service_type=payload.service, **payload.config))
        return BaseResponse(status="success", results=[result])
    
    @interaction_log
    async def get_configs(self, service: str) -> BaseResponse:
        """
        Retrieves configs from the database.
        :param service: Target service type.
            Defaults to None in which case all configs are returned.
        :return: Response.
        """
        if service is None:
            results = [self.database.obj_as_dict(entry) for entry in self.database.get_objects_by_type(object_type="service_config")]
        else:
            results = [self.database.obj_as_dict(entry) for entry in self.database.get_objects_by_filtermasks(object_type="service_config", filtermasks=[FilterMask([["service_type", "==", service]])])]
        return BaseResponse(status="success", results=results)


"""
Backend server
"""
def run() -> None:
    """
    Runs backend server.
    """
    global APP, INTERFACE
    INTERFACE = ServiceRegistry(services=[
        TranscriberService(), 
        ChatService(), 
        SynthesizerService()
    ])
    APP.include_router(INTERFACE.setup_router())
    uvicorn.run("src.services.service_registry:APP",
                host=cfg.BACKEND_HOST,
                port=cfg.BACKEND_PORT,
                log_level="debug")


if __name__ == "__main__":
    run()