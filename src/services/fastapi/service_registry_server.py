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
from enum import Enum
import json
import uvicorn
from pydantic import BaseModel
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi import FastAPI, APIRouter
import asyncio
from queue import Empty
import traceback
from typing import List, Dict, Generator
import traceback
from datetime import datetime as dt
from uuid import UUID
from functools import wraps
import logging
from src.services.services import TranscriberService, ChatService, SynthesizerService
from src.services.service_abstractions import Service, ConcurrentService, ConcurrencyType, ServicePackage, FinalPackage
from src.database.basic_sqlalchemy_interface import BasicSQLAlchemyInterface, FilterMask
from src.database.data_model import populate_data_infrastructure, get_default_entries
from src.configuration import configuration as cfg


APP = FastAPI(title=cfg.PROJECT_NAME, version=cfg.PROJECT_VERSION,
              description=cfg.PROJECT_DESCRIPTION)
INTERFACE: ServiceRegistryServer | None = None
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
            "response": json.dumps(response) if isinstance(response, dict) else str(response),
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


class Endpoints(str, Enum):
    """
    Endpoints config.
    """
    interrupt = "interrupt"
    services_get = "/service/get"
    service_process = "/service/process"
    service_stream = "/service/stream"
    service_run = "/service/run"
    service_reset = "/service/reset"
    service_stop = "/service/stop"
    configs_get = "/configs/get"
    configs_add = "/configs/add"
    configs_patch = "/configs/patch"

    def __str__(self) -> str:
        """
        Returns string representation.
        """
        return str(self.value)


class ServiceRegistryServer(object):
    """
    Service registry.
    """

    def __init__(self, services: List[Service]) -> None:
        """
        Initiation method.
        :param services: Services.
        """
        self.services: Dict[str, Service] = {service.name: service for service in services}
        self.workers: Dict[str, ConcurrentService] = {service.name: None for service in services}
        self.service_uuids = {key: None for key in self.services}
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
        self.router.add_api_route(path="/service/get", endpoint=self.get_services, methods=["GET"])
        self.router.add_api_route(path="/interrupt", endpoint=self.interrupt, methods=["POST"])
        self.router.add_api_route(path="/service/process", endpoint=self.process, methods=["POST"])
        self.router.add_api_route(path="/service/stream", endpoint=self.process_as_stream, methods=["POST"])
        self.router.add_api_route(path="/service/run", endpoint=self.setup_and_run_service, methods=["POST"])
        self.router.add_api_route(path="/service/reset", endpoint=self.reset_and_run_service, methods=["POST"])
        self.router.add_api_route(path="/service/stop", endpoint=self.stop_service, methods=["POST"])
        self.router.add_api_route(path="/configs/get", endpoint=self.get_configs, methods=["POST"])
        self.router.add_api_route(path="/configs/add", endpoint=self.add_config, methods=["POST"])
        self.router.add_api_route(path="/configs/patch", endpoint=self.patch_config, methods=["POST"])
        return self.router
    
    """
    Service interaction
    """
    @interaction_log
    async def interrupt(self) -> BaseResponse:
        """
        Interrupt available services.
        """
        tasks = [asyncio.create_task(self.reset_and_run_service(service=service, config_uuid=self.service_uuids[service])) for service in self.service_uuids if self.service_uuids[service] is not None]
        # wait for tasks to complete
        _ = await asyncio.wait(tasks)
        return BaseResponse(status="success", results=[self.service_uuids])

    @interaction_log
    async def get_services(self) -> BaseResponse:
        """
        Responds available services.
        """
        return BaseResponse(status="success", results=[self.service_uuids])

    @interaction_log
    async def setup_and_run_service(self, service: str, config_uuid: str | UUID, concurrency_type: ConcurrencyType = ConcurrencyType.as_thread) -> BaseResponse:
        """
        Sets up and runs a service.
        :param service: Target service name.
        :param config_uuid: Config UUID.
        :param concurrency_type: Concurrency type.
        :return: Response.
        """
        service = self.services[service]
        if isinstance(config_uuid, str):
            config_uuid = UUID(config_uuid)
        try:
            if config_uuid != self.service_uuids[service.name] or not service.thread.is_alive():
                entry = self.database.obj_as_dict(self.database.get_objects_by_filtermasks(object_type="service_config", filtermasks=[FilterMask([["service_type", "==", service.name], ["uuid", "==", config_uuid]])])[0])
                service.config = entry["config"]
                if self.workers[service.name]:
                    self.workers[service.name].reset_service()
                else:
                    self.workers[service.name] = ConcurrentService(service=service, concurrency_type=concurrency_type)
                self.workers[service.name].run()
                self.service_uuids[service.name] = config_uuid
            while not service.setup_flag:
               await asyncio.sleep(.5)
            return BaseResponse(status="success", results=[{"service": service.name, "config_uuid": config_uuid}])
        except Exception as ex:
            return BaseResponse(status="error", results=[{"service": service.name, "config_uuid": config_uuid}], metadata={
                "error": str(ex), "trace": traceback.format_exc()
            })
    
    @interaction_log
    async def reset_and_run_service(self, service: str, config_uuid: str | UUID) -> BaseResponse:
        """
        Resets a service.
        :param service: Target service name.
        :param config_uuid: Config UUID.
        :return: Response.
        """
        service = self.services[service]
        if isinstance(config_uuid, str):
            config_uuid = UUID(config_uuid)
        try:
            entry = self.database.obj_as_dict(self.database.get_objects_by_filtermasks(object_type="service_config", filtermasks=[FilterMask([["service_type", "==", service], "uuid", "==", config_uuid])]))
            service.config = entry["config"]
            self.workers[service.name].reset_service()
            self.workers[service.name].run()
            while not service.setup_flag:
                await asyncio.sleep(.5)
            service.flush_inputs()
            service.flush_outputs()
            self.service_uuids[service.name] = config_uuid
            return BaseResponse(status="success", results=[{"service": service.name, "config_uuid": config_uuid}])
        except Exception as ex:
            return BaseResponse(status="error", results=[{"service": service.name, "config_uuid": config_uuid}], metadata={
                "error": str(ex), "trace": traceback.format_exc()
            })
    
    @interaction_log
    async def stop_service(self, service: str) -> BaseResponse:
        """
        Stops a service.
        :param service: Target service name.
        :return: Response.
        """
        service = self.services[service]
        try:
            self.workers[service.name].reset_service()
            self.service_uuids[service.name] = None
            return BaseResponse(status="success", results=[{"service": service.name}])
        except Exception as ex:
            return BaseResponse(status="error", results=[{"service": service.name}], metadata={
                "error": str(ex), "trace": traceback.format_exc()
            })
    
    @interaction_log
    async def process(self, service_request: ServiceRequest) -> ServicePackage | List[ServicePackage] | None:
        """
        Runs a service process.
        :param service_request: Service request.
        :return: Service package response.
        """
        service = self.services[service_request.service]
        service.input_queue.put(service_request.input_package)
        try:
            output_package = None
            requeue = []
            responses = []
            while not isinstance(output_package, FinalPackage):
                output_package = service.output_queue.get(timeout=service_request.timeout)
                if output_package.uuid == service_request.input_package.uuid:
                    responses.append(output_package)
                else:
                    requeue.append(output_package)
            for package in requeue:
                service.output_queue.put(package)
            if len(responses) == 1:
                return responses[0]
            else:
                return responses
        except Empty:
            return None

    def stream(self, service_request: ServiceRequest) -> Generator[bytes, None, None]:
        """
        Runs a service process.
        :param service_request: Service request.
        :return: Service package generator.
        """
        service = self.services[service_request.service]
        service.input_queue.put(service_request.input_package)

        finished = False
        while not finished:
            try:
                response = service.output_queue.get(timeout=service_request.timeout)
                if isinstance(response, FinalPackage):
                    finished = True
                yield json.dumps(response.model_dump()).encode("utf-8")
            except Empty:
                finished = True

    @interaction_log
    async def process_as_stream(self, service_request: ServiceRequest) -> StreamingResponse:
        """
        Runs a service process in streamed mode.
        :param service_request: Service request.
        :return: Service package response.
        """
        return StreamingResponse(self.stream(service_request=service_request), media_type="text/plain")

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
        if "uuid" in payload.config:
            payload.config["uuid"] = UUID(payload.config["uuid"])
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
        if "uuid" in payload.config:
            payload.config["uuid"] = UUID(payload.config["uuid"])
        result = self.database.obj_as_dict(self.database.patch_object(object_type="service_config", object_id=payload.config["uuid"], service_type=payload.service, **payload.config))
        return BaseResponse(status="success", results=[result])
    
    @interaction_log
    async def get_configs(self, service: str | None = None) -> BaseResponse:
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
    
    def __del__(self) -> None:
        """
        Deconstructs instance.
        """
        for service in self.services:
            if self.workers[service]:
                self.workers[service].reset_service()
                del self.workers[service]
            del self.services[service]
            del self.service_uuids[service.name]


"""
Backend server
"""
def run() -> None:
    """
    Runs backend server.
    """
    global APP, INTERFACE
    INTERFACE = ServiceRegistryServer(services=[
        TranscriberService(), 
        ChatService(), 
        SynthesizerService()
    ])
    APP.include_router(INTERFACE.setup_router())
    uvicorn.run("src.services.fastapi.service_registry_server:APP",
                host=cfg.BACKEND_HOST,
                port=cfg.BACKEND_PORT,
                log_level="debug")


if __name__ == "__main__":
    run()