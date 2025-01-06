# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2025 Alexander Hering             *
****************************************************
"""
from __future__ import annotations
import os
import json
from pydantic import BaseModel
from queue import Empty
import traceback
from typing import List, Dict, Generator
import traceback
import time
from uuid import UUID
from src.services.abstractions.service_abstractions import Service, ServicePackage, EndOfStreamPackage
from src.database.basic_sqlalchemy_interface import BasicSQLAlchemyInterface, FilterMask
from src.database.data_model import populate_data_infrastructure, get_default_entries
from src.configuration import configuration as cfg


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
        self.services: Dict[str, Service] = {service.name: service for service in services}
        self.service_uuids = {key: None for key in self.services}
        self.working_directory = os.path.join(cfg.PATHS.DATA_PATH, "service_registry")
        self.database = BasicSQLAlchemyInterface(
            working_directory=self.working_directory,
            population_function=populate_data_infrastructure,
            default_entries=get_default_entries()
        )
    
    """
    Service interaction
    """
    def interrupt(self) -> BaseResponse:
        """
        Interrupt available services.
        """
        for service in self.service_uuids:
            if self.service_uuids[service] is not None:
                self.reset_service(service=service, config_uuid=self.service_uuids[service])
        return BaseResponse(status="success", results=[self.service_uuids])

    def get_services(self) -> BaseResponse:
        """
        Responds available services.
        """
        return BaseResponse(status="success", results=[self.service_uuids])

    def setup_and_run_service(self, service: str, config_uuid: str | UUID) -> BaseResponse:
        """
        Sets up and runs a service.
        :param service: Target service name.
        :param config_uuid: Config UUID.
        :return: Response.
        """
        service = self.services[service]
        if isinstance(config_uuid, str):
            config_uuid = UUID(config_uuid)
        try:
            if config_uuid != self.service_uuids[service.name] or not service.thread.is_alive():
                entry = self.database.obj_as_dict(self.database.get_objects_by_filtermasks(object_type="service_config", filtermasks=[FilterMask([["service_type", "==", service.name], ["id", "==", config_uuid]])])[0])
                service.config = entry["config"]
                if service.thread is not None and service.thread.is_alive():
                    service.reset(restart_thread=True)
                else:
                    thread = service.to_thread()
                    thread.start()
                self.service_uuids[service.name] = config_uuid
            while not service.setup_flag:
               time.sleep(.5)
            return BaseResponse(status="success", results=[{"service": service.name, "config_uuid": config_uuid}])
        except Exception as ex:
            return BaseResponse(status="error", results=[{"service": service.name, "config_uuid": config_uuid}], metadata={
                "error": str(ex), "trace": traceback.format_exc()
            })
    
    def reset_service(self, service: str, config_uuid: str | UUID) -> BaseResponse:
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
            entry = self.database.obj_as_dict(self.database.get_objects_by_filtermasks(object_type="service_config", filtermasks=[FilterMask([["service_type", "==", service], "id", "==", config_uuid])]))
            service.config = entry["config"]
            service.reset(restart_thread=True)
            while not service.setup_flag:
                time.sleep(.5)
            self.service_uuids[service.name] = config_uuid
            return BaseResponse(status="success", results=[{"service": service.name, "config_uuid": config_uuid}])
        except Exception as ex:
            return BaseResponse(status="error", results=[{"service": service.name, "config_uuid": config_uuid}], metadata={
                "error": str(ex), "trace": traceback.format_exc()
            })
        
    def clear_queues(self, service: str) -> BaseResponse:
        """
        Clears queues of a service.
        :param service: Target service name.
        :return: Response.
        """
        service = self.services[service]
        if isinstance(config_uuid, str):
            config_uuid = UUID(config_uuid)
        try:
            service.flush_inputs()
            service.flush_outputs()
            return BaseResponse(status="success", results=[{"service": service.name}])
        except Exception as ex:
            return BaseResponse(status="error", results=[{"service": service.name}], metadata={
                "error": str(ex), "trace": traceback.format_exc()
            })
    
    def stop_service(self, service: str) -> BaseResponse:
        """
        Stops a service.
        :param service: Target service name.
        :return: Response.
        """
        service = self.services[service]
        try:
            if service.thread is not None and service.thread.is_alive():
                service.reset()
            self.service_uuids[service.name] = None
            return BaseResponse(status="success", results=[{"service": service.name}])
        except Exception as ex:
            return BaseResponse(status="error", results=[{"service": service.name}], metadata={
                "error": str(ex), "trace": traceback.format_exc()
            })
        
    def process_in_pipeline(self, initial_request: ServiceRequest, pipeline: List[str]) -> ServicePackage | None:
        """
        Runs service processes in a pipeline.
        :param initial_request: Initial service request.
        :param pipeline: Pipeline as a list of subsequent service names.
            The first service is part of the initial request.
        :return: Final service package response.
        """
        next_package = self.process[initial_request]
        for service in pipeline:
            next_package = self.process(
                service_request=ServiceRequest(
                    service=service,
                    input_package=ServicePackage(content=next_package.content, metadata_stack=next_package.metadata_stack),
                    timeout=initial_request.timeout)
            )
        return next_package
    
    def process(self, service_request: ServiceRequest) -> ServicePackage | None:
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

    def process_as_stream(self, service_request: ServiceRequest) -> Generator[bytes, None, None]:
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
                if isinstance(response, EndOfStreamPackage):
                    finished = True
                yield json.dumps(response.model_dump()).encode("utf-8")
            except Empty:
                finished = True

    """
    Config handling
    """
    def add_config(self, payload: ConfigPayload) -> BaseResponse:
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
    
    def patch_config(self, payload: ConfigPayload) -> BaseResponse:
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
    
    def get_configs(self, service: str | None = None) -> BaseResponse:
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
        self.interrupt()
