# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2025 Alexander Hering             *
****************************************************
"""
from __future__ import annotations
import os
from typing import Generator, Any
from functools import partial
from time import time
from fastapi.responses import RedirectResponse
from fastapi import FastAPI, APIRouter
from queue import Empty
from typing import List
import uvicorn
import traceback
from datetime import datetime as dt
from uuid import UUID
from functools import wraps
import logging
from src.services.service_abstractions import Service, ServicePackage
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

    def setup_and_run_service(self, service: Service, process_config: dict) -> bool:
        """
        Sets up and runs a service.
        :param service: Target service name.
        :param process_config: Process config.
        :return: True, if setup was successful else False.
        """
        service = self.services[service]
        try:
            service.config = process_config
            if service.thread is not None and service.thread.is_alive():
                service.reset(restart_thread=True)
            else:
                thread = service.to_thread()
                thread.start()
            return True
        except:
            return False
    
    def reset_service(self, service: str, process_config: dict | None = None) -> bool:
        """
        Resets a service.
        :param service: Target service name.
        :param process_config: Process config for overwriting.
        :return: True, if reset was successful else False.
        """
        service = self.services[service]
        try:
            if process_config is not None:
                service.config = process_config
            service.reset(restart_thread=True)
            return True
        except:
            return False
    
    def stop_service(self, service: Service) -> bool:
        """
        Stops a service.
        :param service: Target service name.
        :return: True, if stopping was successful else False.
        """
        service = self.services[service]
        try:
            service.reset()
            return True
        except:
            return False
    
    def process(self, service: str, input_package: ServicePackage) -> Generator[ServicePackage, None, None]:
        """
        Runs a service process.
        :param service: Target service name.
        :param input_package: Input package.
        :return: True, if stopping was successful else False.
        """
        service = self.services[service]
        input_uuid = input_package.uuid
        service.input_queue.put(input_package)
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
                pass
        for response in wrongly_fetched:
            service.output_queue.put(response)

    def check_connection(self) -> dict:
        """
        Checks connection.
        :return: Connection response.
        """
        return {"status": "success", "message": f"Backend is available!"}

    def setup_router(self) -> APIRouter:
        """
        Sets up an API router.
        :return: API router.
        """
        self.router = APIRouter(prefix=cfg.BACKEND_ENDPOINT_BASE)
        self.router.add_api_route(path="/check", endpoint=self.check_connection, methods=["GET"])

    def register_services_with_router(self, router: APIRouter) -> None:
        """
        Registers services on API router.
        :param router: API router.
        """
        for service in self.services:
            router.add_api_route(path=f"/service/{service.name}/run", endpoint=partial(self.setup_and_run_service, service.name), methods=["POST"])
            router.add_api_route(path=f"/service/{service.name}/reset", endpoint=partial(self.reset_service, service.name), methods=["POST"])
            router.add_api_route(path=f"/service/{service.name}/stop", endpoint=partial(self.stop_service, service.name), methods=["POST"])
            router.add_api_route(path=f"/service/{service.name}/process", endpoint=partial(self.setup_and_run_service, service.name), methods=["POST"])

    """
    Config handling
    """
    @interaction_log
    async def add_config(self,
                         service: str,
                         config: dict) -> dict:
        """
        Adds a config to the database.
        :param service: Target service.
        :param config: Config.
        :return: Response.
        """
        if "id" in config:
            config["id"] = UUID(config["id"])
        result = self.database.obj_as_dict(self.database.put_object(object_type="service_config", service_type=service, **config))
        return {"status": "success", "result": result}
    
    @interaction_log
    async def overwrite_config(self,
                               payload: dict) -> dict:
        """
        Overwrites a config in the database.
        :param service: Target service type.
        :param config: Config.
        :return: Response.
        """
        service = payload["service"]
        config = payload["config"]
        result = self.database.obj_as_dict(self.database.patch_object(object_type="service_config", object_id=UUID(config.pop("id")), service_type=service, **config))
        return {"status": "success", "result": result}
    
    @interaction_log
    async def get_configs(self,
                          service: str | None = None) -> dict:
        """
        Retrieves configs from the database.
        :param service: Target service type.
            Defaults to None in which case all configs are returned.
        :return: Response.
        """
        if service is None:
            result = [self.database.obj_as_dict(entry) for entry in self.database.get_objects_by_type(object_type="service_config")]
        else:
            result = [self.database.obj_as_dict(entry) for entry in self.database.get_objects_by_filtermasks(object_type="service_config", filtermasks=[FilterMask([["service_type", "==", service]])])]
        return {"status": "success", "result": result}


"""
Backend server
"""
def run() -> None:
    """
    Runs backend server.
    """
    global APP, INTERFACE
    INTERFACE = ServiceRegistry()
    APP.include_router(INTERFACE.setup_router())
    uvicorn.run("src.service_interface:APP",
                host=cfg.BACKEND_HOST,
                port=cfg.BACKEND_PORT,
                log_level="debug")


if __name__ == "__main__":
    run()