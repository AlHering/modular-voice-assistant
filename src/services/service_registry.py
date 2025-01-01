# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2025 Alexander Hering             *
****************************************************
"""
from typing import Generator
from functools import partial
from time import time
from fastapi import APIRouter
from queue import Empty
from typing import List
from src.services.service_abstractions import Service, ServicePackage


class ServiceRegistry(object):
    """
    Service registry.
    """

    def __init__(self, services: List[Service]) -> None:
        """
        Initiation method.
        :param services: Services.
        """
        self.services = services

    def setup_and_run_service(self, service: Service, process_config: dict) -> bool:
        """
        Sets up and runs a service.
        :param service: Target service.
        :param process_config: Process config.
        :return: True, if setup was successful else False.
        """
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
    
    def reset_service(self, service: Service, process_config: dict | None = None) -> bool:
        """
        Resets a service.
        :param service: Target service.
        :param process_config: Process config for overwriting.
        :return: True, if reset was successful else False.
        """
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
        :param service: Target service.
        :return: True, if stopping was successful else False.
        """
        try:
            service.reset()
            return True
        except:
            return False
    
    def process(self, service: Service, input_package: ServicePackage) -> Generator[ServicePackage, None, None]:
        """
        Runs a service process.
        :param service: Target service.
        :param input_package: Input package.
        :return: True, if stopping was successful else False.
        """
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

    def register_with_router(self, router: APIRouter) -> None:
        """
        Registers services on API router.
        :param router: API router.
        """
        for service in self.services:
            router.add_api_route(path=f"/service/{service.name}/run", endpoint=partial(self.setup_and_run_service, service), methods=["POST"])
            router.add_api_route(path=f"/service/{service.name}/reset", endpoint=partial(self.reset_service, service), methods=["POST"])
            router.add_api_route(path=f"/service/{service.name}/stop", endpoint=partial(self.stop_service, service), methods=["POST"])
            router.add_api_route(path=f"/service/{service.name}/process", endpoint=partial(self.setup_and_run_service, service), methods=["POST"])