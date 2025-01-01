# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2025 Alexander Hering             *
****************************************************
"""
from __future__ import annotations
from typing import Generator
import requests
from uuid import UUID
import json
from src.services.service_abstractions import ServicePackage
from src.services.service_registry import BaseResponse, ServicePackage, Endpoints


class ServiceRegistryClient(object):
    """
    Service registry client.
    """

    def __init__(self, api_base: str) -> None:
        """
        Initiation method.
        :param api_base: API base.
        :param return_as_dict: Flag for returning responses as dictionaries.
        """
        self.api_base = api_base

    def get_services(self) -> dict:
        """
        Responds available services.
        """
        return requests.get(self.api_base + Endpoints.services_get).json()

    def setup_and_run_service(self, service: str, config_uuid: str | UUID) -> dict:
        """
        Sets up and runs a service.
        :param service: Target service name.
        :param config_uuid: Config UUID.
        :return: Response.
        """
        return requests.post(self.api_base + Endpoints.service_run, params={
            "service": service,
            "config_uuid": config_uuid
        }).json()
    
    def reset_service(self, service: str, config_uuid: str | UUID) -> dict:
        """
        Resets a service.
        :param service: Target service name.
        :param config_uuid: Config UUID.
        :return: Response.
        """
        return requests.post(self.api_base + Endpoints.service_reset, params={
            "service": service,
            "config_uuid": config_uuid
        }).json()
    
    def stop_service(self, service: str) -> BaseResponse:
        """
        Stops a service.
        :param service: Target service name.
        :return: Response.
        """
        return requests.post(self.api_base + Endpoints.service_stop, params={
            "service": service
        }).json()
    
    def process(self, service: str, input_package: ServicePackage, timeout: float | None = None) -> dict | None:
        """
        Runs a service process.
        :param service: Service name.
        :param input_package: Input service package.
        :param timeout: Timeout.
        :return: Response.
        """
        try:
            return requests.post(self.api_base + Endpoints.service_process, json={
                "service": service,
                "input_package": input_package.model_dump(),
                "timeout": timeout
            }).json()
        except json.JSONDecodeError: 
            return None

    def stream(self, service: str, input_package: ServicePackage, timeout: float | None = None) -> Generator[dict, None, None]:
        """
        Runs a service process.
        :param service: Service name.
        :param input_package: Input service package.
        :param timeout: Timeout.
        :return: Response generator.
        """
        with requests.post(self.api_base + Endpoints.service_stream, json={
                "service": service,
                "input_package": input_package.model_dump(),
                "timeout": timeout
            }, stream=True) as response:
            accumulated = ""
            for chunk in response.iter_content():
                decoded_chunk = chunk.decode("utf-8")
                accumulated += decoded_chunk
                if accumulated.endswith("}"):
                    try:
                        json_chunk = json.loads(accumulated)
                        yield json_chunk
                        accumulated = ""
                    except json.JSONDecodeError:
                        pass

    """
    Config handling
    """
    def add_config(self, service: str, config: dict) -> dict:
        """
        Adds a config to the database.
        :param service: Target service.
        :param config: Config.
        :return: Response.
        """
        return requests.post(self.api_base + Endpoints.configs_add, json={
            "service": service,
            "config": config
        }).json()
    
    def patch_config(self, service: str, config: dict) -> dict:
        """
        Adds a config to the database.
        :param service: Target service.
        :param config: Config.
        :return: Response.
        """
        return requests.post(self.api_base + Endpoints.configs_patch, json={
            "service": service,
            "config": config
        }).json()
    
    def get_configs(self, service: str) -> dict:
        """
        Adds a config to the database.
        :param service: Target service.
        :return: Response.
        """
        return requests.post(self.api_base + Endpoints.configs_get, params={
            "service": service
        }).json()
