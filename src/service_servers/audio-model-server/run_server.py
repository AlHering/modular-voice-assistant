# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2025 Alexander Hering             *
****************************************************
"""
from __future__ import annotations
from enum import Enum
import uvicorn
from pydantic import BaseModel
from fastapi.responses import RedirectResponse
from fastapi import FastAPI, APIRouter
from typing import List
import logging
from utility import ENV


"""
Configuration construction
"""
DEFAULT_MODEL_CONFIGS = [
]


APP = FastAPI(title="Audio Model Server", version="v0.1",
              description="Provides access to audio centered generative AI workloads.")
LOGGER = logging.getLogger("uvicorn.error")
LOGGER.setLevel(logging.DEBUG)


@APP.get("/", include_in_schema=False)
async def root() -> dict:
    """
    Redirects to Swagger UI docs.
    :return: Redirect to Swagger UI docs.
    """
    return RedirectResponse(url="/docs")

class ServiceRequest(BaseModel):
    """Config payload class."""
    service: str
    input_package: None
    timeout: float | None = None


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


def setup_router() -> APIRouter:
    """
    Sets up an API router.
    :return: API router.
    """
    router = APIRouter(prefix=ENV.get("API_BASE", "api/v1"))
    #router.add_api_route(path="/service/get", endpoint=get_services, methods=["GET"])
    #router.add_api_route(path="/interrupt", endpoint=interrupt, methods=["POST"])
    return router
    


"""
Backend server
"""
def run() -> None:
    """
    Runs backend server.
    """
    global APP
    router = setup_router()
    APP.include_router(router)
    uvicorn.run("run_server:APP",
                host=ENV.get("HOST", "0.0.0.0"),
                port=ENV.get("PORT", "8125"),
                log_level="debug")


if __name__ == "__main__":
    run()