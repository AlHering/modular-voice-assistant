# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from __future__ import annotations
from typing import Any, Tuple, Generator
from logging import Logger
from abc import abstractmethod
from pydantic import BaseModel, Field
from logging import Logger
from uuid import uuid4
from multiprocessing import Process, Queue, Event
from threading import Thread
from traceback import format_exc
from typing import Any, List
from time import sleep
from copy import deepcopy
from enum import Enum
from gc import collect as collect_garbage
from src.utility.time_utility import get_timestamp


def create_default_metadata() -> List[dict]:
    """
    Creates a default service package metadata stack.
    :return: Default service package dictionary
    """
    return [{"created": get_timestamp()}]


def create_uuid() -> str:
    """
    Creates an UUID for a service package.
    :return: UUID as string.
    """
    return str(uuid4())


class ServicePackage(BaseModel):
    """
    Service package for exchanging data between services.
    """
    uuid: str = Field(default_factory=create_uuid)
    content: Any
    metadata_stack: List[dict] = Field(default_factory=create_default_metadata)


class FinalPackage(ServicePackage):
    """
    Final response service package for exchanging data between services.
    """
    uuid: str = Field(default_factory=create_uuid)
    content: Any
    metadata_stack: List[dict] = Field(default_factory=create_default_metadata)


class InterruptPackage(BaseModel):
    """
    Interrupt service package for sending an interrupt command.
    """
    uuid: str = Field(default_factory=create_uuid)
    metadata_stack: List[dict] = Field(default_factory=create_default_metadata)


class ResetPackage(BaseModel):
    """
    Reset service package for resetting the service (to an optionally given config).
    """
    uuid: str = Field(default_factory=create_uuid)
    content: dict | None = None
    restart_thread: bool = False
    restart_process: bool = False
    metadata_stack: List[dict] = Field(default_factory=create_default_metadata)


class Service(object):
    """
    Service.
    """
    def __init__(self, 
                 name: str, 
                 description: str,
                 config: dict,
                 input_queue: Queue | None = None,
                 output_queue: Queue | None = None,
                 logger: Logger | None = None) -> None:
        """
        Initiates an instance.
        :param name: Service name.
        :param description: Service description.
        :param config: Service config.
        :param input_queue: Input queue.
        :param output_queue: Output queue.
        :param logger: Logger.
        """
        self.name = name
        self.description = description
        self.config = config
        self.cache = {}

        self.interrupt = Event() # setting the interrupt event leaves the processing loop on next iteration
        self.pause = Event() # setting the pause event can be used to pause a single process run and return on clearing
        self.reset = Event()
        self.input_queue = Queue() if input_queue is None else input_queue
        self.output_queue = Queue() if output_queue is None else output_queue
        self.logger = logger

        self.setup_flag = False
        self.received = {}
        self.sent = {}

    @classmethod
    def from_configuration(cls, service_config: dict) -> Any:
        """
        Returns a service instance from configuration.
        :param service_config: Service configuration.
        :return: Service instance.
        """
        return cls(**service_config) 
    
    @classmethod
    def validate_configuration(cls, process_config: dict) -> Tuple[bool | None, str]:
        """
        Validates a process configuration.
        :param process_config: Process configuration.
        :return: True or False and validation report depending on validation success. 
            None and validation report in case of warnings. 
        """
        return None, "Validation method is not implemented."
    
    """
    Utility methods
    """

    def add_uuid(self, store: dict, uuid: str) -> None:
        """
        Adds a UUID to the sent dictionary.
        :param store: UUID dictionary to add UUID to.
        :param uuid: UUID to add.
        """
        if uuid in store:
            store[uuid] += 1
        else:
            store[uuid] = 1

    def _flush_queue(self, queue: Queue) -> None:
        """
        Flushes queue.
        :param queue: Queue to flush.
        """
        while not queue.empty():
            queue.get_nowait()

    def flush_inputs(self) -> None:
        """
        Flushes input queue.
        """
        self._flush_queue(self.input_queue)
        

    def flush_outputs(self) -> None:
        """
        Flushes output queue.
        """
        self._flush_queue(self.output_queue)

    def queues_are_busy(self) -> bool:
        """
        Returns queue status.
        :return: True, if any queue contains elements, else False.
        """
        return self.input_queue.qsize() > 0 or self.output_queue.qsize() > 0
    
    def log_info(self, text: str, as_warning: bool = False) -> None:
        """
        Logs info, if logger is available.
        :param text: Text content to log.
        :param as_warning: Whether to log message as warning.
        """
        if self.logger is not None:
            text = f"[{type(self).__name__}<{self.name}>] " + text
            if as_warning:
                self.logger.warning(text)
            else:
                self.logger.info(text)

    """
    Control methods
    """
    
    def setup_and_loop(self) -> None:
        """
        Method for setting up service and running processing loop.
        """
        if self.setup():
            self.setup_flag = True
            self.log_info(text="Setup succeeded, running loop.")
            self.loop()
        else:
            self.log_info(text="Setup failed.")
    
    def reset_service(self) -> None:
        """
        Resets service.
        """
        self.log_info(text="Resetting service.")
        self.pause.set()
        self.interrupt.set()
        self.input_queue.put(InterruptPackage())
        self.flush_inputs()
        self.flush_outputs()
        self.teardown()
        
        self.setup_flag = False
        self.pause.clear()
        self.interrupt.clear()
        collect_garbage()

    """
    Processing methods
    """

    def loop(self) -> None:
        """
        Starts processing cycle loop.
        """
        while not self.interrupt.is_set():
            self.iterate()
        self.log_info(text="Interrupt received, exiting loop.")
        if self.reset.is_set():
            self.reset()
        
    def iterate(self) -> bool:
        """
        Runs a single processing cycle.
        :returns: True if an element was forwarded, else False. 
            (Note, that a service does not have to forward an element.)
        """
        if not self.pause.is_set():
            input_package = self.input_queue.get(block=True)
            if isinstance(input_package, ResetPackage):
                self.handle_reset_package(input_package=input_package)
            elif isinstance(input_package, InterruptPackage):
                self.handle_interrupt_package(input_package=input_package)
            else:
                return self.handle_service_package(input_package=input_package)
            return False
        else:
            sleep(.1)

    def handle_reset_package(self, input_package: ResetPackage) -> None:
        """
        Handles interrupt package.
        :param input_package: Interrupt package.
        """
        self.log_info("Received Reset Package.")
        if input_package.content:
            if self.validate_configuration(input_package.content):
                self.log_info(f"Adjusting config: {input_package.content}.")
                self.config = deepcopy(input_package.content)
            else:
                self.log_info(f"Config validation failed: {input_package.content}.\nResetting with old config.", as_warning=True)
        self.interrupt.set()
        self.reset.set()

    def handle_interrupt_package(self, input_package: InterruptPackage) -> None:
        """
        Handles interrupt package.
        :param input_package: Interrupt package.
        """
        self.log_info("Received Interrupt Package.")
        self.interrupt.set()

    def handle_service_package(self, input_package: ServicePackage) -> bool:
        """
        Handles service package.
        :param input_package: Service package.
        :returns: True if an element was forwarded, else False. 
            (Note, that a service does not have to forward an element.)
        """
        self.log_info("Received Input Package.")
        result = self.process(input_package=input_package)
        if result is not None:
            if isinstance(result, ServicePackage):
                self.output_queue.put(result)
                self.add_uuid(self.sent, elem.uuid)
                return True
            elif isinstance(result, Generator):
                elem = None
                for elem in result:
                    self.output_queue.put(elem)
                if elem is not None:
                    self.add_uuid(self.sent, elem.uuid)
                    return True
        return False
    
    """
    Methods to potentially overwrite
    """
    
    def setup(self) -> bool:
        """
        Sets up service.
        :returns: True, if successful else False.
        """
        return True
    
    def teardown(self) -> bool:
        """
        Cleans up service cache.
        :returns: True, if successful else False.
        """
        for key in self.cache:
            try:
                del self.cache[key]
            except Exception as ex:
                self.log_info(f"Failed to clean up service cache: {ex}\nTrace: {format_exc()}", as_warning=True)
                return False
        collect_garbage()
        return True

    @abstractmethod
    def process(self, input_package: ServicePackage) -> ServicePackage | Generator[ServicePackage, None, None] | None:
        """
        Processes an input package.
        :param input_package: Input package.
        :returns: Service package, a service package generator or None.
        """
        pass

    def unpack_package(self, package: ServicePackage) -> dict:
        """
        Unpacks a service package.
        :param package: Service package.
        :returns: Unpacked content.
        """
        return package.model_dump()
    

"""
Service wrappers
"""
class ConcurrencyType(Enum):
    """
    Service concurrency type.
    """
    as_thread: int = 0
    as_process: int = 1

class ConcurrentService(object):
    """
    Wrapper class for running service as process.
    """
    def __init__(self, 
                 service: Service,
                 concurrency_type: ConcurrencyType = ConcurrencyType.as_thread) -> None:
        """
        Initiates an instance.
        :param host: Socket server host.
        :param port: Socket server port.
        :param service: Service to wrap into socket interaction.
        """
        self.service = service
        self.concurrency_type = concurrency_type
        self.worker = None

    def run(self) -> None:
        """
        Runs service as thread and/or process.
        """
        if self.concurrency_type == ConcurrencyType.as_thread:
            self.service.log_info(text="Starting as thread...")
            self.to_thread()
        elif self.concurrency_type == ConcurrencyType.as_process:
            self.service.log_info(text="Starting as process...")
            self.to_process()
        self.worker.start()

    def to_thread(self) -> Thread:
        """
        Returns a thread for running service process in loop.
        :return: Thread
        """
        self.worker = Thread(target=self.service.setup_and_loop)
        self.worker.daemon = True
        return self.worker
    
    def to_process(self) -> Process:
        """
        Returns a process for running service process in loop.
        :return: Process.
        """
        self.worker = Process(target=self.service.setup_and_loop)
        self.worker.daemon = True
        return self.worker
    
    def reset_service(self) -> None:
        """
        Resets service.
        """
        self.service.log_info(text="Stopping workers.")
        self.service.reset()
        if self.worker is not None and self.worker.is_alive():
            try:
                self.worker.join(1.0) 
            except RuntimeError:
                self.worker.terminate() 
                self.worker.join(.5) 
