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
from typing import Any, List
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


class EndOfStreamPackage(BaseModel):
    """
    End of stream service package for exchanging data between services.
    """
    uuid: str = Field(default_factory=create_uuid)
    content: Any
    metadata_stack: List[dict] = Field(default_factory=create_default_metadata)


class InterruptPackage(BaseModel):
    """
    Interrupt service package for exchanging data between services.
    """
    uuid: str = Field(default_factory=create_uuid)
    content: None = None
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
        :param name: A name to distinguish log messages.
        """
        self.name = name
        self.description = description
        self.config = config
        self.cache = {}

        self.interrupt = Event()
        self.pause = Event()
        self.input_queue = Queue() if input_queue is None else input_queue
        self.output_queue = Queue() if output_queue is None else output_queue
        self.logger = logger

        self.setup_flag = False
        self.thread = None
        self.process = None
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
    
    def log_info(self, text: str) -> None:
        """
        Logs info, if logger is available.
        :param text: Text content to log.
        """
        if self.logger is not None:
            text = f"[{type(self).__name__}<{self.name}>] " + text
            self.logger.info(text)

    def to_thread(self) -> Thread:
        """
        Returns a thread for running service process in loop.
        :return: Thread
        """
        self.thread = Thread(target=self.setup_and_loop)
        self.thread.daemon = True
        self.mode = "thread"
        return self.thread
    
    def to_process(self) -> Process:
        """
        Returns a process for running service process in loop.
        :return: Process.
        """
        self.process = Process(target=self.setup_and_loop)
        self.process.daemon = True
        return self.process
    
    def reset(self, restart_thread: bool = False, restart_process: bool = False) -> None:
        """
        Resets service.
        :param restart_thread: Flag for restarting thread.
        :param restart_process: Flag for restarting process.
        """
        self.log_info(text="Stopping process.")
        self.pause.set()
        self.interrupt.set()
        self.input_queue.put(InterruptPackage())
        self.flush_inputs()
        self.flush_outputs()
        self.log_info(text="Stopping workers.")
        try:
            for worker in [self.thread, self.process]:
                if worker is not None and worker.is_alive():
                    worker.join(1.0) 
        except RuntimeError:
            if self.process is not None and self.process.is_alive():
                self.process.terminate() 
                self.process.join(.5) 
        self.setup_flag = False
        self.pause.clear()
        self.interrupt.clear()
        if restart_thread or restart_process:
            self.log_info(text="Restarting...")
            if restart_thread:
                self.to_thread()
                self.thread.start()
            self.interrupt.clear()
            if restart_process:
                self.to_process()
                self.process.start()

    def loop(self) -> None:
        """
        Starts processing cycle loop.
        """
        while not self.interrupt.is_set():
            self.iterate()
        self.log_info(text="Interrupt received, exiting loop.")
        
    def iterate(self) -> bool:
        """
        Runs a single processing cycle.
        :returns: True if an element was forwarded, else False. 
            (Note, that a service does not have to forward an element.)
        """
        result = self.run()
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
    
    """
    Methods to potentially overwrite
    """
    
    def setup(self) -> bool:
        """
        Sets up service.
        :returns: True, if successful else False.
        """
        return True

    @abstractmethod
    def run(self) -> ServicePackage | Generator[ServicePackage, None, None] | None:
        """
        Processes queued input.
        :returns: Service package, a service package generator or None.
        """
        pass

    def unpack(self, package: ServicePackage) -> dict:
        """
        Unpacks a service package.
        :param package: Service package.
        :returns: Unpacked content.
        """
        return package.model_dump()
    