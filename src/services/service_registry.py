# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from typing import Any, Tuple
from multiprocessing import Queue, Event
from logging import Logger
from abc import abstractmethod
from pydantic import BaseModel, Field
from logging import Logger
from uuid import uuid4
from typing import Any, Tuple, List, Generator
from threading import Thread
from multiprocessing import Process
from src.utility.time_utility import get_timestamp


def create_default_metadata() -> List[dict]:
    """
    Creates a default pipeline package metadata stack.
    :return: Default pipeline package dictionary
    """
    return [{"created": get_timestamp()}]


def create_uuid() -> str:
    """
    Creates an UUID for a pipeline package.
    :return: UUID as string.
    """
    return str(uuid4())


class ServicePackage(BaseModel):
    """
    Pipeline package for exchanging data between modules.
    """
    uuid: str = Field(default_factory=create_uuid)
    content: Any
    metadata_stack: List[dict] = Field(default_factory=create_default_metadata)


class Service(object):
    """
    Service.
    """
    def __init__(self, name: str, 
                 description: str,
                 input_queue: Queue | None = None,
                 output_queue: Queue | None = None,
                 logger: Logger | None = None) -> None:
        """
        Initiation method.
        :param working_directory: Working directory.
        :param name: Service name.
        :param description: Service description.
        """
        self.name = name
        self.description = description

        self.interrupt = Event()
        self.pause = Event()
        self.input_queue = Queue() if input_queue is None else input_queue
        self.output_queue = Queue() if output_queue is None else output_queue
        self.logger = logger

        self.thread = None
        self.process = None
        self.received = {}
        self.sent = {}

    @classmethod
    def from_configuration(cls, config: dict) -> Any:
        """
        Returns a language model instance from configuration.
        :param config: Module configuration.
        :return: Module instance.
        """
        return cls(**config) 
    
    @classmethod
    def validate_configuration(cls, config: dict) -> Tuple[bool | None, str]:
        """
        Validates an configuration.
        :param config: Module configuration.
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
        queue.notify_all()

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
        Returns a thread for running module process in loop.
        :return: Thread
        """
        self.thread = Thread(target=self.loop)
        self.thread.daemon = True
        self.mode = "thread"
        return self.thread
    
    def to_process(self) -> Process:
        """
        Returns a process for running module process in loop.
        :return: Process.
        """
        self.process = Process(target=self.loop)
        self.process.daemon = True
        return self.process
    
    def reset(self, restart: bool = True) -> None:
        """
        Resets module.
        :param restart: Flag for restarting module.
        """
        self.pause.set()
        self.interrupt.set()
        self.flush_inputs()
        self.flush_outputs()
        try:
            for worker in [self.thread, self.process]:
                if worker is not None and worker.is_alive():
                    worker.join(1.0) 
        except RuntimeError:
            if self.process is not None and self.process.is_alive():
                self.process.terminate() 
                self.process.join(.5) 
        if restart:
            self.pause.clear()
            self.interrupt.clear()
            self.thread = self.to_thread()
            self.thread.start()

    def loop(self) -> None:
        """
        Starts processing cycle loop.
        """
        while not self.interrupt.is_set():
            self.iterate()
        
    def iterate(self) -> bool:
        """
        Runs a single processing cycle.
        :returns: True if an element was forwarded, else False. 
            (Note, that a module does not have to forward an element.)
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

class ServiceRegistry(object):
    """
    Service registry.
    """

    def __init__(self, working_directory: str = None) -> None:
        """
        Initiation method.
        :param working_directory: Working directory.
        """