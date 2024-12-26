# -*- coding: utf-8 -*-
"""
****************************************************
*                      Utility                 
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from logging import Logger
from uuid import uuid4
from typing import Any, Tuple, List, Callable, Generator
import time
from threading import Thread, Event as TEvent
from queue import Empty, Queue as TQueue
from src.utility.time_utility import get_timestamp


def create_default_metadata() -> List[dict]:
    """
    Creates a default VA package metadata stack.
    :return: Default VA package dictionary
    """
    return [{"created": get_timestamp()}]


def create_uuid() -> str:
    """
    Creates an UUID for a VA package.
    :return: UUID as string.
    """
    return str(uuid4())


class VAPackage(BaseModel):
    """
    Voice assistant package for exchanging data between modules.
    """
    uuid: str = Field(default_factory=create_uuid)
    content: Any
    metadata_stack: List[dict] = Field(default_factory=create_default_metadata)


class VAModuleConfig(BaseModel):
    """
    Voice assistant module config class.
    """
    interrupt: TEvent | None = None,
    pause: TEvent | None = None,
    loop_pause: float = 0.1,
    input_timeout: float | None = None, 
    input_queue: TQueue | None = None,
    output_queue: TQueue | None = None,
    logger: Logger | None = None,
    name: str | None = None


class VAModule(ABC):
    """
    Voice assistant module.
    A module can be understood as a pipeline component with an input and output queue, which can be wrapped into a thread.
    It evolves around the central "process"-method which takes an input VAPackage from the input queue and potentially
    puts an output VAPackage back into the output queue, taking over the input package's UUID and previous metadata stack.
    """
    def __init__(self, 
                 interrupt: TEvent | None = None,
                 pause: TEvent | None = None,
                 loop_pause: float = 0.1,
                 input_timeout: float | None = None, 
                 input_queue: TQueue | None = None,
                 output_queue: TQueue | None = None,
                 logger: Logger | None = None,
                 name: str | None = None) -> None:
        """
        Initiates an instance.
        :param interrupt: Interrupt event.
        :param pause: Pause event.
        :param loop_pause: Time to wait between looped runs.
        :param input_timeout: Time to wait for inputs in a single run.
            The module will await an input indefinitely, if set to None.
        :param input_queue: Input queue.
        :param output_queue: Output queue.
        :param logger: Logger.
        :param name: A name to distinguish log messages.
        """
        self.interrupt = TEvent() if interrupt is None else interrupt
        self.pause = TEvent() if pause is None else pause
        self.loop_pause = loop_pause
        self.input_timeout = input_timeout
        self.input_queue = TQueue() if input_queue is None else input_queue
        self.output_queue = TQueue() if output_queue is None else output_queue
        self.logger = logger
        self.name = name or str(self)

        self.thread = None

        self.received = {}
        self.sent = {}

    @classmethod
    def from_configuration(cls, config: VAModuleConfig) -> Any:
        """
        Returns a language model instance from configuration.
        :param config: Module configuration class.
        :return: Module instance.
        """
        return cls(**config.model_dump()) 

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

    def _flush_queue(self, queue: TQueue) -> None:
        """
        Flushes queue.
        :param queue: Queue to flush.
        """
        with queue.mutex:
            queue.clear()
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
        """
        self.thread = Thread(target=self.loop)
        self.thread.daemon = True
        return self.thread

    def loop(self) -> None:
        """
        Starts processing cycle loop.
        """
        while not self.interrupt.is_set():
            self.run()
            time.sleep(self.loop_pause)
        
    def run(self) -> bool:
        """
        Runs a single processing cycle.
        :returns: True if an element was forwarded, else False. 
            (Note, that a module does not have to forward an element.)
        """
        result = self.process()
        if result is not None:
            if isinstance(result, VAPackage):
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
    def process(self) -> VAPackage | Generator[VAPackage, None, None] | None:
        """
        Module processing method.
        :returns: Voice assistant package, a package generator or None.
        """
        pass


class PassiveVAModule(VAModule):
    """
    Passive voice assistant module.
    This module follows the same functionality as a conventional VAModule but forwards the incoming packages in its received state.
    It can be used for fetching and processing pipeline data without the results being fed back into the module pipeline.
    """
    @abstractmethod
    def process(self) -> VAPackage | Generator[VAPackage, None, None] | None:
        """
        Passive module processing method.
        :returns: The input data, as taken from the input queue.
        """
        pass


class BaseModuleSet(object):
    """
    Base module set.
    Holds VAModules in four different categories:
    - input modules resemble a pipeline for inputting user data, e.g. SpeechRecorderModule->TranscriberModule
    - worker modules resemble a pipeline for processing the ingoing user data, e.g. a ChatModelModule
    - output modules resemble a pipeline for outputting the results of the worker module pipeline, e.g. SynthesizerModule->WaveOutputModule
    - additional (passive) modules can be "inserted" into a pipeline to branch out operations, which do not reintroduce transformed data back into 
        the pipeline, e.g. for animating a character alongside the output module pipeline or running additional reporting.
    """
    input_modules: List[VAModule] = []
    worker_modules: List[VAModule] = []
    output_modules: List[VAModule] = []
    additional_modules: List[PassiveVAModule] = []

    @classmethod
    def get_all(cls) -> List[VAModule]:
        """
        Returns all available modules.
        :returns: List of VA modules.
        """
        return cls.input_modules + cls.worker_modules + cls.output_modules + cls.additional_modules
    
    def reroute_pipeline_queues(self) -> List[VAModule]:
        """
        Reroutes queues between the pipelines.
        Note, that the queues of additional (passive) modules are not rerouted.
        """
        for module_list in [self.input_modules, self.worker_modules, self.output_modules]:
            for previous_module_index in range(len(module_list)-1):
                module_list[previous_module_index+1].input_queue = module_list[previous_module_index].output_queue
        self.worker_modules[0].input_queue = self.input_modules[-1].output_queue
        self.output_modules[0].input_queue = self.worker_modules[-1].output_queue