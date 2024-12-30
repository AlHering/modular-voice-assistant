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
import gc
import numpy as np
from typing import Any, Tuple, List, Callable, Generator
import time
from threading import Thread, Event as TEvent
from queue import Empty, Queue as TQueue
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


class PipelinePackage(BaseModel):
    """
    Pipeline package for exchanging data between modules.
    """
    uuid: str = Field(default_factory=create_uuid)
    content: Any
    metadata_stack: List[dict] = Field(default_factory=create_default_metadata)


class PipelineModule(ABC):
    """
    Pipeline module.
    A module can be understood as a pipeline component with an input and output queue, which can be wrapped into a thread.
    It evolves around the central "process"-method which takes an input pipelinePackage from the input queue and potentially
    puts an output pipelinePackage back into the output queue, taking over the input package's UUID and previous metadata stack.
    """
    def __init__(self, 
                 input_queue: TQueue | None = None,
                 output_queue: TQueue | None = None,
                 logger: Logger | None = None,
                 name: str | None = None) -> None:
        """
        Initiates an instance.
        :param input_queue: Input queue.
        :param output_queue: Output queue.
        :param logger: Logger.
        :param name: A name to distinguish log messages.
        """
        self.interrupt = TEvent()
        self.pause = TEvent()
        self.input_queue = TQueue() if input_queue is None else input_queue
        self.output_queue = TQueue() if output_queue is None else output_queue
        self.logger = logger
        self.name = name or str(self)

        self.thread = None

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

    def _flush_queue(self, queue: TQueue) -> None:
        """
        Flushes queue.
        :param queue: Queue to flush.
        """
        with queue.mutex:
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
        """
        self.thread = Thread(target=self.loop)
        self.thread.daemon = True
        return self.thread
    
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
            self.thread.join() 
        except RuntimeError:
            pass
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
            self.run()
        
    def run(self) -> bool:
        """
        Runs a single processing cycle.
        :returns: True if an element was forwarded, else False. 
            (Note, that a module does not have to forward an element.)
        """
        result = self.process()
        if result is not None:
            if isinstance(result, PipelinePackage):
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
    def process(self) -> PipelinePackage | Generator[PipelinePackage, None, None] | None:
        """
        Module processing method.
        :returns: Pipeline package, a package generator or None.
        """
        pass


class BasicHandlerModule(PipelineModule):
    """
    Basic handler module.
    """
    def __init__(self, 
                 handler_method: Callable, 
                 *args: Any | None, 
                 **kwargs: Any | None) -> None:
        """
        Initiates an instance.
        :param handler_method: Handler method.
        :param args: Arbitrary arguments.
        :param kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.handler_method = handler_method

    def process(self) -> PipelinePackage | Generator[PipelinePackage, None, None] | None:
        """
        Module processing method.
        :returns: Pipeline package, package generator or None.
        """
        if not self.pause.is_set():
            try:
                input_package: PipelinePackage = self.input_queue.get(block=True)
                self.add_uuid(self.received, input_package.uuid)
                self.log_info(f"Received input:\n'{input_package.content}'")
                valid_input = (isinstance(input_package.content, np.ndarray) and input_package.content.size > 0) or input_package.content

                if valid_input:
                    result = self.handler_method(input_package.content)
                    if isinstance(result, Generator):
                        for response_tuple in result:
                            self.log_info(f"Received response shard\n'{response_tuple[0]}'.")   
                            yield PipelinePackage(uuid=input_package.uuid, content=response_tuple[0], metadata_stack=input_package.metadata_stack + [response_tuple[1]])
                    else:
                        self.log_info(f"Received response\n'{result[0]}'.")             
                        yield PipelinePackage(uuid=input_package.uuid, content=result[0], metadata_stack=input_package.metadata_stack + [result[1]])
            except Empty:
                pass


class ModularPipeline(object):
    """
    Modular Pipeline class.
    Holds PipelineModules in four different categories:
    - input modules resemble a pipeline for inputting user data, e.g. SpeechRecorderModule->TranscriberModule
    - worker modules resemble a pipeline for processing the ingoing user data, e.g. a ChatModelModule
    - output modules resemble a pipeline for outputting the results of the worker module pipeline, e.g. SynthesizerModule->WaveOutputModule
    """
    def __init__(self,
                 input_modules: List[PipelineModule] = [],
                 worker_modules: List[PipelineModule] = [],
                 output_modules: List[PipelineModule] = [],
                 base_loop_pause: float = 0.1,
                 logger: Logger | None = None) -> None:
        """
        Initiation method.
        :param input_modules: Input modules.
        :param worker_modules: Worker modules. 
        :param output_modules: Output modules.
        :param base_loop_pause: Pipeline base loop pause.
        :param logger: Logger if logging is desired.
        """
        self.input_modules = input_modules
        self.worker_modules = worker_modules
        self.output_modules = output_modules
        self.base_loop_pause = base_loop_pause
        self.logger = logger

        self.stop = TEvent()
        self.setup_modules()

    """
    Pipeline management
    """

    def get_all(self) -> List[PipelineModule]:
        """
        Returns all available modules.
        :returns: List of pipeline modules.
        """
        return self.input_modules + self.worker_modules + self.output_modules
    
    def reroute_pipeline_queues(self) -> List[PipelineModule]:
        """
        Reroutes queues between the pipelines.
        """
        for module_list in [self.input_modules, self.worker_modules, self.output_modules]:
            for previous_module_index in range(len(module_list)-1):
                module_list[previous_module_index+1].input_queue = module_list[previous_module_index].output_queue
        if self.input_modules and self.worker_modules:
            self.worker_modules[0].input_queue = self.input_modules[-1].output_queue
        if self.worker_modules and self.output_modules:
            self.output_modules[0].input_queue = self.worker_modules[-1].output_queue

    """
    Module management
    """
    def setup_modules(self) -> None:
        """
        Sets up modules.
        """
        self.stop = TEvent()
        for module in self.get_all():
            module.pause.clear()
            module.interrupt.clear()
            module.to_thread()

    def stop_modules(self) -> None:
        """
        Stops process modules.
        """
        self.stop.set()
        for module in self.get_all():
            module.reset(restart=False)

    def reset(self) -> None:
        """
        Sets up and resets handler. 
        """
        if self.logger:
            self.logger.info("(Re)setting pipeline...")
        self.stop_modules()
        gc.collect()
        self.setup_modules()
        if self.logger:
            self.logger.info("(Re)setting pipeline...")

    def get_all_threads(self) -> List[Thread]:
        """
        Returns all threads.
        :returns: List of threads.
        """
        return [module.thread for module in self.get_all()]

    def get_partition_and_index(self, module: PipelineModule) -> Tuple[List[PipelineModule], int] | None:
        """
        Fetches module partition list and module index. 
        :param module: Target module.
        :return: Partition (input, worker or output modules) and corresponding index.
            None, if module was not found in partitions.
        """
        for partition in [self.input_modules, self.worker_modules, self.output_modules]:
            if module in partition:
                return partition, partition.index(module)

    def reset_modules(self, modules: List[PipelineModule] | None = None) -> None:
        """
        Interrupts modules.
        :param modules: Modules to interrupt.
            Defaults to None in which case all modules are interrupted.
        """
        for module in self.get_all():
            if modules is None or module in modules:
                module.reset()
            
    def _run_nonblocking_pipeline(self, loop: bool) -> None:
        """
        Runs a non-blocking pipeline.
        :param loop: Declares, whether to loop pipeline or stop after a single interaction.
        """
        for module in self.worker_modules + self.input_modules:
            module.thread.start()         
        if not loop:
            while self.input_modules[0].output_queue.qsize() == 0 and self.input_modules[-1].output_queue.qsize() == 0:
                time.sleep(self.base_loop_pause/16)
            self.input_modules[0].pause.set()
            while self.worker_modules[-1].output_queue.qsize() == 0:
                time.sleep(self.base_loop_pause/16)
            while self.output_modules[-1].input_queue.qsize() > 0 or self.output_modules[-1].pause.is_set():
                time.sleep(self.base_loop_pause/16)
            self.reset()
    
    def _run_blocking_pipeline(self, loop: bool) -> None:
        """
        Runs a blocking pipeline.
        :param loop: Declares, whether to loop pipeline or stop after a single interaction.
        """
        # TODO: Trace inputs via PipelinePackage UUIDs
        while not self.stop.is_set():
            for module in self.input_modules:
                while not module.run():
                    time.sleep(self.base_loop_pause//16)
            for module in self.worker_modules:
                while not module.run():
                    time.sleep(self.base_loop_pause//16)
            while any(module.queues_are_busy() for module in self.output_modules):
                time.sleep(self.base_loop_pause//16)
            if not loop:
                self.stop.set()
        self.reset()

    def run_pipeline(self,
                     blocking: bool = True, 
                     loop: bool = True, 
                     greeting: str = "Hello there, how may I help you today?",
                     report: bool = False) -> None:
        """
        Runs pipeline.
        :param blocking: Declares, whether or not to wait for each step.
            Defaults to True.
        :param loop: Declares, whether to loop pipeline or stop after a single interaction.
            Defaults to True.
        :param greeting: Assistant greeting.
            Defaults to "Hello there, how may I help you today?".
        :param report: Flag for logging reports.
        """
        if self.logger:
            self.logger.info(f"Starting pipeline loop...")
        if report:
            self.run_report_thread()

        for output_module in self.output_modules:
            output_module.thread.start()
        if self.output_modules:
            self.output_modules[0].input_queue.put(PipelinePackage(content=greeting))

        try:
            if not blocking:
                self._run_nonblocking_pipeline(loop=loop)
            else:
                self._run_blocking_pipeline(loop=loop)
        except KeyboardInterrupt:
            if self.logger:
                self.logger.info(f"Received keyboard interrupt, shutting down handler ...")
            self.stop_modules()

    def run_report_thread(self) -> None:
        """
        Runs a thread for logging reports.
        """ 
        def log_report(wait_time: float = 15.0) -> None:
            while not self.stop.is_set():
                module_info = "\n".join([
                    "==========================================================",
                    f"#                {get_timestamp()}                       ",
                    f"#                {self}                                  ",
                    f"#                Running: {not self.stop.is_set()}       ",
                    "=========================================================="
                ])
                for partition in ["input_modules", "worker_modules", "output_modules"]:
                    module_info += f"\n{partition}"
                    for module in getattr(self, partition):
                        thread = module.thread
                        module_info += f"\n\t[{type(module).__name__}<{module.name}>] Thread '{thread}: {thread.is_alive()}'"
                        module_info += f"\n\t\t Inputs: {module.input_queue.qsize()}'"
                        module_info += f"\n\t\t Outputs: {module.output_queue.qsize()}'"
                        module_info += f"\n\t\t Received: {module.received}'"
                        module_info += f"\n\t\t Sent: {module.sent}'"
                        module_info += f"\n\t\t Pause: {module.pause.is_set()}'"
                        module_info += f"\n\t\t Interrupt: {module.interrupt.is_set()}'"
                if self.logger:
                    self.logger.info(module_info)
                else:
                    print(module_info)
                time.sleep(wait_time)
        thread = Thread(target=log_report)
        thread.daemon = True
        thread.start()    