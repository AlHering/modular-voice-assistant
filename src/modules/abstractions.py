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
            None and validation report in case of no implemented validation method. 
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


class PassivePipelineModule(PipelineModule):
    """
    Passive pipeline module.
    This module follows the same functionality as a conventional pipelineModule but forwards the incoming packages in its received state.
    It can be used for fetching and processing pipeline data without the results being fed back into the module pipeline.
    """
    @abstractmethod
    def process(self) -> PipelinePackage | Generator[PipelinePackage, None, None] | None:
        """
        Passive module processing method.
        :returns: The input data, as taken from the input queue.
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
                input_package: PipelinePackage = self.input_queue.get(block=True, timeout=self.input_timeout)
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
    - additional (passive) modules can be "inserted" into a pipeline to branch out operations, which do not reintroduce transformed data back into 
        the pipeline, e.g. for animating a character alongside the output module pipeline or running additional reporting.
    """
    def __init__(self,
                 input_modules: List[PipelineModule] = [],
                 worker_modules: List[PipelineModule] = [],
                 output_modules: List[PipelineModule] = [],
                 additional_modules: List[PassivePipelineModule] = [],
                 base_loop_pause: float = 0.1,
                 logger: Logger | None = None) -> None:
        """
        Initiation method.
        :param input_modules: Input modules.
        :param worker_modules: Worker modules. 
        :param output_modules: Output modules.
        :param additional_modules: Additional modules.
        :param base_loop_pause: Pipeline base loop pause.
        :param logger: Logger if logging is desired.
        """
        self.input_modules = input_modules
        self.worker_modules = worker_modules
        self.output_modules = output_modules
        self.additional_modules = additional_modules
        self.base_loop_pause = base_loop_pause
        self.logger = logger

        self.input_threads = None
        self.worker_threads = None
        self.output_threads = None
        self.additional_threads = None
        self.stop = None
        self.setup_modules()

    """
    Pipeline management
    """

    @classmethod
    def get_all(cls) -> List[PipelineModule]:
        """
        Returns all available modules.
        :returns: List of pipeline modules.
        """
        return cls.input_modules + cls.worker_modules + cls.output_modules + cls.additional_modules
    
    def reroute_pipeline_queues(self) -> List[PipelineModule]:
        """
        Reroutes queues between the pipelines.
        Note, that the queues of additional (passive) modules are not rerouted.
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
    def stop_modules(self) -> None:
        """
        Stops process modules.
        """
        self.stop.set()
        for module in self.get_all():
            module.interrupt.set()
            module.pause.set()
        for thread in self.get_all_threads():
            try:
                thread.join(.12) 
            except RuntimeError:
                pass

    def setup_modules(self) -> None:
        """
        Sets up modules.
        """
        self.stop = TEvent()
        for module in self.get_all():
            module.pause.clear()
            module.interrupt.clear()
        self.input_threads = [module.to_thread() for module in self.input_modules]
        self.worker_threads = [module.to_thread() for module in self.worker_modules]
        self.output_threads = [module.to_thread() for module in self.output_modules]
        self.additional_threads = [module.to_thread() for module in self.additional_modules]

    def get_all_threads(self) -> List[Thread]:
        """
        Returns all threads.
        :returns: List of threads.
        """
        res = []
        for threads in [self.input_threads, self.worker_threads, self.output_threads, self.additional_threads]:
            if threads is not None:
                res.extend(threads)
        return threads

    def stop_modules(self) -> None:
        """
        Stops process modules.
        """
        self.stop.set()
        for module in self.get_all():
            module.interrupt.set()
            module.pause.set()
        for thread in self.get_all_threads():
            try:
                thread.join(.12) 
            except RuntimeError:
                pass

    def reset(self) -> None:
        """
        Sets up and resets handler. 
        """
        if self.logger:
            self.logger.info("(Re)setting Conversation Handler...")
        self.stop_modules()
        gc.collect()
        self.setup_modules()
        if self.logger:
            self.logger.info("(Re)setting Conversation Handler...")

    def _run_nonblocking_pipeline(self, loop: bool) -> None:
        """
        Runs a non-blocking pipeline.
        :param loop: Declares, whether to loop pipeline or stop after a single interaction.
        """
        for thread in self.worker_threads + self.input_threads:
            thread.start()         
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

        for thread in self.output_threads:
            thread.start()
        if self.output_modules:
            self.output_modules[0].input_queue.put(PipelinePackage(content=greeting))

        try:
            if not blocking:
                self._run_nonblocking_conversation(loop=loop)
            else:
                self._run_blocking_conversation(loop=loop)
        except KeyboardInterrupt:
            if self.logger:
                self.logger.info(f"Received keyboard interrupt, shutting down handler ...")
            self.stop_modules()

    def run_report_thread(self) -> None:
        """
        Runs a thread for logging reports.
        """ 
        def log_report(wait_time: float = 10.0) -> None:
            while not self.stop.is_set():
                module_info = "\n".join([
                    "==========================================================",
                    f"#                    {get_timestamp()}                   ",
                    f"#                    {self}                              ",
                    f"#                Running: {not self.stop.is_set()}       ",
                    "=========================================================="
                ])
                for threads in ["input_threads", "worker_threads", "output_threads", "additional_threads"]:
                    for thread_index, thread in enumerate(getattr(self, threads)):
                        module = getattr(self, f"{threads.split('_')[0]}_modules")[thread_index]
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