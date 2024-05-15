# -*- coding: utf-8 -*-
"""
****************************************************
*                     Utility                      *
*            (c) 2023-2024 Alexander Hering        *
****************************************************
"""
import sys
from typing import Optional, Any, Callable, Union, Dict, Generator
import time
import functools
from abc import ABC, abstractmethod
from uuid import uuid4
from queue import Empty, Queue as TQueue
import multiprocessing
from multiprocessing import Process, Queue as MPQueue, Event as mp_get_event
from multiprocessing.synchronize import Event as MPEvent
from threading import Thread, Event as TEvent
from src.utility.bronze import dictionary_utility


class Task(ABC):
    """
    Class for representing a worker's task.
    """

    @classmethod 
    @abstractmethod
    def run(switch: Union[TEvent, MPEvent], configuration: dict, input_queue: Union[TQueue, MPQueue], output_queue: Union[TQueue, MPQueue]) -> None:
        """
        Task running method.
        :param switch: Pool killswitch event.
        :param configuration: Configuration for the process.
        :param input_queue: Input queue.
        :param output_queue: Output queue.
        """
        pass


class WorkerPool(ABC):
    """
    Class for handling a pool of worker instances.
    """

    def __init__(self, task: Task, queue_spawns: bool = False, processing_timeout: float = None) -> None:
        """
        Initiation method.
        :param task: Task which implements the run-method.
        :param queue_spawns: Queue up instanciation until resources are available.
            Defaults to False.
        :param processing_timeout: Timeout for processing tasks.
            Defaults to None in which case the processing task potentially runs indefinitly.
            If set, a None value will be returned if the timeout value is passed.
        """
        # TODO: Add prioritization and potentially interrupt concept
        self.task = task
        self.queue_spawns = queue_spawns
        self.processing_timeout = processing_timeout
        self.workers = {}

    def stop_all(self) -> None:
        """
        Method for stopping workers.
        """
        for worker_uuid in self.workers:
            self._unload_worker(worker_uuid)
            self.workers[worker_uuid]["running"] = False

    def stop(self, target_worker: str) -> None:
        """
        Method for stopping a workers.
        :param target_worker: Worker to stop.
        """
        if self.is_running(target_worker):
            self._unload_worker(target_worker)
            self.workers[target_worker]["running"] = False

    def start(self, target_worker: str) -> None:
        """
        Method for stopping a workers.
        :param target_worker: Worker to stop.
        """
        if not self.is_running(target_worker):
            self._load_worker(target_worker)
            self.workers[target_worker]["running"] = True

    def is_running(self, target_worker: str) -> bool:
        """
        Method for checking whether worker is running.
        :param target_worker: Worker to check.
        :return: True, if worker is running, else False.
        """
        return self.workers[target_worker]["running"]

    def validate_resources(self, worker_configuration: dict, queue_spawns: bool) -> bool:
        """
        Method for validating resources before worker instantiation.
        :param worker_configuration: Worker configuration.
            Dictionary containing "model_path" and "model_config".
        :param queue_spawns: Queue up instanciation until resources are available.
            Defaults to False.
        :return: True, if resources are available, else False.
        """
        # TODO: Implement
        pass

    def reset_worker(self, target_worker: str, worker_configuration: dict) -> str:
        """
        Method for resetting worker instance to a new config.
        :param target_worker: Worker of instance.
        :param worker_configuration: Worker configuration.
            Dictionary containing "model_path" and "model_config".
        :return: Worker UUID.
        """
        if not dictionary_utility.check_equality(self.workers[target_worker]["config"], worker_configuration):
            if self.workers[target_worker]["running"]:
                self._unload_worker(target_worker)
            self.workers[target_worker]["config"] = worker_configuration
        return target_worker

    def prepare_worker(self, worker_configuration: dict, given_uuid: str = None) -> str:
        """
        Method for preparing worker instance.
        :param worker_configuration: Worker configuration.
        :param given_uuid: Given UUID to run worker under.
            Defaults to None in which case a new UUID is created.
        :return: Worker UUID.
        """
        uuid = uuid4() if given_uuid is None else given_uuid
        if uuid not in self.workers:
            self.workers[uuid] = {
                "config": worker_configuration,
                "running": False
            }
        else:
            self.reset_worker(uuid, worker_configuration)
        return uuid

    @abstractmethod
    def _load_worker(self, target_worker: str) -> None:
        """
        Internal method for loading worker.
        :param target_worker: Worker to start.
        """
        pass

    @abstractmethod
    def _unload_worker(self, target_worker: str) -> None:
        """
        Internal method for unloading worker.
        :param target_worker: Worker to stop.
        """
        pass

    @abstractmethod
    def process(self, target_worker: str, prompt: str) -> Optional[Any]:
        """
        Request processing response for query from target worker.
        :param target_worker: Target worker.
        :param prompt: Prompt to send.
        :return: Response.
        """
        pass


class ThreadedWorkerPool(WorkerPool):
    """
    Class for handling a pool of worker instances in separated threads for leightweight non-blocking I/O.
    """

    def _load_worker(self, target_worker: str) -> None:
        """
        Internal method for loading worker.
        :param target_worker: Worker to start.
        """
        self.workers[target_worker]["switch"] = TEvent()
        self.workers[target_worker]["input"] = TQueue()
        self.workers[target_worker]["output"] = TQueue()
        self.workers[target_worker]["worker"] = Thread(
            target=self.task.run,
            args=(
                self.workers[target_worker]["switch"],
                self.workers[target_worker]["config"],
                self.workers[target_worker]["input"],
                self.workers[target_worker]["output"],
            )
        )
        self.workers[target_worker]["worker"].daemon = True
        self.workers[target_worker]["worker"].start()
        self.workers[target_worker]["running"] = True

    def _unload_worker(self, target_worker: str) -> None:
        """
        Internal method for unloading worker.
        :param target_worker: Worker to stop.
        """
        self.workers[target_worker]["switch"].set()
        self.workers[target_worker]["worker"].join(1)

    def process(self, target_worker: str, prompt: str) -> Optional[Any]:
        """
        Request processing response for query from target worker.
        :param target_worker: Target worker.
        :param prompt: Prompt to send.
        :return: Response.
        """
        self.workers[target_worker]["input"].put(prompt)
        try:
            return self.workers[target_worker]["output"].get(timeout=self.processing_timeout)
        except Empty:
            return None


class MulitprocessingWorkerPool(WorkerPool):
    """
    Class for handling a pool of worker instances in separate processes for actual concurrency on capable devices.
    """

    def _load_worker(self, target_worker: str) -> None:
        """
        Internal method for loading worker.
        :param target_worker: Worker to start.
        """
        self.workers[target_worker]["switch"] = mp_get_event()
        self.workers[target_worker]["input"] = MPQueue()
        self.workers[target_worker]["output"] = MPQueue()
        self.workers[target_worker]["worker"] = Process(
            target=self.task.run,
            args=(
                self.workers[target_worker]["switch"],
                self.workers[target_worker]["config"],
                self.workers[target_worker]["input"],
                self.workers[target_worker]["output"],
            )
        )
        self.workers[target_worker]["worker"].start()
        self.workers[target_worker]["running"] = True

    def _unload_worker(self, target_worker: str) -> None:
        """
        Internal method for unloading worker.
        :param target_worker: Worker to stop.
        """
        self.workers[target_worker]["switch"].set()
        self.workers[target_worker]["worker"].join(1)
        if self.workers[target_worker]["worker"].exitcode != 0:
            self.workers[target_worker]["worker"].kill()
        self.workers[target_worker]["running"] = False

    def process(self, target_worker: str, prompt: str) -> Optional[Any]:
        """
        Request processing response for query from target worker.
        :param target_worker: Target worker.
        :param prompt: Prompt to send.
        :return: Response.
        """
        self.workers[target_worker]["input"].put(prompt)
        try:
            return self.workers[target_worker]["output"].get(timeout=self.processing_timeout)
        except Empty:
            return None


class PipelineComponentThread(Thread):
    """
    Represents a threaded pipeline component.
    """
    def __init__(self, 
                 pipeline_function: Callable,
                 input_queue: TQueue = None,
                 output_queue: TQueue = None,
                 interrupt: TEvent = None,
                 loop_pause: float = .1,
                 validation_function: Callable = None,
                 *thread_args: Optional[Any], 
                 **thread_kwargs: Optional[Any]) -> None:
        """
        Initiation method.
        :param pipeline_function: Pipeline function.
        :param input_queue: Input queue.
        :param output_queue: Output queue.
        :param interrupt: Interrupt event.
        :param loop_pause: Processing loop pause.
            Defaults to 0.1 seconds.
        :param validation_function: Validation function for ingoing data.
            Defaults to None in which case the pipeline function should check for valid inputs.
        :param thread_args: Thread constructor arguments.
        :param thread_kwargs: Thread constructor keyword arguments.
        """
        super().__init__(*thread_args, **thread_kwargs)
        self.busy = TEvent()
        self.pipeline_function = pipeline_function
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.interrupt = TEvent() if interrupt is None else interrupt
        self.loop_pause = loop_pause
        self.validation_function = validation_function

    def run(self) -> None:
        """
        Main runner method.
        """
        while not self.interrupt.is_set():
            try:
                res = None
                if self.input_queue is None:
                    self.busy.set()
                    res = self.pipeline_function()
                else:
                    input_data = self.input_queue.get(self.loop_pause/8)
                    if self.validation_function is None or self.validation_function(input_data):
                        self.busy.set()
                        res = self.pipeline_function(input_data)
                if self.output_queue is not None:
                    if isinstance(res, Generator):
                        for elem in res:
                            self.output_queue.put(elem)
                            
                    else:
                        self.output_queue.put(res)
                self.busy.clear()
                time.sleep(self.loop_pause*7/8)
            except Empty:
                time.sleep(self.loop_pause)


class Pipeline(object):
    """
    Represents a pipeline of threaded components.
    """
    def __init__(self, 
                 component_functions: Dict[str, Callable],
                 links: Dict[str, str],
                 validation_functions: Dict[str, Callable] = {},
                 loop_pause: float = .1
                 ) -> None:
        """
        Initiation method.
        :param component_functions: Dictionary of component functions with component name as key.
        :param links: Dictionary of links that connect the output of the key component to the value component.
            Note, that the "*" component resembles a global input/output for the pipeline.
        :param loop_pause: Processing loop pause.
            Defaults to 0.1 seconds.
        :param validation_functions: Dictionary of validation functions with component name as key.
        """
        self.loop_pause = loop_pause
        self.interrupts = {
            key: TEvent() for key in [key for key in component_functions if key != "*"]
        }
        self.input_queues = {}
        self.output_queues = {}

        for link in links:
            q = TQueue()
            self.input_queues[link] = q
            self.output_queues[links[link]] = q
            
        self.threads = {}
        for component in component_functions:
            self.threads[component] = PipelineComponentThread(
                pipeline_function=component_functions[component],
                input_queue=self.input_queues.get(component),
                output_queue=self.output_queues.get(component),
                interrupt=self.interrupts[component],
                loop_pause=self.loop_pause,
                validation_function=validation_functions.get(component)
            )
            self.threads[component].daemon = True

    def start(self, component: str = None) -> None:
        """
        Starts component thread(s).
        :param component: Specific component to start thread for.
            Defaults to None in which case all components are started.
        """
        if component is None:
            for component in self.threads:
                self.threads[component].start()
        else:
            self.threads[component].start()

    def stop(self, component: str = None) -> None:
        """
        Stops component thread(s).
        :param component: Specific component to stop thread for.
            Defaults to None in which case all components are stopped.
        """
        if component is None:
            for component in self.threads:
                self.interrupts[component].set()
                self.threads[component].join(1)
        else:
            self.interrupts[component].set()
            self.threads[component].join(1)

    def put(self, input_data: Any) -> None:
        """
        Puts data into the global input queue.
        :param input_data: Input data.
        """
        if "*" in self.input_queues:
            self.input_queues["*"].put(input_data)

    def get(self) -> Optional[Any]:
        """
        Retrieves data from the global output queue.
        :return: Global output queue element or None if empty.
        """
        if "*" in self.output_queues:
            try:
                self.output_queues["*"].get(self.loop_pause)
            except Empty:
                return None


def timeout(max_timeout: float) -> Any:
    """
    Timeout decorator, parameter in seconds.
    Taken from https://stackoverflow.com/questions/492519/timeout-on-a-function-call.
    :param max_timeout: Maximum timeout.
    :raises multiprocessing.context.TimeoutError: In case of timeout.
    :returns: Return value of the decorated callable.
    """
    def timeout_decorator(item):
        """Wrap the original function."""
        @functools.wraps(item)
        def func_wrapper(*args: Optional[Any], **kwargs: Optional[Any]):
            """Closure for function."""
            pool = multiprocessing.pool.ThreadPool(processes=1)
            async_result = pool.apply_async(item, args, kwargs)
            # raises a TimeoutError if execution exceeds max_timeout
            return async_result.get(max_timeout)
        
        return func_wrapper
    return timeout_decorator