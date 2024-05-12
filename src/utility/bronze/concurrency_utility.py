# -*- coding: utf-8 -*-
"""
****************************************************
*                     Utility                      *
*            (c) 2023-2024 Alexander Hering        *
****************************************************
"""
import sys
from typing import Optional, Any, Callable, Union
from abc import ABC, abstractmethod
from uuid import uuid4
from queue import Empty, Queue as TQueue
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
