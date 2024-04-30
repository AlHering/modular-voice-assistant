# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import sys
from typing import Optional, Any
from abc import ABC, abstractmethod
from uuid import uuid4
from queue import Empty, Queue as TQueue
from multiprocessing import Process, Queue as MPQueue, Event as mp_get_event
from multiprocessing.synchronize import Event as MPEvent
from threading import Thread, Event as TEvent
from src.utility.gold.text_generation.language_model_abstractions import LanguageModelInstance
from src.utility.bronze import dictionary_utility


def run_threaded_llm(switch: TEvent, llm_configuration: dict, input_queue: TQueue, output_queue: TQueue) -> None:
    """
    Function for running LLM instance in threading mode.
    :param switch: Pool killswitch event.
    :param llm_configuration: Configuration to instantiate LLM.
        Either containing an available template under the key "template" or containing valid instantiation parameters.
    :param input_queue: Input queue.
    :param output_queue: Output queue.
    """
    llm = LanguageModelInstance(**llm_configuration)
    while not switch.wait(0.5):
        output_queue.put(llm.generate(input_queue.get()))


def run_multiprocessed_llm(switch: MPEvent, llm_configuration: dict, input_queue: MPQueue, output_queue: MPQueue) -> None:
    """
    Function for running LLM instance in multiprocessing mode.
    :param switch: Pool killswitch event.
    :param llm_configuration: Configuration to instantiate LLM.
            Dictionary containing "model_path" and "model_config".
    :param input_queue: Input queue.
    :param output_queue: Output queue.
    """
    try:
        llm = LanguageModelInstance(**llm_configuration)
        while not switch.wait(0.5):
            output_queue.put(llm.generate(input_queue.get()))
    except:
        sys.exit(1)
    sys.exit(0)


class LLMPool(ABC):
    """
    Class for handling a pool of LLM instances.
    """

    def __init__(self, queue_spawns: bool = False, generation_timeout: float = None) -> None:
        """
        Initiation method.
        :param queue_spawns: Queue up instanciation until resources are available.
            Defaults to False.
        :param generation_timeout: Timeout for generation tasks.
            Defaults to None in which case the generation task potentially runs indefinitly.
            If set, a None value will be returned if the timeout value is passed.
        """
        # TODO: Add prioritization and potentially interrupt concept
        self.queue_spawns = queue_spawns
        self.generation_timeout = generation_timeout
        self.workers = {}

    def stop_all(self) -> None:
        """
        Method for stopping workers.
        """
        for worker_uuid in self.workers:
            self._unload_llm(worker_uuid)
            self.workers[worker_uuid]["running"] = False

    def stop(self, target_worker: str) -> None:
        """
        Method for stopping a worker.
        :param target_worker: Worker to stop.
        """
        if self.is_running(target_worker):
            self._unload_llm(target_worker)
            self.workers[target_worker]["running"] = False

    def start(self, target_worker: str) -> None:
        """
        Method for stopping a worker.
        :param target_worker: Worker to stop.
        """
        if not self.is_running(target_worker):
            self._load_llm(target_worker)
            self.workers[target_worker]["running"] = True

    def is_running(self, target_worker: str) -> bool:
        """
        Method for checking whether worker is running.
        :param target_worker: Worker to check.
        :return: True, if worker is running, else False.
        """
        return self.workers[target_worker]["running"]

    def validate_resources(self, llm_configuration: dict, queue_spawns: bool) -> bool:
        """
        Method for validating resources before LLM instantiation.
        :param llm_configuration: LLM configuration.
            Dictionary containing "model_path" and "model_config".
        :param queue_spawns: Queue up instanciation until resources are available.
            Defaults to False.
        :return: True, if resources are available, else False.
        """
        # TODO: Implement
        pass

    def reset_llm(self, target_worker: str, llm_configuration: dict) -> str:
        """
        Method for resetting LLM instance to a new config.
        :param target_worker: Worker of instance.
        :param llm_configuration: LLM configuration.
            Dictionary containing "model_path" and "model_config".
        :return: Worker UUID.
        """
        if not dictionary_utility.check_equality(self.workers[target_worker]["config"], llm_configuration):
            if self.workers[target_worker]["running"]:
                self._unload_llm(target_worker)
            self.workers[target_worker]["config"] = llm_configuration
        return target_worker

    def prepare_llm(self, llm_configuration: dict, given_uuid: str = None) -> str:
        """
        Method for preparing LLM instance.
        :param llm_configuration: LLM configuration.
        :param given_uuid: Given UUID to run worker under.
            Defaults to None in which case a new UUID is created.
        :return: Worker UUID.
        """
        uuid = uuid4() if given_uuid is None else given_uuid
        if uuid not in self.workers:
            self.workers[uuid] = {
                "config": llm_configuration,
                "running": False
            }
        else:
            self.reset_llm(uuid, llm_configuration)
        return uuid

    @abstractmethod
    def _load_llm(self, target_worker: str) -> None:
        """
        Internal method for loading LLM.
        :param target_worker: Worker to start.
        """
        pass

    @abstractmethod
    def _unload_llm(self, target_worker: str) -> None:
        """
        Internal method for unloading LLM.
        :param target_worker: Worker to stop.
        """
        pass

    @abstractmethod
    def generate(self, target_worker: str, prompt: str) -> Optional[Any]:
        """
        Request generation response for query from target LLM.
        :param target_worker: Target worker.
        :param prompt: Prompt to send.
        :return: Response.
        """
        pass


class ThreadedLLMPool(LLMPool):
    """
    Class for handling a pool of LLM instances in separated threads for leightweight non-blocking I/O.
    """

    def _load_llm(self, target_worker: str) -> None:
        """
        Internal method for loading LLM.
        :param target_worker: Worker to start.
        """
        self.workers[target_worker]["switch"] = TEvent()
        self.workers[target_worker]["input"] = TQueue()
        self.workers[target_worker]["output"] = TQueue()
        self.workers[target_worker]["worker"] = Thread(
            target=run_threaded_llm,
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

    def _unload_llm(self, target_worker: str) -> None:
        """
        Internal method for unloading LLM.
        :param target_worker: Worker to stop.
        """
        self.workers[target_worker]["switch"].set()
        self.workers[target_worker]["worker"].join(1)

    def generate(self, target_worker: str, prompt: str) -> Optional[Any]:
        """
        Request generation response for query from target LLM.
        :param target_worker: Target worker.
        :param prompt: Prompt to send.
        :return: Response.
        """
        self.workers[target_worker]["input"].put(prompt)
        try:
            return self.workers[target_worker]["output"].get(timeout=self.generation_timeout)
        except Empty:
            return None


class MulitprocessingLLMPool(LLMPool):
    """
    Class for handling a pool of LLM instances in separate processes for actual concurrency on heavy devices.
    """

    def _load_llm(self, target_worker: str) -> None:
        """
        Internal method for loading LLM.
        :param target_worker: Worker to start.
        """
        self.workers[target_worker]["switch"] = mp_get_event()
        self.workers[target_worker]["input"] = MPQueue()
        self.workers[target_worker]["output"] = MPQueue()
        self.workers[target_worker]["worker"] = Process(
            target=run_multiprocessed_llm,
            args=(
                self.workers[target_worker]["switch"],
                self.workers[target_worker]["config"],
                self.workers[target_worker]["input"],
                self.workers[target_worker]["output"],
            )
        )
        self.workers[target_worker]["worker"].start()
        self.workers[target_worker]["running"] = True

    def _unload_llm(self, target_worker: str) -> None:
        """
        Internal method for unloading LLM.
        :param target_worker: Worker to stop.
        """
        self.workers[target_worker]["switch"].set()
        self.workers[target_worker]["worker"].join(1)
        if self.workers[target_worker]["worker"].exitcode != 0:
            self.workers[target_worker]["worker"].kill()
        self.workers[target_worker]["running"] = False

    def generate(self, target_worker: str, prompt: str) -> Optional[Any]:
        """
        Request generation response for query from target LLM.
        :param target_worker: Target worker.
        :param prompt: Prompt to send.
        :return: Response.
        """
        self.workers[target_worker]["input"].put(prompt)
        try:
            return self.workers[target_worker]["output"].get(timeout=self.generation_timeout)
        except Empty:
            return None
