# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from typing import Tuple, List
import os
import gc
import time
from src.configuration import configuration as cfg
from threading import Thread, Event as TEvent
from src.utility.time_utility import get_timestamp
from src.utility.string_utility import separate_pattern_from_text, extract_matches_between_bounds, remove_multiple_spaces, EMOJI_PATTERN
from src.modules.abstractions import BaseModuleSet, VAPackage


class ModularConversationHandler(object):
    """
    Represents a modular conversation handler for handling audio based interaction.
    A conversation handler manages the following modules:
        - speech_recorder: A recorder for spoken input.
        - transcriber: A transcriber to transcribe spoken input into text.
        - worker: A worker to compute an output for the given user input.
        - synthesizer: A synthesizer to convert output texts to sound.
    """

    def __init__(self, 
                 working_directory: str,
                 module_set: BaseModuleSet,
                 loop_pause: float = 0.1) -> None:
        """
        Initiation method.
        :param working_directory: Directory for productive files.
        :param module_set: Module set.
        :param loop_pause: Loop pause.
        """
        cfg.LOGGER.info("Initiating Conversation Handler...")
        self.working_directory = working_directory
        os.makedirs(self.working_directory, exist_ok=True)
        self.loop_pause = loop_pause

        self.module_set = module_set
        self.input_threads = None
        self.worker_threads = None
        self.output_threads = None
        self.additional_threads = None
        self.stop = None
        self.setup_modules()

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
        for module in self.module_set.get_all():
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
        for module in self.module_set.get_all():
            module.pause.clear()
            module.interrupt.clear()
        self.input_threads = [module.to_thread() for module in self.module_set.input_modules]
        self.worker_threads = [module.to_thread() for module in self.module_set.worker_modules]
        self.output_threads = [module.to_thread() for module in self.module_set.output_modules]
        self.additional_threads = [module.to_thread() for module in self.module_set.additional_modules]

    def reset(self) -> None:
        """
        Sets up and resets handler. 
        """
        cfg.LOGGER.info("(Re)setting Conversation Handler...")
        self.stop_modules()
        gc.collect()
        self.setup_modules()
        cfg.LOGGER.info("Reset is done.")

    def _run_nonblocking_conversation(self, loop: bool) -> None:
        """
        Runs a non-blocking conversation.
        :param loop: Declares, whether to loop conversation or stop after a single interaction.
        """
        for thread in self.worker_threads + self.input_threads:
            thread.start()         
        if not loop:
            while self.module_set.input_modules[0].output_queue.qsize() == 0 and self.module_set.input_modules[-1].output_queue.qsize() == 0:
                time.sleep(self.loop_pause/16)
            self.module_set.input_modules[0].pause.set()
            while self.module_set.worker_modules[-1].output_queue.qsize() == 0:
                time.sleep(self.loop_pause/16)
            while self.module_set.output_modules[-1].input_queue.qsize() > 0 or self.module_set.output_modules[-1].pause.is_set():
                time.sleep(self.loop_pause/16)
            self.reset()
    
    def _run_blocking_conversation(self, loop: bool) -> None:
        """
        Runs a blocking conversation.
        :param loop: Declares, whether to loop conversation or stop after a single interaction.
        """
        # TODO: Trace inputs via VAPackage UUIDs
        while not self.stop.is_set():
            for module in self.module_set.input_modules:
                while not module.run():
                    time.sleep(self.loop_pause//16)
            for module in self.module_set.worker_modules:
                while not module.run():
                    time.sleep(self.loop_pause//16)
            while any(module.queues_are_busy() for module in self.module_set.output_modules):
                time.sleep(self.loop_pause//16)
            if not loop:
                self.stop.set()
        self.reset()

    def run_conversation(self, 
                         blocking: bool = True, 
                         loop: bool = True, 
                         greeting: str = "Hello there, how may I help you today?",
                         report: bool = False) -> None:
        """
        Runs conversation.
        :param blocking: Declares, whether or not to wait for each step.
            Defaults to True.
        :param loop: Declares, whether to loop conversation or stop after a single interaction.
            Defaults to True.
        :param greeting: Assistant greeting.
            Defaults to "Hello there, how may I help you today?".
        :param report: Flag for logging reports.
        """
        cfg.LOGGER.info(f"Starting conversation loop...")
        if report:
            self.run_report_thread()

        for thread in self.output_threads:
            thread.start()
        if self.module_set.output_modules:
            self.module_set.output_modules[0].input_queue.put(VAPackage(content=greeting))

        try:
            if not blocking:
                self._run_nonblocking_conversation(loop=loop)
            else:
                self._run_blocking_conversation(loop=loop)
        except KeyboardInterrupt:
            cfg.LOGGER.info(f"Received keyboard interrupt, shutting down handler ...")
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
                        module = getattr(self.module_set, f"{threads.split('_')[0]}_modules")[thread_index]
                        module_info += f"\n\t[{type(module).__name__}<{module.name}>] Thread '{thread}: {thread.is_alive()}'"
                        module_info += f"\n\t\t Inputs: {module.input_queue.qsize()}'"
                        module_info += f"\n\t\t Outputs: {module.output_queue.qsize()}'"
                        module_info += f"\n\t\t Received: {module.received}'"
                        module_info += f"\n\t\t Sent: {module.sent}'"
                        module_info += f"\n\t\t Pause: {module.pause.is_set()}'"
                        module_info += f"\n\t\t Interrupt: {module.interrupt.is_set()}'"
                cfg.LOGGER.info(module_info)
                time.sleep(wait_time)
        thread = Thread(target=log_report)
        thread.daemon = True
        thread.start()    
