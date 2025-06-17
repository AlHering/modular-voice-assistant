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
import socket
from threading import Thread
from traceback import format_exc
from typing import Any, List, Callable
import json
from gc import collect as collect_garbage
from src.utility.time_utility import get_timestamp


SOCKET_BUFFER_SIZE = 4096


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
    Thread and process control
    """

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
        self.teardown()
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
            self.log_info(text="Restarting as thread...")
            if restart_thread:
                self.to_thread()
                self.thread.start()
        if restart_process:
            self.log_info(text="Restarting as process...")
            self.to_process()
            self.process.start()

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
                self.service.log_info(f"Failed to clean up service cache: {ex}\nTrace: {format_exc()}", as_warning=True)
                return False
        collect_garbage()
        return True

    @abstractmethod
    def run(self) -> ServicePackage | Generator[ServicePackage, None, None] | None:
        """
        Processes queued input.
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
    

def receive_from_socket(receiving_socket: socket.socket, encoding: str = "utf-8") -> str:
    """
    Retrieves data from socket.
    :param receiving_socket: Receiving socket.
    :param encoding: Data encoding.
        Defaults to utf-8.
    :return: Received data as string.
    """
    buffer = b""
    while True:
        part = receiving_socket.recv(SOCKET_BUFFER_SIZE)
        if not part:
            break
        buffer += part
        if b"\n" in part:
            break
    return buffer.decode(encoding=encoding).strip()


def interact_with_service_socket(host: str, port: int, package: ServicePackage) -> ServicePackage:
    """
    Sends a request service package to a service socket and returns response. 
    :param host: Socket server host.
    :param port: Socket server port.
    :param package: Package to send to service socket.
    """
    with socket.create_connection((host, port)) as sock:
        sock.sendall((json.dumps(package) + "\n").encode())
        data = receive_from_socket(receiving_socket=sock)
    return ServicePackage(**json.loads(data))


class SocketService(object):
    """
    Wrapper class for interacting with services via sockets.
    """
    def __init__(self, 
                 host: str,
                 port: int,
                 service: Service) -> None:
        """
        Initiates an instance.
        :param host: Socket server host.
        :param port: Socket server port.
        :param service: Service to wrap into socket interaction.
        """
        self.host = host
        self.port = port
        self.service = service
        self.server_socket = None
        self.connection_thread = None
        self.client_threads = []

    def setup_socket(self) -> bool:
        """
        Sets up socket and connection handling.
        :returns: True if setup was successful, else False.
        """
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.service.log_info(f"Listening on {self.host}:{self.port}.")
            self.connection_thread = Thread(target=self._connection_loop)
            self.connection_thread.daemon = True
            self.connection_thread.start()
            return True
        except Exception as ex:
            self.service.log_info(f"Failed to set up socket server: {ex}\nTrace: {format_exc()}", as_warning=True)
            return False
        
    def shutdown_socket(self) -> None:
        """
        Shuts down the socket server and closes connections.
        """
        self.service.interrupt.set()
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.host, self.port))
                s.close()
            self.server_socket.close()
        except Exception:
            pass

        self.connection_thread.join()
        for thread in self.client_threads:
            thread.join()

        self.client_threads.clear()
        self.connection_thread = None
        self.service.interrupt.clear()

    def _connection_loop(self) -> None:
        """
        Runs connection accept loop.
        """
        while not self.service.interrupt.is_set():
            try:
                client_socket, addr = self.server_socket.accept()
                self.service.log_info(f"Accepted connection from {addr}.")
                thread = Thread(target=self._handle_client, args=(client_socket,))
                thread.daemon = True
                thread.start()
                self.client_threads.append(thread)
            except socket.error as ex:
                self.service.log_info(f"Failed to set up socket server: {ex}\nTrace: {format_exc()}", as_warning=True)

    def _handle_client(self, client_socket: socket.socket) -> None:
        """
        Handles client interaction.
        :param client_socket: Client socket.
        """
        try:
            input_package = receive_from_socket(receiving_socket=client_socket)
            self.service.add_uuid(self.service.received, input_package.uuid)

            def send_back(package: ServicePackage):
                serialized = json.dumps(package.model_dump()) + "\n"
                client_socket.sendall(serialized.encode("utf-8"))

            self.service.input_queue.put(input_package)
            self._iterate(callback_function=send_back)
        except Exception as ex:
            client_socket.sendall(json.dumps({"error": str(ex), "trace": format_exc()}).encode("utf-8"))
        client_socket.close()

    def _iterate(self, callback_function: Callable) -> bool:
        """
        Runs a single processing cycle.
        :param callback_function: Callback function for returning results.
        :returns: True if an element was forwarded, else False. 
            (Note, that a service does not have to forward an element.)
        """    
        result = self.service.run()
        if result is not None:
            if isinstance(result, ServicePackage):
                callback_function(result)
                self.service.add_uuid(self.service.sent, elem.uuid)
                return True
            elif isinstance(result, Generator):
                elem = None
                for elem in result:
                    callback_function(elem)
                if elem is not None:
                    self.service.add_uuid(self.service.sent, elem.uuid)
                    return True
        return False
        