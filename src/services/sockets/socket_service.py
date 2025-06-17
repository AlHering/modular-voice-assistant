# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from __future__ import annotations
from typing import Generator
from pydantic import BaseModel
import socket
from threading import Thread
from traceback import format_exc
from typing import Callable
import json
from enum import Enum
from src.services.service_abstractions import Service, ServicePackage, FinalPackage, InterruptPackage, ResetPackage


SOCKET_BUFFER_SIZE = 4096


class PackageType(str, Enum):
    """
    Socket service package type.
    """
    service_package: str = "service_package"
    final_package: str = "final_package"
    interrupt_package: str = "interrupt_package"
    reset_package: str = "reset_package"


class SocketServicePackage(BaseModel):
    """
    Service package for exchanging data between services.
    """
    package_type: PackageType = PackageType.service_package
    package: ServicePackage | FinalPackage | InterruptPackage | ResetPackage
    

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
            client_socket.sendall("END\n".encode("utf-8"))
        except Exception as ex:
            client_socket.sendall(json.dumps({"error": str(ex), "trace": format_exc()}).encode("utf-8"))
        client_socket.close()

    def _iterate(self, callback_function: Callable) -> None:
        """
        Runs a single processing cycle.
        :param callback_function: Callback function for returning results.
        """
        self.service.iterate()
        new_response = self.service.output_queue.get()
        while not isinstance(new_response, FinalPackage):
            callback_function(new_response)
            new_response = self.service.output_queue.get()
        callback_function(new_response)


def send_service_request(host: str, port: int, request_package: ServicePackage) -> Generator[ServicePackage, None, None]:
    """
    Exemplary function for sending of a service request.
    :param host: Socket server host.
    :param port: Socket server port.
    :param request_package: Request service package.
    :return: Received package(s).
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((host, port))
        buffer = ""
        while True:
            data = sock.recv(SOCKET_BUFFER_SIZE).decode()
            if not data:
                break
            buffer += data
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if line == "END":
                    break
                if line.strip():
                    package_data = json.loads(line)
                    yield ServicePackage(**package_data)