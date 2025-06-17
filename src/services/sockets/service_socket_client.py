# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2025 Alexander Hering             *
****************************************************
"""
from __future__ import annotations


class ServiceRegistryClient(object):
    """
    Service registry client.
    """

    def __init__(self, socket_config: dict) -> None:
        """
        Initiation method.
        :param socket_config: Socket config.
        """
        self.socket_config = socket_config
