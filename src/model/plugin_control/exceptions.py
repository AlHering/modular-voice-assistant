# -*- coding: utf-8 -*-
"""
****************************************************
*                  EntityProtocol                  *
*            (c) 2020-2023 Alexander Hering        *
****************************************************
"""


class PluginImportException(Exception):
    """
    PluginImportException class.
    """
    def __init__(self, plugin: str, plugin_type: str,  message: str = "exception occurred while importing plugin ") -> None:
        """
        Initiation method for Plugin Import Exception.
        :param plugin: Plugin name, that caused exception
        :param plugin_type: Plugin type.
        :param message: Message to include in exception.
        """
        self.plugin = plugin
        self.plugin_type = plugin_type
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        """
        Method for encoding exception as string.
        :return: Exception string.
        """
        return f"{self.message} : {self.plugin} as {self.plugin_type}"


class PluginRuntimeException(Exception):
    """
    PluginRuntimeException class.
    """
    def __init__(self, plugin: str, plugin_type: str,  message: str = "exception occurred while using plugin ") -> None:
        """
        Initiation method for Exception.
        :param plugin: Plugin name, that caused exception
        :param plugin_type: Plugin type.
        :param message: Message to include in exception.
        """
        self.plugin = plugin
        self.plugin_type = plugin_type
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        """
        Method for encoding exception as string.
        :return: Exception string.
        """
        return f"{self.message} : {self.plugin} as {self.plugin_type}"
    