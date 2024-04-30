# -*- coding: utf-8 -*-
"""
****************************************************
*                  EntityProtocol                  *
*            (c) 2020-2023 Alexander Hering        *
****************************************************
"""
import os
import sys
from typing import Any, Optional, List
from src.model.plugin_control.exceptions import PluginImportException, PluginRuntimeException
from src.utility.bronze import json_utility
from src.utility.silver import file_system_utility, environment_utility


class GenericPlugin(object):
    """
    Generic Plugin class.
    """

    def __init__(self, info: dict, path: str, security_hash: str = None, install_dependencies: bool = False) -> None:
        """
        Representation of a generic plugin.
        :param info: Plugin info.
        :param path: Path to plugin.
        :param security_hash: Hash that can be used for authentication purposes.
            Defaults to None.
        :param install_dependencies: Flag, declaring whether to dynamically install dependencies. Defaults to False.
            Only set this flag to True, if the code loaded through plugins is actively checked.
        """
        self.info = info
        self.path = path
        self.security_hash = security_hash
        self.name = self.info.get("name")
        self.type = self.info.get("type")
        self.dependencies = self.info.get("dependencies", [])
        for dependency in self.dependencies:
            if not environment_utility.check_module_availability(dependency) and (
                    not install_dependencies or environment_utility.safely_import_package(dependency) is None):
                raise PluginImportException(
                    self.name, self.type, f"Dependency '{dependency}' is not available!")
        if self.info.get("data_path"):
            if "./" in self.info["data_path"]:
                self.info["data_path"] = os.path.normpath(
                    os.path.join(path, self.info["data_path"]))
            file_system_utility.safely_create_path(self.info["data_path"])
        self._basic_validate()
        self.validate()

    def _basic_validate(self) -> None:
        """
        Internal method for validating plugin configuration.
        """
        if not isinstance(self.info, dict):
            raise PluginImportException(
                self.name, self.type, "Plugin configuration is no valid JSON.")
        if not os.path.exists(self.path):
            raise PluginImportException(
                self.name, self.type, "Plugin path is not valid.")
        if self.name is None:
            raise PluginImportException(
                "Unknown Plugin", self.type, "Plugin name not found.")
        if self.type is None:
            raise PluginImportException(
                self.name, "unknown type", "Plugin type not found.")

    def validate(self, *args: Optional[Any], **kwargs: Optional[Any]) -> None:
        """
        Method for validating plugin health.
        :param args: Arbitrary arguments.
        :param kwargs: Arbitrary keyword arguments.
        """
        pass

    def save(self, path: str = None) -> None:
        """
        Method for saving plugin info to file.
        :param path: Plugin path to safe plugin info file to, defaults to None in which case the import path is taken.
        """
        json_utility.save(self.info,
                          os.path.join(path, "info.json") if path is not None else os.path.join(self.path, "info.json"))
