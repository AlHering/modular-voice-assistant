# -*- coding: utf-8 -*-
"""
****************************************************
*                  EntityProtocol                  *
*            (c) 2020-2023 Alexander Hering        *
****************************************************
"""
import os
import logging
import traceback
from typing import Tuple, Optional, Any, List
from src.configuration import paths
from src.model.plugin_control.exceptions import PluginImportException
from src.model.plugin_control.plugins import GenericPlugin
from src.utility.bronze import json_utility
from src.utility.silver import file_system_utility


class PluginController(object):
    """
    Class, representing Plugin Controller objects.
    """
    def __init__(self, plugin_class_dictionary: dict, plugin_folders: List[str] = None, security_hashes: dict = None,
                 supported_types: List[str] = None, ignore_failed_imports: bool = True) -> None:
        """
        Plugin controller for importing and managing plugins.
        :param plugin_class_dictionary: Dictionary, mapping plugin type to plugin class.
        :param plugin_folders: Plugin folder list. 
            Defaults to None, in which case local plugin folder is used.
        :param security_hashes: Security hash of dynamically imported files under security_hashes[<plugin type>][<plugin name>].
            Defaults to None, in which case hashes are not checked!
        :param supported_types: Allowed types. Defaults to None in which case all types are allowed.
        :param ignore_failed_imports: Flag declaring whether to raise Exceptions or to just ignore failed imports.
            Exceptions are still logged, if ignore-flag is set.
        """
        self._logger = logging.Logger("[PluginController]")
        self._logger.info("initiating ...")

        self.plugin_classes = plugin_class_dictionary
        self.security_hashes = security_hashes
        self.ignore_failed_imports = ignore_failed_imports
        self.supported_types = supported_types
        if plugin_folders:
            self.plugin_folders = plugin_folders
        else:
            self.plugin_folders = [folder for folder in paths.PLUGIN_FOLDERS]
        self.plugin_paths = None
        self.plugins = None
        self._cache = {
            "dynmically_loaded": []
        }
        self._logger.info("importing plugins ...")

        self.reload()
        self._logger.info("processed plugins!")

    def reload(self) -> None:
        """
        Method for reloading all paths
        """
        self.plugin_paths = []
        self.plugins = {}
        for plugin_folder in self.plugin_folders:
            self.reload_from_path(plugin_folder)
        for plugin_path in self._cache["dynmically_loaded"]:
            self.dynamically_load_plugin(plugin_path)

    def reload_from_path(self, path: str) -> None:
        """
        Method for loading in plugins from plugin folders.
        :param path: Plugin folder path to load in.
        """
        for root, dirs, files in os.walk(path):
            for folder in dirs:
                info_path = file_system_utility.clean_path(
                    os.path.join(root, folder, "info.json"))
                if os.path.isfile(info_path):
                    self._logger.info(f"found plugin at {info_path}")
                    self.plugin_paths.append(
                        file_system_utility.clean_path(os.path.join(root, folder)))
                    plugin_info = json_utility.load(info_path)

                    self._logger.info(
                        f"collecting {plugin_info['type']}-plugin: {plugin_info['name']} ...")
                    self.import_plugin(plugin_info)
            break

    def import_plugin(self, plugin_info: dict) -> Optional[Any]:
        """
        Import separate plugin types and create representation.
        :param plugin_info: Info dictionary of plugin to be imported.
        :return: Plugin instance (which is also added to local plugin storage).
        """
        # Check import premises and select class wrapper
        if self.supported_types is not None and plugin_info["type"] not in self.supported_types:
            self._logger.info(
                f"not importing {plugin_info['name']} due to type limitations.")
            return None
        elif " " in plugin_info["name"]:
            self._logger.warning(
                f"illegal name found for {plugin_info['name']}.")
            if not self.ignore_failed_imports:
                raise PluginImportException(
                    plugin_info["name"], plugin_info["type"], "Plugin name contains spaces!")
        elif plugin_info["type"] in self.plugin_classes:
            wrapper = self.plugin_classes[plugin_info["type"]]
        else:
            wrapper = GenericPlugin

        # compute security hash
        security_hash = None if self.security_hashes is None else self.security_hashes.get(
            plugin_info["type"], {}).get(plugin_info["name"], None)

        # create, catalogue and return plugin instance
        try:
            plugin = wrapper(
                plugin_info, self.plugin_paths[-1], security_hash)
            if plugin_info["type"] in self.plugins:
                if plugin_info["name"] in self.plugins[plugin_info["type"]]:
                    self._logger.warning(
                        f"Name {plugin_info['name']} already registered and will be overwritten")
                self.plugins[plugin_info["type"]][plugin_info["name"]] = plugin
            else:
                self.plugins[plugin_info["type"]] = {
                    plugin_info["name"]: plugin}
        except PluginImportException as ex:
            if self.ignore_failed_imports:
                self._logger.warning(
                    f"Error occured while importing {plugin_info['name']}.\n{traceback.format_exc}")
            else:
                raise ex
        return plugin

    def save_plugin_info(self, plugin_type: str = None, plugin_name: str = None) -> None:
        """
        Method for saving plugin info files back to disk.
        :param plugin_type: Plugin type of target plugin. Defaults to None in which case all types are potentially saved.
        :param plugin_name: Plugin name of target plugin. Defaults to None in which case all names are potentially saved.
        """
        if plugin_type is None:
            for plugin_type in self.plugins:
                self.save_plugin_info(plugin_type, plugin_name)
        elif plugin_name is None:
            for plugin_name in self.plugins[plugin_type]:
                self.save_plugin_info(plugin_type, plugin_name)
        else:
            self.plugins[plugin_type][plugin_name].save()

    def get_plugin(self, plugin_type: str = None, plugin_name: str = None) -> Any:
        """
        Method for getting plugin instance.
        :param plugin_type: Plugin type of target plugin.
        :param plugin_name: Plugin name of target plugin.
        """
        return self.plugins.get(plugin_type, {}).get(plugin_name)

    def dynamically_load_plugin_folder(self, plugin_folder: str) -> Optional[dict]:
        """
        Method for dynamically loading additional plugins from a plugin folder at runtime, not loaded via configured
        plugin paths.
        :param plugin_folder: Plugin folder path.
        :return: List of tuples of plugin types and names.
        """
        if os.path.exists(plugin_folder) and plugin_folder not in self.plugin_folders:
            return self.reload_from_path(plugin_folder)
        else:
            return None

    def dynamically_load_plugin(self, plugin_path: str) -> Optional[Tuple[str]]:
        """
        Method for dynamically loading additional plugins at runtime, not loaded via configured plugin paths.
        :param plugin_path: Plugin folder path.
        :return: Tuple of plugin type and name.
        """
        plugin_info_path = os.path.join(plugin_path, "info.json")
        if os.path.exists(plugin_info_path) and plugin_path not in self.plugin_paths:
            self.plugin_paths.append(plugin_path)
            if plugin_path not in self._cache["dynmically_loaded"]:
                self._cache["dynmically_loaded"].append(plugin_path)
            return self.import_plugin(json_utility.load(plugin_info_path))
        else:
            return None
