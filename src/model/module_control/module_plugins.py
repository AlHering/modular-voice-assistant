# -*- coding: utf-8 -*-
"""
****************************************************
*             Modular Voice Assistant              *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
from src.model.plugin_control.plugins import GenericPlugin, PluginImportException, PluginRuntimeException


class TextGenerationPlugin(GenericPlugin):
    """
    Text Generation Plugin class.
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
        super().__init__(info=info, path=path, security_hash=security_hash, install_dependencies=install_dependencies)


    