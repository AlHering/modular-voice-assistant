# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os
import requests
import json
import copy
from time import sleep
from urllib.parse import urlparse
import shutil
from typing import Any, Optional, List, Tuple
from src.utility.bronze import time_utility, json_utility, requests_utility
from src.utility.silver import image_utility, internet_utility, file_system_utility, environment_utility
from src.model.plugin_control.plugins import GenericPlugin, PluginImportException
from src.configuration import configuration as cfg
import abc


class AbstractAPIWrapper(abc.ABC):
    """
    Abstract class, representing a API wrapper object.
    Such wrappers are used for connecting to model services.
    """

    def __init__(self, access_token: str = None) -> None:
        """
        Initiation method.
        :param access_token: Access token, if necessary.
            Defaults to None.
        """
        self.access_token = access_token

    @abc.abstractclassmethod
    def get_source_name(cls) -> str:
        """
        Abstract classmethod for retrieving source name.
        :return: Source name.
        """
        pass

    def check_connection(self, **kwargs: Optional[dict]) -> bool:
        """
        Abstract method for checking connection.
        :param kwargs: Arbitrary keyword arguments.
        :return: True if connection was established successfuly else False.
        """
        pass

    @abc.abstractmethod
    def validate_url_responsiblity(self, url: str, **kwargs: Optional[dict]) -> bool:
        """
        Abstract method for validating the responsiblity for a URL.
        :param url: Target URL.
        :param kwargs: Arbitrary keyword arguments.
        :return: True, if wrapper is responsible for URL else False.
        """
        pass

    @abc.abstractmethod
    def scrape_available_targets(self, target_type: str, **kwargs: Optional[dict]) -> List[dict]:
        """
        Abstract method for acquring available targets.
        :param target_type: Type of target object.
        :param kwargs: Arbitrary keyword arguments.
        :return: List of entries of given target type.
        """
        pass

    @abc.abstractmethod
    def get_api_url(self, target_type: str, target_object: Any, **kwargs: Optional[dict]) -> str:
        """
        Abstract method for acquring API URL for a given object.
        :param target_type: Type of target object.
        :param target_object: Target object.
        :param kwargs: Arbitrary keyword arguments.
        :return: API URL for given object.
        """
        pass

    @abc.abstractmethod
    def collect_metadata(self, target_type: str, target_object: Any, **kwargs: Optional[dict]) -> dict:
        """
        Abstract method for acquring model data by target type and object.
        :param target_type: Type of target object.
        :param target_object: Target object.
        :param kwargs: Arbitrary keyword arguments.
        :return: Metadata for given model ID.
        """
        pass

    @abc.abstractmethod
    def normalize_metadata(self, target_type: str, metadata: dict, **kwargs: Optional[dict]) -> dict:
        """
        Abstract method for normalizing metadata.
        :param target_type: Type of target object.
        :param metadata: Metadata.
        :param kwargs: Arbitrary keyword arguments.
        :return: Normalized metadata.
        """
        pass

    @abc.abstractmethod
    def download_model(self, model: Any, path: str, **kwargs: Optional[dict]) -> None:
        """
        Abstract method for downloading a model.
        :param model: Model object.
        :param path: Path to download assets to.
        :param kwargs: Arbitrary keyword arguments.
        """
        pass

    @abc.abstractmethod
    def download_modelversion(self, modelversion: Any, path: str, **kwargs: Optional[dict]) -> None:
        """
        Abstract method for downloading a model.
        :param modelversion: Model version object.
        :param path: Path to download assets to.
        :param kwargs: Arbitrary keyword arguments.
        """
        pass

    @abc.abstractmethod
    def download_assets(self, target_type: str, target_object: Any, asset_type: str, path: str, **kwargs: Optional[dict]) -> List[dict]:
        """
        Abstract method for downloading assets for a target object.
        :param target_type: Type of target object.
        :param target_object: Target object.
        :param asset_type: Asset type.
        :param path: Path to download assets to.
        :param kwargs: Arbitrary keyword arguments.
        :return: Asset data dictionaries.
        """
        pass


class APIWrapperPlugin(GenericPlugin):
    """
    Plugin class for API wrappers.
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
        self.module = None
        super().__init__(info, path, security_hash, install_dependencies)

    # Override
    def validate(self) -> None:
        """
        Method for validating plugin health.
        """
        module_path = self.info.get("module_path", os.path.join(
            self.path, "wrapper.py"))
        if module_path.startswith("./"):
            module_path = os.path.join(self.path, module_path[2:])
        if not os.path.exists(module_path):
            self.module = environment_utility.get_module(module_path)
            if not hasattr(self.module, "spawn_wrapper"):
                raise PluginImportException(
                    self.name, self.type, "Wrapper module has no 'spawn_wrapper' function")
        else:
            raise PluginImportException(
                self.name, self.type, f"Wrapper module not available under '{module_path}'")

    def spawn_wrapper(self, access_token: str = None) -> AbstractAPIWrapper:
        """
        Method for spawning API wrapper.
        :param access_token: Access token.
            Defaults to None.
        """
        return self.module.spawn_wrapper(access_token)


class CivitaiAPIWrapper(AbstractAPIWrapper):
    """
    Class, representing civitai API wrapper.
    """

    def __init__(self) -> None:
        """
        Initiation method.
        """
        self._logger = cfg.LOGGER
        self.authorization = cfg.ENV["CIVITAI_API_KEY"]
        self.headers = {"Authorization": self.authorization}
        self.base_url = "https://civitai.com/"
        self.api_base_url = f"{self.base_url}api/v1"
        self.modelversion_api_endpoint = f"{self.api_base_url}/model-versions/"
        self.modelversion_by_hash_endpoint = f"{self.modelversion_api_endpoint}/by-hash/"
        self.model_api_endpoint = f"{self.api_base_url}/models"
        self.wait = 1.5

    def get_source_name(self) -> str:
        """
        Classmethod for retrieving source name.
        :return: Source name.
        """
        return "civitai"

    def check_connection(self, **kwargs: Optional[dict]) -> bool:
        """
        Method for checking connection.
        :param kwargs: Arbitrary keyword arguments.
        :return: True if connection was established successfuly else False.
        """
        result = requests.get(self.base_url).status_code == 200
        self._logger.info("Connection was successfuly established.") if result else self._logger.warn(
            "Connection could not be established.")
        return result

    def validate_url_responsiblity(self, url: str, **kwargs: Optional[dict]) -> bool:
        """
        Method for validating the responsiblity for a URL.
        :param url: Target URL.
        :param kwargs: Arbitrary keyword arguments.
        :return: True, if wrapper is responsible for URL else False.
        """
        return urlparse(url).netloc in self.base_url

    def scrape_available_targets(self, target_type: str, **kwargs: Optional[dict]) -> List[dict]:
        """
        Abstract method for acquring available targets.
        :param target_type: Type of target object.
        :param kwargs: Arbitrary keyword arguments.
            'callback': A callback for adding batches of scraping results while scraping process runs.
                If a callback for adding results is given, this method will return an empty list.
            'model': A target model for constraining target model versions to be scraped.
            'start_url': A URL to start at.
        :return: List of entries of given target type.
        """
        result = []
        callback = kwargs.get("callback")
        if callback is None:
            def callback(x: Any) -> None: result.extend(
                x) if isinstance(x, list) else result.append(x)
        if target_type == "model":
            self.collect_models_via_api(callback, kwargs.get("start_url"))
        elif target_type == "modelversion":
            target_model = kwargs.get("model")
            if target_model is not None:
                metadata = self.safely_fetch_api_data(
                    target_model.url) if target_model.metadata is None else target_model.metadata
                if metadata is not None:
                    callback(metadata["modelVersions"])
            else:
                def modelversion_callback_gateway(x: Any) -> None:
                    if not isinstance(x, list):
                        x = [x]
                    for entry in x:
                        callback(entry["modelVersions"])
                self.collect_models_via_api(modelversion_callback_gateway)
        return result

    def collect_models_via_api(self, callback: Any, start_url: str = None) -> None:
        """
        Method for collecting model data via api.
        :param callback: Callback to call with collected model data batches.
        :param start_url: Starting URL. Defaults to None.
        """
        next_url = f"{self.model_api_endpoint}?limit=100&nsfw=true" if start_url is None else start_url
        while next_url:
            sleep(self.wait)
            data = self.safely_fetch_api_data(next_url, current_try=1)
            next_url = False
            if isinstance(data, dict):
                metadata = data["metadata"]
                self._logger.info(f"Fetched metadata: {metadata}.")
                next_url = metadata.get("nextPage")
                if next_url and "limit=" not in next_url:
                    next_url += "&limit=100"
                if next_url and "nsfw=" not in next_url:
                    next_url += "&nsfw=true"
                callback(data["items"])
            else:
                self._logger.warning(f"Fetched data is no dictionary: {data}")

    
    def collect_models_via_api_pagination(self, callback: Any, start_url: str = None) -> None:
        """
        Method for collecting model data via api.
        :param callback: Callback to call with collected model data batches.
        :param start_url: Starting URL. Defaults to None.
        """
        next_url = f"{self.model_api_endpoint}?limit=100&page=1" if start_url is None else start_url
        while next_url:
            sleep(self.wait)
            data = self.safely_fetch_api_data(next_url, current_try=1)
            next_url = False
            if isinstance(data, dict):
                metadata = data["metadata"]
                self._logger.info(f"Fetched metadata: {metadata}.")
                if metadata:
                    next_url = f"{self.model_api_endpoint}?limit=100&page={metadata['currentPage']+1}"
                if "items" in data:
                    callback(data["items"])
                elif data.get("error") == "You've requested too many pages, please use cursors instead":
                    self._logger.info(f"Collecting seems to be at last page...")
                    next_url = False
            else:
                self._logger.warning(f"Fetched data is no dictionary: {data}")

    def get_api_url(self, target_type: str, target_object: Any, **kwargs: Optional[dict]) -> Optional[str]:
        """
        Abstract method for acquring API URL for a given object.
        :param target_type: Type of target object.
        :param target_object: Target object.
        :param kwargs: Arbitrary keyword arguments.
        :return: API URL for given object.
        """
        if target_type == "modelversion":
            return self.model_by_versionhash_url + str(target_object.sha256)

    def collect_metadata(self, target_type: str, target_object: Any, **kwargs: Optional[dict]) -> dict:
        """
        Abstract method for acquring model data by identifier.
        :param target_type: Type of target object.
        :param target_object: Target object.
        :param kwargs: Arbitrary keyword arguments.
        :return: Metadata for given model ID.
        """
        return self.safely_fetch_api_data(target_object.url)

    def safely_fetch_api_data(self, url: str, current_try: int = 3, max_tries: int = 3) -> dict:
        """
        Method for fetching API data.
        :param url: Target URL.
        :param current_try: Current try.
            Defaults to 3, which results in a single fetching try with max_tries at 3.
        :param max_tries: Maximum number of tries.
            Defaults to 3.
        :return: Fetched data or empty dictionary.
        """
        self._logger.info(
            f"Fetching data for '{url}'...")
        resp = requests.get(url, headers=self.headers)
        try:
            data = json.loads(resp.content)
            if data is not None and not "error" in data:
                self._logger.info(f"Fetching content was successful.")
                return data
            else:
                self._logger.warn(f"Fetching metadata failed.")
        except json.JSONDecodeError:
            self._logger.warn(f"Response content could not be deserialized.")
            if current_try < 3:
                sleep(self.wait*2)
                return self.safely_fetch_api_data(url, current_try+1)
            else:
                return {}

    def normalize_metadata(self, target_type: str, metadata: dict, **kwargs: Optional[dict]) -> dict:
        """
        Abstract method for normalizing metadata.
        :param target_type: Type of target object.
        :param metadata: Metadata.
        :param kwargs: Arbitrary keyword arguments.
        :return: Normalized metadata.
        """
        normalized = {}
        if target_type == "model":
            normalized = {
                "name": metadata["name"],
                "type": metadata["type"].upper(),
                "task": {
                    "checkpoint": "image_generation",
                    "textualinversion": "image_generation_guidance",
                    "hypernetwork": "image_generation_guidance",
                    "aestheticgradient": "image_generation_guidance",
                    "lora": "image_generation_guidance",
                    "locon": "image_generation_guidance",
                    "loha": "image_generation_guidance",
                    "embedding": "image_generation_guidance",
                    "lycoris": "image_generation_guidance",
                    "vae": "image_generation_guidance",
                    "controlnet": "image_generation_guidance",
                    "poses": "image_generation_guidance",
                    "wildcards": "image_generation_guidance",
                    "upscaler": "image_upscaling",
                    "other": None,
                }[metadata["type"].lower()],
                "architecture": "stablediffusion",
                "meta_data": copy.deepcopy(metadata),
                "url": f"{self.model_api_endpoint}{metadata['id']}",
                "source": self.get_source_name()
            }
        elif target_type == "modelversion":
            primary_file = [file for file in metadata["files"]
                            if file.get("primary", False)]
            primary_file = primary_file[0] if primary_file else metadata["files"][0] if metadata["files"] else {
            }
            normalized = {
                "name": metadata["name"],
                "basemodel": metadata["baseModel"],
                "format": primary_file.get("metadata", {}).get("format"),
                "type": primary_file.get("type"),
                "meta_data": copy.deepcopy(metadata),
                "url": f"{self.modelversion_api_endpoint}{metadata['id']}",
                "source": self.get_source_name()
            }
        return normalized if normalized else metadata

    def download_model(self, model: Any, **kwargs: Optional[dict]) -> None:
        """
        Abstract method for downloading a model.
        :param model: Model object.
        :param kwargs: Arbitrary keyword arguments.
            "path": Path to use instead of model object path.
            "download_assets": Flag, declaring whether to download assets for model.
        """
        path = kwargs.get("path", model.path)
        download_assets = kwargs.get("download_assets", False)
        _, backup_path = self._create_default_model_folder(path)

        for file in file_system_utility.get_all_files(path, include_root=False):
            if file.startswith(f"m{model.id}_modelcard_"):
                shutil.move(os.path.join(path, file),
                            os.path.join(backup_path, file))
        json_utility.save(model.metadata, os.path.join(
            path, f"m{model.id}_metadata.json"))
        resp = requests.get(
            f"{self.base_url}/models/{model.metadata['id']}", headers=self.headers)
        open(os.path.join(
            path, f"m{model.id}_modelcard_{time_utility.get_timestamp()}.html"), "w").write(resp.text)

        if download_assets:
            # No direct assets for model
            pass

    def download_modelversion(self, modelversion: Any, **kwargs: Optional[dict]) -> None:
        """
        Abstract method for downloading a model.
        :param modelversion: Model version object.
        :param kwargs: Arbitrary keyword arguments.
            "path": Path to use instead of model object path.
            "download_assets": Flag, declaring whether to download assets for model.
            "file_ids": List of files to download. If not given, the primary modelversion is downloaded.
        """
        path = kwargs.get("path", modelversion.model.path)
        download_assets = kwargs.get("download_assets", False)
        _, backup_path = self._create_default_model_folder(path)

        if "file_ids" in kwargs:
            files = [file for file in modelversion.metadata["files"]
                     if file["id"] in kwargs["file_ids"]]
        else:
            files = [file for file in modelversion.metadata["files"]
                     if file.get("primary", False)]

        for file in file_system_utility.get_all_files(path, include_root=False):
            if file.startswith(f"mv{modelversion.id}_modelcard_"):
                shutil.move(os.path.join(path, file),
                            os.path.join(backup_path, file))
            elif any(file == target_file["name"] for target_file in files):
                backupped_file_path = os.path.join(backup_path, file)
                if os.path.exists(backupped_file_path):
                    os.remove(backupped_file_path)
                shutil.move(os.path.join(path, file), backupped_file_path)
            json_utility.save(modelversion.metadata, os.path.join(
                path, f"mv{modelversion.id}_metadata.json"))
        resp = requests.get(
            f"{self.base_url}/models/{modelversion.metadata['modelId']}?modelVersionId={modelversion.metadata['id']}", headers=self.headers)
        open(os.path.join(
            path, f"mv{modelversion.id}_modelcard_{time_utility.get_timestamp()}.html"), "w").write(resp.text)

        for file in files:
            json_utility.save(file, os.path.join(
                path, f"mvf{file['id']}_metadata.json"))
            requests_utility.download_web_asset(file["downloadUrl"],
                                                os.path.join(path, file["name"]), headers=self.headers)

        if download_assets:
            self.download_assets(
                "modelversion", modelversion, "image", **kwargs)

    def download_assets(self, target_type: str, target_object: Any, **kwargs: Optional[dict]) -> List[dict]:
        """
        Abstract method for downloading assets for a target object.
        :param target_type: Type of target object.
        :param target_object: Target object.
        :param asset_type: Asset type.
        :param kwargs: Arbitrary keyword arguments.
            "path": Path to use instead of model object path.
        :return: Asset data dictionaries.
        """
        path = kwargs.get("path", target_object.path if target_type ==
                          "model" else target_object.model.path)
        asset_path, _ = self._create_default_model_folder(path)
        for asset in target_object.assets:
            json_utility.save(asset.metadata, os.path.join(
                asset_path, f"a{asset.id}_metadata.json"))
            requests_utility.download_web_asset(
                asset.url, os.path.join(asset_path, asset.url.split("/")[-1]), headers=self.headers)

    def _create_default_model_folder(self, path: str) -> Tuple[str]:
        """
        Internal method for creating default model folder structure.
        :param path: Path for creating model folder.
        :return: Asset path and backup path.
        """
        file_system_utility.create_folder_tree(path, ["_assets", "_backups"])
        return os.path.join(path, "_assets"), os.path.join(path, "_backups")


class HuggingfaceAPIWrapper(AbstractAPIWrapper):
    """
    Class, representing civitai API wrapper.
    """

    def __init__(self) -> None:
        """
        Initiation method.
        """
        self._logger = cfg.LOGGER
        self.authorization = cfg.ENV["HUGGINGFACE_API_KEY"]
        self.base_url = "https://huggingface.co/"
        self.api_base_url = f"{self.base_url}api"
        self.model_api_endpoint = f"{self.api_base_url}/models/"
        self.wait = 3.0

    def get_source_name(self) -> str:
        """
        Classmethod for retrieving source name.
        :return: Source name.
        """
        return "huggingface"

    def check_connection(self, **kwargs: Optional[dict]) -> bool:
        """
        Method for checking connection.
        :param kwargs: Arbitrary keyword arguments.
        :return: True if connection was established successfuly else False.
        """
        result = requests.get(self.base_url).status_code == 200
        self._logger.info("Connection was successfuly established.") if result else self._logger.warn(
            "Connection could not be established.")
        return result

    def validate_url_responsiblity(self, url: str, **kwargs: Optional[dict]) -> bool:
        """
        Method for validating the responsiblity for a URL.
        :param url: Target URL.
        :param kwargs: Arbitrary keyword arguments.
        :return: True, if wrapper is responsible for URL else False.
        """
        return urlparse(url).netloc in self.base_url

    def scrape_available_targets(self, target_type: str, **kwargs: Optional[dict]) -> List[dict]:
        """
        Abstract method for acquring available targets.
        :param target_type: Type of target object.
        :param kwargs: Arbitrary keyword arguments.
            'callback': A callback for adding batches of scraping results while scraping process runs.
                If a callback for adding results is given, this method will return an empty list.
            'model': A target model for constraining target model versions to be scraped.
        :return: List of entries of given target type.
        """
        result = []
        callback = kwargs.get("callback")
        if callback is None:
            def callback(x: Any) -> None: result.extend(
                x) if isinstance(x, list) else result.append(x)
        if target_type == "model":
            self.collect_models_via_api(callback)
        elif target_type == "modelversion":
            target_model = kwargs.get("model")
            if target_model is not None:
                metadata, _ = self.safely_fetch_api_data(
                    target_model.url) if target_model.metadata is None else target_model.metadata
                if metadata is not None:
                    callback([metadata])
            else:
                def modelversion_callback_gateway(x: Any) -> None:
                    if not isinstance(x, list):
                        x = [x]
                    for entry in x:
                        callback(self.safely_fetch_api_data(
                            self.model_api_endpoint + f"{entry['id']}")[0])
                self.collect_models_via_api(modelversion_callback_gateway)

        return result

    def collect_models_via_api(self, callback: Any) -> None:
        """
        Method for collecting model data via api.
        :param callback: Callback to call with collected model data batches.
        """
        next_url = self.model_api_endpoint + "?full=true&config=true"
        page = 1
        fetched_last_url = False
        while next_url:
            sleep(self.wait)
            data, header_data = self.safely_fetch_api_data(
                next_url, current_try=1)
            next_url = False
            if isinstance(data, list):
                if not fetched_last_url:
                    page += 1
                    next_url = header_data.get("link", False)
                    if next_url:
                        next_url = next_url[1:-1]
                        next_url, rel = next_url.split('>; rel="')
                        if rel == "last":
                            fetched_last_url = True
                        self._logger.info(
                            f"Fetched next url: '{next_url}' with relation '{rel}' as page {page}.")

                callback(data)
            else:
                self._logger.warning(
                    f"Fetched data is no list of entries: {data}")

    def get_api_url(self, target_type: str, target_object: Any, **kwargs: Optional[dict]) -> Optional[str]:
        """
        Abstract method for acquring API URL for a given object.
        :param target_type: Type of target object.
        :param target_object: Target object.
        :param kwargs: Arbitrary keyword arguments.
        :return: API URL for given object.
        """
        pass

    def collect_metadata(self, target_type: str, target_object: Any, **kwargs: Optional[dict]) -> dict:
        """
        Abstract method for acquring model data by identifier.
        :param target_type: Type of target object.
        :param target_object: Target object.
        :param kwargs: Arbitrary keyword arguments.
        :return: Metadata for given model ID.
        """
        return self.safely_fetch_api_data(target_object.url)[0]

    def safely_fetch_api_data(self, url: str, current_try: int = 3, max_tries: int = 3) -> Tuple[dict]:
        """
        Method for fetching API data.
        :param url: Target URL.
        :param current_try: Current try.
            Defaults to 3, which results in a single fetching try with max_tries at 3.
        :param max_tries: Maximum number of tries.
            Defaults to 3.
        :return: Fetched data or empty dictionary and header data or empty dictionary.
        """
        self._logger.info(
            f"Fetching data for '{url}'...")
        resp = requests.get(url, headers={
                            "Authorization": self.authorization})
        try:
            data = json.loads(resp.content)
            if data is not None and not "error" in data:
                self._logger.info(
                    f"Fetching content was successful with headers: {resp.headers}.")
                return data, resp.headers if resp.headers else {}
            else:
                self._logger.warn(f"Fetching metadata failed.")
        except json.JSONDecodeError:
            self._logger.warn(f"Response content could not be deserialized.")
            if current_try < 3:
                sleep(self.wait*2)
                return self.safely_fetch_api_data(url, current_try+1)
            else:
                return {}

    def normalize_metadata(self, target_type: str, metadata: dict, **kwargs: Optional[dict]) -> dict:
        """
        Abstract method for normalizing metadata.
        :param target_type: Type of target object.
        :param metadata: Metadata.
        :param kwargs: Arbitrary keyword arguments.
        :return: Normalized metadata.
        """
        normalized = {}
        resp = requests.get(f"{self.base_url}{metadata['id']}")
        metadata["modelcard"] = resp.text
        if target_type == "model":
            normalized = {
                "name": metadata["id"],
                "type": metadata.get("config", {}).get("model_type"),
                "task": metadata.get("pipeline_tag", "").lower().replace("-", "_"),
                "architecture": metadata.get("library_name"),
                "meta_data": copy.deepcopy(metadata),
                "url": f"{self.model_api_endpoint}{metadata['id']}",
                "source": self.get_source_name()
            }
        elif target_type == "modelversion":
            normalized = {
                "name": metadata["id"],
                "meta_data": copy.deepcopy(metadata),
                "url": f"{self.model_api_endpoint}{metadata['id']}",
                "source": self.get_source_name()
            }
            normalized.update(
                self._extract_condense_modelversion_data(metadata))

        return normalized if normalized else metadata

    def _extract_condense_modelversion_data(self, metadata: dict) -> dict:
        """
        Internal method for extracting condense modelversion data.
        :param metadata: Raw modelversion metadata.
        :return: Condense modelversion data.
        """
        format = None
        for option in ["safetensors", "bin", "pt", "pth"]:
            if any(file.get("rfilename", "").endswith("option") for file in metadata.get("siblings", [])):
                format = option
                break
        config = metadata.get("config", {})
        return {
            "basemodel": config.get("model_type"),
            "type": config.get("architectures"),
            "format": format
        }

    def download_model(self, model: Any, path: str, **kwargs: Optional[dict]) -> None:
        """
        Abstract method for downloading a model.
        :param model: Model object.
        :param path: Absolute path to download model to.
        :param kwargs: Arbitrary keyword arguments.
        """
        pass

    def download_modelversion(self, modelversion: Any, path: str, **kwargs: Optional[dict]) -> None:
        """
        Abstract method for downloading a model.
        :param modelversion: Model version object.
        :param path: Absolute path to download model version to.
        :param kwargs: Arbitrary keyword arguments.
        """
        pass

    def download_assets(self, target_type: str, target_object: Any, asset_type: str, path: str, **kwargs: Optional[dict]) -> List[dict]:
        """
        Abstract method for downloading assets for a target object.
        :param target_type: Type of target object.
        :param target_object: Target object.
        :param asset_type: Asset type.
        :param path: Path to download assets to.
        :param kwargs: Arbitrary keyword arguments.
        :return: Asset data dictionaries.
        """
        pass
