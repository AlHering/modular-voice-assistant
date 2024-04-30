# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os
from typing import Any, Optional, List, Tuple, Dict
import traceback
from src.configuration import configuration as cfg
from src.utility.bronze import json_utility, hashing_utility, dictionary_utility
from src.utility.silver import file_system_utility
from src.utility.gold.filter_mask import FilterMask
from src.model.model_control.api_wrapper import AbstractAPIWrapper
import sqlalchemy
from src.model.model_control.model_database import ModelDatabase


class GenericModelHandler(object):
    """
    Class, representing ML Model Handler objects.
    A Model Handler utilizes API Wrappers for collecting metadata and downloading assets from
    model services and managing updates, sorter and usage.
    """

    def __init__(self, database: ModelDatabase, model_folder: str, cache_path: str, apis: Dict[str, AbstractAPIWrapper] = None, sorting_field: str = "task", sorters: List[str] = None) -> None:
        """
        Initiation method.
        :param database: Database for model data.
        :param model_folder: Model folder under which the handlers models are kept.
        :param cache_path: Cache path for handler.
        :param apis: API wrappers.
            Defaults to None which results in empty list.
        :param sorting_field: Sorting field.
            Defaults to "task".
        :param sorters: Model sorting buckets.
            Defaults to None which results in empty list.
        """
        self._logger = cfg.LOGGER
        self.database = database
        self.model_folder = model_folder
        self.cache_path = cache_path
        self.apis = [] if apis is None else apis
        self.sorting_field = sorting_field
        self.sorters = [] if sorters is None else sorters
        self._cache = {}
        if os.path.exists(self.cache_path):
            self.import_cache()

    def import_cache(self, import_path: str = None) -> None:
        """
        Method for importing cache data.
        :param import_path: Import path.
            Defaults to handler's cache path.
        """
        import_path = self.cache_path if import_path is None else import_path
        self._logger.log(f"Importing cache from '{import_path}'...")
        self._cache = json_utility.load(import_path)

    def export_cache(self, export_path: str) -> None:
        """
        Method for exporting cache data.
        :param export_path: Export path.
            Defaults to handler's cache path.
        """
        export_path = self.cache_path if export_path is None else export_path
        self._logger.log(f"Exporting cache to '{export_path}'...")
        json_utility.save(self._cache, export_path)

    def add_api_wrapper(self, wrapper: AbstractAPIWrapper) -> None:
        """
        Method for adding API wrapper.
        :param wrapper: API wrapper to add.
        """
        self.apis[wrapper.get_source_name()] = wrapper

    def get_unlinked_models(self, sorter: str = None) -> List[Any]:
        """
        Method for acquiring unlinked models.
        :param sorter: sorter constraint.
            Defaults to None.
        """
        filtermask_expressions = [["url", "==", None]]
        if sorter is not None:
            filtermask_expressions.append([self.sorting_field, "==", sorter])
        return self.database.get_objects_by_filtermasks(
            "model",
            [FilterMask(filtermask_expressions)]
        )

    def get_unlinked_modelversions(self) -> List[Any]:
        """
        Method for acquiring unlinked modelversions.
        """
        return self.database.get_objects_by_filtermasks(
            "model",
            [FilterMask([["url", "==", None]])]
        )

    def download_assets(self, target_type: str, target_id: int) -> None:
        """
        Method for downloading assets for target.
        :param target_type: Target object type.
        :param target_id: Target ID.
        """
        obj, wrapper = self.get_object_and_wrapper(target_type, target_id)
        if obj is not None and wrapper is not None:
            self.apis[wrapper].download_asset(obj)

    def download_modelversion(self, modelversion_id: int) -> None:
        """
        Method for downloading a model.
        :param modelversion_id: Modelversion ID.
        """
        obj, wrapper = self.get_object_and_wrapper(
            "modelversion", modelversion_id)
        if obj is not None and wrapper is not None:
            self.apis[wrapper].download_modelversion(obj)

    def get_object_and_wrapper(self, target_type: str, target_id: int) -> Tuple[Any, str]:
        """
        Method for downloading assets for target.
        :param target_type: Target object type.
        :param target_id: Target ID.
        :return: Target object and API wrapper.
        """
        obj = None
        wrapper = None

        obj = self.database.get_object_by_id(target_type, target_id)
        if obj is not None and obj.url is None:
            if target_type == "model":
                self.link_model(target_id)
            elif target_type == "modelversion":
                self.link_modelversion(target_id)
            obj = self.database.get_object_by_id(target_type, target_id)

        if obj is not None and obj.url is not None:
            if obj.source is not None:
                wrapper = self.apis.get(obj.source)
            if wrapper is None:
                wrapper = self.get_api_wrapper_for_url(obj.url)

        return obj, wrapper

    def link_object(self, target_type: str, target_id: int, api_wrapper_name: str = None) -> None:
        """
        Method for linking objects.
        :param target_type: Target object type.
        :param target_id: Target ID.
        :param api_wrapper_name: Name of API wrapper to use.   
            Defaults to None in which case all APIs are tested.
        """
        obj = self.database.get_object_by_id(target_type, target_id)
        wrapper_options = list(self.apis.keys()) if api_wrapper_name is None else [
            api_wrapper_name]
        for api_wrapper in wrapper_options:
            result = self.apis[api_wrapper].get_api_url(target_type, target_id)
            if result is not None:
                self.database.patch_object(
                    target_type, target_id, url=result, source=api_wrapper)
                break

    def update_metadata(self, target_type: str, target_id: int) -> None:
        """
        Method for updating metadata.
        :param target_type: Target object type.
        :param target_id: Target ID.
        """
        obj, wrapper = self.get_object_and_wrapper(target_type, target_id)
        if obj is not None:
            if obj.url is None:
                if target_type == "model":
                    self.link_model(target_id)
                elif target_type == "modelversion":
                    self.link_modelversion(target_id)
                obj = self.database.get_object_by_id(target_type, target_id)

        if obj.url is not None and wrapper is not None:
            metadata = self.apis[wrapper].collect_metadata(obj)
            if metadata is not None:
                self.database.patch_object(
                    target_type, target_id, meta_data=metadata)

    def create_model_folder(self, model_id: int, api_wrapper_name: str = None) -> None:
        """
        Method for creating a model folder.
        :param model_id: Model ID.
        :param api_wrapper_name: Name of API wrapper to use.   
            Defaults to None in which case all APIs are tested.
        """
        obj, wrapper = self.get_object_and_wrapper("model", model_id)
        if api_wrapper_name is not None:
            wrapper = api_wrapper_name
        if obj:
            if not os.path.exists(obj.path):
                os.makedirs(obj.path)
        if wrapper is not None:
            self.apis[wrapper].download_model(obj)

    def get_api_wrapper_for_url(self, url: str) -> Optional[str]:
        """
        Method for getting API wrapper to handle URL.
        :param url: URL to get wrapper for.
        :return: Appropriate API wrapper name.
        """
        for wrapper in self.apis:
            if self.apis[wrapper].validate_url_responsiblity(url):
                return wrapper

    def scrape_available_models(self, target_api_wrapper: str, callback: Any = None) -> List[dict]:
        """
        Method for scraping available models.
        :param target_api_wrapper: Target API wrapper.
        :param callback: Callback to pass scraped entry batches through.
            If a callback is given, the entries are usually not accumulated for the final return.
        :return: List of model entries.
        """
        return self.apis[target_api_wrapper].scrape_available_targets(
            "model",
            callback=self.get_scraping_callback(
                target_api_wrapper,
                "model")
            if callback is None else callback)

    def scrape_available_modelversions(self, target_api_wrapper: str, target_model_id: int = None, callback: Any = None) -> List[dict]:
        """
        Method for scraping available model versions.
        :param target_api_wrapper: Target API wrapper.
        :param target_model_id: Model ID constraint for scraping model versions.
        :param callback: Callback to pass scraped entry batches through.
            If a callback is given, the entries are usually not accumulated for the final return.
        :return: List of model version entries.
        """
        target_model = None
        if target_model_id is not None:
            target_model = self.database.get_object_by_id(
                "model", target_model_id)
        return self.apis[target_api_wrapper].scrape_available_targets(
            "modelversion",
            model=target_model,
            callback=self.get_scraping_callback(
                target_api_wrapper,
                "modelversion")
            if callback is None else callback)

    def patch_object_from_metadata(self, target_type: str, target_id: int, source: str, metadata: dict) -> None:
        """
        Method for pathing object from metadata.
        :param target_type: Target object type.
        :param target_id: Target ID.
        :param source: Source / API wrapper name.
        :param metadata: Metadata.
        """
        normalized_data = self.apis[source].normalize_metadata(
            target_type, metadata)
        self.database.patch_object(target_type, target_id, **normalized_data)

    def get_scraping_callback(self, api_wrapper: str, target_type: str) -> Optional[Any]:
        """
        Method for acquiring a callback method.
        :param api_wrapper: API wrapper name.
        :param target_type: Target object type.
        :return: Callback function if existing, else None.
        """
        def callback(entries: List[Any]) -> None:
            for entry in entries:
                failed = False
                exction_data = None
                normalized_data = {}
                try:
                    normalized_data = self.apis[api_wrapper].normalize_metadata(
                        target_type, entry
                    )
                except Exception as ex:
                    failed = True
                    exction_data = {
                        "exception": str(ex),
                        "traceback": traceback.format_exc()
                    }
                if not failed:
                    try:
                        obj = self.database.get_objects_by_filtermasks(
                            target_type, [FilterMask(
                                [["url", "==", normalized_data["url"]]])]
                        )
                        if obj:
                            self.database.patch_object(
                                target_type, obj[0].id, **normalized_data)
                        else:
                            self.database.post_object(
                                target_type, **normalized_data)
                    except Exception as ex:
                        failed = True
                        exction_data = {
                            "exception": str(ex),
                            "traceback": traceback.format_exc()
                        }
                if failed:
                    self.database.post_object(
                        "scraping_fail",
                        url=normalized_data.get("url"),
                        source=api_wrapper,
                        fetched_data=entry,
                        normalized_data=normalized_data,
                        exception_data=exction_data)
        return callback

    def load_model_folder(self) -> None:
        """
        Method for loading model folder.
        :param args: Arbitrary arguments.
        :param kwargs: Arbitrary keyword arguments.
        """
        self._logger.info(f"Loading model folder '{self.model_folder}'...")
        possible_sorters = file_system_utility.get_all_folders(
            self.model_folder)
        self.sorters.extend(
            [folder for folder in possible_sorters if folder not in self.sorters])
        for sorter in self.sorters:
            self._logger.info(f"Checking on '{sorter}'...")
            index_path = os.path.join(
                self.model_folder, sorter, "model_index.json")
            index = None
            if os.path.exists(index_path):
                index = json_utility.load(index_path)
            else:
                self._logger.warning(f"Not index found under '{index_path}'.")
            for folder in (file_system_utility.get_all_folders(
                    os.path.join(self.model_folder, sorter), include_root=False)):
                if not self.database.get_objects_by_filtermasks(
                    "model",
                    [FilterMask([["path", "==", folder], [
                                self.sorting_field, "==", sorter]])]
                ):
                    self.database.post_object("model", {
                        "path": folder,
                        self.sorting_field: sorter
                    } if index is None or folder not in index else index[folder])
