# -*- coding: utf-8 -*-
"""
****************************************************
*                   Utility
*            (c) 2023 Alexander Hering             *
****************************************************
"""
# In-depth documentation can be found under utility/docs/entity_data_interfaces.md
from abc import ABC, abstractmethod
import copy
import hashlib
from typing import List, Optional, Union, Any
from ..silver import environment_utility
from .filter_mask import FilterMask


def get_authorization_token(password: str) -> str:
    """
    Function for getting an authorization token.
    :param password: Password to get token.
    :return: Token.
    """
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


def handle_gateways(filter_index: int = None, data_index: Union[int, List[int]] = None, skip: bool = False) -> Any:
    """
    Decorator method for wrapping interfacing methods and handling defaults, obfuscation and deobuscation.
    :param filter_index: Index of filter argument.
    :param data_index: Index (or indices) of data argument(s).
    :param skip: Skip gateways.
    :return: Function wrapper.
    """

    def decorator(func: Any) -> Any:
        """
        Decorator method for decorating interfacing functions.
        :param func: Interfacing function.
        :return: Interfacing function decorator.
        """
        batch = func.__name__.endswith("_batch")
        interface_method = func.__name__.replace("_batch", "")

        def func_wrapper(*args: Optional[Any], **kwargs: Optional[Any]) -> Any:
            """
            Function wrapper for wrapping decorated function.
            :param args: Arguments. Arguments should contain interface instance as first argument and target entity type
                as second argument.
            :param kwargs: Arbitrary keyword arguments.
            :return: Result of wrapped function.
            """
            if not skip:
                res = None
                if len(args) >= 2 and isinstance(args[0], EntityDataInterface):
                    instance = args[0]
                    entity_type = args[1]

                    if instance.authorize(entity_type, kwargs.get("authorize")):
                        if filter_index is not None:
                            instance.obfuscate_filters(
                                entity_type, args[filter_index], batch)
                        if data_index is not None:
                            if isinstance(data_index, int):
                                data_index = [data_index]
                            for index in data_index:
                                instance.set_defaults(
                                    entity_type, interface_method, args[index], batch)
                                instance.obfuscate_entity_data(
                                    entity_type, args[index], batch)

                        res = instance.deobfuscate_entity_data(
                            entity_type, func(*args, **kwargs), batch)
                return res
            else:
                return func(*args, **kwargs)

        return func_wrapper

    return decorator


class EntityDataInterface(ABC):
    """
    Abstract class for data handle objects.
    Based off of the Common Data Controller and streamlined for faster prototyping.
    In contrast to the Common Entity Data Interfaces, this solution is not split into authentication, transformation and backend layers!
    In contrast to the Common Entity Data Interfaces, this solution does not support Plugins!
    """

    def __init__(self, environment_profile: dict, entity_profiles: dict, linkage_profiles: dict = None,
                 view_profiles: dict = None) -> None:
        """
        Method for initiating data handle object.
        :param environment_profile: Environment profile for entity profiles.
        :param entity_profiles: Physical Data Model for data entities.
        :param linkage_profiles: Physical Data Model for connecting to other data entities.
        :param view_profiles: Visual representation profiles.
        """
        self._environment_profile = copy.deepcopy(environment_profile)
        self._entity_profiles = copy.deepcopy(entity_profiles)
        self._linkage_profiles = copy.deepcopy(linkage_profiles)
        self._view_profiles = copy.deepcopy(view_profiles)

        if "targets" in self._environment_profile and self._environment_profile["targets"] != "*":
            self._remove_untracked_entity_types()

        self.handle_as_objects = self._environment_profile.get(
            "handle_as_objects", False)

        self.cache = {
            "keys": {
                entity_type: [] for entity_type in self._entity_profiles
            }
        }

        self._gateways = self._populate_gateway_barriers()
        self._defaults = self._populate_default_parsers()

    """
    Initiation methods
    """

    @abstractmethod
    def initiate_infrastructure(self) -> None:
        """
        Abstract method for initiating infrastructure.
        """
        pass

    def _remove_untracked_entity_types(self) -> None:
        """
        Method for removing untracked entity types from configuration.
        """
        removed_linkages = []
        for entity_type in self._entity_profiles:
            if entity_type not in self._environment_profile["targets"]:
                self._entity_profiles.pop(entity_type)
                for linkage in self._linkage_profiles:
                    if self._linkage_profiles[linkage]["source"] == entity_type or self._linkage_profiles[linkage]["target"] == entity_type:
                        self._linkage_profiles.pop(linkage)
                        removed_linkages.append(linkage)
                for view in self._view_profiles:
                    if self._view_profiles[view]["root"] == entity_type or any(removed_linkage in self._view_profiles[view]["linkages"] for removed_linkage in removed_linkages):
                        self._view_profiles.pop(view)

    def _populate_gateway_barriers(self) -> dict:
        """
        Method for populating gateway barriers.
        :return: Gateway dictionary.
        """
        gateways = {
            entity_type: {
            } for entity_type in self._entity_profiles
        }
        for entity_type in self._entity_profiles:
            obfuscate = self._entity_profiles[entity_type].get(
                "#meta", {}).get("obfuscate")
            deobfuscate = self._entity_profiles[entity_type].get(
                "#meta", {}).get("deobfuscate")
            authorize = self._entity_profiles[entity_type].get(
                "#meta", {}).get("authorize")
            if obfuscate is not None:
                gateways[entity_type]["obfuscate"] = environment_utility.get_lambda_function_from_string(
                    obfuscate)
            if deobfuscate is not None:
                gateways[entity_type]["deobfuscate"] = environment_utility.get_lambda_function_from_string(
                    deobfuscate)
            if authorize is not None:
                gateways[entity_type]["authorize"] = authorize
        return gateways

    def _populate_default_parsers(self) -> dict:
        """
        Method for initiating argument parser for default value insertion.
        :return: Default parser dictionary.
        """
        argument_parsers = {
            entity_type: {
                "post": {},
                "patch": {},
                "delete": {}
            } for entity_type in self._entity_profiles
        }
        for entity_type in self._entity_profiles:
            for key in [key for key in self._entity_profiles[entity_type]]:
                for option in [opt for opt in ["post", "patch", "delete"] if
                               opt in self._entity_profiles[entity_type][key]]:
                    argument_parsers[entity_type][option][key] = \
                        self._entity_profiles[entity_type][key][option]
                if self._entity_profiles[entity_type][key].get("key", False) and key not in self.cache["keys"][entity_type]:
                    self.cache["keys"][entity_type].append(key)
        return argument_parsers

    """
    Gateway methods
    """

    def authorize(self, entity_type: str, password: str) -> bool:
        """
        Method for checking authorized access to an entity type.
        :param entity_type: Entity type.
        :param password: Authorization password.
        :return: Authorization status.
        """
        return "authorize" not in self._gateways[entity_type] or get_authorization_token(password) == self._gateways[entity_type]("authorize")

    def set_defaults(self, entity_type: str, method_type: str, data: Union[list, dict, Any], batch: bool = False) \
            -> None:
        """
        Method for setting default value.
        :param entity_type: Entity type.
        :param method_type: Method type out of 'post', 'patch' and 'delete'
        :param data: Data to set standard values for.
        :param batch: Flag, declaring whether data contains multiple entries. Defaults to False.
        """
        if self._defaults[entity_type].get(method_type, False):
            if isinstance(data, dict):
                for key in self._defaults[entity_type][method_type]:
                    data[key] = self._defaults[entity_type][method_type][key](
                        data)
            elif not batch:
                for key in self._defaults[entity_type][method_type]:
                    setattr(
                        data, key, self._defaults[entity_type][method_type][key](data))
            else:
                for data_entry in data:
                    self.set_defaults(entity_type, data_entry, data_entry)

    def obfuscate_filters(self, entity_type: str, filters: Union[List[FilterMask], List[List[FilterMask]]], batch: bool = False) -> None:
        """
        Method for obfuscating filters.
        :param entity_type: Entity type.
        :param filters: List of FilterMasks or list of lists of FilterMasks in case of batch filtering.
        :param batch: Flag, declaring whether filters contain multiple entries. Defaults to False.
        """
        if "obfuscate" in self._gateways[entity_type] and filters:
            if not batch:
                for filtermask in filters:
                    filtermask.transform(
                        self._gateways[entity_type]["obfuscate"])
            else:
                for filter_list in filters:
                    self.obfuscate_filters(entity_type, filter_list)

    def obfuscate_entity_data(self, entity_type: str, data: Union[dict, list, Any], batch: bool = False) -> None:
        """
        Method for obfuscating entity data.
        :param entity_type: Entity type.
        :param data: Entity data or list of entity data entries.
        :param batch: Flag, declaring whether data contains multiple entries. Defaults to False.
        """
        if "obfuscate" in self._gateways[entity_type] and data:
            if batch:
                for entry in data:
                    self.obfuscate_entity_data(entity_type, entry)
            else:
                data = self._gateways[entity_type]["obfuscate"](data)

    def deobfuscate_entity_data(self, entity_type: str, data: Union[dict, list, Any], batch: bool = False) -> None:
        """
        Method for deobfuscating data.
        :param entity_type: Entity type.
        :param data: Entity data or list of entity data entries.
        :param batch: Flag, declaring whether data contains multiple entries. Defaults to False.
        """
        if "deobfuscate" in self._gateways[entity_type] and data:
            if batch:
                for entry in data:
                    self.deobfuscate_entity_data(entity_type, entry)
            else:
                data = self._gateways[entity_type]["deobfuscate"](data)

    def filters_from_data(self, entity_type: str, data: Any) -> list:
        """
        Method to derive filters from data.
        :param entity_type: Entity type.
        :param data: Entity data.
        :return: Filter masks for data.
        """
        if self.cache["keys"].get(entity_type, False):
            return FilterMask([[key, "==", data[key] if isinstance(data, dict) else getattr(data, key)] for key in
                               self.cache["keys"][entity_type]])
        else:
            return FilterMask([[key, "==", data[key] if isinstance(data, dict) else getattr(data, key)] for key in data])

    def obj_to_dictionary(self, entity_type: str, obj: Any) -> dict:
        """
        Method for transforming data object to dictionary.
        :param entity_type: Entity type.
        :param data: Data object.
        :return: Dictionary, representing data object.
        """
        return obj if isinstance(obj, dict) else {key: getattr(obj, key) for key in
                                                  self._entity_profiles[entity_type] if key != "#meta"}

    """
    Interfacing methods
    """

    @abstractmethod
    @handle_gateways(filter_index=2, data_index=None, skip=False)
    def _get(self, entity_type: str, filters: List[FilterMask], **kwargs: Optional[Any]) -> Optional[Any]:
        """
        Abstract method for acquring entity as object.
        :param entity_type: Entity type.
        :param filters: A list of Filtermasks declaring constraints.
        :param kwargs: Arbitrary keyword arguments.
        :return: Target entity.
        """
        pass

    @handle_gateways(filter_index=2, data_index=None, skip=True)
    def _get_batch(self, entity_type: str, filters: List[List[FilterMask]], **kwargs: Optional[Any]) -> List[Any]:
        """
        Method for acquring entities as object.
        :param entity_type: Entity type.
        :param filters: A list of lists of Filtermasks declaring constraints.
        :param kwargs: Arbitrary keyword arguments.
        :return: Target entities.
        """
        res = [self._get(entity_type, filtermasks,  **kwargs)
               for filtermasks in filters]
        return [entry for entry in res if res is not None]

    def get(self, batch: bool, *args: Optional[Any], **kwargs: Optional[Any]) -> Optional[Any]:
        """
        Method for acquring entities.
        :param batch: Flag, declaring whether to handle operation as batch-operation.
        :param args: Arbitrary arguments.
        :param kwargs: Arbitrary keyword arguments.
            'mode': Overwrite class flag for handling entities via modes "as_object", "as_dict".
        :return: Target entities.
        """
        if batch:
            self._get_batch(*args, **kwargs)
        else:
            self._get(*args, **kwargs)

    @abstractmethod
    @handle_gateways(filter_index=None, data_index=2, skip=False)
    def _post(self, entity_type: str, entity: Any, **kwargs: Optional[Any]) -> Optional[Any]:
        """
        Abstract method for adding a new entity.
        :param entity_type: Entity type.
        :param entity: Entity object.
        :param kwargs: Arbitrary keyword arguments.
        :return: Target entity data if existing, else None.
        """
        pass

    @handle_gateways(filter_index=None, data_index=2, skip=True)
    def _post_batch(self, entity_type: str, entities: List[Any], **kwargs: Optional[Any]) -> List[Any]:
        """
        Method for adding new entities.
        :param entity_type: Entity type.
        :param entity: Entity objects.
        :param kwargs: Arbitrary keyword arguments.
        :return: Target entities.
        """
        res = [self._post(entity_type, entity,  **kwargs)
               for entity in entities]
        return [entry for entry in res if res is not None]

    def post(self, batch: bool, *args: Optional[Any], **kwargs: Optional[Any]) -> Optional[Any]:
        """
        Method for adding a new entity.
        :param batch: Flag, declaring whether to handle operation as batch-operation.
        :param args: Arbitrary arguments.
        :param kwargs: Arbitrary keyword arguments.
            'mode': Overwrite class flag for handling entities via modes "as_object", "as_dict".
        :return: Target entity if existing, else None.
        """
        if batch:
            self._post_batch(*args, **kwargs)
        else:
            self._post(*args, **kwargs)

    @abstractmethod
    @handle_gateways(filter_index=None, data_index=[2, 3], skip=False)
    def _patch(self, entity_type: str, entity: Any, patch: Optional[dict] = None, **kwargs: Optional[Any]) -> Optional[Any]:
        """
        Abstract method for patching an existing entity.
        :param entity_type: Entity type.
        :param entity: Entity object to patch.
        :param patch: Patch as dictionary, if entity is not already patched.
        :param kwargs: Arbitrary keyword arguments.
        :return: Target entity data if existing, else None.
        """
        pass

    @abstractmethod
    @handle_gateways(filter_index=None, data_index=[2, 3], skip=True)
    def _patch_batch(self, entity_type: str, entities: List[Any], patches: List[dict] = [], **kwargs: Optional[Any]) -> List[Any]:
        """
        Method for patching existing entities.
        :param entity_type: Entity type.
        :param entity: Entity objects to patch.
        :param patches: Patches as dictionaries, if entities are not already patched.
        :param kwargs: Arbitrary keyword arguments.
        :return: Target entities.
        """
        res = [self._patch(entity_type, entity, patches[index] if patches else None, **kwargs)
               for index, entity in enumerate(entities)]
        return [entry for entry in res if res is not None]

    @abstractmethod
    def patch(self, batch: bool, *args: Optional[Any], **kwargs: Optional[Any]) -> Optional[Any]:
        """
        Method for patching an existing entities.
        :param batch: Flag, declaring whether to handle operation as batch-operation.
        :param args: Arbitrary arguments.
        :param kwargs: Arbitrary keyword arguments.
            'mode': Overwrite class flag for handling entities via modes "as_object", "as_dict".
        :return: Target entities.
        """
        if batch:
            self._patch_batch(*args, **kwargs)
        else:
            self._patch(*args, **kwargs)

    @abstractmethod
    @handle_gateways(filter_index=None, data_index=2, skip=False)
    def _delete(self, entity_type: str, entity: Any, **kwargs: Optional[Any]) -> Optional[Any]:
        """
        Abstract method for deleting an entity.
        :param entity_type: Entity type.
        :param entity: Entity to delete.
        :param kwargs: Arbitrary keyword arguments.
        :return: Target entity data if existing, else None.
        """
        pass

    @abstractmethod
    @handle_gateways(filter_index=None, data_index=2, skip=True)
    def _delete_batch(self, entity_type: str, entities: List[Any], **kwargs: Optional[Any]) -> List[Any]:
        """
        Abstract method for deleting entities.
        :param entity_type: Entity type.
        :param entities: Entities to delete.
        :param kwargs: Arbitrary keyword arguments.
        :return: Target entities.
        """
        res = [self._delete(entity_type, entity,  **kwargs)
               for entity in entities]
        return [entry for entry in res if res is not None]

    @abstractmethod
    def delete(self, batch: bool, *args: Optional[Any], **kwargs: Optional[Any]) -> Optional[Any]:
        """
        Method for deleting entities.
        :param batch: Flag, declaring whether to handle operation as batch-operation.
        :param args: Arbitrary arguments.
        :param kwargs: Arbitrary keyword arguments.
            'mode': Overwrite class flag for handling entities via modes "as_object", "as_dict".
        :return: Target entities.
        """
        if batch:
            self._delete_batch(*args, **kwargs)
        else:
            self._delete(*args, **kwargs)

    """
    Linkage methods
    """

    @abstractmethod
    def get_linked_entities(self, linkage: str, *args: Optional[Any], **kwargs: Optional[Any]) -> List[Any]:
        """
        Method for getting linked entities.
        :param args: Arbitrary arguments.
        :param kwargs: Arbitrary keyword arguments.
        :return: Linked entities.
        """
        pass

    @abstractmethod
    def link_entities(self, linkage: str, *args: Optional[Any], **kwargs: Optional[Any]) -> List[Any]:
        """
        Method for linking entities.
        :param args: Arbitrary arguments.
        :param kwargs: Arbitrary keyword arguments.
        :return: Linked entities.
        """
        pass
