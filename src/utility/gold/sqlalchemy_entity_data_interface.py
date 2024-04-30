# -*- coding: utf-8 -*-
"""
****************************************************
*                   Utility
*            (c) 2023 Alexander Hering             *
****************************************************
"""
# In-depth documentation can be found under utility/docs/entity_data_interfaces.md
import copy
from sqlalchemy import and_, or_, not_
from typing import Optional, Any, List, Union
from ..bronze import sqlalchemy_utility
from .filter_mask import FilterMask
from ..bronze.dictionary_utility import get_filter_depth
from ..bronze.comparison_utility import COMPARISON_METHOD_DICTIONARY as CMD
from .entity_data_interface import EntityDataInterface, handle_gateways

# Dictionary, mapping filter types of filters to SQLAlchemy-compatible filters
SQLALCHEMY_FILTER_CONVERTER = {
    "equals": lambda x, y: x == y,
    "not_equals": lambda x, y: not_(x == y),
    "contains": lambda x, y: x.contains(y),
    "not_contains": lambda x, y: not_(x.contains(y)),
    "is_contained": lambda x, y: x.in_(y),
    "not_is_contained": lambda x, y: not_(x.in_(y)),
    "==": lambda x, y: x == y,
    "!=": lambda x, y: or_(x != y, and_(x is None, y is not None)),
    "has": lambda x, y: x.contains(y),
    "not_has": lambda x, y: not_(x.contains(y)),
    "in": lambda x, y: x.in_(y),
    "not_in": lambda x, y: not_(x.in_(y))
}

# Dictionary, defining table for manual linking
MANUAL_LINKAGE = {
    "id": {
        "type": "int",
        "primary_key": True,
        "autoincrement": True,
        "required": True
    },
    "linkage": {
        "type": "str_180",
        "required": True,
        "description": "Linkage type."
    },
    "source_key": {
        "type": "text",
        "required": True,
        "description": "Source key."
    },
    "target_key": {
        "type": "text",
        "required": True,
        "description": "Target key."
    }
}


class SQLAlchemyEntityInterface(EntityDataInterface):
    """
    Class, representing SQLAlchemy Entity Interface.
    """

    def __init__(self, environment_profile: dict, entity_profiles: dict, linkage_profiles: dict,
                 view_profiles: dict = None) -> None:
        """
        Initiation method for SQLAlchemy Entity Interface.
        :param environment_profile: Environment profile.
        :param entity_profiles: Entity profiles under entity name.
        :param linkage_profiles: Physical Data Model linkage for connecting to other data entities.
        :param view_profiles: Visual representation profiles.
        """
        super().__init__(environment_profile, entity_profiles, linkage_profiles, view_profiles)
        self.engine = sqlalchemy_utility.get_engine(environment_profile["arguments"]["database"],
                                                    encoding=environment_profile["arguments"].get("encoding", "utf-8"))
        self.base = sqlalchemy_utility.get_automapped_base(self.engine)
        self.model = sqlalchemy_utility.get_classes_from_base(self.base)
        self.session_factory = None

    """
    Initiation methods
    """

    def initiate_infrastructure(self) -> None:
        """
        Method for initiating infrastructure.
        """
        # add profile for manual linking
        global MANUAL_LINKAGE
        self._entity_profiles["MANUAL_LINKAGE"] = MANUAL_LINKAGE

        # add dataclasses, based off of with schema args enriched profiles, to model
        for profile in [p for p in self._entity_profiles if p not in self.model]:
            self._create_dataclass(profile)

        # create infrastructure and define session factory
        self.base.metadata.create_all(self.engine)
        self.session_factory = sqlalchemy_utility.get_session_factory(
            self.engine)

    def _create_dataclass(self, entity_type: str) -> None:
        """
        Internal method for creating dataclass for the given entity type.
        :param entity_type: Entity type.
        """
        mapping_profile = copy.deepcopy(
            {key: {"type": self._entity_profiles[entity_type][key]["type"],
                   "schema_args": {
                       "primary_key": self._entity_profiles[entity_type][key].get("primary_key", False),
                       "nullable": not self._entity_profiles[entity_type][key].get("not_null", False),
                       "comment": self._entity_profiles[entity_type][key].get("description", ""),
            }} for key in
                self._entity_profiles[entity_type] if
                key != "#meta"})
        desc = self._entity_profiles[entity_type].get(
            "#meta", {}).get("description", False)
        schema = self._entity_profiles[entity_type].get(
            "#meta", {}).get("schema", False)

        mapping_profile["#meta"] = {}
        if desc:
            mapping_profile["#meta"]["comment"] = desc
        if schema:
            mapping_profile["#meta"]["schema"] = schema

        for key in mapping_profile:
            if "autoincrement" in mapping_profile[key]:
                mapping_profile[key]["schema_args"]["autoincrement"] = mapping_profile[key]["autoincrement"]
            if "unique" in mapping_profile[key]:
                mapping_profile[key]["schema_args"]["unique"] = mapping_profile[key]["unique"]
        self.model[entity_type] = sqlalchemy_utility.create_mapping_for_dictionary(self.base, entity_type, mapping_profile,
                                                                                   self._linkage_profiles)

    """
    Gateway methods
    """

    def convert_filters(self, entity_type: str, filters: List[FilterMask]) -> list:
        """
        Method for coverting common FilterMasks to SQLAlchemy-filter expressions.
        :param entity_type: Entity type.
        :param filters: A list of Filtermasks declaring constraints.
        :return: Filter expressions.
        """
        filter_expressions = []
        for filtermask in filters:
            filter_expressions.extend([
                SQLALCHEMY_FILTER_CONVERTER[exp[1]](getattr(self.model[entity_type], exp[0]),
                                                    exp[2]) for exp in filtermask.expressions])
        return filter_expressions

    """
    Interfacing methods
    """
    # override
    @handle_gateways(filter_index=2, data_index=None, skip=False)
    def _get(self, entity_type: str, filters: List[FilterMask], **kwargs: Optional[Any]) -> Optional[Any]:
        """
        Method for acquring entity as object.
        :param entity_type: Entity type.
        :param filters: A list of Filtermasks declaring constraints.
        :param kwargs: Arbitrary keyword arguments.
        :return: Target entity.
        """
        with self.session_factory() as session:
            result = session.query(self.model[entity_type]).filter(
                *self.convert_filters(entity_type, filters)
            ).first()
        return result

    # override
    @handle_gateways(filter_index=2, data_index=None, skip=False)
    def _get_batch(self, entity_type: str, list_of_filters: List[List[FilterMask]], **kwargs: Optional[Any]) -> List[Any]:
        """
        Method for acquring entities as object.
        :param entity_type: Entity type.
        :param filters: A list of lists of Filtermasks declaring constraints.
        :param kwargs: Arbitrary keyword arguments.
        :return: Target entities.
        """
        converted_filters = [
            or_(*self.convert_filters(entity_type, filters)) for filters in list_of_filters]
        with self.session_factory() as session:
            result = session.query(self.model[entity_type]).filter(or_(
                *converted_filters)
            ).all()
        return result

    # override
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

    # override
    @handle_gateways(filter_index=None, data_index=2, skip=False)
    def _post(self, entity_type: str, entity: Any, **kwargs: Optional[Any]) -> Optional[Any]:
        """
        Method for adding a new entity.
        :param entity_type: Entity type.
        :param entity: Entity object.
        :param kwargs: Arbitrary keyword arguments.
        :return: Target entity data if existing, else None.
        """
        with self.session_factory() as session:
            session.add(entity)
            session.commit()
            session.refresh(entity)
        return entity

    # override
    @handle_gateways(filter_index=None, data_index=2, skip=False)
    def _post_batch(self, entity_type: str, entities: List[Any], **kwargs: Optional[Any]) -> List[Any]:
        """
        Method for adding new entities.
        :param entity_type: Entity type.
        :param entity: Entity objects.
        :param kwargs: Arbitrary keyword arguments.
        :return: Target entities.
        """
        with self.session_factory() as session:
            for entity in entities:
                session.add(entity)
            session.commit()
            for entity in entities:
                session.refresh(entity)
        return entities

    # override
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

    # override
    @handle_gateways(filter_index=None, data_index=[2, 3], skip=False)
    def _patch(self, entity_type: str, entity: Any, patch: Optional[dict] = None, **kwargs: Optional[Any]) -> Optional[Any]:
        """
        Method for patching an existing entity.
        :param entity_type: Entity type.
        :param entity: Entity object to patch.
        :param patch: Patch as dictionary, if entity is not already patched.
        :param kwargs: Arbitrary keyword arguments.
        :return: Target entity data if existing, else None.
        """
        if patch is not None:
            for key in patch:
                setattr(entity, key, patch[key])
        with self.session_factory() as session:
            session.update(entity)
            session.commit()
            session.refresh(entity)
        return entity

    # override
    @handle_gateways(filter_index=None, data_index=[2, 3], skip=False)
    def _patch_batch(self, entity_type: str, entities: List[Any], patches: List[dict] = [], **kwargs: Optional[Any]) -> List[Any]:
        """
        Method for patching existing entities.
        :param entity_type: Entity type.
        :param entity: Entity objects to patch.
        :param patches: Patches as dictionaries, if entities are not already patched.
        :param kwargs: Arbitrary keyword arguments.
        :return: Target entities.
        """
        if patches:
            for index, patch in enumerate(patches):
                for key in patch:
                    setattr(entities[index], key, patch[key])
        with self.session_factory() as session:
            for entity in entities:
                session.update(entity)
            session.commit()
            for entity in entities:
                session.refresh(entity)
        return entities

    # override
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

    # override
    @handle_gateways(filter_index=None, data_index=3, skip=False)
    def _delete(self, entity_type: str, entity: Any, **kwargs: Optional[Any]) -> Optional[Any]:
        """
        Method for deleting an entity.
        :param entity_type: Entity type.
        :param entity: Entity to delete.
        :param kwargs: Arbitrary keyword arguments.
        :return: Target entity data if existing, else None.
        """
        with self.session_factory() as session:
            if self._entity_profiles[entity_type].get(
                    "#meta", {}).get("keep_deleted", False):
                session.update(entity)
                session.commit()
                session.refresh(entity)
            else:
                session.delete(entity)
                session.commit()
        return entity

    # override
    @handle_gateways(filter_index=None, data_index=2, skip=False)
    def _delete_batch(self, entity_type: str, entities: List[Any], **kwargs: Optional[Any]) -> List[Any]:
        """
        Method for deleting entities.
        :param entity_type: Entity type.
        :param entities: Entities to delete.
        :param kwargs: Arbitrary keyword arguments.
        :return: Target entities.
        """
        with self.session_factory() as session:
            if self._entity_profiles[entity_type].get(
                    "#meta", {}).get("keep_deleted", False):
                for entity in entities:
                    session.update(entity)
                    session.commit()
                    session.refresh(entity)
            else:
                for entity in entities:
                    session.delete(entity)
                    session.commit()
        return entities

    # override
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

    # override
    def get_linked_entities(self, linkage: str, source: Any, filters: List[FilterMask] = None, **kwargs: Optional[Any]) -> List[Any]:
        """
        Method for getting linked entities.
        :param source: Source entity.
        :param filters: List of FilterMasks if linkage type is 'filter_mask'.
        :param kwargs: Arbitrary keyword arguments.
        :return: Linked entities.
        """
        for filter_mask in filters:
            filter_mask.reference = source
        if self._linkage_profiles[linkage]["linkage_type"] == "manual":
            source_key = str(
                getattr(source, self._linkage_profiles[linkage]["source_key"][1]))
            return self._get_batch("MANUAL_LINKAGE", [
                FilterMask([["linkage", "==", linkage], ["source_key", "==", source_key]])])
        elif self._linkage_profiles[linkage]["linkage_type"] == "foreign_key":
            linked_entities = getattr(source, linkage)
            return linked_entities if isinstance(linked_entities, list) else [linked_entities]
        elif self._linkage_profiles[linkage]["linkage_type"] == "filter_masks":
            return self._get_batch(self._linkage_profiles[linkage]["target"], filters)

    # override
    def link_entities(self, linkage: str, source_entity: Any, target_entity: Any, **kwargs: Optional[Any]) -> None:
        """
        Method for linking entities.
        :param source_entity: Source entity.
        :param target_entity: Target entity.
        """
        if self._linkage_profiles[linkage]["linkage_type"] == "manual":
            source_key = str(
                getattr(source_entity, self._linkage_profiles[linkage]["source_key"][1]))
            target_key = str(
                getattr(source_entity, self._linkage_profiles[linkage]["target_key"][1]))
            if not self._get("MANUAL_LINKAGE", [FilterMask([["source_key", "==", source_key], ["target_key", "==", target_key]])]):
                self._post(
                    "MANUAL_LINKAGE", self.model["MANUAL_LINKAGE"]({
                        "linkage": linkage,
                        "source_key": source_key,
                        "target_key": target_key
                    })
                )
        elif self._linkage_profiles[linkage]["linkage_type"] == "foreign_key":
            if not self._linkage_profiles[linkage]["relation"].endswith("1"):
                getattr(source_entity, linkage).append(target_entity)
            else:
                setattr(source_entity, linkage, target_entity)
            self._patch(
                self._linkage_profiles[linkage]["source"], source_entity)
