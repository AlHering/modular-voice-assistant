# -*- coding: utf-8 -*-
"""
****************************************************
*                   Utility                        *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
from src.utility.filter_mask_utility import FilterMask
from src.utility import sqlalchemy_utility
from src.utility import time_utility
from uuid import UUID
from datetime import datetime as dt, timedelta
from typing import Optional, Any, List, Dict, Callable 


class BasicSQLAlchemyInterface(object):
    """
    Class, representing a basic SQLAlchemy interface.
    """

    def __init__(self, 
                 database_uri: str | None = None, 
                 working_directory: str | None  = None, 
                 population_function: Any = None, 
                 default_entries: Dict[str, List[dict]] = None,
                 schema: str = "",
                 engine_kwargs: dict = {"pool_recycle": 280, "encoding": "utf-8", "timeout": 60},
                 session_kwargs: dict = {"autocommit": False, "autoflush": False, "expire_on_commit": False},
                 logger: Any = None) -> None:
        """
        Initiation method.
        :param database_uri: Database URI. Defaults to SQLite DB in the working directory.
        :param working_directory: Working directory.
        :param population_function: A function, taking an engine, schema and a dataclass dictionary (later one can be empty and is to be populated).
            Defaults to None.
        :param default_entries: Default entries to populate database with.
        :param engine_kwargs: Engine instantiation parameters.
        :param session_kwargs: Session maker instantiation parameters.
        :param logger: Logger instance. 
            Defaults to None in which case separate logging is disabled.
        """
        if database_uri is None and working_directory is None:
            raise ValueError("Either database URI or working directory must be given.")
        self.logger = logger
        self.database_uri = f"sqlite:///{working_directory}/database.db" if database_uri is None else database_uri
        self.population_function = population_function
        self.default_entries = default_entries

        # Database infrastructure
        self.engine_kwargs = engine_kwargs
        self.session_kwargs = session_kwargs
        self.base = None
        self.engine = None
        self.model = None
        self.schema = schema
        self.session_factory = None
        self.primary_keys = None
        self._setup_database()

    def _setup_database(self) -> None:
        """
        Internal method for setting up database infrastructure.
        """
        if self.logger is not None:
            self.logger.info("Automapping existing structures")
        self.base = sqlalchemy_utility.automap_base()
        self.engine = sqlalchemy_utility.get_engine(self.database_uri)
        self.base.prepare(autoload_with=self.engine, reflect=True)
        self.model = sqlalchemy_utility.get_classes_from_base(self.base)
        if self.schema:
            schema = str(self.schema)
            if schema and not schema.endswith("."):
                schema += "."
            self.model = {
                table.replace(schema, ""): self.model[table] for table in self.model
            }

        if self.logger is not None:
            self.logger.info(
                f"Generating model tables for website with schema '{self.schema}'")
            
        if self.population_function is not None:
            self.population_function(
                self.engine, self.schema, self.model)
            
        self.session_factory = sqlalchemy_utility.get_session_factory(
            self.engine)
        if self.logger is not None:
            self.logger.info("base created with")
            self.logger.info(f"Classes: {self.base.classes.keys()}")
            self.logger.info(f"Tables: {self.base.metadata.tables.keys()}")

        self.primary_keys = {
            object_class: self.model[object_class].__mapper__.primary_key[0].name for object_class in self.model}
        if self.logger is not None:
            self.logger.info(f"Datamodel after addition: {self.model}")
            for object_class in self.model:
                self.logger.info(
                    f"Object type '{object_class}' currently has {self.get_object_count_by_type(object_class)} registered entries.")
                
        if self.logger is not None:
            self.logger.info(f"Inserting default entries: {self.default_entries}")
        for object_type in self.default_entries:
            for entry in self.default_entries[object_type]:
                if all(key in entry for key in self.primary_keys[object_type]):
                    self.put_object(object_type=object_type, reference_attributes=self.primary_keys[object_type], **entry)
                else:
                    self.put_object(object_type=object_type, **entry)

                
    """
    Gateway methods
    """

    def convert_filters(self, entity_type: str, filters: List[FilterMask]) -> list:
        """
        Method for converting common FilterMasks to SQLAlchemy-filter expressions.
        :param entity_type: Entity type.
        :param filters: A list of Filtermasks declaring constraints.
        :return: Filter expressions.
        """
        converted_filtermasks = []
        for filtermask in filters:
            converted_filtermasks.append(sqlalchemy_utility.SQLALCHEMY_FILTER_CONVERTER["and"](
                sqlalchemy_utility.SQLALCHEMY_FILTER_CONVERTER[exp[1]](getattr(self.model[entity_type], exp[0]),
                                                                       exp[2]) for exp in filtermask.expressions))
        return converted_filtermasks
    
    def obj_as_dict(self, obj: Any, convert_timestamps: bool = False, convert_uuids: bool = False) -> dict:
        """
        Method for converting SQLAlchemy ORM object to a dictionary.
        :param obj: Object.
        :param convert_timestamps: Declares, whether to convert timestamps into strings if objects are handled as dictionaries.
        :param convert_uuids: Declares, whether to convert UUIDs into strings if objects are handled as dictionaries.
        :return: Object entry as dictionary.
        """
        data = {
            col.key: getattr(obj, col.key) for col in sqlalchemy_utility.inspect(obj).mapper.column_attrs
        }
        if convert_timestamps is not None:
            for key in data:
                if isinstance(data[key], dt):
                    data[key] = data[key].strftime(
                        time_utility.DEFAULTS_TS_FORMAT)
        if convert_uuids:
            for key in data:
                if isinstance(data[key], UUID):
                    data[key] = str(data[key])
        return data

    """
    Default object interaction.
    """
    def get_model_representation(self,
                                 ignore_object_types: List[str] = [],
                                 ignore_columns: List[str] = [], 
                                 types_as_strings: bool = True) -> dict:
        """
        Method for acquiring model representation.
        :param ignore_object_types: List of ignored object types, ["logs"].
            Defaults to an empty list.
        :param ignore_columns: List of ignored columns, e.g. ["created", "updated", "inactive"].
            Defaults to an empty list.
        :param types_as_strings: Flag for declaring whether to return types as strings.
            Defaults to True.
        :return: Model representation as dictionary.
        """
        return {
            object_type: {
                "description": self.model[object_type].__table__.description,
                "table_name": self.model[object_type].__table__.name,
                "table_description": self.model[object_type].__table__.comment,
                "entry_count": self.get_object_count_by_type(object_type),
                "parameters": [{
                    "name": column.name,
                    "type": str(sqlalchemy_utility.SQLALCHEMY_TYPING_FROM_COLUMN_DICTIONARY.get(type(column.type))) if types_as_strings
                        else sqlalchemy_utility.SQLALCHEMY_TYPING_FROM_COLUMN_DICTIONARY.get(type(column.type)),
                    "table_type": str(type(column.type)) if types_as_strings else type(column.type),
                    "description": column.comment,
                    "nullable": column.nullable,
                    "unique": column.unique,
                    "default": column.default,
                    "server_default": str(column.server_default),
                    "server_onupdate": str(column.server_onupdate)
                } for column in self.model[object_type].__table__.columns if column.name not in ignore_columns]
            } for object_type in self.model if object_type not in ignore_object_types
        }

    def get_object_count_by_type(self, object_type: str) -> int:
        """
        Method for acquiring object count.
        :param object_type: Target object type.
        :return: Number of objects.
        """
        return int(self.engine.connect().execute(sqlalchemy_utility.select(sqlalchemy_utility.func.count()).select_from(
            self.model[object_type])).scalar())

    def get_objects_by_type(self, object_type: str) -> List[Any]:
        """
        Method for acquiring objects.
        :param object_type: Target object type.
        :return: List of objects of given type.
        """
        return self.session_factory().query(self.model[object_type]).all()
    
    def get_objects_by_slice(self, object_type: str, start_index: int, end_index: int) -> List[Any]:
        """
        Method for acquiring objects.
        :param object_type: Target object type.
        :param start_index: Starting index of slice.
        :param end_index: Ending index of slice.
        :return: List of objects of given type.
        """
        return self.session_factory().query(self.model[object_type]).slice(start_index, end_index)
    
    def get_objects_by_offset(self, object_type: str, offset: int, limit: int) -> List[Any]:
        """
        Method for acquiring objects.
        :param object_type: Target object type.
        :param offset: Starting index of slice.
        :param limit: Number of objects to fetch.
        :return: List of objects of given type.
        """
        return self.session_factory().query(self.model[object_type]).offset(offset).fetch(limit)

    def get_object_by_id(self, object_type: str, object_id: Any) -> Optional[Any]:
        """
        Method for acquiring objects.
        :param object_type: Target object type.
        :param object_id: Target ID.
        :return: An object of given type and ID, if found.
        """
        return self.session_factory().query(self.model[object_type]).filter(
            getattr(self.model[object_type],
                    self.primary_keys[object_type]) == object_id
        ).first()
    
    def get_objects_by_timedelta(self, object_type: str, datetime_attribute: str, timedelta: timedelta, comparison_function: Callable = lambda timestamp, delta: timestamp >= delta) -> List[Any]:
        """
        Method for acquiring objects by timedelta on a certain attribute.
        :param object_type: Target object type.
        :param datetime_attribute: Target datetime attribute.
        :param timedelta: Time delta for querying.
        :param comparison_function: Comparison function taking timestamp and delta as parameters.
            Defaults to 'lambda timestamp, delta: timestamp >= delta'
        :return: A list of objects of the given type, based on the given timedelta.
        """
        return self.session_factory().query(self.model[object_type]).filter(
            comparison_function(getattr(self.model[object_type],
                    datetime_attribute), timedelta)
        ).all()

    def get_objects_by_filtermasks(self, object_type: str, filtermasks: List[FilterMask]) -> List[Any]:
        """
        Method for acquiring objects.
        :param object_type: Target object type.
        :param filtermasks: Filtermasks.
        :return: A list of objects, meeting filtermask conditions.
        """
        converted_filters = self.convert_filters(object_type, filtermasks)
        with self.session_factory() as session:
            result = session.query(self.model[object_type]).filter(sqlalchemy_utility.SQLALCHEMY_FILTER_CONVERTER["or"](
                *converted_filters)
            ).all()
        return result

    def post_object(self, object_type: str, **object_attributes: Optional[Any]) -> Optional[Any]:
        """
        Method for adding an object.
        :param object_type: Target object type.
        :param object_attributes: Object attributes.
        :return: Added object, if adding was successful.
        """
        obj = self.model[object_type](**object_attributes)
        with self.session_factory() as session:
            session.add(obj)
            session.commit()
            session.refresh(obj)
        return obj

    def patch_object(self, object_type: str, object_id: Any, **object_attributes: Optional[Any]) -> Optional[Any]:
        """
        Method for patching an object.
        :param object_type: Target object type.
        :param object_id: Target ID.
        :param object_attributes: Object attributes.
        :return: Patched object, if patching was successful.
        """
        with self.session_factory() as session:
            obj = session.query(self.model[object_type]).filter(
                getattr(self.model[object_type],
                        self.primary_keys[object_type]) == object_id
            ).first()
            if obj:
                if hasattr(obj, "updated"):
                    obj.updated = dt.now()
                for attribute in object_attributes:
                    setattr(obj, attribute, object_attributes[attribute])
                session.add(obj)
                session.commit()
        return obj

    def delete_object(self, object_type: str, object_id: Any, force: bool = False) -> Optional[Any]:
        """
        Method for deleting an object.
        :param object_type: Target object type.
        :param object_id: Target ID.
        :param force: Force deletion of the object instead of setting inactivity flag.
        :return: Deleted object, if deletion was successful.
        """
        with self.session_factory() as session:
            obj = session.query(self.model[object_type]).filter(
                getattr(self.model[object_type],
                        self.primary_keys[object_type]) == object_id
            ).first()
            if obj:
                if hasattr(obj, "inactive") and not force:
                    if hasattr(obj, "updated"):
                        obj.updated = dt.now()
                    obj.inactive = True
                    session.add(obj)
                else:
                    session.delete(obj)
                session.commit()
        return obj
    
    def put_object(self, object_type: str, reference_attributes: List[str] = None,  **object_attributes: Optional[Any]) -> Optional[Any]:
            """
            Method for putting in an object.
            :param object_type: Target object type.
            :param reference_attributes: Reference attributes for finding already existing objects.
                Defaults to None in which case all given attributes are checked.
            :param object_attributes: Object attributes.
            :return: Added/patched object, if adding/patching was successful.
            """
            if reference_attributes is None:
                reference_attributes = list(object_attributes.keys())
            objs = self.get_objects_by_filtermasks(object_type,
                                                [FilterMask([[key, "==", object_attributes[key]] for key in reference_attributes])])
            if not objs:
                return self.post_object(object_type, **object_attributes)
            else:
                return self.patch_object(object_type, getattr(objs[0], self.primary_keys[object_type]), **object_attributes)
    