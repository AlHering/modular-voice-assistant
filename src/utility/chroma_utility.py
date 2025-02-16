# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2025 Alexander Hering             *
****************************************************
"""
from chromadb import Settings, PersistentClient
from src.utility.filter_mask_utility import FilterMask
from typing import List, Union


class ChromaStorage(object):
    """
    Represents a chroma based storage.
    """
    def __init__(self,
                 storage_path: str,
                 storage_parameters: dict | None = None,
                 collection_parameters: dict | None = None,
                 retrieval_parameters: dict | None = None) -> None:
        """
        Initiates chroma based storage.
        :param storage_path: Knowledgebase path for permanent storage on disk.
        :param storage_parameters: Knowledgebase instantiation parameters.
        :param collection_parameters: Collection parameters.
        :param retrieval_parameters: Default retrieval parameters.
        """
        self.storage_path = storage_path
        self.storage_parameters = {} if storage_parameters is None else storage_parameters
        self.collection_parameters = {"name": "base"} if collection_parameters is None else collection_parameters
        self.retrieval_parameters = {} if retrieval_parameters is None else retrieval_parameters

        settings = Settings(
            persist_directory=self.storage_path,
            is_persistent=True,
            anonymized_telemetry=False
        )
        for parameter in [param for param in self.storage_parameters if hasattr(settings, param)]:
            setattr(settings, parameter, self.storage_parameters[parameter])
        self.client = PersistentClient(
            path=storage_path,
            settings=settings)

        self.collection = self.client.get_or_create_collection(
            **self.collection_parameters
        )

        self.operation_translation = {
            "equals": "$eq",
            "not_equals": "$neq",
            "contains": "$contains",
            "not_contains": "$not_contains",
            "is_contained": "$in",
            "not_is_contained": "$nin",
            "==": "$eq",
            "!=": "$neq",
            "has": "$contains",
            "not_has":"$not_contains",
            "in": "$in",
            "not_in": "$nin",
            "and": lambda *x: {"$and": [*x]},
            "or": lambda *x: {"$or": [*x]},
            "not": lambda x: {"$ne": x},
            "&&": lambda *x: {"$and": [*x]},
            "||": lambda *x: {"$or": [*x]},
            "!": lambda x: {"$ne": x},
            "smaller": lambda x: {"$lt": x},
            "greater": lambda x: {"$gt": x},
            "smaller_or_equal": lambda x: {"$lte": x},
            "greater_or_equal": lambda x: {"$gte": x},
            "<": lambda x: {"$lt": x},
            ">": lambda x: {"$gt": x},
            "<=": lambda x: {"$lte": x},
            ">=": lambda x: {"$gte": x},
        }

    """
    Conversion functionality
    """

    def filtermasks_conversion(self, filtermasks: List[FilterMask]) -> dict:
        """
        Function for converting Filtermasks to ChromaDB filters.
        :param filtermasks: Filtermasks.
        :return: Query keyword arguments.
        """
        return {
                "$or": [
                    {
                        "$and": [
                            { 
                                expression[0]: {
                                    self.operation_translation[expression[1]]: expression[2]
                                }
                            } for expression in filtermask
                        ] 
                    } for filtermask in filtermasks 
                ] 
            }

    """
    Extended access
    """

    def store_embeddings(self,
                        ids: List[Union[int, str]],
                        embeddings: List[int | float | List[int | float]], 
                        contents: List[str] = None, 
                        metadatas: List[list] = None) -> None:
        """
        Method for storing embeddings.
        :param ids: IDs to store the embedding of the same index under.
        :param embeddings: Embeddings to store.
        :param contents: Content entries to attach to embedding of the same index.
        :param metadatas: Metadata entries to attach to embedding of the same index.
        :param embeddings: Entries to embed.
        """
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            entries=contents,
            metadatas=metadatas
        )
