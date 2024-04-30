# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import abc
import os
import traceback
from typing import List, Tuple, Any, Callable, Optional, Union
from src.utility.bronze import langchain_utility
from ..filter_mask import FilterMask
from src.utility.gold.text_generation.language_model_abstractions import LanguageModelInstance


"""
Vector DB backend overview
------------------------------------------
ChromaDB
 
SQLite-VSS
 
FAISS

PGVector
 
Qdrant
 
Pinecone

Redis

Langchain Vector DB Zoo
"""


"""
Model instantiation functions
"""


def load_chromadb_knowledgebase(embedding_function: Callable,
                 knowledgebase_path: str = None,
                 knowledgebase_parameters: dict = None,
                 preprocessing_parameters: dict = None,
                 embedding_parameters: dict = None,
                 retrieval_method: str = "similarity",
                 retrieval_parameters: dict = None) -> Tuple:
    """
    Function for loading ChromaDB based knowledgebase instance.S
    :param embedding_model: Embedding model instance.
    :param knowledgebase_path: Knowledgebase path for permanent storage on disk.
        Defaults to None.
    :param knowledgebase_parameters: Knowledgebase instantiation parameters.
        Defaults to None.
    :param preprocessing_parameters: Document preprocessing paramters.
        Defaults to None.
    :param embedding_parameters: Embedding parameters.
        Defaults to None.
    :param retrieval_method: Retrieval method.
        Defaults to "similarity".
    :param retrieval_parameters: Retrieval parameters.
        Defaults to None.
    :return: Tuple of knowledgebase handle (client), collection handle dictionary, filtermasks conversion function.
    """
    from chromadb.config import Settings
    from chromadb import Client

    settings = Settings(
        persist_directory=knowledgebase_path,
        chroma_db_impl="duckdb+parquet",
        anonymized_telemetry=False
    )
    for parameter in [param for param in knowledgebase_parameters if hasattr(settings, param)]:
        setattr(settings, parameter, knowledgebase_parameters[parameter])
    client = Client(settings=settings)

    collections = knowledgebase_parameters.get("collections", {
        "base": {
            "embedding_function": embedding_function
        }
    })
    collections = {
        collection: client.get_or_create_collection(
            name=collection,
            **collections[collection])
            for collection in collections
    }

    operation_translation = {
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
        "!": lambda x: {"$ne": x}
    }

    def filtermasks_conversion(filtermasks: List[FilterMask]) -> dict:
        """
        Function for converting Filtermasks to ChromaDB filters.
        :param filtermasks: Filtermasks.
        :return: Query keyword arguments.
        """
        return {
            "where": {
                "$or": [
                    {
                        "$and": [
                            { 
                                expression[0]: {
                                    operation_translation[expression[1]]: expression[2]
                                }
                            } for expression in filtermask
                        ] for filtermask in filtermasks 
                    }
                ] 
            }
        }
    
    return (client, collections, filtermasks_conversion)
