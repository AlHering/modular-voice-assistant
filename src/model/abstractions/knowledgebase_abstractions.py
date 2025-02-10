# -*- coding: utf-8 -*-
"""

WARNING: LEGACY CODE - just for reference

****************************************************
*          Basic Language Model Backend            *
*            (c) 2025 Alexander Hering             *
****************************************************
"""
import copy
from abc import ABC, abstractmethod
from typing import List, Callable, Union, Optional, Any
from src.utility.filter_mask_utility import FilterMask
from src.model.abstractions.language_model_abstractions import LanguageModelInstance
from chromadb import Settings, PersistentClient, EmbeddingFunction as ChromaEmbeddingFunction, QueryResult as ChromaQueryResult
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction as ChromaDefaultEmbeddingFunction


"""
Abstractions
"""
class Entry(object):
    """
    Class, representing entries.
    """
    def __init__(self, id: Union[int, str], content: str, metadata: dict | None = None, embedding: List[float] | None = None) -> None:
        """
        Initiation method.
        :param id: ID of the entry.
        :param content: Textual content of the entry.
        :param metadata: Metadata of the entry.
            Defaults to None.
        :param embedding: Embedding of the entry.
            Defaults to None.
        """
        self.id = id
        self.content = content
        self.metadata = {} if metadata is None else metadata
        self.embedding = embedding

    def __dict__(self) -> dict:
        """
        Returns dictionary representation of a entry.
        :return: Dictionary representation.
        """
        return {"id": self.id, "content": self.content, "metadata": self.metadata, "embedding": self.embedding}


class EmbeddingFunction(object):
    """
    Class, representing embedding functions.
    """

    def __init__(self,
                 single_target_function: Callable | None = None,
                 multi_target_function: Callable | None = None,
                 language_model_instance: LanguageModelInstance = None) -> None:
        """
        Initiation method. 
        Needs at least one of the following parameters.
        :param single_target_function: Single target embedding function.
            Should be callable with an input as string,
            encoding parameters and embedding parameters as dictionaries.
        :param multi_target_function: Multitarget embedding function.
            Should be callable with an input as list of strings,
            encoding parameters and embedding parameters as dictionaries.
        :param language_model_instance: Language model instance for embedding.
        """
        if all(elem is None for elem in [single_target_function, 
                                         multi_target_function, 
                                         language_model_instance]):
            raise ValueError(
                "At least one of the initiation parameters needs to be given.")

        if single_target_function is not None:
            self.single_target_function = single_target_function
        elif multi_target_function is not None:
            self.single_target_function = lambda *args: multi_target_function(
                [args[0]], *args[1:])[0]
        elif language_model_instance:
            self.single_target_function = lambda *args: language_model_instance.embed(
                *args)

        if multi_target_function is not None:
            self.multi_target_function = multi_target_function
        else:
            self.multi_target_function = lambda *args: [
                self.single_target_function(elem, *args[1:]) for elem in args[0]]

    def __call__(self,
                 input: Union[str, List[str]],
                 encoding_parameters: dict | None = None,
                 embedding_parameters: dict | None = None,
                 ) -> Union[List[float], List[List[float]]]:
        """
        Method for embedding an input.
        :param input: Input to embed as string or list of strings.
        :param encoding_parameters: Kwargs for encoding as dictionary.
            Defaults to None.
        :param embedding_parameters: Kwargs for embedding as dictionary.
            Defaults to None.
        """
        if isinstance(input, str):
            return self.single_target_function(input, encoding_parameters, embedding_parameters)
        else:
            return self.multi_target_function(input, encoding_parameters, embedding_parameters)


class Knowledgebase(ABC):
    """
    Abstract class, representing knowledgebases.
    """
    
    @abstractmethod
    def retrieve_entries(self, 
                           query: str, 
                           filtermasks: List[FilterMask] | None = None, 
                           retrieval_method: str | None = None, 
                           retrieval_parameters: dict | None = None,
                           entry_type: str = "base") -> List[Entry]:
        """
        Method for retrieving entries.
        :param query: Retrieval query.
        :param filtermasks: List of filtermasks.
            Defaults to None.
        :param retrieval_method: Retrieval method.
            Defaults to None.
        :param retrieval_parameters: Retrieval parameters.
            Defaults to None.
        :param entry_type: Target entry type.
            Defaults to "base".
        :return: Retrieved entries.
        """
        pass

    @abstractmethod
    def embed_entries(self,
                        entries: List[Entry], 
                        embedding_parameters: dict | None = None, 
                        entry_type: str = "base") -> None:
        """
        Method for embedding entries.
        :param entries: Entries to embed.
        :param embedding_parameters: Embedding parameters.
            Defaults to None.
        :param entry_type: Target entry type.
            Defaults to "base".
        """
        pass

    @abstractmethod
    def store_embeddings(self,
                        embeddings: List[list], 
                        metadatas: List[list] = None, 
                        ids: List[Union[int, str]] = None, 
                        entry_type: str = "base") -> None:
        """
        Method for storing embeddings.
        :param embeddings: Embeddings to store.
        :param metadatas: Metadata entries to attach to embedding of the same index.
            Defaults to None.
        :param ids: IDs to store the embedding of the same index under.
            Defaults to None.
        :param embeddings: Entries to embed.
        :param entry_type: Target entry type.
            Defaults to "base".
        """
        pass

    @abstractmethod
    def update_entry(self, 
                        entry: Entry, 
                        entry_type: str = "base") -> None:
        """
        Abstract method for updating a entry in the knowledgebase.
        :param entry: Entry update.
        :param entry_type: Target entry type.
            Defaults to "base".
        """
        pass

    @abstractmethod
    def delete_entry(self, 
                        entry_id: Union[int, str], 
                        entry_type: str = "base") -> None:
        """
        Abstract method for deleting a entry from the knowledgebase.
        :param entry_id: Entry ID.
        :param entry_type: Target entry type.
            Defaults to "base".
        """
        pass

    @abstractmethod
    def get_all_entries(self,
                         entry_type: str = "base") -> List[Entry]:
        """
        Method for retrieving all entries.
        :param entry_type: Target entry type.
            Defaults to "base".
        :return: Retrieved entries.
        """
        pass

    @abstractmethod
    def create_entry_storage(self,
                             entry_type: str,
                             *args: Optional[Any],
                             **kwargs: Optional[Any]) -> None:
        """
        Abstract method for creating entry storages for the knowledgebase.
        :param entry_type: Collection name.
        :param args: Arbitrary arguments.
        :param kwargs: Arbitrary keyword arguments.
        """
        pass

    @abstractmethod
    def delete_entry_storage(self,
                             entry_type: str | None = None) -> None:
        """
        Abstract method for deleting entry storages from the knowledgebase.
        :param entry_type: Target entry type.
            Defaults to None in which case all entry types are wiped.
        """
        pass


class ChromaKnowledgebase(Knowledgebase):
    """
    Represents a chroma based knowledgebase.
    """
    def __init__(self,
                 knowledgebase_path: str,
                 embedding_function: ChromaEmbeddingFunction | None = None,
                 knowledgebase_parameters: dict | None = None,
                 retrieval_parameters: dict | None = None) -> None:
        """
        Initiates chroma based knowledgebase.
        :param knowledgebase_path: Knowledgebase path for permanent storage on disk.
        :param embedding_model: Embedding function.
            Defaults to None.
        :param knowledgebase_parameters: Knowledgebase instantiation parameters.
            Defaults to None.
        :param retrieval_parameters: Retrieval parameters.
            Defaults to None.
        """
        self.knowledgebase_path = knowledgebase_path
        self.embedding_function = ChromaDefaultEmbeddingFunction() if embedding_function is None else embedding_function
        self.knowledgebase_parameters = {} if knowledgebase_parameters is None else knowledgebase_parameters
        self.retrieval_parameters = {} if retrieval_parameters is None else retrieval_parameters

        settings = Settings(
            persist_directory=self.knowledgebase_path,
            is_persistent=True,
            anonymized_telemetry=False
        )
        for parameter in [param for param in self.knowledgebase_parameters if hasattr(settings, param)]:
            setattr(settings, parameter, self.knowledgebase_parameters[parameter])
        self.client = PersistentClient(
            path=knowledgebase_path,
            settings=settings)

        collections = self.knowledgebase_parameters.get("collections", {
            "base": {
                "embedding_function": self.embedding_function
            }
        })
        self.collections = {
            collection: self.client.get_or_create_collection(
                name=collection,
                **collections[collection])
                for collection in collections
        }

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
    
    def query_result_conversion(self, query_result: ChromaQueryResult, query: str | None = None, filtermasks: List[FilterMask] | None = None) -> List[Entry]:
        """
        Method for converting a ChromaDB query result to entries.
        :param query_result: ChromaDB query result.
        :param query: Retrieval query.
        :param filtermasks: List of retrieval filter masks.
        :return: List of entries.
        """
        entries = []
        metadata_update = {} if query is None else {"retrieval_query": query}
        if filtermasks is not None:
            metadata_update["retrieval_filtermasks"] = filtermasks
        for index, id in enumerate(query_result["ids"]):
            for doc_index, text in enumerate(query_result["entries"][index]):
                new_entry = Entry(id=id[index],
                                        content=text, 
                                        metadata=query_result["metadatas"][index][doc_index],
                                        embedding=query_result["embeddings"][index][doc_index] if query_result["embeddings"] else None)
                if "distances" in query_result:
                    metadata_update["query_distance"] = query_result["distances"][index][doc_index]
                new_entry.metadata.update(metadata_update)
                entries.append(new_entry)

        return entries
    
    def retrieve_entries(self, 
                           query: str | None = None, 
                           filtermasks: List[FilterMask] | None = None, 
                           retrieval_parameters: dict | None = None,
                           collection: str = "base") -> List[Entry]:
        """
        Method for retrieving entries.
        :param query: Retrieval query.
            Defaults to None.
        :param filtermasks: List of filtermasks.
            Defaults to None.
        :param retrieval_method: Retrieval method.
            Defaults to None.
        :param retrieval_parameters: Retrieval parameters.
            Defaults to None.
        :param collection: Target collection.
            Defaults to "base".
        :return: Retrieved entries.
        """
        retrieval_parameters = copy.deepcopy(self.retrieval_parameters) if retrieval_parameters is None else copy.deepcopy(retrieval_parameters)
        if query is not None:
            retrieval_parameters["query_texts"] = [query]
        if filtermasks is not None:
            retrieval_parameters["where"] = self.filtermasks_conversion(filtermasks)
        if "include" not in retrieval_parameters:
            retrieval_parameters["include"] = ["embeddings", "metadatas", "entries", "distances"]

        result = self.collections[collection].query(
            **retrieval_parameters
        )
        return self.query_result_conversion(result)

    def embed_entries(self,
                        entries: List[Entry], 
                        embedding_parameters: dict | None = None, 
                        collection: str = "base") -> None:
        """
        Method for embedding entries.
        :param entries: Entries to embed.
        :param embedding_parameters: Embedding parameters.
            Defaults to None and is not used for the ChromaDB backend.
        :param collection: Target collection.
            Defaults to "base".
        """
        data = {
            "embeddings": {"ids": [], "entries": [], "metadatas": [], "embeddings":[]},
            "no_embeddings": {"ids": [], "entries": [], "metadatas": []}
        }
        
        for entry in entries:
            if entry.embedding is None:
                data["no_embeddings"]["ids"].append(entry.id)
                data["no_embeddings"]["entries"].append(entry.content)
                data["no_embeddings"]["metadatas"].append(entry.metadata)
            else:
                data["with_embeddings"]["ids"].append(entry.id)
                data["with_embeddings"]["entries"].append(entry.content)
                data["with_embeddings"]["metadatas"].append(entry.metadata)
                data["with_embeddings"]["embeddings"].append(entry.embedding)

        if data["no_embeddings"]["ids"]:
            self.collections[collection].add(**data["no_embeddings"])
        if data["with_embeddings"]["ids"]:
            self.collections[collection].add(**data["with_embeddings"])

    def store_embeddings(self,
                        ids: List[Union[int, str]],
                        embeddings: List[list], 
                        contents: List[str] = None, 
                        metadatas: List[list] = None, 
                        collection: str = "base") -> None:
        """
        Method for storing embeddings.
        :param ids: IDs to store the embedding of the same index under.
        :param embeddings: Embeddings to store.
        :param contents: Content entries to attach to embedding of the same index.
            Defaults to None.
        :param metadatas: Metadata entries to attach to embedding of the same index.
            Defaults to None.
        :param embeddings: Entries to embed.
        :param collection: Target collection.
            Defaults to "base".
        """
        self.collections[collection].add(
            ids=ids,
            embeddings=embeddings,
            entries=contents,
            metadatas=metadatas
        )

    def update_entry(self, 
                        entry: Entry, 
                        collection: str = "base") -> None:
        """
        Method  for updating a entry in the knowledgebase.
        :param entry: Entry update.
        :param collection: Target collection.
            Defaults to "base".
        """
        self.collections[collection].update(
            ids=entry.id,
            entries=entry.content,
            metadatas=entry.metadata,
            embeddings=entry.embedding
        )

    def delete_entry(self, 
                        entry_id: Union[int, str], 
                        collection: str = "base") -> None:
        """
        Method  for deleting a entry from the knowledgebase.
        :param entry_id: Entry ID.
        :param collection: Target collection.
            Defaults to "base".
        """
        self.collections[collection].delete(
            ids=entry_id
        )

    def get_all_entries(self,
                          collection: str = "base") -> List[Entry]:
        """
        Method for retrieving all entries.
        :param collection: Target collection.
            Defaults to "base".
        :return: Retrieved entries.
        """
        return self.query_result_conversion(self.collections[collection].get())

    def create_entry_storage(self,
             collection: str,
             metadata: dict | None = None,
             embedding_function: ChromaEmbeddingFunction = None) -> None:
        """
        Method for creating collections for the knowledgebase.
        :param collection: Collection name.
        :param metadata: Collection metadata.
        :param embedding_function: Collection embedding function.
        """
        self.collections[collection] = self.client.get_or_create_collection(
            name=collection,
            metadata=metadata,
            embedding_function=embedding_function
        )

    def delete_entry_storage(self,
             collection: str | None = None) -> None:
        """
        Method for deleting collections from the knowledgebase.
        :param collection: Target collection.
            Defaults to None in which case all collections are wiped.
        """
        if collection is None:
            for collection in self.collections:
                self.client.delete_collection(collection)
        else:
            self.client.delete_collection(collection)

