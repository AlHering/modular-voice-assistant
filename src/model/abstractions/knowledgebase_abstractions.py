# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2025 Alexander Hering             *
****************************************************
"""
import copy
from abc import ABC, abstractmethod
from uuid import UUID
from typing import List, Union
from src.utility.chroma_utility import ChromaStorage
from src.utility.neo4j_utility import neo4j, Neo4jStorage
from src.utility.filter_mask_utility import FilterMask
from chromadb import QueryResult as ChromaQueryResult


"""
Abstractions
"""
class Entry(object):
    """
    Class, representing entries.
    """
    def __init__(self, id: int | str | UUID, 
                 content: str, metadata: dict | None = None, 
                 embeddings: List[int | float | List[int | float]] | None = None) -> None:
        """
        Initiation method.
        :param id: ID of the entry.
        :param content: Textual content of the entry.
        :param metadata: Metadata of the entry.
        :param embeddings: Embeddings of the entry.
        """
        self.id: int | str | UUID  = id
        self.content: str = content
        self.metadata: dict = {} if metadata is None else metadata
        self.embeddings: List[int | float | List[int | float]] | None = embeddings

    def __dict__(self) -> dict:
        """
        Returns dictionary representation of a entry.
        :return: Dictionary representation.
        """
        return {"id": self.id, "content": self.content, "metadata": self.metadata, "embeddings": self.embeddings}

    def __call__(self,
                 input: Union[str, List[str]],
                 encoding_parameters: dict | None = None,
                 embedding_parameters: dict | None = None,
                 ) -> Union[List[float], List[List[float]]]:
        """
        Method for embedding an input.
        :param input: Input to embed as string or list of strings.
        :param encoding_parameters: Kwargs for encoding as dictionary.
        :param embedding_parameters: Kwargs for embedding as dictionary.
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
    def store_entry(self, 
                    entry: Entry) -> None:
        """
        Method for storing entries.
        :param entry: Entry to store.
        """
        pass

    @abstractmethod
    def update_entry(self,
                     entry_or_id: int | str | UUID | Entry,
                     patch: dict | Entry) -> None:
        """
        Abstract method for updating a entry in the knowledgebase.
        :param entry_or_id: Entry or entry ID.
        :param patch: Entry or dictionary for patching values.
        """
        pass

    @abstractmethod
    def delete_entry(self, 
                     entry_or_id: int | str | UUID | Entry) -> None:
        """
        Abstract method for deleting a entry from the knowledgebase.
        :param entry_or_id: Entry or entry ID.
        """
        pass
    
    @abstractmethod
    def retrieve_entries(self, 
                         query: str | None = None, 
                         filtermasks: List[FilterMask] | None = None, 
                         retrieval_parameters: dict | None = None) -> List[Entry]:
        """
        Method for retrieving entries.
        :param query: Optional retrieval query.
        :param filtermasks: List of filtermasks.
        :param retrieval_parameters: Retrieval parameters.
        :return: Retrieved entries.
        """
        pass

    @abstractmethod
    def get_entry_by_id(self, entry_id: int | str | UUID) -> Entry:
        """
        Method for retrieving entries by ID.
        :param entry_id: Entry ID.
        :return: Target entry.
        """
        pass

    @abstractmethod
    def get_all_entries(self) -> List[Entry]:
        """
        Method for retrieving all entries.
        :param entry_type: Target entry type.
        :return: Retrieved entries.
        """
        pass


class ChromaKnowledgeBase(ChromaStorage):
    """
    Chroma based knowledgebase.
    """
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
    
    def store_entry(self, 
                    entry: Entry) -> None:
        """
        Method for storing entries.
        :param entry: Entry to store.
        """
        self.collection.add(
            ids=entry.id, 
            embeddings=entry.embeddings,
            metadatas=entry.metadata,
            documents=entry.content
        )

    def update_entry(self,
                     entry_or_id: int | str | UUID | Entry,
                     patch: dict) -> None:
        """
        Abstract method for updating a entry in the storage.
        :param entry_or_id: Entry or entry ID.
        :param patch: Dictionary for patching values.
        """
        entry = entry_or_id if isinstance(entry_or_id, Entry) else self.get_entry_by_id(entry_id=entry_or_id)
        self.collection.update(
            ids=entry_or_id.id if isinstance(entry_or_id, Entry) else entry_or_id,
            entries=patch.get("content", entry.content),
            metadatas=patch.get("metadata", entry.metadata),
            embeddings=patch.get("embeddings", entry.embeddings)
        )

    def delete_entry(self, 
                     entry_or_id: int | str | UUID | Entry) -> None:
        """
        Abstract method for deleting a entry from the storage.
        :param entry_or_id: Entry or entry ID.
        """
        entry = entry_or_id if isinstance(entry_or_id, Entry) else self.get_entry_by_id(entry_id=entry_or_id)
        self.collection.delete(
            ids=entry.id
        )
    
    def retrieve_entries(self, 
                         query: str | None = None, 
                         filtermasks: List[FilterMask] | None = None, 
                         retrieval_parameters: dict | None = None) -> List[Entry]:
        """
        Method for retrieving entries.
        :param query: Optional retrieval query.
        :param filtermasks: List of filtermasks.
        :param retrieval_parameters: Retrieval parameters.
        :return: Retrieved entries.
        """
        retrieval_parameters = copy.deepcopy(self.retrieval_parameters) if retrieval_parameters is None else copy.deepcopy(retrieval_parameters)
        if query is not None:
            retrieval_parameters["query_texts"] = query
        if filtermasks is not None:
            retrieval_parameters["where"] = self.filtermasks_conversion(filtermasks)
        if "include" not in retrieval_parameters:
            retrieval_parameters["include"] = ["embeddings", "metadatas", "entries", "distances"]

        result = self.collection.query(
            **retrieval_parameters
        )
        return self.query_result_conversion(result)

    def get_entry_by_id(self, 
                        entry_id: int | str | UUID) -> Entry:
        """
        Method for retrieving entries by ID.
        :param entry_id: Entry ID.
        :param collection: Target collection.
        :return: Target entry.
        """
        return self.query_result_conversion(self.collection.get(ids=entry_id,
                                            include=["embeddings", "metadatas", "entries", "distances"])) 

    def get_all_entries(self) -> List[Entry]:
        """
        Method for retrieving all entries.
        :param entry_type: Target entry type.
        :return: Retrieved entries.
        """
        return self.query_result_conversion(self.collection.get(include=["embeddings", "metadatas", "entries", "distances"])) 


class Neo4jKnowledgebase(Neo4jStorage):
    """
    Neo4j based knowledgebase.
    """
    def query_result_conversion(self, query_result: neo4j.Result, query: str | None = None, filtermasks: List[FilterMask] | None = None) -> List[Entry]:
        """
        Method for converting a Neo4j query result to entries.
        :param query_result: Neo4j query result.
        :param query: Retrieval query.
        :param filtermasks: List of retrieval filter masks.
        :return: List of entries.
        """
        return []

    def store_entry(self, 
                    entry: Entry) -> None:
        """
        Method for storing entries.
        :param entry: Entry to store.
        """
        pass

    def update_entry(self,
                     entry_or_id: int | str | UUID | Entry,
                     patch: dict | Entry) -> None:
        """
        Abstract method for updating a entry in the knowledgebase.
        :param entry_or_id: Entry or entry ID.
        :param patch: Entry or dictionary for patching values.
        """
        pass

    def delete_entry(self, 
                     entry_or_id: int | str | UUID | Entry) -> None:
        """
        Abstract method for deleting a entry from the knowledgebase.
        :param entry_or_id: Entry or entry ID.
        """
        pass
    
    def retrieve_entries(self, 
                         query: str | None = None, 
                         filtermasks: List[FilterMask] | None = None, 
                         retrieval_parameters: dict | None = None) -> List[Entry]:
        """
        Method for retrieving entries.
        :param query: Optional retrieval query.
        :param filtermasks: List of filtermasks.
        :param retrieval_parameters: Retrieval parameters.
        :return: Retrieved entries.
        """
        pass

    def get_entry_by_id(self, entry_id: int | str | UUID) -> Entry:
        """
        Method for retrieving entries by ID.
        :param entry_id: Entry ID.
        :return: Target entry.
        """
        pass

    def get_all_entries(self) -> List[Entry]:
        """
        Method for retrieving all entries.
        :param entry_type: Target entry type.
        :return: Retrieved entries.
        """
        pass