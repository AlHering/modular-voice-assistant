# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import traceback
import copy
from abc import ABC, abstractmethod
from datetime import datetime as dt
from pydantic import BaseModel, Field
from typing import List, Any, Callable, Optional, Union, Tuple
from ..filter_mask import FilterMask
from uuid import uuid4
from src.utility.gold.text_generation.language_model_abstractions import LanguageModelInstance
from chromadb import Settings, PersistentClient, EmbeddingFunction as ChromaEmbeddingFunction, QueryResult as ChromaQueryResult
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction as ChromaDefaultEmbeddingFunction


"""
Abstractions
"""
class Document(object):
    """
    Class, representing documents.
    """
    def __init__(self, id: Union[int, str], content: str, metadata: dict | None = None, embedding: List[float] | None = None) -> None:
        """
        Initiation method.
        :param id: ID of the document.
        :param content: Textual content of the document.
        :param metadata: Metadata of the document.
            Defaults to None.
        :param embedding: Embedding of the document.
            Defaults to None.
        """
        self.id = id
        self.content = content
        self.metadata = {} if metadata is None else metadata
        self.embedding = embedding


class EmbeddingFunction(object):
    """
    Class, representing embedding functions.
    """

    def __init__(self,
                 single_target_function: Callable | None = None,
                 multi_target_function: Callable | None = None,
                 language_model_instance: LanguageModelInstance = None) -> None:
        """
        Intiation method. 
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
    def retrieve_documents(self, 
                           query: str, 
                           filtermasks: List[FilterMask] | None = None, 
                           retrieval_method: str | None = None, 
                           retrieval_parameters: dict | None = None,
                           collection: str = "base") -> List[Document]:
        """
        Method for retrieving documents.
        :param query: Retrieval query.
        :param filtermasks: List of filtermasks.
            Defaults to None.
        :param retrieval_method: Retrieval method.
            Defaults to None.
        :param retrieval_parameters: Retrieval parameters.
            Defaults to None.
        :param collection: Target collection.
            Defaults to "base".
        :return: Retrieved documents.
        """
        pass

    @abstractmethod
    def embed_documents(self,
                        documents: List[Document], 
                        embedding_parameters: dict | None = None, 
                        collection: str = "base") -> None:
        """
        Method for embedding documents.
        :param documents: Documents to embed.
        :param embedding_parameters: Embedding parameters.
            Defaults to None.
        :param collection: Target collection.
            Defaults to "base".
        """
        pass

    @abstractmethod
    def store_embeddings(self,
                        embeddings: List[list], 
                        metadatas: List[list] = None, 
                        ids: List[Union[int, str]] = None, 
                        collection: str = "base") -> None:
        """
        Method for storing embeddings.
        :param embeddings: Embeddings to store.
        :param metadatas: Metadata entries to attach to embedding of the same index.
            Defaults to None.
        :param ids: IDs to store the embedding of the same index under.
            Defaults to None.
        :param embeddings: Documents to embed.
        :param collection: Target collection.
            Defaults to "base".
        """
        pass

    @abstractmethod
    def update_document(self, 
                        document: Document, 
                        collection: str = "base") -> None:
        """
        Abstract method for updating a document in the knowledgebase.
        :param document: Document update.
        :param collection: Target collection.
            Defaults to "base".
        """
        pass

    @abstractmethod
    def delete_document(self, 
                        document_id: Union[int, str], 
                        collection: str = "base") -> None:
        """
        Abstract method for deleting a document from the knowledgebase.
        :param document_id: Document ID.
        :param collection: Target collection.
            Defaults to "base".
        """
        pass

    @abstractmethod
    def get_all_documents(self,
                         collection: str = "base") -> List[Document]:
        """
        Method for retrieving all documents.
        :param collection: Target collection.
            Defaults to "base".
        :return: Retrieved documents.
        """
        pass

    @abstractmethod
    def create_collection(self,
             collection: str) -> None:
        """
        Abstract method for creating collections for the knowledgebase.
        :param collection: Collection name.
        """
        pass

    @abstractmethod
    def delete_collection(self,
             collection: str | None = None) -> None:
        """
        Abstract method for deleting collections from the knowledgebase.
        :param collection: Target collection.
            Defaults to None in which case all collections are wiped.
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
            "!": lambda x: {"$ne": x}
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
                        ] for filtermask in filtermasks 
                    }
                ] 
            }
    
    def query_result_conversion(self, query_result: ChromaQueryResult) -> List[Document]:
        """
        Method for converting a ChromaDB query result to documents.
        :param query_result: ChromaDB query result.
        :return: List of documents.
        """
        documents = []
        for index, id in enumerate(query_result.ids):
            documents.extend([
                Document(id=id, 
                         content=doc, 
                         metadata=query_result.metadatas[index][doc_index],
                         embedding=query_result.embeddings[index][doc_index] if query_result.embeddings else None
                ) for doc_index, doc in enumerate(query_result.documents[index])
            ])
        return documents
    
    def retrieve_documents(self, 
                           query: str | None = None, 
                           filtermasks: List[FilterMask] | None = None, 
                           retrieval_parameters: dict | None = None,
                           collection: str = "base") -> List[Document]:
        """
        Method for retrieving documents.
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
        :return: Retrieved documents.
        """
        retrieval_parameters = copy.deepcopy(self.retrieval_parameters) if retrieval_parameters is None else copy.deepcopy(retrieval_parameters)
        if query is not None:
            retrieval_parameters["query_texts"] = [query]
        if filtermasks is not None:
            retrieval_parameters["where"] = self.filtermasks_conversion(filtermasks)
        if "include" not in retrieval_parameters:
            retrieval_parameters["include"] = ["embeddings", "metadatas", "documents"]
        result = self.collections[collection].query(
            **retrieval_parameters
        )
        return self.query_result_conversion(result)

    def embed_documents(self,
                        documents: List[Document], 
                        embedding_parameters: dict | None = None, 
                        collection: str = "base") -> None:
        """
        Method for embedding documents.
        :param documents: Documents to embed.
        :param embedding_parameters: Embedding parameters.
            Defaults to None and is not used for the ChromaDB backend.
        :param collection: Target collection.
            Defaults to "base".
        """
        data = {
            "embeddings": {"ids": [], "documents": [], "metadatas": [], "embeddings":[]},
            "no_embeddings": {"ids": [], "documents": [], "metadatas": []}
        }
        
        for document in documents:
            if document.embedding is None:
                data["no_embeddings"]["ids"].append(document.id)
                data["no_embeddings"]["documents"].append(document.content)
                data["no_embeddings"]["metadatas"].append(document.metadata)
            else:
                data["embeddings"]["ids"].append(document.id)
                data["embeddings"]["documents"].append(document.content)
                data["embeddings"]["metadatas"].append(document.metadata)
                data["embeddings"]["embeddings"].append(document.embedding)

        if data["no_embeddings"]["ids"]:
            self.collections[collection].add(**data["no_embeddings"])
        if data["embeddings"]["ids"]:
            self.collections[collection].add(**data["embeddings"])

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
        :param embeddings: Documents to embed.
        :param collection: Target collection.
            Defaults to "base".
        """
        self.collections[collection].add(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas
        )

    def update_document(self, 
                        document: Document, 
                        collection: str = "base") -> None:
        """
        Method  for updating a document in the knowledgebase.
        :param document: Document update.
        :param collection: Target collection.
            Defaults to "base".
        """
        self.collections[collection].update(
            ids=document.id,
            documents=document.content,
            metadatas=document.metadata,
            embeddings=document.embedding
        )

    def delete_document(self, 
                        document_id: Union[int, str], 
                        collection: str = "base") -> None:
        """
        Method  for deleting a document from the knowledgebase.
        :param document_id: Document ID.
        :param collection: Target collection.
            Defaults to "base".
        """
        self.collections[collection].delete(
            ids=document_id
        )

    def get_all_documents(self,
                          collection: str = "base") -> List[Document]:
        """
        Method for retrieving all documents.
        :param collection: Target collection.
            Defaults to "base".
        :return: Retrieved documents.
        """
        return self.query_result_conversion(self.collections[collection].get())

    def create_collection(self,
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

    def delete_collection(self,
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


def create_memory_timestamp() -> str:
    """
    Returns memory timestamp.
    :return: Ctime formated timestamp.
    """
    return dt.ctime(dt.now())


class MemoryMetadata(BaseModel):
    """
    Represents a memory entry metadata.
    """
    timestamp: str = Field(default_factory=create_memory_timestamp)
    importance: int = -1
    layer: int = 0


class MemoryEntry(BaseModel):
    """
    Represents a memory entry.
    """
    id: Union[int, str]
    content: str
    embedding: List[float] | None = None
    metadata: MemoryMetadata = MemoryMetadata()


class Memory(object):
    """
    Represents a knowledgebase based memory.
    """
    def __init__(self, knowledgebase: Knowledgebase, memories: List[MemoryEntry] | None = None) -> None:
        """
        Initiation method.
        :param knowledgebase: Knowledgebase for managing memories.
        :param memories: List of memory entries for initialization.
            Defaults to None.
        """
        self.knowledgebase = knowledgebase
        self._initiate_memory(memories=memories)

    @classmethod
    def from_knowledgebase(cls, knowledgebase: Knowledgebase) -> Any:
        """
        Returns session instance from a json file.
        :param file_path: Path to json file.
        :returns: VoiceAssistantSession instance.
        """
        return cls(knowledgebase=knowledgebase)

    def _initiate_memory(self, memories: List[MemoryEntry] | None = None) -> None:
        """
        Method for initiating memories.
        :param memories: List of memory entries for initialization.
            Defaults to None.
        """
        if memories is not None:
            for memory in memories:
                self.add_memory(memory)

    def memory_to_document(self, memory: MemoryEntry) -> Document:
        """
        Method for converting memory entries to documents.
        :param memory: Memory entry.
        :return: Document.
        """
        return Document(
            id=memory.id,
            content=memory.content,
            metadata=memory.metadata.model_dump(),
            embedding=memory.embedding
        )
    
    def document_to_memory(self, document: Document) -> MemoryEntry:
        """
        Method for converting documents to memory entries.
        :param document: Document.
        :return: Memory entry.
        """
        return MemoryEntry(
            id=document.id,
            content=document.content,
            metadata=MemoryMetadata(**document.metadata),
            embedding=document.embedding
        )

    def memorize(self, content: str, metadata: dict) -> None:
        """
        Method for memorizing something.
        This method should be used for memory model agnostic usage.
        :param content: Memory content.
        :param metadata: Metadata for memory.
        """
        self.add_memory(MemoryEntry(
            id=str(uuid4()),
            content=content,
            metadata=MemoryMetadata()
        ))

    def remember(self, 
                 reference: str, 
                 min_importance: int | None = None, 
                 min_layer: int | None = None) -> Optional[List[Tuple[str, dict]]]:
        """
        Method for remembering something.
        This method should be used for memory model agnostic usage.
        :param reference: Recall reference.
        :param min_importance: Minimum importance of the memory.
            Defaults to None.
        :param min_layer: Minimum layer of the memory.
            Defaults to None.
        :return: Memory contents as list of strings.
        """
        filtermask = []
        if min_importance is not None:
            filtermask.append(["importance", ">=", min_importance])
        if min_layer is not None:
            filtermask.append(["layer", ">=", min_layer])
        if filtermask:
            memories = self.retrieve_memories_by_similarity(
                    reference=reference,
                    filtermasks=[filtermask])
        else:
            memories = self.retrieve_memories_by_similarity(
                    reference=reference)
        return [(memory.content, memory.metadata.model_dump()) for memory in memories]

    def add_memory(self, memory: MemoryEntry) -> None:
        """
        Method to add a memory.
        :param memory: Memory to add.
        """
        self.knowledgebase.embed_documents(
            documents=[Document(
                id=memory.id,
                content=memory.content,
                metadata=memory.metadata.model_dump(),
                embedding=memory.embedding
            )]
        )

    def retrieve_memories(self) -> List[MemoryEntry]:
        """
        Method to retrieve memories.
        :return: List of memories.
        """
        return [self.document_to_memory(doc) for doc in self.knowledgebase.get_all_documents()]

    def retrieve_memories_by_similarity(self, reference: str, filtermasks: List[FilterMask] | None = None, retrieval_parameters: dict | None = None) -> List[MemoryEntry]:
        """
        Method for retrieving memories by similarity.
        :param reference: Reference for similarity search.
        :param filtermasks: List of filtermasks for additional filering.
            Defaults to None.
        :param retrieval_parameters: Keyword arguments for retrieval.
            Defaults to None.
        """
        return [self.document_to_memory(doc) for doc in
            self.knowledgebase.retrieve_documents(query=reference, filtermasks=filtermasks, retrieval_parameters=retrieval_parameters)]

"""
Templates
"""
TEMPLATES = {

}


"""
Interfacing
"""
def spawn_knowledgebase_instance(template: str) -> Union[Any, dict]:
    """
    Function for spawning knowledgebase instances based on configuration templates.
    :param template: Instance template.
    :return: Knowledgebase instance if configuration was successful else an error report.
    """
    # TODO: Research common parameter pattern for popular knowledgebase backends
    # TODO: Update interfacing and move to gold utility
    # TODO: Support ChromaDB, SQLite-VSS, FAISS, PGVector, Qdrant, Pinecone, Redis, Langchain Vector DB Zoo(?)
    try:
        return Knowledgebase(**TEMPLATES[template])
    except Exception as ex:
        return {"exception": ex, "trace": traceback.format_exc()}
