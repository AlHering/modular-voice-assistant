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
Abstractions
"""


class Document(object):
    """
    Document abstraction, following common data model.
    """
    page_content: str
    metadata: dict


class EmbeddingFunction(object):
    """
    Embedding function abstraction.
    """

    def __init__(self,
                 single_target_function: Callable = None,
                 multi_target_function: Callable = None,
                 language_model_instance: LanguageModelInstance = None
                 ) -> None:
        """
        Intiation method. 
        Needs at least one of the following paramters.
        :param single_target_function: Single target embedding function.
            Should be callable with an input as string,
            encoding parameters and embedding paramters as dictionaries.
        :param multi_target_function: Multitarget embedding function.
            Should be callable with an input as list of strings,
            encoding parameters and embedding paramters as dictionaries.
        :param language_model_instance: Language model instance for embedding.
        """
        if all(elem is None for elem in [single_target_function, multi_target_function, language_model_instance]):
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
                 encoding_parameters: dict = None,
                 embedding_parameters: dict = None,
                 ) -> Union[List[float], List[List[float]]]:
        """
        Method for embedding an input.
        :param input: Input to embed as string or list of strings.
        :param encoding_parameters: Kwargs for encoding as dictionary.
            Defaults to None.
        :param embedding_paramters: Kwargs for embedding as dictionary.
            Defaults to None.
        """
        if isinstance(input, str):
            return self.single_target_function(input, encoding_parameters, embedding_parameters)
        else:
            return self.multi_target_function(input, encoding_parameters, embedding_parameters)


class VectorStore(abc.ABC):
    """
    Abstract class for vector stores.
    backend = Column(String, nullable=False,
                         comment="Backend of the knowledgebase instance.")
        knowledgebase_path = Column(String, nullable=False,
                                    comment="Path of the knowledgebase instance.")
        knowledgebase_parameters = Column(JSON,
                                          comment="Parameters for the knowledgebase instantiation.")
        preprocessing_parameters = Column(JSON,
                                          comment="Parameters for document preprocessing.")
        embedding_parameters = Column(JSON,
                                      comment="Parameters for document embedding.")
        retrieval_parameters = Column(JSON,
                                      comment="Parameters for the document retrieval.")
    """

    def __init__(self,
                 backend: str,
                 embedding_function: EmbeddingFunction,
                 knowledgebase_path: str = None,
                 knowledgebase_parameters: dict = None,
                 preprocessing_parameters: dict = None,
                 embedding_parameters: dict = None,
                 retrieval_parameters: dict = None) -> None:
        """
        Initiation method.
        :param backend: Knowledgebase backend.
        :param embedding_model: Embedding model instance.
        :param knowledgebase_path: Knowledgebase path for permanent storage on disk.
            Defaults to None.
        :param knowledgebase_parameters: Knowledgebase instantiation parameters.
            Defaults to None.
        :param preprocessing_parameters: Document preprocessing paramters.
            Defaults to None.
        :param embedding_parameters: Embedding parameters.
            Defaults to None.
        :param retrieval_parameters: Retrieval parameters.
            Defaults to None.
        """
        self.backend = backend
        self.embedding_function = embedding_function
        self.knowledgebase_path = knowledgebase_path
        self.knowledgebase_parameters = {
        } if knowledgebase_parameters is None else knowledgebase_parameters
        self.preprocessing_parameters = {
        } if preprocessing_parameters is None else preprocessing_parameters
        self.embedding_parameters = {
        } if embedding_parameters is None else embedding_parameters
        self.retrieval_parameters = {
        } if retrieval_parameters is None else retrieval_parameters

    @abc.abstractmethod
    def get_or_create_collection(self, collection: str, metadata: dict = None, embedding_function: EmbeddingFunction = None) -> Any:
        """
        Method for retrieving or creating a collection.
        :param collection: Collection collection.
        :param metadata: Embedding collection metadata. Defaults to None.
        :param embedding_function: Embedding function for the collection. Defaults to base embedding function.
        :return: Database API.
        """
        pass

    @abc.abstractmethod
    def get_retriever(self, collection: str, search_type: str = "similarity", search_kwargs: dict = {"k": 4, "include_metadata": True}) -> langchain_utility.VectorStoreRetriever:
        """
        Method for acquiring a retriever.
        :param collection: Collection to use.
        :param search_type: The retriever's search type. Defaults to "similarity".
        :param search_kwargs: The retrievery search keyword arguments. Defaults to {"k": 4, "include_metadata": True}.
        :return: Retriever instance.
        """
        pass

    @abc.abstractmethod
    def retrieve_documents(self, query: str, collection: str, metadata_constraints: dict = None, search_type: str = "similarity", search_kwargs: dict = {"k": 4, "include_metadata": True}) -> List[Document]:
        """
        Method for acquiring documents.
        :param query: Retrieval query.
        :param metadata_constraints: Metadata constraints.
        :param collection: Collection to use.
        :param search_type: The retriever's search type. Defaults to "similarity".
        :param search_kwargs: The retrievery search keyword arguments. Defaults to {"k": 4, "include_metadata": True}.
        :return: Retrieved documents.
        """
        pass

    @abc.abstractmethod
    def embed_documents(self, documents: List[Document], metadatas: List[dict] = None, ids: List[str] = None, collection: str = "base", compute_additional_metadata: bool = False) -> None:
        """
        Method for embedding documents.
        :param documents: Documents to embed.
        :param metadatas: Metadata entries. 
            Defaults to None.
        :param ids: Custom IDs to add. 
            Defaults to the hash of the document contents.
        :param collection: Collection to use.
            Defaults to "base".
        :param compute_additional_metadata: Flag for declaring, whether to compute additional metadata.
            Defaults to False.
        """
        pass

    @abc.abstractmethod
    def delete_document(self, document_id: Any, collection: str = "base") -> None:
        """
        Abstract method for deleting a document from the knowledgebase.
        :param document_id: Document ID.
        :param collection: Collection to remove document from.
        """
        pass

    @abc.abstractmethod
    def wipe_knowledgebase(self) -> None:
        """
        Abstract method for wiping knowledgebase.
        """
        pass

    def compute_additional_metadata(self, doc_content: str, collection: str = "base", **kwargs: Optional[Any]) -> dict:
        """
        Method for computing additional metadata from content.
        :param doc_content: Document content.
        :param collection: Target collection.
            Defaults to "base".
        :param kwargs: Arbitary keyword arguments.
        """
        return {}

    def load_folder(self, folder: str, target_collection: str = "base", splitting: Tuple[int] = None, compute_additional_metadata: bool = False) -> None:
        """
        Method for (re)loading folder contents.
        :param folder: Folder path.
        :param target_collection: Collection to handle folder contents. Defaults to "base".
        :param splitting: A tuple of chunk size and overlap for splitting. Defaults to None in which case the documents are not split.
        :param compute_additional_metadata: Flag for declaring, whether to compute additionail metadata.
            Defaults to False.
        """
        file_paths = []
        for root, dirs, files in os.walk(folder, topdown=True):
            file_paths.extend([os.path.join(root, file) for file in files])

        self.load_files(file_paths, target_collection, splitting)

    def load_files(self, file_paths: List[str], target_collection: str = "base", splitting: Tuple[int] = None, compute_additional_metadata: bool = False) -> None:
        """
        Method for (re)loading file paths.
        :param file_paths: List of file paths.
        :param target_collection: Collection to handle folder contents. Defaults to "base".
        :param splitting: A tuple of chunk size and overlap for splitting. Defaults to None in which case the documents are not split.
        :param compute_additional_metadata: Flag for declaring, whether to compute additional metadata.
            Defaults to False.
        """
        document_paths = [file for file in file_paths if any(file.lower().endswith(
            supported_extension) for supported_extension in langchain_utility.DOCUMENT_LOADERS)]
        documents = []

        for index, document_path in enumerate(document_paths):
            documents.append(reload_document(document_path))

        if splitting is not None:
            documents = self.split_documents(documents, *splitting)

        self.embed_documents(target_collection, documents)

    def split_documents(self, documents: List[Document], split: int, overlap: int) -> List[Document]:
        """
        Method for splitting document content.
        :param documents: Documents to split.
        :param split: Chunk size to split documents into.
        :param overlap: Overlap for split chunks.
        :return: Split documents.
        """
        pass


"""
Utility
"""


def reload_document(document_path: str) -> Document:
    """
    Function for (re)loading document content.
    :param document_path: Document path.
    :return: Document object.
    """
    res = langchain_utility.DOCUMENT_LOADERS[os.path.splitext(document_path)[
        1]](document_path).load()
    return res[0] if isinstance(res, list) and len(res) == 1 else res


"""
Vector store instantiation functions
"""


"""
Interfacing
"""


def spawn_knowledgebase_instance(*args: Optional[Any], **kwargs: Optional[Any]) -> Union[Any, dict]:
    """
    Function for spawning knowledgebase instances based on configuration arguments.
    :param args: Arbitrary initiation arguments.
    :param kwargs: Arbitrary initiation keyword arguments.
    :return: Language model instance if configuration was successful else an error report.
    """
    # TODO: Research common parameter pattern for popular knowledgebase backends
    # TODO: Update interfacing and move to gold utility
    # TODO: Support ChromaDB, SQLite-VSS, FAISS, PGVector, Qdrant, Pinecone, Redis, Langchain Vector DB Zoo(?)
    try:
        pass
    except Exception as ex:
        return {"exception": ex, "trace": traceback.format_exc()}
