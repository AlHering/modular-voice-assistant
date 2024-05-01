# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2024 Alexander Hering             *
****************************************************
"""
import traceback
from typing import List, Any, Callable, Optional, Union
from ..filter_mask import FilterMask
from src.utility.gold.text_generation.language_model_abstractions import LanguageModelInstance


"""
Abstractions
"""
class Document(object):
    """
    Class, representing documents.
    """
    def __init__(self, id: Union[int, str], content: str, metadata: dict) -> None:
        """
        Initiation method.
        :param id: ID of the document.
        :param content: Textual content of the document.
        :param metadata: Metadata of the document.
        """
        self.id = id
        self.content = content
        self.metadata = metadata


class EmbeddingFunction(object):
    """
    Class, representing embedding functions.
    """

    def __init__(self,
                 single_target_function: Callable = None,
                 multi_target_function: Callable = None,
                 language_model_instance: LanguageModelInstance = None) -> None:
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


class Knowledgebase(object):
    """
    Class for knowledgebases.
    """
    supported_backends: List[str] = []

    def __init__(self,
                 backend: str,
                 embedding_function: EmbeddingFunction,
                 knowledgebase_path: str = None,
                 knowledgebase_parameters: dict = None,
                 preprocessing_parameters: dict = None,
                 embedding_parameters: dict = None,
                 retrieval_method: str = "similarity",
                 retrieval_parameters: dict = None) -> None:
        """
        Initiation method.
        :param backend: Knowledgebase backend.
            Check Knowledgebase.supported_backends for supported backends.
        :param embedding_function: Default embedding function.
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
        self.retrieval_method = retrieval_method
        self.retrieval_parameters = {
        } if retrieval_parameters is None else retrieval_parameters

    def retrieve_documents(self, 
                           query: str, 
                           filtermasks: List[FilterMask] = None, 
                           retrieval_method: str = None, 
                           retrieval_paramters: dict = None,
                           collection: str = "base") -> List[Document]:
        """
        Method for retrieving documents.
        :param query: Retrieval query.
        :param filtermasks: List of filtermasks.
            Defaults to None.
        :param retrieval_method: Retrieval method.
            Defaults to None.
        :param retrieval_paramters: Retrieval paramters.
            Defaults to None.
        :param collection: Collection to retrieve documents from.
            Defaults to "base".
        :return: Retrieved documents.
        """
        pass

    def embed_documents(self,
                        documents: List[Document], 
                        embedding_paramters: dict = None, 
                        collection: str = "base") -> None:
        """
        Method for embedding documents.
        :param documents: Documents to embed.
        :param embedding_paramters: Embedding parameters.
            Defaults to None.
        :param collection: Collection to embed to.
            Defaults to "base".
        """
        pass

    def store_embeddings(self,
                        embeddings: List[list], 
                        metadatas: List[list] = None, 
                        ids: List[Union[int, str]] = None, 
                        embedding_paramters: dict = None, 
                        collection: str = "base") -> None:
        """
        Method for storing embeddings.
        :param embeddings: Embeddings to store.
        :param metadatas: Metadata entries to attach to embedding of the same index.
            Defaults to None.
        :param ids: IDs to store the embedding of the same index under.
            Defaults to None.
        :param embeddings: Documents to embed.
        :param embedding_paramters: Embedding parameters.
            Defaults to None.
        :param collection: Collection to embed to.
            Defaults to "base".
        """
        pass

    def load_documents_from_file(self,
                                file_path: str,
                                preprocessing_parameters: dict = None,
                                embed_after_loading: bool = False,
                                collection: str = "base") -> List[Document]:
        """
        Method for loading documents from a file.
        :param file_path: File path to load documents from.
        :param preprocessing_parameters: Preprocessing parameters.
            Defaults to None in which case the default preprocessing parameters are used.
        :param embed_after_loading: Flag for declaring whether to directly embed the loaded documents.
            Defaults to False.
        :param collection: Collection to embed to. Only is relevant if embed_after_loading is set to True.
            Defaults to "base".
        """
        pass

    def update_document(self, document: Document) -> None:
        """
        Abstract method for deleting a document from the knowledgebase.
        :param document: Document update.
        """
        pass

    def delete_document(self, 
                        document_id: Union[int, str], 
                        collection: str = "base") -> None:
        """
        Abstract method for deleting a document from the knowledgebase.
        :param document_id: Document ID.
        :param collection: Collection to embed to.
            Defaults to "base".
        """
        pass

    def get_all_documents(self,
                         collection: str = "base") -> List[Document]:
        """
        Method for retrieving all documents.
        :param collection: Collection to retrieve documents from.
            Defaults to "base".
        :return: Retrieved documents.
        """
        pass

    def wipe_knowledgebase(self) -> None:
        """
        Abstract method for wiping knowledgebase.
        """
        pass

    def write_to_storage(self) -> None:
        """
        Abstract method for writing knowledgebase to persistant storage.
        """
        pass

    def read_from_storage(self) -> None:
        """
        Abstract method for reading knowledgebase from persistant storage.
        """
        pass


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
