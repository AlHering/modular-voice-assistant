# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os
from time import sleep
from datetime import datetime as dt
from typing import Optional, Any, List, Dict, Union
from src.configuration import configuration as cfg
from src.utility.gold.basic_sqlalchemy_interface import BasicSQLAlchemyInterface
from src.utility.bronze.hashing_utility import hash_text_with_sha256
from src.model.text_generation_control.data_model import populate_data_instrastructure
from src.model.text_generation_control.llm_pool import ThreadedLLMPool
from src.utility.silver import embedding_utility
from src.utility.bronze.hashing_utility import hash_text_with_sha256
from src.utility.silver.file_system_utility import safely_create_path
from src.utility.gold.text_generation.knowledgebase_abstractions import Knowledgebase, Document, spawn_knowledgebase_instance
from src.utility.gold.text_generation.language_model_abstractions import LanguageModelInstance, spawn_language_model_instance


class TextGenerationController(BasicSQLAlchemyInterface):
    """
    Controller class for handling text generation tasks.
    """
    
    def __init__(self, working_directory: str = None, database_uri: str = None) -> None:
        """
        Initiation method.
        :param working_directory: Working directory.
            Defaults to configured backend folder.
        :param database_uri: Database URI.
            Defaults to 'backend.db' file under configured backend folder.
        """
        # Main instance variables
        self._logger = cfg.LOGGER
        self.working_directory = cfg.PATHS.BACKEND_PATH if working_directory is None else working_directory
        if not os.path.exists(self.working_directory):
            os.makedirs(self.working_directory)
        self.database_uri = f"sqlite:///{os.path.join(cfg.PATHS.BACKEND_PATH, 'backend.db')}" if database_uri is None else database_uri

        # Database infrastructure
        super().__init__(self.working_directory, self.database_uri,
                         populate_data_instrastructure, "text_generation.", self._logger)

        # LLM infrastructure
        self.llm_pool = None

        # Cache
        self._cache = None

    """
    Setup and shutdown methods
    """
    def setup(self) -> None:
        """
        Method for running setup process.
        """
        # Knowledgebase infrastructure
        self.knowledgebase_directory = os.path.join(
            self.working_directory, "knowledgebases")
        self.file_directory = os.path.join(
            self.working_directory, "files")
        safely_create_path(self.knowledgebase_directory)
        safely_create_path(self.file_directory)
        self.kbs: Dict[str, Knowledgebase] = {}
        self.documents = {}

        # LLM infrastructure
        self.llm_pool = ThreadedLLMPool()

        # Cache
        self._cache = {
            "active": {}
        }

    def shutdown(self) -> None:
        """
        Method for running shutdown process.
        """
        self.llm_pool.stop_all()
        while any(self.llm_pool.is_running(instance_id) for instance_id in self._cache):
            sleep(2.0)

    """
    LLM handling methods
    """ 
    def load_instance(self, instance_id: Union[str, int]) -> Optional[int]:
        """
        Method for loading a configured language model instance.
        :param instance_id: Instance ID.
        :return: Instance ID if process as successful.
        """
        instance_id = str(instance_id)
        if instance_id in self._cache:
            if not self.llm_pool.is_running(instance_id):
                self.llm_pool.start(instance_id)
                self._cache[instance_id]["restarted"] += 1
        else:
            self._cache[instance_id] = {
                "started": None,
                "restarted": 0,
                "accessed": 0,
                "inactive": 0
            }
            instance = self.get_object_by_id("lm_instance", int(instance_id))
            llm_config = {
                attribute: getattr(instance, attribute) for attribute in [
                    "backend",
                    "model_path",
                    "model_file",
                    "model_parameters",
                    "tokenizer_path",
                    "tokenizer_parameters",
                    "config_path",
                    "config_parameters",
                    "default_system_prompt",
                    "use_history",
                    "encoding_parameters",
                    "generating_parameters",
                    "decoding_parameters",
                    "resource_requirements"
                ]
            }

            self.llm_pool.prepare_llm(llm_config, instance_id)
            self.llm_pool.start(instance_id)
            self._cache[instance_id]["started"] = dt.now()
        return int(instance_id)

    def unload_instance(self, instance_id: Union[str, int]) -> Optional[str]:
        """
        Method for unloading a configured language model instance.
        :param instance_id: Instance ID.
        :return: Instance ID if process as successful.
        """
        instance_id = str(instance_id)
        if instance_id in self._cache:
            if self.llm_pool.is_running(instance_id):
                self.llm_pool.stop(instance_id)
            return instance_id
        else:
            return None

    def forward_generate(self, instance_id: Union[str, int], prompt: str) -> Optional[str]:
        """
        Method for forwarding a generate request to an instance.
        :param instance_id: Instance ID.
        :param prompt: Prompt.
        :return: Response.
        """
        instance_id = str(instance_id)
        self.load_instance(instance_id)
        return self.llm_pool.generate(instance_id, prompt)

    """
    Knowledgebase handling methods
    """
    def load_knowledgebase(self, kb_id: Union[str, int]) -> int:
        """
        Method for loading knowledgebase.
        :param kb_id: Knowledgebase ID.
        :return: Knowledgebase ID.
        """
        kb_id = str(kb_id)
        if kb_id not in self.kbs:
            instance = self.get_object_by_id("kb_instance", int(kb_id))
            kb_config = {
                attribute: getattr(instance, attribute) for attribute in [
                    "backend",
                    "knowledgebase_path",
                    "knowledgebase_parameters",
                    "preprocessing_parameters",
                    "embedding_parameters",
                    "default_retrieval_method",
                    "retrieval_parameters",
                ]
            }
            self.kbs = Knowledgebase(**kb_config)
        return int(kb_id)
    
    def embed_documents(
            self,
            target_kb_id: Union[str, int], 
            documents: List[Document], 
            collection: str = "base") -> None:
        """
        Method for embedding documents.
        :param target_kb_id: Target knowledgebase ID.
        :param documents: Documents to embed.
        :param collection: Target collection.
            Defaults to "base".
        """
        target_kb_id = str(target_kb_id)
        if target_kb_id not in self.kbs:
            self.load_knowledgebase(target_kb_id)
        self.kbs[target_kb_id].embed_documents(documents=documents, collection=collection)

    def embed_documents_from_file(
            self,
            target_kb_id: Union[str, int], 
            file_path: str,
            collection: str = "base") -> None:
        """
        Method for embedding documents from file.
        :param target_kb_id: Target knowledgebase ID.
        :param file_path: File path of target file.
        :param collection: Target collection.
            Defaults to "base".
        """
        self.kbs[target_kb_id].load_documents_from_file(
            file_path=file_path,
            embed_after_loading=True,
            collection=collection)

    def embed_document_data(self, 
                        target_kb_id: Union[str, int], 
                        contents: List[str], 
                        metadatas: List[dict] = None, 
                        hashes: List[str] = None, 
                        collection: str = "base") -> None:
        """
        Method for embedding documents.
        :param target_kb_id: Target knowledgebase ID.
        :param contents: Document contents to embed.
        :param metadatas: Metadata entries for documents.
            Defaults to None.
        :param ids: Custom IDs to add. 
            Defaults to the hash of the document contents.
        :param hashes: Content hashes.
            Defaults to None in which case hashes are computet.
        :param collection: Target collection.
            Defaults to "base".
        """
        hashes = [hash_text_with_sha256(content)
                  for content in contents] if hashes is None else hashes
        if metadatas is None:
            metadatas = [{"hash": hash} for hash in hashes]

        documents = [Document(id=hash, content=contents[index], metadata=metadatas[index]) for index, hash in enumerate(hashes)]
        self.embed_documents(target_kb_id, documents, collection)
        
    def delete_documents(self, target_kb_id: Union[str, int], document_ids: List[Union[str, int]], collection: str = "base") -> None:
        """
        Method for deleting a document from the knowledgebase.
        :param target_kb_id: Target knowledgebase ID.
        :param document_ids: Target document IDs.
        :param collection: Collection to remove document from.
            Defaults to "base" collection.
        """
        for document_id in document_ids:
            self.kbs[str(target_kb_id)].delete_document(document_id, collection)

    def wipe_knowledgebase(self, target_kb_id: Union[str, int]) -> None:
        """
        Method for wiping a knowledgebase.
        :param target_kb_id: Target knowledgebase ID.
        """
        self.kbs[str(target_kb_id)].wipe_knowledgebase()

    def migrate_knowledgebase(self, 
                              source_kb_id: Union[str, int], 
                              target_kb_id: Union[str, int], 
                              collection: str = "base") -> None:
        """
        Method for migrating knowledgebase.
        :param source_kb: Source knowledgebase.
        :param target_kb: Target knowledgebase.
        :param collection: Collection to migrate.
            Defaults to "base" collection.
        """
        documents = self.kbs[str(source_kb_id)].get_all_documents(collection=collection)
        self.kbs[str(source_kb_id)].embed_documents(documents=documents, collection=collection)

    """
    Additional methods
    """
    def forward_document_qa(self, 
                            llm_id: Union[int, str], 
                            kb_id: Union[int, str], 
                            query: str, 
                            collection: str = "base", 
                            include_sources: bool = True) -> dict:
        """
        Method for posting query.
        :param llm_id: LLM ID.
        :param kb_id: Knowledgebase ID.
        :param query: Query.
        :param collection: Collection to retrieve from.
            Defaults to "base" collection.
        :param include_sources: Flag declaring, whether to include sources.
        :return: Response.
        """
        docs = self.kbs[kb_id].retrieve_documents(query=query, collection=collection)

        document_list = "'''" + "\n\n '''".join(
            [doc.content for doc in docs]) + "'''"
        generation_prompt = f"Answer the question '''{query}''' with the following information: \n\n {document_list}"
        answer = self.forward_generate(llm_id, generation_prompt)

        return {
            "answer": answer, 
            "sources": {
                doc.id: {"metadata": doc.metadata, "content": doc.content} 
                for doc in docs
            } if include_sources else {}}
