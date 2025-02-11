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
from typing import List, Callable, Union, Any, Dict
from enum import Enum
import json
import neo4j
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
        :param knowledgebase_parameters: Knowledgebase instantiation parameters.
        :param retrieval_parameters: Default retrieval parameters.
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

        self.collection = self.client.get_or_create_collection(
            name="base",
            embedding_function= self.embedding_function
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

    """
    Extended access
    """

    def store_embeddings(self,
                        ids: List[Union[int, str]],
                        embeddings: List[list], 
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

    """
    Knowledgebase access
    """
    
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
        Abstract method for updating a entry in the knowledgebase.
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
        Abstract method for deleting a entry from the knowledgebase.
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


class RelationshipDirection(Enum):
    """
    Enum class for relationship direction options.
    """
    outgoing = ("-", "->")
    incoming = ("<-", "-")
    bidirectional = ("-", "-")


class Neo4jKnowledgebase(Knowledgebase):
    """
    Represents a neo4j based knowledgebase.
    """
    def __init__(self,
                 neo4j_uri: str,
                 neo4j_username: str,
                 neo4j_password: str,
                 neo4j_driver_params: dict = None) -> None:
        """
        Initiates KnowledgeGraph instance.
        :param neo4j_uri: Neo4j database URI.
        :param neo4j_username: Neo4j database user.
        :param neo4j_password: Neo4j database password.
        :param neo4j_driver_params: Neo4j database driver parameters.
        """
        db_params = {} if db_params is None else db_params
        self.neo4j_uri = neo4j_uri
        self.database = neo4j.GraphDatabase.driver(
            uri=self.neo4j_uri,
            auth=(neo4j_username, neo4j_password),
            **neo4j_driver_params 
        )
        self.prepare_labels = lambda labels: ":".join(labels) if isinstance(labels, list) else labels
        self.prepare_properties = lambda properties: None if properties is None else {k: v for k, v in properties.items() if v is not None}
    
    def __del__(self) -> None:
        """
        Closes down the instance. 
        """
        self.database.close()

    def _build_query(self, query: str | List[str]) -> str:
        """
        Refines a query.
        :param query: Raw query as string or list of strings.
        :return: Refined query.
        """
        if isinstance(query, list):
            query = "\n".join(query)
        return neo4j.Query(query)
    
    """
    Conversion functionality
    """

    def filtermasks_conversion(self, filtermasks: List[FilterMask]) -> str:
        """
        Function for converting Filtermasks to Neo4j filters.
        :param filtermasks: Filtermasks.
        :return: Query component.
        """
        return ""
    
    def query_result_conversion(self, query_result: neo4j.Result, query: str | None = None, filtermasks: List[FilterMask] | None = None) -> List[Entry]:
        """
        Method for converting a Neo4j query result to entries.
        :param query_result: Neo4j query result.
        :param query: Retrieval query.
        :param filtermasks: List of retrieval filter masks.
        :return: List of entries.
        """
        return []
    
    """
    Node and relationship management
    """

    def create_node(self, node_labels: str | List[str], node_properties: dict) -> List[Dict[str, Any]]:
        """
        Creates a knowledge graph node.
        :param node_labels: Label(s) of the node.
        :param node_properties: Node properties as dictionary.
        :return: Node creation query result.
        """
        node_label = self.prepare_labels(node_labels)
        node_properties = self.prepare_properties(node_properties)

        already_existing = self.retrieve_node(
            target_labels=node_label, 
            target_properties=node_properties)
        if already_existing:
            return already_existing
        else:
            query = f"CREATE (target:{node_label} $node_properties) RETURN target"
            return self.database.session().run(self._build_query(query), node_properties=node_properties).data()
    
    def create_relationship(self, 
                            relation_labels: str | List[str], 
                            relation_properties: dict | None, 
                            from_labels: str | List[str], 
                            from_properties: dict | None, 
                            to_labels: str | List[str], 
                            to_properties: dict | None, 
                            direction: RelationshipDirection = RelationshipDirection.outgoing) -> List[Dict[str, Any]]:
        """
        Creates a knowledge graph relationship.
        :param relation_labels: Label(s) of the relationship.
        :param relation_properties: Properties of the relationship.
        :param from_label: Label(s) of the source node.
        :param from_properties: Properties of the source node.
        :param to_label: Label(s) of the target node.
        :param to_properties: Properties of the target node.
        :return: Relationship creation query result.
        """
        already_existing = self.retrieve_relation(
            relation_labels=relation_labels,
            relation_properties=relation_properties,
            from_labels=from_labels,
            from_properties=from_properties,
            to_labels=to_labels,
            to_properties=to_properties
        )
        
        if not any([record["relation"] for record in already_existing]):
            relation_label = self.prepare_labels(relation_labels)
            relation_properties = self.prepare_properties(relation_properties)
            from_label = self.prepare_labels(from_labels)
            from_properties = self.prepare_properties(from_properties)
            to_label = self.prepare_labels(to_labels)
            to_properties = self.prepare_properties(to_properties)

            from_prop_clause = f"{{ {', '.join([f'{key}: $from_{key}' for key in from_properties])} }}" if from_properties else ""
            relation_prop_clause = f"{{ {', '.join([f'{key}: $rel_{key}' for key in relation_properties])} }}" if relation_properties else ""
            to_prop_clause = f"{{ {', '.join([f'{key}: $to_{key}' for key in to_properties])} }}" if to_properties else ""
            
            query = [
                f"MATCH (source:{from_label} {from_prop_clause}), "
                f"(target:{to_label} {to_prop_clause}) "
                f"CREATE (source){direction.value[0]}[relation:{relation_label} {relation_prop_clause}]{direction.value[1]}(target) RETURN relation"
            ]
            query_params = {f"from_{k}": v for k, v in from_properties.items()}
            query_params.update({f"to_{k}": v for k, v in to_properties.items()})
            query_params.update({f"rel_{k}": v for k, v in relation_properties.items()})
            return self.database.session().run(self._build_query(query), 
                                            **query_params).data()
    
    def update_node(self, 
                    target_labels: str | List[str], 
                    target_properties: dict,
                    target_update: dict) -> List[Dict[str, Any]]:
        """
        Updates a graph node.
        :param target_labels: Label(s) of the node.
        :param target_properties: Target properties as dictionary.
        :param target_update: Target property update as dictionary.
        :return: Retrieval query result.
        """
        target_label = self.prepare_labels(target_labels)
        target_properties = self.prepare_properties(target_properties)
        target_update = self.prepare_properties(target_update)

        target_prop_clause = f"{{ {', '.join([f'{key}: ${key}' for key in target_properties])} }}" if target_properties else ""

        query = [
            f"MATCH (target:{target_label} {target_prop_clause})"
            "SET target+= $target_update",
            "RETURN target"
        ]
        target_properties["target_update"] = target_update

        query = self._build_query(query)
        return self.database.session().run(query, **target_properties).data()
    
    def retrieve_node(self, 
                      target_labels: str | List[str], 
                      target_properties: dict) -> List[Dict[str, Any]]:
        """
        Retrieves a graph node.
        :param target_labels: Label(s) of the node.
        :param target_properties: Target properties as dictionary.
        :return: Retrieval query result.
        """
        target_label = self.prepare_labels(target_labels)
        target_properties = self.prepare_properties(target_properties)

        target_prop_clause = f"{{ {', '.join([f'{key}: ${key}' for key in target_properties])} }}" if target_properties else ""
        query = [
            f"MATCH (target:{target_label} {target_prop_clause})"
        ]
        
        query.append(f"RETURN target")
        query = self._build_query(query)
        return self.database.session().run(query, **target_properties).data()

    def retrieve_relation(self, 
                          relation_labels: str | List[str], 
                          relation_properties: dict | None, 
                          from_labels: str | List[str], 
                          from_properties: dict | None, 
                          to_labels: str | List[str], 
                          to_properties: dict | None) -> List[Dict[str, Any]]:
        """
        Retrieves a graph relation.
        :param relation_labels: Label(s) of the relationship.
        :param relation_properties: Properties of the relationship.
        :param from_label: Label(s) of the source node.
        :param from_properties: Properties of the source node.
        :param to_label: Label(s) of the target node.
        :param to_properties: Properties of the target node.
        :return: Retrieval query result.
        """
        relation_label = self.prepare_labels(relation_labels)
        relation_properties = self.prepare_properties(relation_properties)
        from_label = self.prepare_labels(from_labels)
        from_properties = self.prepare_properties(from_properties)
        to_label = self.prepare_labels(to_labels)
        to_properties = self.prepare_properties(to_properties)

        from_prop_clause = f"{{ {', '.join([f'{key}: $from_{key}' for key in from_properties])} }}" if from_properties else ""
        relation_prop_clause = f"{{ {', '.join([f'{key}: $rel_{key}' for key in relation_properties])} }}" if relation_properties else ""
        to_prop_clause = f"{{ {', '.join([f'{key}: $to_{key}' for key in to_properties])} }}" if to_properties else ""
        query = [
            f"MATCH (source:{from_label} {from_prop_clause})",
            f"-[relation:{relation_label} {relation_prop_clause}]-",
            f"(target:{to_label} {to_prop_clause})"
        ]
        query_params = {f"from_{k}": v for k, v in from_properties.items()}
        query_params.update({f"to_{k}": v for k, v in to_properties.items()})
        query_params.update({f"rel_{k}": v for k, v in relation_properties.items()})
        
        query.append(f"RETURN relation")
        query = self._build_query(query)
        return self.database.session().run(query, **query_params).data()

    def retrieve_network(self, 
                 source_labels: str | List[str], 
                 source_properties: dict | None, 
                 allowed_nodes: List[str] | None = None, 
                 allowed_relationships: List[str] | None = None, 
                 min_depth: int = 1,
                 max_depth: int = 1,
                 limit: int = -1) -> List[Dict[str, Any]]:
        """
        Retrieves a knowledge graph network.
        :param source_labels: Label(s) of the node.
        :param source_properties: Source properties as dictionary.
        :param allowed_nodes: List of allowed node labels.
        :param allowed_relationships: List of allowed relationship labels.
        :param min_depth: Minimum depth of related nodes.
        :param max_depth: Maximum depth of related nodes.
        :param limit: Maximum number of retrieved nodes.
            Will not be taken into account if a negative value is given.
        :return: Retrieval query result.
        """
        source_label = self.prepare_labels(source_labels)
        source_properties = self.prepare_properties(source_properties)
        
        allowed_nodes = allowed_nodes or []
        allowed_relationships = allowed_relationships or []

        source_prop_clause = f"{{ {', '.join([f'{key}: $source_{key}' for key in source_properties])} }}" if source_properties else ""
        query = [
            f"MATCH (source:{source_label} {source_prop_clause})"
        ]
        if source_properties:
            query_params = {"source_" + key: source_properties[key] for key in source_properties}
        else:
            query_params = {}

        depth_clause = f"*{min_depth}..{max_depth}"
        relationship_pattern = "relation:" + '|'.join(allowed_relationships) if allowed_relationships else "relation"
        depth_relationship_clause = f"-[{relationship_pattern}{depth_clause}]-"
        node_pattern = "target:" + '|'.join(allowed_nodes) if allowed_nodes else "target"
        query.append(f"{depth_relationship_clause}({node_pattern})")
        
        query.append(f"RETURN source, target, relation")
        if limit > 0:
            query.append(f"LIMIT {limit}")
        query = self._build_query(query)
        return self.database.session().run(query, **query_params).data()
    
    def get_full_graph(self, limit: int = -1) -> List[Dict[str, Any]]:
        """
        Retrieves all nodes and relationships.
        :param limit: Element limit.
        :return: Network graph in Neo4j path structure.
        """
        retrieval_query = """
        MATCH (source)-[relation]->(target) 

        RETURN
        {elementId: ELEMENTID(source), labels: LABELS(source), properties: PROPERTIES(source)} AS source,
        {elementId: ELEMENTID(relation), type: TYPE(relation), properties: PROPERTIES(relation)} AS relation,
        {elementId: ELEMENTID(target), labels: LABELS(target), properties: PROPERTIES(target)} AS target
        """
        if limit > 0:
            retrieval_query += f"\nLIMIT {limit}"
        return self.run_query(retrieval_query)
    
    def flush_graph(self) -> List[Dict[str, Any]]:
        """
        Flushes nodes and relationships.
        :return: Deletion query result.
        """
        return self.database.session().run(self._build_query(
            "MATCH (n) OPTIONAL MATCH (n)-[r]-() DELETE n,r"
        )).data()
    
    """
    Vector index management
    """

    def create_vector_index(self, 
                            source_labels:  str | List[str],
                            index_name:  str,
                            vector_property: str,
                            index_config: dict) -> List[Dict[str, Any]]:
        """
        Creates a vector index.
        :param source_labels: Label(s) of the node.
        :param index_name: Index name.
        :param vector_property: Name of the vector property.
        :param index_config: The index config.
        """
        source_label = self.prepare_labels(source_labels)
        index_config = json.dumps(index_config).replace('{"', '{`').replace(', "', ', `').replace('":', '`:')
        query = [
            f"CREATE VECTOR INDEX {index_name} IF NOT EXISTS",
            f"FOR (source:{source_label})",
            f"ON source.{vector_property}",
            f"OPTIONS {{ indexConfig: {index_config}}}"
        ]
        query = self._build_query(query)
        return self.database.session().run(query).data()
    
    def retrieve_vector_indexes(self) -> List[Dict[str, Any]]:
        """
        Retrieves available vector indexes.
        """
        query =  [
            "SHOW VECTOR INDEXES YIELD name, type, entityType, labelsOrTypes, properties",
            "RETURN name, type, entityType, labelsOrTypes, properties"
        ]
        query = self._build_query(query)
        return self.database.session().run(query).data()
        
    def retrieve_nearest_neighbors(self,
                                    source_labels:  str | List[str],
                                    source_properties: dict,
                                    index_name:  str,
                                    embeddings: List[List[float]],
                                    max_neighbors: int = 5) -> List[Dict[str, Any]]:
        """
        Creates a vector index.
        :param source_labels: Label(s) of the node.
        :param source_properties: Source properties as dictionary.
        :param index_name: Index name.
        :param embeddings: Embeddings to score against.
        :param index_config: The index config.
        :param max_neighbors: Max number of neighbors.
        """
        source_label = self.prepare_labels(source_labels)
        source_properties = self.prepare_properties(source_properties)

        source_prop_clause = f"{{ {', '.join([f'{key}: ${key}' for key in source_properties])} }}" if source_properties else ""
        query = [
            f"MATCH (source:{source_label} {source_prop_clause})"
            f"CALL db.index.vector.queryNodes('{index_name}', {max_neighbors}, {embeddings})",
            "YIELD node AS target, score",
            "RETURN target, score"
        ]
        query = self._build_query(query)
        return self.database.session().run(query).data()    

    def flush_vector_index(self, index_name: str) -> List[Dict[str, Any]]:
        """
        Flushes vector index.
        :return: Deletion query result.
        """
        return self.database.session().run(self._build_query(f"DROP INDEX {index_name}")).data()
    
    """
    General access
    """

    def change_setting(self, setting: str, value: str) -> List[Dict[str, Any]]:
        """
        Changes Neo4j setting.
        :param setting: Neo4j setting.
        :param value: Target value.
        """
        self.run_query(f"CALL dbms.setConfigValue(‘{setting}’,’{value}');")

    def run_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Runs a query.
        :return: Query result.
        """
        return self.database.session().run(query).data()
    
    """
    Knowledgebase access
    """

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