# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2025 Alexander Hering             *
****************************************************
"""
from enum import Enum
from typing import List, Dict, Any
from src.utility.filter_mask_utility import FilterMask
import json
import neo4j


class RelationshipDirection(Enum):
    """
    Enum class for relationship direction options.
    """
    outgoing = ("-", "->")
    incoming = ("<-", "-")
    bidirectional = ("-", "-")


class Neo4jStorage(object):
    """
    Represents a neo4j based storage.
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
        
    def retrieve_by_semantic_similarity(self,
                                    source_labels:  str | List[str],
                                    source_properties: dict,
                                    index_name:  str,
                                    embeddings: List[List[float]],
                                    max_nodes: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves nodes by semantic similarity.
        :param source_labels: Label(s) of the node.
        :param source_properties: Source properties as dictionary.
        :param index_name: Index name.
        :param embeddings: Embeddings to score against.
        :param index_config: The index config.
        :param max_nodes: Max number of retrieved nodes.
        """
        source_label = self.prepare_labels(source_labels)
        source_properties = self.prepare_properties(source_properties)

        source_prop_clause = f"{{ {', '.join([f'{key}: ${key}' for key in source_properties])} }}" if source_properties else ""
        query = [
            f"MATCH (source:{source_label} {source_prop_clause})"
            f"CALL db.index.vector.queryNodes('{index_name}', {max_nodes}, {embeddings})",
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
    