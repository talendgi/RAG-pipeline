import logging
from neo4j import GraphDatabase
from ingestion.mysql_loader import load_raw_dataframe
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Builds a Neo4j knowledge graph from MySQL data loaded via SQLAlchemy.

    Typical usage
    -------------
    g = GraphBuilder()
    g.clear_graph()
    g.create_nodes("customers", id_col="customer_id", name_col="name")
    g.create_nodes("products",  id_col="product_id",  name_col="name")
    g.create_nodes("orders",    id_col="order_id")
    g.create_relationships("orders", "order_id", "customers", "customer_id", "PLACED_BY")
    g.create_relationships("orders", "order_id", "products",  "product_id",  "CONTAINS")
    g.close()
    """

    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        logger.info("Connected to Neo4j at %s", NEO4J_URI)

    def close(self):
        self.driver.close()
        logger.info("Neo4j connection closed")

    def clear_graph(self):
        """Delete ALL nodes and relationships. Use with caution."""
        with self.driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")
        logger.info("Graph cleared")

    def create_nodes(
        self,
        table_name: str,
        id_col: str,
        name_col: str | None = None,
        extra_props: list[str] | None = None,
        where_clause: str | None = None,
    ):
        """
        Create one Neo4j node per row in the given MySQL table.

        Parameters
        ----------
        table_name   : MySQL table to read (via SQLAlchemy)
        id_col       : column used as the node's unique :id property
        name_col     : optional column stored as :name (human-readable label)
        extra_props  : additional columns to store on the node
        where_clause : optional SQL WHERE clause to filter rows
        """
        cols_to_load = [id_col]
        if name_col:
            cols_to_load.append(name_col)
        if extra_props:
            cols_to_load.extend(extra_props)

        df = load_raw_dataframe(
            table_name,
            where_clause=where_clause,
            columns=cols_to_load,
        )

        node_label = table_name.rstrip("s").capitalize()  # orders→Order, customers→Customer
        created = 0

        with self.driver.session() as s:
            for _, row in df.iterrows():
                props = {
                    "id":    str(row[id_col]),
                    "_table": table_name,
                }
                if name_col and name_col in row:
                    props["name"] = str(row[name_col])
                if extra_props:
                    for col in extra_props:
                        if col in row and row[col] is not None:
                            props[col] = str(row[col])

                s.run(
                    f"MERGE (n:{node_label} {{id: $id}}) SET n += $props",
                    id=props["id"],
                    props=props,
                )
                created += 1

        logger.info("Created/merged %d :%s nodes from table '%s'",
                    created, node_label, table_name)

    def create_relationships(
        self,
        from_table: str,
        from_id_col: str,
        to_table: str,
        to_id_col: str,
        rel_type: str,
        where_clause: str | None = None,
    ):
        """
        Create relationships between nodes based on a foreign key column.

        Example: orders.customer_id → (:Order)-[:PLACED_BY]->(:Customer)

        Parameters
        ----------
        from_table   : table that owns the FK (e.g. 'orders')
        from_id_col  : PK column of from_table (e.g. 'order_id')
        to_table     : referenced table (e.g. 'customers')
        to_id_col    : FK column in from_table pointing to to_table (e.g. 'customer_id')
        rel_type     : Neo4j relationship type string (e.g. 'PLACED_BY')
        where_clause : optional SQL WHERE to limit rows
        """
        df = load_raw_dataframe(
            from_table,
            where_clause=where_clause,
            columns=[from_id_col, to_id_col],
        )

        from_label = from_table.rstrip("s").capitalize()
        to_label   = to_table.rstrip("s").capitalize()
        created    = 0

        with self.driver.session() as s:
            for _, row in df.iterrows():
                if row[to_id_col] is None:
                    continue
                s.run(
                    f"""
                    MATCH (a:{from_label} {{id: $fid}})
                    MATCH (b:{to_label}   {{id: $tid}})
                    MERGE (a)-[:{rel_type}]->(b)
                    """,
                    fid=str(row[from_id_col]),
                    tid=str(row[to_id_col]),
                )
                created += 1

        logger.info(
            "Created %d [:%s] relationships: %s → %s",
            created, rel_type, from_table, to_table,
        )

    def run_cypher(self, cypher: str, params: dict | None = None) -> list[dict]:
        """Run any Cypher statement and return results as a list of dicts."""
        with self.driver.session() as s:
            result = s.run(cypher, **(params or {}))
            return [dict(r) for r in result]
