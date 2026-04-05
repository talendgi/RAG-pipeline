import json
import logging
from neo4j import GraphDatabase
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Thin wrapper around the Neo4j driver for use inside agent tools."""

    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    def query(self, cypher: str, params: dict | None = None, limit: int = 20) -> str:
        """Run a Cypher query and return JSON-formatted results (capped at `limit` rows)."""
        with self.driver.session() as s:
            result = s.run(cypher, **(params or {}))
            rows = [dict(r) for r in result][:limit]
        return json.dumps(rows, default=str, indent=2)

    # ── Pre-built Cypher helpers used by the agent tools ──────────────────────

    def related_entities(self, entity_id: str) -> str:
        cypher = """
        MATCH (n {id: $eid})-[r]-(m)
        RETURN type(r)        AS relationship,
               labels(m)[0]   AS entity_type,
               m.id            AS entity_id,
               m.name          AS entity_name
        LIMIT 20
        """
        return self.query(cypher, {"eid": entity_id})

    def top_connected(self, node_label: str, top_n: int = 10) -> str:
        cypher = f"""
        MATCH (n:{node_label})-[r]-()
        RETURN n.id   AS id,
               n.name AS name,
               count(r) AS connections
        ORDER BY connections DESC
        LIMIT {top_n}
        """
        return self.query(cypher)

    def keyword_pattern(self, keyword: str) -> str:
        cypher = """
        MATCH (n)-[r]-(m)
        WHERE toLower(n.name) CONTAINS toLower($kw)
           OR toLower(m.name) CONTAINS toLower($kw)
        RETURN labels(n)[0] AS from_type, n.name AS from_name,
               type(r)       AS relationship,
               labels(m)[0] AS to_type,  m.name AS to_name
        LIMIT 20
        """
        return self.query(cypher, {"kw": keyword})

    def shortest_path(self, from_id: str, to_id: str) -> str:
        cypher = """
        MATCH p = shortestPath(
            (a {id: $fid})-[*..6]-(b {id: $tid})
        )
        RETURN [n IN nodes(p) | coalesce(n.name, n.id)] AS path,
               length(p) AS hops
        """
        return self.query(cypher, {"fid": from_id, "tid": to_id})


def build_trend_agent() -> AgentExecutor:
    """
    Build and return the Trend Research Agent.

    This agent uses Neo4j graph traversal tools to:
      - Find entities related to a given node
      - Identify the most connected / influential nodes
      - Discover patterns matching a keyword
      - Run arbitrary Cypher queries
      - Find shortest paths between two entities
    """
    llm = ChatGroq(api_key=GROQ_API_KEY, model_name=GROQ_MODEL, temperature=0)
    neo = Neo4jClient()

    tools = [
        Tool(
            name="graph_related_entities",
            func=neo.related_entities,
            description=(
                "Find all nodes related to a specific entity in the knowledge graph. "
                "Input: an entity ID string (e.g. a customer_id or order_id). "
                "Output: JSON list of connected nodes with relationship types."
            ),
        ),
        Tool(
            name="graph_top_connected",
            func=lambda x: neo.top_connected(x.strip()),
            description=(
                "Find the most connected (influential) nodes of a given type. "
                "Useful for identifying top customers, popular products, or key hubs. "
                "Input: a Neo4j node label string, e.g. 'Customer' or 'Product'. "
                "Output: JSON ranked list with connection counts."
            ),
        ),
        Tool(
            name="graph_keyword_pattern",
            func=neo.keyword_pattern,
            description=(
                "Search the graph for relationships involving a keyword or concept. "
                "Input: a keyword or short phrase. "
                "Output: JSON list of matching entity pairs and their relationships."
            ),
        ),
        Tool(
            name="graph_shortest_path",
            func=lambda x: neo.shortest_path(*[s.strip() for s in x.split(",", 1)]),
            description=(
                "Find the shortest path between two entities in the graph. "
                "Input: two entity IDs separated by a comma, e.g. 'C001, P042'. "
                "Output: JSON showing the path and number of hops."
            ),
        ),
        Tool(
            name="graph_cypher",
            func=neo.query,
            description=(
                "Run a raw Cypher query on the Neo4j graph for custom analysis. "
                "Input: a valid Cypher query string. "
                "Output: JSON query results. Use LIMIT to keep results manageable."
            ),
        ),
    ]

    prompt = hub.pull("hwchase17/react")
    agent  = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=6,
        handle_parsing_errors=True,
        return_intermediate_steps=False,
    )

    logger.info("Trend Research Agent ready (%d tools)", len(tools))
    return executor
