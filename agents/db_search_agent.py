import logging
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain import hub
from ingestion.vector_store import similarity_search
from config import MYSQL_URL, GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)


def _build_vector_tool() -> Tool:
    """
    Tool 1 — semantic search across all embedded MySQL rows.
    The agent calls this when the question is fuzzy or conceptual.
    """
    def vector_search(query: str) -> str:
        docs = similarity_search(query, k=5)
        if not docs:
            return "No relevant records found."
        results = []
        for i, doc in enumerate(docs, 1):
            table = doc.metadata.get("source_table", "unknown")
            results.append(f"[{i}] ({table}) {doc.page_content}")
        return "\n".join(results)

    return Tool(
        name="vector_search",
        func=vector_search,
        description=(
            "Search all database records semantically using natural language. "
            "Use this when the question is conceptual, fuzzy, or you are unsure "
            "which table to query. "
            "Input: a natural language question or keywords. "
            "Output: the most relevant rows across all tables."
        ),
    )


def _build_sql_tools(llm: ChatGroq) -> list[Tool]:
    """
    Tool 2+ — LangChain SQL toolkit that generates and runs SQL via SQLAlchemy.
    Gives the agent: sql_db_query, sql_db_schema, sql_db_list_tables, sql_db_query_checker.
    """
    # SQLDatabase uses SQLAlchemy under the hood — same engine URL from config
    db = SQLDatabase.from_uri(
        MYSQL_URL,
        sample_rows_in_table_info=3,   # show 3 sample rows in schema context
    )
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    return toolkit.get_tools()


def build_db_search_agent() -> AgentExecutor:
    """
    Build and return the Database Search Agent.

    This agent combines:
      - Vector similarity search (semantic / fuzzy lookups)
      - SQL execution via SQLAlchemy (exact / aggregate queries)

    The agent decides which tool to use based on the question.
    """
    llm = ChatGroq(api_key=GROQ_API_KEY, model_name=GROQ_MODEL, temperature=0)

    vector_tool = _build_vector_tool()
    sql_tools   = _build_sql_tools(llm)
    all_tools   = [vector_tool] + sql_tools

    # ReAct prompt from LangChain hub — works well with tool-use LLMs
    prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(llm=llm, tools=all_tools, prompt=prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=all_tools,
        verbose=True,
        max_iterations=6,
        handle_parsing_errors=True,
        return_intermediate_steps=False,
    )

    logger.info("DB Search Agent ready (%d tools)", len(all_tools))
    return executor
