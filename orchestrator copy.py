import logging
from langchain_groq import ChatGroq
from langchain_core.messages  import HumanMessage, SystemMessage
from agents.db_search_agent import build_db_search_agent
from agents.trend_agent import build_trend_agent
from config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a data analyst assistant with two capabilities:
1. Database search — exact lookups, aggregates, and semantic search across MySQL records
2. Trend research — graph-based pattern discovery across entity relationships

You will receive context gathered by specialist agents. Use it to produce a clear,
concise, and accurate answer. Always cite the source table or entity when referencing data."""

# ── Query router ──────────────────────────────────────────────────────────────

TREND_KEYWORDS = [
    "trend", "pattern", "relationship", "connected", "network",
    "influence", "top", "most", "popular", "related to", "path",
    "hub", "cluster", "community", "between",
]
DB_KEYWORDS = [
    "find", "list", "count", "average", "sum", "total", "show me",
    "what is", "who is", "how many", "where", "when", "latest",
    "filter", "search", "lookup",
]


def classify_query(question: str) -> str:
    """
    Route the question to the right agent(s).
    Returns: 'db_search' | 'trend' | 'both'
    """
    q = question.lower()
    is_trend = any(k in q for k in TREND_KEYWORDS)
    is_db    = any(k in q for k in DB_KEYWORDS)

    if is_trend and is_db:
        return "both"
    if is_trend:
        return "trend"
    return "db_search"   # default fallback


# ── Orchestrator ──────────────────────────────────────────────────────────────

class RAGOrchestrator:
    """
    Main entry point for the RAG pipeline.

    - Routes each question to DB Search Agent, Trend Agent, or both.
    - Merges the agent outputs into a single context block.
    - Calls Groq LLM to produce a final synthesized answer.
    """

    def __init__(self):
        logger.info("Initializing RAG orchestrator...")
        self.db_agent    = build_db_search_agent()
        self.trend_agent = build_trend_agent()
        self.llm         = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL,
            temperature=0.2,
        )
        logger.info("Orchestrator ready")

    def run(self, question: str) -> str:
        """
        Run the full RAG pipeline for a user question.

        Parameters
        ----------
        question : natural language question from the user

        Returns
        -------
        Final answer string from the LLM.
        """
        route = classify_query(question)
        logger.info("Question routed to: [%s]", route)

        contexts: list[str] = []

        # ── DB Search Agent ────────────────────────────────────────────────
        if route in ("db_search", "both"):
            try:
                result = self.db_agent.invoke({"input": question})
                output = result.get("output", "")
                if output:
                    contexts.append(f"[Database search]\n{output}")
            except Exception as e:
                logger.warning("DB agent error: %s", e)
                contexts.append(f"[Database search]\nAgent encountered an error: {e}")

        # ── Trend Research Agent ───────────────────────────────────────────
        if route in ("trend", "both"):
            try:
                result = self.trend_agent.invoke({"input": question})
                output = result.get("output", "")
                if output:
                    contexts.append(f"[Graph trend analysis]\n{output}")
            except Exception as e:
                logger.warning("Trend agent error: %s", e)
                contexts.append(f"[Graph trend analysis]\nAgent encountered an error: {e}")

        if not contexts:
            contexts.append("No relevant data was found by either agent.")

        combined_context = "\n\n---\n\n".join(contexts)

        # ── Final LLM synthesis ────────────────────────────────────────────
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Context gathered by agents:\n\n{combined_context}\n\n"
                f"User question: {question}\n\n"
                "Provide a clear, synthesized answer based on the context above."
            )),
        ]

        response = self.llm.invoke(messages)
        return response.content
