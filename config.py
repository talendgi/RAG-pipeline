import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

load_dotenv(override=True)

# ── MySQL ──────────────────────────────────────────────────────────────────────
MYSQL_USER     = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_HOST     = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT     = os.getenv("MYSQL_PORT", "3306")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

MYSQL_URL = (
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}"
    f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
)

# ── Neo4j ──────────────────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# ── Groq LLM ───────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = "llama3-8b-8192"   # swap to mixtral-8x7b-32768 for longer context

# ── Vector store ───────────────────────────────────────────────────────────────
CHROMA_DIR  = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ── SQLAlchemy engine (singleton) ──────────────────────────────────────────────
_mysql_engine: Engine | None = None


def get_mysql_engine() -> Engine:
    """
    Returns a module-level singleton SQLAlchemy engine for MySQL.
    Creates it on the first call and reuses it on every subsequent call.

    Pool settings:
      pool_pre_ping  — test the connection before handing it out (avoids stale connections)
      pool_recycle   — recycle connections after 1 hour (prevents 'MySQL server has gone away')
      pool_size      — keep 5 connections open in the background
      max_overflow   — allow up to 10 extra connections under load
    """
    global _mysql_engine
    if _mysql_engine is None:
        _mysql_engine = create_engine(
            MYSQL_URL,
            pool_pre_ping=True,
            pool_recycle=3600,
            pool_size=5,
            max_overflow=10,
            echo=False,  # set True to log every SQL query for debugging
        )
    return _mysql_engine


def test_mysql_connection() -> bool:
    """Quick connectivity check. Call this at startup to fail fast."""
    import logging
    logger = logging.getLogger(__name__)
    try:
        with get_mysql_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("MySQL connection OK")
        return True
    except Exception as e:
        logger.error("MySQL connection FAILED: %s", e)
        return False
