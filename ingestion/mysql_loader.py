import logging
import pandas as pd
from sqlalchemy import text, inspect
from langchain_core.documents import Document
from config import get_mysql_engine, MYSQL_DATABASE

logger = logging.getLogger(__name__)


# ── Schema inspection ──────────────────────────────────────────────────────────

def list_tables() -> list[str]:
    """Return all table names in the configured MySQL database."""
    inspector = inspect(get_mysql_engine())
    tables = inspector.get_table_names(schema=MYSQL_DATABASE)
    logger.info("Found %d tables: %s", len(tables), tables)
    return tables


def get_table_schema(table_name: str) -> list[dict]:
    """
    Return column metadata for a table.
    Each dict has: column_name, data_type, is_nullable, column_key
    """
    engine = get_mysql_engine()
    query = text("""
        SELECT column_name, data_type, is_nullable, column_key,
               character_maximum_length
        FROM information_schema.columns
        WHERE table_schema = :schema
          AND table_name   = :table
        ORDER BY ordinal_position
    """)
    with engine.connect() as conn:
        rows = conn.execute(query, {"schema": MYSQL_DATABASE, "table": table_name}).mappings().fetchall()

    # Normalize keys to lowercase for consistent access everywhere
    columns = [{k.lower(): v for k, v in dict(r).items()} for r in rows]
    logger.info("Schema for '%s': %d columns", table_name, len(columns))
    return columns


# ── Data loading ───────────────────────────────────────────────────────────────

def load_table_as_documents(
    table_name: str,
    text_columns: list[str],
    metadata_columns: list[str] | None = None,
    where_clause: str | None = None,
) -> list[Document]:
    """
    Load every row of a MySQL table and convert each row into a
    LangChain Document for embedding.

    Parameters
    ----------
    table_name        : MySQL table name
    text_columns      : columns whose values are joined to form the document text
    metadata_columns  : columns stored as Document metadata (used for filtering)
    where_clause      : optional SQL WHERE clause, e.g. "status = 'active'"

    Returns
    -------
    List of LangChain Document objects ready for embedding.
    """
    engine = get_mysql_engine()

    sql = f"SELECT * FROM {MYSQL_DATABASE}.{table_name} LIMIT 200"
    if where_clause:
        sql += f" WHERE {where_clause}"

    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)

    # Normalize column names
    df.columns = [c.lower() for c in df.columns]
    text_columns     = [c.lower() for c in text_columns]
    metadata_columns = [c.lower() for c in (metadata_columns or [])]

    documents = []
    for _, row in df.iterrows():
        # Build the text the LLM will read for this row
        text_parts = [
            f"{col}: {row[col]}"
            for col in text_columns
            if col in row and row[col] is not None
        ]
        page_text = " | ".join(text_parts)

        # Build metadata dict (used for vector store filtering)
        meta = {"source_table": table_name}
        # try to grab a primary key for reference
        for pk_candidate in ["id", f"{table_name}_id", f"{table_name[:-1]}_id"]:
            if pk_candidate in row:
                meta["row_id"] = str(row[pk_candidate])
                break

        for col in metadata_columns:
            if col in row and row[col] is not None:
                meta[col] = str(row[col])

        documents.append(Document(page_content=page_text, metadata=meta))

    logger.info("Loaded %d documents from '%s'", len(documents), table_name)
    return documents


def load_raw_dataframe(
    table_name: str,
    where_clause: str | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load a MySQL table directly as a Pandas DataFrame.
    Useful for graph building and data inspection.

    Parameters
    ----------
    table_name   : MySQL table name
    where_clause : optional SQL WHERE clause
    columns      : optional list of columns to SELECT (defaults to *)
    """
    engine = get_mysql_engine()

    col_expr = ", ".join(columns) if columns else "*"
    sql = f"SELECT {col_expr} FROM {MYSQL_DATABASE}.{table_name} LIMIT 200"
    if where_clause:
        sql += f" WHERE {where_clause}"

    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)

    df.columns = [c.lower() for c in df.columns]
    logger.info("Loaded DataFrame from '%s': %d rows x %d cols",
                table_name, len(df), len(df.columns))
    return df


def run_custom_query(sql: str, params: dict | None = None) -> pd.DataFrame:
    """
    Execute any raw SQL query and return the result as a DataFrame.
    Use SQLAlchemy text() with named params to avoid SQL injection.

    Example
    -------
    run_custom_query(
        "SELECT * FROM orders WHERE status = :status AND amount > :amount",
        {"status": "pending", "amount": 100}
    )
    """
    engine = get_mysql_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params or {})
    df.columns = [c.lower() for c in df.columns]
    return df
