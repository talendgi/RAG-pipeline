
import argparse
import logging
from config import test_mysql_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── Ingestion ──────────────────────────────────────────────────────────────────

def run_ingestion():
    """
    Full ingestion pipeline:
      1. Load rows from MySQL via SQLAlchemy
      2. Embed and store in Chroma vector store
      3. Build Neo4j knowledge graph

    Customize the table names, columns, and relationships below
    to match your actual MySQL schema.
    """
    from ingestion.mysql_loader import load_table_as_documents
    from ingestion.vector_store import build_vector_store
    from ingestion.graph_builder import GraphBuilder

    logger.info("=== Ingestion START ===")

    # ── 1. Load documents for embedding ───────────────────────────────────
    #  table_name, text_columns, and metadata_columns to match  schema
    all_docs = []
    all_docs += load_table_as_documents(
    table_name="customers",
    text_columns=[
        "customer_id",
        "customer_city",
        "customer_state"
    ],
    metadata_columns=[
        "customer_state",
        "customer_city"
    ])
    all_docs += load_table_as_documents(
    table_name="orders",
    text_columns=[
        "order_id",
        "customer_id",
        "order_status",
        "order_purchase_timestamp"
    ],
    metadata_columns=[
        "order_status",
        "customer_id"
    ])
    all_docs += load_table_as_documents(
        table_name="order_items",
        text_columns=[
            "order_id",
            "product_id",
            "price",
            "freight_value",
            "shipping_limit_date"
        ],
        metadata_columns=[
            "product_id",
            "order_id"
        ])
    all_docs += load_table_as_documents(
    table_name="products",
    text_columns=[
        "product_id",
        "product_category_name",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm"
    ],
    metadata_columns=[
        "product_category_name"
    ])

    logger.info("Total documents to embed: %d", len(all_docs))

    # ── 2. Build vector store ──────────────────────────────────────────────
    build_vector_store(all_docs)

    # ── 3. Build Neo4j graph ───────────────────────────────────────────────
    g = GraphBuilder()
    g.clear_graph()
    g.create_nodes(
        "customers",
        id_col="customer_id",
        name_col="customer_id",
        extra_props=["customer_city", "customer_state"]
    )

    g.create_nodes(
        "orders",
        id_col="order_id",
        name_col="order_id",
        extra_props=["order_status", "order_purchase_timestamp"]
    )

    g.create_nodes(
        "order_items",
        id_col="order_item_id",   
        name_col="order_item_id",
        extra_props=["order_id", "product_id", "price"]
    )

    g.create_nodes(
        "products",
        id_col="product_id",
        name_col="product_id",
        extra_props=["product_category_name"]
    )
    # Create nodes — adjust id_col and name_col to your PK and display column
        # Orders → Customers
    g.create_relationships(
        from_table="orders",
        from_id_col="customer_id",
        to_table="customers",
        to_id_col="customer_id",
        rel_type="PLACED_BY",
    )
        # Orders → Order Items
    g.create_relationships(
        from_table="order_items",
        from_id_col="order_id",
        to_table="orders",
        to_id_col="order_id",
        rel_type="BELONGS_TO",
    )
        # Order Items → Products
    g.create_relationships(
        from_table="order_items",
        from_id_col="product_id",
        to_table="products",
        to_id_col="product_id",
        rel_type="CONTAINS",
    )
    g.close()
#     MATCH (c:customers)-[:PLACED_BY]-(o:orders)
#       -[:BELONGS_TO]-(oi:order_items)
#       -[:CONTAINS]-(p:products)
# WHERE c.customer_city = "Sao Paulo"
# RETURN p.product_category_name
    logger.info("=== Ingestion COMPLETE ===")


# ── Query ──────────────────────────────────────────────────────────────────────

def run_query(question: str) -> str:
    from orchestrator import RAGOrchestrator
    orchestrator = RAGOrchestrator()
    answer = orchestrator.run(question)
    print(f"\nQ: {question}\nA: {answer}\n")
    return answer


def run_interactive():
    from orchestrator import RAGOrchestrator
    orchestrator = RAGOrchestrator()
    print("\nRAG Pipeline — interactive mode. Type 'exit' to quit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break
        answer = orchestrator.run(question)
        print(f"\nAssistant: {answer}\n")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphDB Vectorized RAG Pipeline")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ingest",      action="store_true",
                       help="Run full ingestion (MySQL → Chroma + Neo4j)")
    group.add_argument("--query",       type=str, metavar="QUESTION",
                       help="Run a single query and print the answer")
    group.add_argument("--interactive", action="store_true",
                       help="Start interactive query mode")
    args = parser.parse_args()

    # Always verify DB connectivity first
    if not test_mysql_connection():
        raise SystemExit("Cannot connect to MySQL. Check your .env settings.")

    if args.ingest:
        run_ingestion()
    elif args.query:
        run_query(args.query)
    elif args.interactive:
        run_interactive()
