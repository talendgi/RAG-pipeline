import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from config import CHROMA_DIR, EMBED_MODEL

logger = logging.getLogger(__name__)


def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the HuggingFace embedding model (downloaded once, cached locally)."""
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(documents: list[Document]) -> Chroma:
    """
    Embed all documents and persist them to Chroma.
    Returns
    A live Chroma vectorstore instance.
    """
    if not documents:
        raise ValueError("No documents provided to build_vector_store.")

    logger.info("Embedding %d documents...", len(documents))
    embeddings = get_embeddings()

    store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    logger.info("Vector store persisted to '%s'", CHROMA_DIR)
    return store


def load_vector_store() -> Chroma:
    """
    Load an already-built Chroma store from disk.
    Call this at query time — no re-embedding needed.
    """
    logger.info("Loading vector store from '%s'", CHROMA_DIR)
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings(),
    )


def similarity_search(query: str, k: int = 5,
                      filter_metadata: dict | None = None) -> list[Document]:
    """
    Convenience wrapper: load the store and run a similarity search.

    Parameters
    ----------
    query           : natural language query string
    k               : number of results to return
    filter_metadata : optional Chroma metadata filter dict,
                      e.g. {"source_table": "orders"}
    """
    store = load_vector_store()
    kwargs = {"k": k}
    if filter_metadata:
        kwargs["filter"] = filter_metadata

    docs = store.similarity_search(query, **kwargs)
    logger.info("Vector search returned %d docs for query: '%s'", len(docs), query)
    return docs
