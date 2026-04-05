"""
main.py — entry point for the GraphDB Vectorized RAG Pipeline

Usage
-----
### Step 1 (first time / data refresh):
    python main.py --ingest

### Step 2 (query):
    python main.py --query "Who are the top 5 most connected customers?"

### Step 3 (interactive mode):
    python main.py --interactive
"""


pip install langchain langchain-groq langchain-community langchain-huggingface \
            chromadb neo4j sqlalchemy pymysql python-dotenv sentence-transformers
```

###  — Project structure
```
rag_pipeline/
├── .env
├── config.py
├── ingestion/
│   ├── mysql_loader.py       # Load rows from MySQL
│   ├── vector_store.py       # Embed + store in Chroma
│   └── graph_builder.py      # Build Neo4j graph from data
├── agents/
│   ├── db_search_agent.py    # SQL + vector hybrid agent
│   └── trend_agent.py        # Graph traversal + trend agent
├── orchestrator.py           # LangChain agent router
└── main.py                   # Entry point

### neo4j installation via docker

    docker run -d    -p 7474:7474 -p 7687:7687 -v neo4j_data:/data  -e NEO4J_AUTH=neo4j/Andorokta!321    --name neo4j neo4j:latest

    docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/Andorokta!321  neo4j:latest

    http://127.0.0.1:7474
    