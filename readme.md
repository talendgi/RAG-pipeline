main.py — entry point for the GraphDB Vectorized RAG Pipeline

Usage
-----
### Step 1 (first time / data refresh):
    python main.py --ingest

### Step 2 (query):
    python main.py --query "Who are the top 5 most connected customers?"

### Step 3 (interactive mode):
    python main.py --interactive



pip install langchain langchain-groq langchain-community langchain-huggingface \
            chromadb neo4j sqlalchemy pymysql python-dotenv sentence-transformers


###  Project structure
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
```
### neo4j installation via docker

    docker run -d    -p 7474:7474 -p 7687:7687 -v neo4j_data:/data  -e NEO4J_AUTH=neo4j/Andorokta!321    --name neo4j neo4j:latest

    docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/Andorokta!321  neo4j:latest

    http://127.0.0.1:7474


<img width="1383" height="815" alt="image" src="https://github.com/user-attachments/assets/86774bda-a326-45e6-844b-2b93d130f1ba" />





<img width="1246" height="880" alt="Screenshot 2026-04-04 221707" src="https://github.com/user-attachments/assets/add39239-8c31-4df5-ad0a-d3570434c7a0" />


```

(airflow_venv) C:\D-Disk\Visual Studio\airflow>python rag_pipeline\main.py --ingest
2026-04-05 09:02:16,180 [INFO] config — MySQL connection OK
2026-04-05 09:02:30,607 [INFO] __main__ — === Ingestion START ===
2026-04-05 09:02:30,681 [INFO] ingestion.mysql_loader — Loaded 200 documents from 'customers'
2026-04-05 09:02:30,757 [INFO] ingestion.mysql_loader — Loaded 200 documents from 'orders'
2026-04-05 09:02:30,830 [INFO] ingestion.mysql_loader — Loaded 200 documents from 'order_items'
2026-04-05 09:02:30,904 [INFO] ingestion.mysql_loader — Loaded 200 documents from 'products'
2026-04-05 09:02:30,905 [INFO] __main__ — Total documents to embed: 800
2026-04-05 09:02:30,905 [INFO] ingestion.vector_store — Embedding 800 documents...
2026-04-05 09:02:41,275 [INFO] sentence_transformers.SentenceTransformer — Load pretrained SentenceTransformer: sentence-transformers/all-MiniLM-L6-v2
2026-04-05 09:02:43,087 [INFO] httpx — HTTP Request: HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/modules.json "HTTP/1.1 307 Temporary Redirect"
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
2026-04-05 09:02:43,090 [WARNING] huggingface_hub.utils._http — Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
2026-04-05 09:02:43,120 [INFO] httpx — HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/modules.json "HTTP/1.1 200 OK"
2026-04-05 09:02:43,410 [INFO] httpx — HTTP Request: HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config_sentence_transformers.json "HTTP/1.1 307 Temporary Redirect"
2026-04-05 09:02:43,436 [INFO] httpx — HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/config_sentence_transformers.json "HTTP/1.1 200 OK"
2026-04-05 09:02:43,720 [INFO] httpx — HTTP Request: HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config_sentence_transformers.json "HTTP/1.1 307 Temporary Redirect"
2026-04-05 09:02:43,750 [INFO] httpx — HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/config_sentence_transformers.json "HTTP/1.1 200 OK"
2026-04-05 09:02:44,044 [INFO] httpx — HTTP Request: HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/README.md "HTTP/1.1 307 Temporary Redirect"
2026-04-05 09:02:44,071 [INFO] httpx — HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/README.md "HTTP/1.1 200 OK"
2026-04-05 09:02:44,370 [INFO] httpx — HTTP Request: HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/modules.json "HTTP/1.1 307 Temporary Redirect"
2026-04-05 09:02:44,400 [INFO] httpx — HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/modules.json "HTTP/1.1 200 OK"
2026-04-05 09:02:44,704 [INFO] httpx — HTTP Request: HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/sentence_bert_config.json "HTTP/1.1 307 Temporary Redirect"
2026-04-05 09:02:44,736 [INFO] httpx — HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/sentence_bert_config.json "HTTP/1.1 200 OK"
2026-04-05 09:02:45,022 [INFO] httpx — HTTP Request: HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/adapter_config.json "HTTP/1.1 404 Not Found"
2026-04-05 09:02:45,303 [INFO] httpx — HTTP Request: HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
2026-04-05 09:02:45,329 [INFO] httpx — HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/config.json "HTTP/1.1 200 OK"
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 857.80it/s]
BertModel LOAD REPORT from: sentence-transformers/all-MiniLM-L6-v2
Key                     | Status     |  |
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  |

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
2026-04-05 09:02:46,192 [INFO] httpx — HTTP Request: HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
2026-04-05 09:02:46,220 [INFO] httpx — HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/config.json "HTTP/1.1 200 OK"
2026-04-05 09:02:46,506 [INFO] httpx — HTTP Request: HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer_config.json "HTTP/1.1 307 Temporary Redirect"
2026-04-05 09:02:46,532 [INFO] httpx — HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/tokenizer_config.json "HTTP/1.1 200 OK"
2026-04-05 09:02:46,836 [INFO] httpx — HTTP Request: GET https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2/tree/main/additional_chat_templates?recursive=false&expand=false "HTTP/1.1 404 Not Found"
2026-04-05 09:02:47,115 [INFO] httpx — HTTP Request: GET https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2/tree/main?recursive=true&expand=false "HTTP/1.1 200 OK"
2026-04-05 09:02:47,554 [INFO] httpx — HTTP Request: HEAD https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/1_Pooling/config.json "HTTP/1.1 307 Temporary Redirect"
2026-04-05 09:02:47,581 [INFO] httpx — HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/sentence-transformers/all-MiniLM-L6-v2/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/1_Pooling%2Fconfig.json "HTTP/1.1 200 OK"
2026-04-05 09:02:47,865 [INFO] httpx — HTTP Request: GET https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2 "HTTP/1.1 200 OK"
2026-04-05 09:03:21,553 [INFO] ingestion.vector_store — Vector store persisted to './chroma_db'
2026-04-05 09:03:21,555 [INFO] ingestion.graph_builder — Connected to Neo4j at bolt://localhost:7687
2026-04-05 09:03:23,910 [INFO] ingestion.graph_builder — Graph cleared
2026-04-05 09:03:23,917 [INFO] ingestion.mysql_loader — Loaded DataFrame from 'customers': 200 rows x 4 cols
2026-04-05 09:03:27,041 [INFO] ingestion.graph_builder — Created/merged 200 :Customer nodes from table 'customers'
2026-04-05 09:03:27,046 [INFO] ingestion.mysql_loader — Loaded DataFrame from 'orders': 200 rows x 4 cols
2026-04-05 09:03:28,697 [INFO] ingestion.graph_builder — Created/merged 200 :Order nodes from table 'orders'
2026-04-05 09:03:28,702 [INFO] ingestion.mysql_loader — Loaded DataFrame from 'order_items': 200 rows x 5 cols
2026-04-05 09:03:30,766 [INFO] ingestion.graph_builder — Created/merged 200 :Order_item nodes from table 'order_items'
2026-04-05 09:03:30,770 [INFO] ingestion.mysql_loader — Loaded DataFrame from 'products': 200 rows x 3 cols
2026-04-05 09:03:32,323 [INFO] ingestion.graph_builder — Created/merged 200 :Product nodes from table 'products'
2026-04-05 09:03:28,697 [INFO] ingestion.graph_builder — Created/merged 200 :Order nodes from table 'orders'
2026-04-05 09:03:28,702 [INFO] ingestion.mysql_loader — Loaded DataFrame from 'order_items': 200 rows x 5 cols
2026-04-05 09:03:30,766 [INFO] ingestion.graph_builder — Created/merged 200 :Order_item nodes from table 'order_items'
2026-04-05 09:03:30,770 [INFO] ingestion.mysql_loader — Loaded DataFrame from 'products': 200 rows x 3 cols
2026-04-05 09:03:28,697 [INFO] ingestion.graph_builder — Created/merged 200 :Order nodes from table 'orders'
2026-04-05 09:03:28,702 [INFO] ingestion.mysql_loader — Loaded DataFrame from 'order_items': 200 rows x 5 cols
2026-04-05 09:03:30,766 [INFO] ingestion.graph_builder — Created/merged 200 :Order_item nodes from table 'order_items'
2026-04-05 09:03:28,697 [INFO] ingestion.graph_builder — Created/merged 200 :Order nodes from table 'orders'
2026-04-05 09:03:28,702 [INFO] ingestion.mysql_loader — Loaded DataFrame from 'order_items': 200 rows x 5 cols
2026-04-05 09:03:28,697 [INFO] ingestion.graph_builder — Created/merged 200 :Order nodes from table 'orders'
2026-04-05 09:03:28,697 [INFO] ingestion.graph_builder — Created/merged 200 :Order nodes from table 'orders'
2026-04-05 09:03:28,702 [INFO] ingestion.mysql_loader — Loaded DataFrame from 'order_items': 200 rows x 5 cols
2026-04-05 09:03:30,766 [INFO] ingestion.graph_builder — Created/merged 200 :Order_item nodes from table 'order_items'
2026-04-05 09:03:28,697 [INFO] ingestion.graph_builder — Created/merged 200 :Order nodes from table 'orders'
2026-04-05 09:03:28,702 [INFO] ingestion.mysql_loader — Loaded DataFrame from 'order_items': 200 rows x 5 cols
2026-04-05 09:03:30,766 [INFO] ingestion.graph_builder — Created/merged 200 :Order_item nodes from table 'order_items'
2026-04-05 09:03:28,697 [INFO] ingestion.graph_builder — Created/merged 200 :Order nodes from table 'orders'
2026-04-05 09:03:28,697 [INFO] ingestion.graph_builder — Created/merged 200 :Order nodes from table 'orders'
2026-04-05 09:03:28,697 [INFO] ingestion.graph_builder — Created/merged 200 :Order nodes from table 'orders'
2026-04-05 09:03:28,697 [INFO] ingestion.graph_builder — Created/merged 200 :Order nodes from table 'orders'
2026-04-05 09:03:28,697 [INFO] ingestion.graph_builder — Created/merged 200 :Order nodes from table 'orders'
2026-04-05 09:03:28,697 [INFO] ingestion.graph_builder — Created/merged 200 :Order nodes from table 'orders'
2026-04-05 09:03:28,697 [INFO] ingestion.graph_builder — Created/merged 200 :Order nodes from table 'orders'
2026-04-05 09:03:28,702 [INFO] ingestion.mysql_loader — Loaded DataFrame from 'order_items': 200 rows x 5 cols
2026-04-05 09:03:30,766 [INFO] ingestion.graph_builder — Created/merged 200 :Order_item nodes from table 'order_items'
2026-04-05 09:03:28,697 [INFO] ingestion.graph_builder — Created/merged 200 :Order nodes from table 'orders'
2026-04-05 09:03:28,697 [INFO] ingestion.graph_builder — Created/merged 200 :Order nodes from table 'orders'
2026-04-05 09:03:28,697 [INFO] ingestion.graph_builder — Created/merged 200 :Order nodes from table 'orders'
2026-04-05 09:03:28,702 [INFO] ingestion.mysql_loader — Loaded DataFrame from 'order_items': 200 rows x 5 cols
2026-04-05 09:03:30,766 [INFO] ingestion.graph_builder — Created/merged 200 :Order_item nodes from table 'order_items'
2026-04-05 09:03:30,770 [INFO] ingestion.mysql_loader — Loaded DataFrame from 'products': 200 rows x 3 cols
2026-04-05 09:03:32,323 [INFO] ingestion.graph_builder — Created/merged 200 :Product nodes from table 'products'
2026-04-05 09:03:32,324 [INFO] ingestion.mysql_loader — Loaded DataFrame from 'orders': 200 rows x 2 cols
2026-04-05 09:03:33,582 [INFO] ingestion.graph_builder — Created 200 [:PLACED_BY] relationships: orders → customers
2026-04-05 09:03:33,585 [INFO] ingestion.mysql_loader — Loaded DataFrame from 'order_items': 200 rows x 2 cols
2026-04-05 09:03:35,268 [INFO] ingestion.graph_builder — Created 200 [:BELONGS_TO] relationships: order_items → orders
2026-04-05 09:03:35,271 [INFO] ingestion.mysql_loader — Loaded DataFrame from 'order_items': 200 rows x 2 cols
2026-04-05 09:03:37,135 [INFO] ingestion.graph_builder — Created 200 [:CONTAINS] relationships: order_items → products
2026-04-05 09:03:37,136 [INFO] ingestion.graph_builder — Neo4j connection closed
2026-04-05 09:03:37,136 [INFO] __main__ — === Ingestion COMPLETE ===
```
