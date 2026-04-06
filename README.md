# langchain-singlestore

This package provides the LangChain integration with SingleStore, enabling efficient storage, retrieval, and management of documents, embeddings, and chat message history using SingleStore's high-performance distributed SQL database.

## Installation

To install the package, run:

```bash
pip install -U langchain-singlestore
```

## Features

This package includes the following components:

### Chat Message History

The `SingleStoreChatMessageHistory` class provides persistent storage for chat message history in SingleStore. This is essential for AI applications that need to maintain conversation context across sessions. It seamlessly integrates with LangChain's chat models and chains.

**Key Features:**
- Automatic schema creation and management
- Support for multiple conversation sessions
- Efficient message retrieval and storage
- Easy integration with LangChain chat models

### Semantic Cache

The `SingleStoreSemanticCache` class implements semantic caching for LLM responses using SingleStore's vector capabilities. Instead of exact string matching, it uses embeddings to find semantically similar cached queries, dramatically reducing API costs and improving performance for similar questions.

**Key Features:**
- Vector-based semantic similarity for cache hits
- Reduces LLM API calls for similar queries
- Configurable similarity threshold
- Thread-safe caching operations

### Vector Store

The `SingleStoreVectorStore` class provides a powerful document storage and retrieval system with combined vector and full-text search capabilities. It supports multiple search strategies, advanced metadata filtering, and both vector and text-based indexing for optimal performance.

**Key Features:**
- Hybrid search combining vector and text indexes
- Multiple search strategies (VECTOR_ONLY, TEXT_ONLY, FILTER_BY_TEXT, FILTER_BY_VECTOR, WEIGHTED_SUM)
- Simple and advanced metadata filtering
- Efficient document management (add, delete, update)
- Configurable distance metrics (DOT_PRODUCT, EUCLIDEAN_DISTANCE)
- Full-text index versions (V1, V2) with different capabilities
- Multiple text scoring algorithms (MATCH, BM25, BM25_GLOBAL)

### SQL Database Retriever

The `SingleStoreSQLDatabaseRetriever` enables LangChain agents and chains to execute SQL queries directly against SingleStore and retrieve results as structured documents.

**Key Features:**
- Execute SQL queries and convert results to documents
- Flexible row-to-document conversion with custom handlers
- Connection pooling for efficient resource management
- Integration with LangChain agents for database-aware AI
- Support for complex queries with JSON results

## Usage Examples

### Vector Store

#### Basic Usage

```python
from langchain_singlestore.vectorstores import SingleStoreVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# Initialize vector store
vector_store = SingleStoreVectorStore(
    embeddings=OpenAIEmbeddings(),
    host="127.0.0.1:3306/db",
    table_name="documents",
    metric="EUCLIDEAN_DISTANCE"  # or "DOT_PRODUCT"
)

# Add documents
documents = [
    Document(
        page_content="The Eiffel Tower is an iconic landmark in Paris.",
        metadata={"category": "landmark", "country": "France", "year_built": 1889}
    ),
    Document(
        page_content="The Louvre is the world's largest art museum.",
        metadata={"category": "museum", "country": "France", "year_built": 1793}
    ),
    Document(
        page_content="Big Ben is a famous clock tower in London.",
        metadata={"category": "landmark", "country": "UK", "year_built": 1859}
    )
]
vector_store.add_documents(documents)

# Basic similarity search
results = vector_store.similarity_search("famous landmarks", k=2)
for doc in results:
    print(doc.page_content)
```

#### Simple Metadata Filtering

Filter documents using simple dictionary-style syntax for backward compatibility:

```python
# Filter by exact match
results = vector_store.similarity_search(
    query="European landmarks",
    k=5,
    filter={"category": "landmark"}  # Simple dict filter
)

# Filter by multiple fields (implicit AND)
results = vector_store.similarity_search(
    query="museums",
    k=5,
    filter={"category": "museum", "country": "France"}
)
```

#### Advanced Metadata Filtering with FilterTypedDict

Use `FilterTypedDict` for complex filtering with operators like `$eq`, `$gt`, `$in`, `$and`, `$or`, etc.:

```python
from langchain_singlestore import FilterTypedDict

# Comparison operators: $eq, $ne, $gt, $gte, $lt, $lte
results = vector_store.similarity_search(
    query="old structures",
    k=10,
    filter={"year_built": {"$lt": 1900}}  # Built before 1900
)

# Collection operators: $in, $nin
results = vector_store.similarity_search(
    query="landmarks",
    k=10,
    filter={"country": {"$in": ["France", "UK"]}}  # In France or UK
)

# Existence check: $exists
results = vector_store.similarity_search(
    query="heritage sites",
    k=10,
    filter={"heritage_status": {"$exists": True}}  # Must have heritage_status field
)

# Logical operators: $and, $or
results = vector_store.similarity_search(
    query="european landmarks",
    k=10,
    filter={
        "$and": [
            {"category": "landmark"},
            {"year_built": {"$gte": 1800}},
            {"country": {"$in": ["France", "UK"]}}
        ]
    }
)

# Complex nested queries
results = vector_store.similarity_search(
    query="cultural sites",
    k=10,
    filter={
        "$or": [
            {
                "$and": [
                    {"category": "museum"},
                    {"country": "France"}
                ]
            },
            {
                "$and": [
                    {"category": "landmark"},
                    {"year_built": {"$lt": 1900}}
                ]
            }
        ]
    }
)
```

#### Search Strategies and Indexes

Configure different search strategies based on your use case:

```python
from langchain_singlestore._utils import SearchStrategy

# Strategy 1: Vector search only (fastest)
results = vector_store.similarity_search(
    query="landmarks",
    k=5,
    search_strategy=SearchStrategy.VECTOR_ONLY
)

# Strategy 2: Full-text search only (best for keyword matching)
results = vector_store.similarity_search(
    query="Eiffel",
    k=5,
    search_strategy=SearchStrategy.TEXT_ONLY
)

# Strategy 3: Filter by text, then rank by vector (hybrid)
results = vector_store.similarity_search(
    query="landmarks in paris",
    k=5,
    search_strategy=SearchStrategy.FILTER_BY_TEXT  # Text match required
)

# Strategy 4: Filter by vector, then rank by text
results = vector_store.similarity_search(
    query="iconic structures",
    k=5,
    search_strategy=SearchStrategy.FILTER_BY_VECTOR
)

# Strategy 5: Weighted combination (balanced approach)
results = vector_store.similarity_search(
    query="famous landmarks",
    k=5,
    search_strategy=SearchStrategy.WEIGHTED_SUM  # Combines vector + text scores
)
```

#### Full-Text Index Versions

SingleStore supports two versions of full-text indexes with different capabilities:

```python
from langchain_singlestore.vectorstores import SingleStoreVectorStore
from langchain_singlestore._utils import FullTextIndexVersion
from langchain_openai import OpenAIEmbeddings

# Version 1 (V1) - Compatible with all SingleStore versions
# Uses the original full-text index implementation
vector_store_v1 = SingleStoreVectorStore(
    embedding=OpenAIEmbeddings(),
    host="127.0.0.1:3306/db",
    use_full_text_search=True,
    full_text_index_version=FullTextIndexVersion.V1,  # Default
)

# Version 2 (V2) - Requires SingleStore 8.7+
# Offers improved performance and additional features like BM25 scoring
vector_store_v2 = SingleStoreVectorStore(
    embedding=OpenAIEmbeddings(),
    host="127.0.0.1:3306/db",
    use_full_text_search=True,
    full_text_index_version=FullTextIndexVersion.V2,
)
```

**Version Comparison:**

| Feature | V1 | V2 |
|---------|----|----|  
| SingleStore Compatibility | All versions | 8.7+ |
| MATCH scoring | ✓ | ✓ |
| BM25 scoring | ✗ | ✓ |
| BM25_GLOBAL scoring | ✗ | ✓ |

#### Full-Text Scoring Modes

When using full-text search strategies (TEXT_ONLY, FILTER_BY_TEXT, FILTER_BY_VECTOR, WEIGHTED_SUM), you can choose different scoring algorithms:

```python
from langchain_singlestore.vectorstores import SingleStoreVectorStore
from langchain_singlestore._utils import FullTextIndexVersion, FullTextScoringMode
from langchain_openai import OpenAIEmbeddings

# Initialize with V2 full-text index (required for BM25 modes)
vector_store = SingleStoreVectorStore(
    embedding=OpenAIEmbeddings(),
    host="127.0.0.1:3306/db",
    use_full_text_search=True,
    full_text_index_version=FullTextIndexVersion.V2,
)

# MATCH mode (default) - Works with V1 and V2
# Uses native MATCH() AGAINST() function
results = vector_store.similarity_search(
    query="famous landmarks",
    k=5,
    search_strategy=SingleStoreVectorStore.SearchStrategy.TEXT_ONLY,
    full_text_scoring_mode=FullTextScoringMode.MATCH,
)

# BM25 mode - Requires V2
# Uses BM25 algorithm with TF-IDF and document length normalization
results = vector_store.similarity_search(
    query="famous landmarks",
    k=5,
    search_strategy=SingleStoreVectorStore.SearchStrategy.TEXT_ONLY,
    full_text_scoring_mode=FullTextScoringMode.BM25,
)

# BM25_GLOBAL mode - Requires V2
# Computes IDF statistics across the entire dataset (more accurate in distributed environments)
results = vector_store.similarity_search(
    query="famous landmarks",
    k=5,
    search_strategy=SingleStoreVectorStore.SearchStrategy.TEXT_ONLY,
    full_text_scoring_mode=FullTextScoringMode.BM25_GLOBAL,
)
```

**Scoring Mode Comparison:**

| Mode | Description | Index Version | Use Case |
|------|-------------|---------------|----------|
| MATCH | Native SingleStore MATCH() function | V1, V2 | General text search, backward compatibility |
| BM25 | Best Matching 25 algorithm | V2 only | More accurate relevance scoring with TF-IDF |
| BM25_GLOBAL | BM25 with global IDF statistics | V2 only | Consistent scoring in distributed/sharded environments |

**When to use each mode:**
- **MATCH**: Default mode. Fast, simple keyword matching. Good for basic search needs.
- **BM25**: Better relevance ranking by considering term frequency, inverse document frequency, and document length. Recommended for most text search applications.
- **BM25_GLOBAL**: Same as BM25 but calculates statistics globally across all partitions. Use when you need consistent scoring across a distributed SingleStore cluster.

### Document Loader

The `SingleStoreLoader` class provides efficient loading of documents directly from SingleStore database tables. This is ideal for applications that need to process documents stored in your database without intermediate file exports, enabling seamless ETL workflows.

**Key Features:**
- Load documents from any database table
- Configurable content and metadata fields
- Efficient batch processing
- Support for complex metadata structures

```python
from langchain_singlestore.document_loaders import SingleStoreLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize loader
loader = SingleStoreLoader(
    host="127.0.0.1:3306/db",
    table_name="documents",
    content_field="content",       # Column containing document text
    metadata_field="metadata"       # Column containing metadata JSON
)

# Load all documents
documents = loader.load()
print(f"Loaded {len(documents)} documents")
print(documents[0].page_content[:100])

# Use with text splitter for chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunked_docs = splitter.split_documents(documents)

# Add chunked documents to vector store
vector_store.add_documents(chunked_docs)
```

### SQL Database Retriever

The `SingleStoreSQLDatabaseRetriever` enables LangChain agents and chains to execute SQL queries directly against a SingleStore database and retrieve results formatted as documents. This is perfect for building database-aware AI applications that need to query structured data.

**Key Features:**
- Execute SQL queries and retrieve results as documents
- Flexible row-to-document conversion with custom handlers
- Connection pooling for efficient resource management
- Clean integration with LangChain agents and chains
- Support for complex queries with JSON results

#### Basic Usage

```python
from langchain_singlestore import SingleStoreSQLDatabaseRetriever

# Initialize retriever
retriever = SingleStoreSQLDatabaseRetriever(
    host="127.0.0.1:3306/db",
    user="root",
    password="your_password",
    database="my_database"
)

# Execute a query and get results as documents
docs = retriever.invoke("SELECT id, name, email FROM users LIMIT 10")

# Each row becomes a document
for doc in docs:
    print(doc.page_content)
    print(doc.metadata)
```

#### Using with LangChain Agents

```python
from langchain_singlestore import SingleStoreSQLDatabaseRetriever
from langchain.agents import create_tool_use_agent
from langchain_openai import ChatOpenAI

# Create retriever
retriever = SingleStoreSQLDatabaseRetriever(
    host="127.0.0.1:3306/db",
    user="root",
    password="your_password",
    database="my_database"
)

# Create agent with database query tool
llm = ChatOpenAI(model="gpt-4")

# Build a tool that executes queries
def query_database(query: str) -> str:
    """Execute SQL query and return formatted results."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

# Use in agent
agent = create_tool_use_agent(
    llm,
    tools=[
        {
            "type": "function",
            "function": {
                "name": "query_database",
                "description": "Execute SQL queries against the database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "SQL query to execute"
                        }
                    }
                }
            }
        }
    ]
)
```

#### Custom Row Conversion

Convert database rows to documents using a custom function:

```python
from langchain_core.documents import Document

def custom_row_converter(row_dict: dict, row_index: int) -> Document:
    """Custom converter for database rows."""
    content = f"Record {row_index}: {row_dict.get('name', 'Unknown')}"
    metadata = {
        "index": row_index,
        "record_id": row_dict.get("id"),
        "source": "database"
    }
    return Document(page_content=content, metadata=metadata)

retriever = SingleStoreSQLDatabaseRetriever(
    host="127.0.0.1:3306/db",
    user="root",
    password="your_password",
    database="my_database",
    row_to_document_fn=custom_row_converter
)

docs = retriever.invoke("SELECT id, name FROM customers")
```

#### Query with Result Limits

```python
from langchain_singlestore import SingleStoreSQLDatabaseChain

# Execute query with automatic LIMIT
docs = SingleStoreSQLDatabaseChain.query_to_document(
    query="SELECT * FROM orders",
    host="root:password@127.0.0.1:3306/db",
    row_limit=100  # Automatically adds LIMIT 100
)

print(f"Retrieved {len(docs)} records")
```

For detailed documentation, visit the [LangChain documentation](https://python.langchain.com/).

# Development and Testing

To set up the development environment and run tests, follow these steps:

## Installation

Install all dependencies, including those for linting, typing, and testing, using `poetry`:

```bash
poetry install --with lint,typing,test,test_integration
```

## Linting

Before committing any changes, ensure that the code passes all linting checks. Run the following command:

```bash
make lint
```

This will check the code for style and formatting issues.

## Running Tests

Run all integration tests to verify that the code works as expected:

```bash
make tests
make integration_tests
```

### Note on Integration Tests

The `test_add_image2` integration test for `SingleStoreVectorStore` downloads data to your local machine. The first run may take a significant amount of time due to the data download process. Subsequent runs will be faster as the data will already be available locally.

## Contribution

We welcome contributions to the `langchain-singlestore` project! Please refer to the [CONTRIBUTE.md](./CONTRIBUTE.md) file for detailed guidelines on how to contribute, including instructions for running tests, linting, and publishing new package versions.

