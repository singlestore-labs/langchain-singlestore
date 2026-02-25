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

```python
from langchain_singlestore import SingleStoreChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# Initialize chat history
chat_history = SingleStoreChatMessageHistory(
    host="127.0.0.1:3306/db",
    table_name="chat_history",
    session_id="user_123"
)

# Add messages to the chat history
chat_history.add_message(HumanMessage(content="Hello, how are you?"))
chat_history.add_message(AIMessage(content="I'm doing well, thank you for asking!"))

# Retrieve chat history
messages = chat_history.get_messages()
for message in messages:
    print(f"{message.type}: {message.content}")

# Use with a chat chain
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
chain = LLMChain(llm=llm, memory=chat_history)
response = chain.run(input="What was the last thing we discussed?")
```

### Semantic Cache

The `SingleStoreSemanticCache` class implements semantic caching for LLM responses using SingleStore's vector capabilities. Instead of exact string matching, it uses embeddings to find semantically similar cached queries, dramatically reducing API costs and improving performance for similar questions.

**Key Features:**
- Vector-based semantic similarity for cache hits
- Reduces LLM API calls for similar queries
- Configurable similarity threshold
- Thread-safe caching operations

```python
from langchain_singlestore import SingleStoreSemanticCache
from langchain_core.globals import set_llm_cache
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Configure semantic caching
set_llm_cache(
    SingleStoreSemanticCache(
        embedding=OpenAIEmbeddings(),
        host="root:pass@localhost:3306/db",
        table_name="llm_semantic_cache",
        distance_threshold=0.2  # Similarity threshold
    )
)

# Now all LLM calls will use semantic caching
llm = ChatOpenAI(model="gpt-3.5-turbo")

# First call will invoke the LLM
response1 = llm.invoke("What is the capital of France?")
print(response1.content)  # Makes API call

# Similar query will use cached response
response2 = llm.invoke("What city is the capital of France?")  
print(response2.content)  # Uses cache - no API call!
```

### Vector Store

The `SingleStoreVectorStore` class provides a powerful document storage and retrieval system with combined vector and full-text search capabilities. It supports multiple search strategies, advanced metadata filtering, and both vector and text-based indexing for optimal performance.

**Key Features:**
- Hybrid search combining vector and text indexes
- Multiple search strategies (VECTOR_ONLY, TEXT_ONLY, FILTER_BY_TEXT, FILTER_BY_VECTOR, WEIGHTED_SUM)
- Simple and advanced metadata filtering
- Efficient document management (add, delete, update)
- Configurable distance metrics

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

