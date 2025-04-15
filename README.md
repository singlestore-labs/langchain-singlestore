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

The `SingleStoreChatMessageHistory` class allows you to store and retrieve chat message history in SingleStore. 

```python
from langchain_singlestore import SingleStoreChatMessageHistory

chat_history = SingleStoreChatMessageHistory(
    host="127.0.0.1:3306/db",
    table_name="chat_history"
)

# Add a message to the chat history
chat_history.add_message({"role": "user", "content": "Hello, how are you?"})

# Retrieve chat history
messages = chat_history.get_messages()
print(messages)
```

### Semantic Cache

The `SingleStoreSemanticCache` class provides a semantic caching mechanism using SingleStore as the backend. It stores embeddings and allows for efficient retrieval based on similarity.

```python
from langchain_singlestore import SingleStoreSemanticCache
from langchain_core.globals import set_llm_cache

set_llm_cache(
    SingleStoreSemanticCache(
        embedding=YourEmbeddings(),
        host="root:pass@localhost:3306/db",
    )
)
```

### Vector Store

The `SingleStoreVectorStore` class enables storing document embeddings and performing fast vector and full-text searches. It supports advanced search strategies like `VECTOR_ONLY`, `TEXT_ONLY`, and hybrid approaches.

```python
from langchain_singlestore.vectorstores import SingleStoreVectorStore
from langchain_core.documents import Document

vector_store = SingleStoreVectorStore(
    embeddings=OpenAIEmbeddings(),
    host="127.0.0.1:3306/db",
    table_name="vector_store"
)

# Add documents to the vector store
documents = [
    Document(page_content="The Eiffel Tower is in Paris.", metadata={"category": "landmark"}),
    Document(page_content="The Louvre is a famous museum in Paris.", metadata={"category": "museum"})
]
vector_store.add_documents(documents)

# Perform a similarity search
results = vector_store.similarity_search(query="famous landmarks in Paris", k=1)
print(results[0].page_content)
```

### Document Loader

The `SingleStoreLoader` class allows you to load documents directly from a SingleStore database table. This is useful for applications that need to process large datasets stored in SingleStore.

```python
from langchain_singlestore.document_loaders import SingleStoreLoader

loader = SingleStoreLoader(
    host="127.0.0.1:3306/db",
    table_name="documents",
    content_field="content",
    metadata_field="metadata"
)

# Load documents
documents = loader.load()
print(documents[0].page_content)
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
make integration_tests
```

### Note on Integration Tests

The `test_add_image2` integration test for `SingleStoreVectorStore` downloads data to your local machine. The first run may take a significant amount of time due to the data download process. Subsequent runs will be faster as the data will already be available locally.

