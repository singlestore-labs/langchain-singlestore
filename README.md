# langchain-singlestore

This package contains the LangChain integration with SingleStore

## Installation

```bash
pip install -U langchain-singlestore
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatSingleStore` class exposes chat models from SingleStore.

```python
from langchain_singlestore import ChatSingleStore

llm = ChatSingleStore()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`SingleStoreEmbeddings` class exposes embeddings from SingleStore.

```python
from langchain_singlestore import SingleStoreEmbeddings

embeddings = SingleStoreEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`SingleStoreLLM` class exposes LLMs from SingleStore.

```python
from langchain_singlestore import SingleStoreLLM

llm = SingleStoreLLM()
llm.invoke("The meaning of life is")
```
