# singlestore-langchain-core

Shared, internal helpers used by the SingleStore integrations for LangChain
and LangGraph:

- Connection attribute helpers (`set_connector_attributes`,
  `compute_connector_version`).
- SingleStore capability enums (`DistanceStrategy`, `FullTextIndexVersion`,
  `FullTextScoringMode`).
- Metadata filter DSL (`FilterTypedDict`, `_parse_filter`) shared by vector
  stores and stores.

This package is not intended to be depended on directly by application code.
Install one of the higher-level packages instead:

- [`langchain-singlestore`](../langchain-singlestore)
- [`langgraph-singlestore`](../langgraph-singlestore)
