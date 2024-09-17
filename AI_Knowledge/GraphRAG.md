# GraphRAG

**Graph RAG** (Graph-based Retrieval-Augmented Generation) is a novel approach that combines the power of **graph-based indexing** with **retrieval-augmented generation (RAG)**, aimed at improving the ability of **large language models (LLMs)** to answer complex, global queries over large text corpora. Here's a breakdown of how it works and what makes it innovative:

### Key Concepts:

1. **Graph-Based Text Index**:

   - Instead of treating documents as isolated texts, Graph RAG creates a **graph of entities and their relationships** extracted from the text corpus.
   - This process involves identifying entities (like people, places, concepts) and creating **nodes** for them, as well as establishing **edges** that represent relationships between them (e.g., "Person A works for Company B").
2. **Hierarchical Community Detection**:

   - The graph is further **partitioned into communities** using algorithms like **Leiden**, allowing related entities and relationships to form meaningful clusters. These clusters represent different levels of detail, from broader groups to more focused subtopics.
   - The hierarchical nature of the graph allows for **multi-level summarization**—responses can be tailored to include more or fewer details depending on the user's query.
3. **Summarization and Query Answering**:

   - Using this graph, LLMs can perform **query-focused summarization** by retrieving relevant nodes and relationships, summarizing them into partial answers, and combining them into a final, coherent response.
   - This approach allows the model to generate **detailed, context-rich answers** to global queries, especially in cases where documents or entities are related across multiple contexts.
4. **Efficiency**:

   - Graph RAG allows for more **efficient information retrieval** since it reduces the need for LLMs to process entire documents repeatedly. Instead, relevant entities and relationships are pre-extracted and indexed, making it easier for the model to retrieve and generate responses.

### Benefits:

- **Better handling of complex, cross-document queries** by leveraging relationships between entities.
- **Improved summarization** that can provide different levels of detail depending on the context of the query.
- **Efficient querying** over large, unstructured datasets through the use of pre-built graph indices, rather than starting from scratch with each query.

### Applications:

Graph RAG is especially useful in tasks like **sensemaking**, where global understanding across a corpus is needed, such as summarizing long documents, exploring relationships in news articles, or answering high-level questions across multiple data sources.

This technique outperforms baseline RAG models on complex datasets, demonstrating its ability to capture and organize rich relational information across large corpora.

### LINKS

* Graph RAG: **[https://www.youtube.com/watch?v=r09tJfON6kE](https://www.youtube.com/watch?v=r09tJfON6kE)**
* GraphRAG: The Marriage of Knowledge Graphs and RAG: Emil Eifrem: [link](https://www.youtube.com/watch?v=knDDGYHnnSI)
* GraphRAG: Unlocking LLM discovery on narrative private data: [link](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)
* (To READ) End-to-End implementation GraphRAG: [link](https://medium.com/@vinodkumargr/graphrag-graphs-retreival-augmented-generation-unlocking-llm-discovery-on-narrative-private-1bf977dadcdd)
  * https://github.com/ApexIQ/End-to-End-Graphrag-implementation?source=post_page-----1bf977dadcdd--------------------------------
  * https://pypi.org/project/graphrag/
* (To READ) Avanced RAG techniques: [link](https://github.com/NirDiamant/RAG_Techniques)
* (To WATCH) Local GraphRAG with Llama 3.1: [link](https://www.youtube.com/watch?v=nkbyD4joa0A&t=20s)
