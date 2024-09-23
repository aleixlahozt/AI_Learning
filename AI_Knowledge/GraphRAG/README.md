# GraphRAG

## Introduction to RAG and Its Challenges: [article](https://medium.com/@zilliz_learn/graphrag-explained-enhancing-rag-with-knowledge-graphs-3312065f99e1)

[Retrieval Augmented Generation](https://zilliz.com/learn/Retrieval-Augmented-Generation) (RAG) is a technique that connects external data sources to enhance the output of **[large language models](https://zilliz.com/glossary/large-language-models-(llms)) (LLMs). This technique is perfect for LLMs to access private or domain-specific data and address** **[hallucination](https://zilliz.com/glossary/ai-hallucination) issues. Therefore, RAG has been widely used to power many GenAI applications, such as AI chatbots and** [recommendation systems](https://zilliz.com/vector-database-use-cases/recommender-system).

A baseline RAG usually integrates a vector database and an LLM, where the [vector database](https://zilliz.com/learn/what-is-vector-database) stores and retrieves contextual information for user queries, and the LLM generates answers based on the retrieved context. While this approach works well in many cases, it struggles with complex tasks like multi-hop reasoning or answering questions that require connecting disparate pieces of information.

For example, consider this question: “*What name was given to the son of the man who defeated the usurper Allectus?”*

A baseline RAG would generally follow these steps to answer this question:

1. Identify the Man: Determine who defeated Allectus.
2. Research the Man’s Son: Look up information about this person’s family, specifically his son.
3. Find the Name: Identify the name of the son

The challenge usually arises at the first step because a baseline RAG retrieves text based on [semantic similarity](https://zilliz.com/glossary/semantic-similarity), not directly answering complex queries where specific details may not be explicitly mentioned in the dataset. This limitation makes it difficult to find the exact information needed, often requiring expensive and impractical solutions like manually creating Q&A pairs for frequent queries.

To address such challenges, Microsoft Research introduced **[GraphRAG](https://microsoft.github.io/graphrag/), a brand-new method that augments RAG retrieval and generation with knowledge graphs.**

## What is Graph RAG?

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

## Baseline GraphRAG: [step-by-step](https://medium.com/@zilliz_learn/graphrag-explained-enhancing-rag-with-knowledge-graphs-3312065f99e1)

A GraphRAG pipeline usually consists of two fundamental processes: indexing and querying.

![img](https://miro.medium.com/v2/resize:fit:700/0*CFrSdpijjpq7HD3h.png)

### 1. INDEXING:

**1.1. Text Unit Segmentation:** Divide the entire corpus documents into text chunks. By segmenting long documents into smaller chunks, we can extract and preserve more detailed information about this input data.

**1.2. Entity, Relationship, and Claims Extraction:** Use LLMs to identify and extract all entites, relationships and key claims. This information is used to construct the knowledge graph.

**1.3. Hierarchical Clustering:** GraphRAG typically uses the Leiden technique to perform hierarchical clustering on the initial knowledge graphs. Leiden is a community detection algorithm that can effectively discover community structures within the graph. Entities in each cluster are assigned to different communities for more in-depth analysis.

**1.4. Community Summary Generation:** GraphRAG generates summaries for each community and its members using a bottom-up approach. These summaries include the main entities within the community, their relationships, and key claims. This step gives an overview of the entire dataset and provides useful contextual information for subsequent queries.

### **2. QUERYING:**

GraphRAG has two different querying workflows tailored for different queries.

* [Global Search](https://microsoft.github.io/graphrag/posts/query/0-global_search) for reasoning about holistic questions related to the whole data corpus by leveraging the community summaries.
* [Local Search](https://microsoft.github.io/graphrag/posts/query/1-local_search) for reasoning about specific entities by fanning out to their neighbors and associated concepts.

**2.1. Global Search:**

![](https://miro.medium.com/v2/resize:fit:700/0*b4NUADKOIYWUB544.png)

**2.2. Local Search**

![img](https://miro.medium.com/v2/resize:fit:700/0*47MgjBZ3g7brNXnG.png)

1. **User Query:** First, the system receives a user query, which could be a simple question or a more complex query.
2. **Similar Entity Search:** The system identifies a set of entities from the knowledge graph that are semantically related to the user input. These entities serve as entry points into the knowledge graph. This step uses a vector database like Milvus to conduct [text similarity searches](https://zilliz.com/learn/vector-similarity-search).
3. **Entity-Text Unit Mapping:** The extracted text units are mapped to the corresponding entities, removing the original text information.
4. **Entity-Relationship Extraction:** The step extracts specific information about the entities and their corresponding relationships.
5. **Entity-Covariate Mapping:** This step maps entities to their covariates, which may include statistical data or other relevant attributes.
6. **Entity-Community Report Mapping:** Community reports are integrated into the search results, incorporating some global information.
7. **Utilization of Conversation History:** If provided, the system uses conversation history to better understand the user’s intent and context.
8. **Response Generation:** Finally, the system constructs and responds to the user query based on the filtered and sorted data generated in the previous steps.

---

## To learn GraphRAG in detail

1. First, watch this two videos to have a general understanding of what GraphRAG is:

* [Graph RAG]([https://www.youtube.com/watch?v=r09tJfON6kE](https://www.youtube.com/watch?v=r09tJfON6kE)): LLM-Derived Knowledge Graphs for RAG
* [GraphRAG](https://www.youtube.com/watch?v=knDDGYHnnSI): The Marriage of Knowledge Graphs and RAG: Emil Eifrem

2. Read the original [paper](https://arxiv.org/pdf/2404.16130) of GraphRAG published by Microsoft researchers.
3. Also, read this short article from Microsoft Research Blog

* GraphRAG: Unlocking LLM discovery on narrative private data: [link](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)

**Also:**

* **PYTHON PACKAGE / REPO:** Checkout [GraphRAG](https://pypi.org/project/graphrag/), a data pipeline and transformation suite that is designed to extract meaningful, structured data from unstructured text using the power of LLMs
* **REPO:** Microsoft Azure GraphRAG [Accelerator](https://github.com/Azure-Samples/graphrag-accelerator)
* **ARTICLE:**[ GraphRAG](https://medium.com/data-science-in-your-pocket/graphrag-using-langchain-31b1ef8328b9) using LangChain
* **VIDEO:** Run local [GraphRAG](https://www.youtube.com/watch?v=nkbyD4joa0A) with LLaMa 3.1, Langchain and Neo4j

Last:

* **REPO:** Avanced RAG techniques: [link](https://github.com/NirDiamant/RAG_Techniques)

## Hands-on with GraphRAG

**Read this very detailed article from Medium, it includes an explanatory video**

* End-to-End implementation GraphRAG: [link](https://medium.com/@vinodkumargr/graphrag-graphs-retreival-augmented-generation-unlocking-llm-discovery-on-narrative-private-1bf977dadcdd)

From this article, we can access this github repository we can use to build an End-to-End GraphRAG implementation:

* End-to-End GitHub [repo](https://github.com/ApexIQ/End-to-End-Graphrag-implementation?source=post_page-----1bf977dadcdd--------------------------------)
