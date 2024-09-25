## RAG (Retrieval Augmented Generation)

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXe0-JqD-TYU62yym64zlK8BnWZr3FFkjuEqhbwtrfcfH7bu4-zw7T7lafwOQV7RF3D5DB54fJYcibo2GFEl46u4yZDYIPwZVzB6a5upz0gE_OJjo6Q6Qu7S5SrKOZebVM8tQNgf_E773jxKo5do9UJk64Z-?key=9uyNk7FoZciHwUo0N8S5pQ)

**RAG (Retrieval-Augmented Generation)** is a technique that combines two AI systems: a **retrieval system** and a **generative model**, to enhance the ability of large language models (LLMs) to generate factually accurate and contextually relevant content. It is particularly useful in tasks where the model needs access to external knowledge, like open-domain question answering.

* **RAG Foundations summary and advenced techniques: [Miro Board](https://miro.com/app/board/uXjVLbpeOd4=/)**
* **Reduce LLM Hallucinations (also improve translations) with [RAGFix](https://www.ragfix.ai), article [here](https://medium.com/@JamesStakelum/the-end-of-ai-hallucinations-a-breakthrough-in-accuracy-for-data-engineers-e67be5cc742a)**

### **Why RAG?**

* [Documentation](https://python.langchain.com/v0.2/docs/tutorials/rag/) from Langchain
* [Video](https://www.youtube.com/watch?v=Q7CSm-Gl0RA) explanation

### How RAG Works:

1. **Retrieval Component**:

   - The retrieval system, often a dense retriever, searches through a **knowledge base** (e.g., documents, Wikipedia, databases) to find relevant information related to a given query.
   - This component identifies the most relevant passages or documents based on semantic similarity between the query and the documents, usually by embedding them into a vector space and finding the nearest neighbors.
2. **Augmentation of Information**:

   - Once the relevant documents are retrieved, they are **fed into the generative model** along with the original query. This gives the model access to external knowledge that was not encoded during its pre-training phase.
3. **Generative Model**:

   - The **generative model** (e.g., GPT-4 or BART) processes both the query and the retrieved documents to generate a well-informed, natural language response.
   - The generative model augments its internal knowledge with the **retrieved information** to produce a more factually accurate and contextually relevant output.

### Key Advantages of RAG:

- **Improved factual accuracy**: Since the generative model has access to real-time retrieved knowledge, it can produce more factually grounded responses.
- **Open-domain capabilities**: It can answer questions on any topic by retrieving information from vast, external knowledge sources.
- **Scalability**: RAG can scale across large knowledge bases, and the retrieval system narrows down the information to what is relevant, making it more efficient.

### Example Workflow:

1. A user asks a question: "What is the capital of France?"
2. The retrieval system searches through a corpus (e.g., Wikipedia) and retrieves documents mentioning "Paris" and "capital of France."
3. These documents are fed into the generative model, which produces a precise response like, "The capital of France is Paris."

RAG is an effective approach when factual accuracy and up-to-date information are critical for tasks such as customer service, knowledge management, or research assistance.

### **What is semantic / similarity search?**

* Article explanation: [link](https://medium.com/@sudhiryelikar/understanding-similarity-or-semantic-search-and-vector-databases-5f9a5ba98acb)
* What is semantic search? [link](https://cohere.com/llmu/what-is-semantic-search)

### What is a Reranker?

A reranker is a crucial component in the information retrieval (IR) ecosystem that evaluates and reorders search results or passages to enhance their relevance to a specific query. In RAG, this tool builds on the primary vector Approximate Nearest Neighbor ([ANN](https://zilliz.com/glossary/anns)) search, improving search quality by more effectively determining the semantic relevance between documents and queries.

Rerankers fall into two main categories: **Score-based and Neural Network-based** .

**Score-based rerankers** work by aggregating multiple candidate lists from various sources, applying weighted scoring or Reciprocal Rank Fusion (RRF) to unify and reorder these candidates into a single, prioritized list based on their score or relative position in the original list. This type of reranker is known for its efficiency and is widely used in traditional search systems due to its lightweight nature.

On the other hand,**Neural Network-based reranker**s, often called cross-encoder rerankers, leverage a neural network to analyze the relevance between a query and a document. They are specifically designed to compute a similarity score that reflects the semantic proximity between the two, allowing for a refined rearrangement of results from single or multiple sources. This method ensures more semantic relatedness, thus providing useful search results and enhancing the overall effectiveness of the retrieval system.

* Cohere ReRank-3: [here](https://cohere.com/blog/rerank-3)

### Hybrid Search:

* Optimizing RAG with Hybrid Search & Reranking: [article](https://superlinked.com/vectorhub/articles/optimizing-rag-with-hybrid-search-reranking?utm_source=youtube&utm_medium=video&utm_campaign=vh-summary-video)
* Explanation from Microsoft Learn: [article and video](https://learn.microsoft.com/en-us/azure/search/hybrid-search-overview)

#### Advanced Techniques: [GraphRAG](GraphRAG)
