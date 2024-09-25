# RAG with Contextual Retrieval: [article](https://www.anthropic.com/news/contextual-retrieval)

* Anthropic [cookbook](https://github.com/anthropics/anthropic-cookbook/tree/main/skills/contextual-embeddings) for Contextual Retrieval

> Sometimes the simplest solution is the best. If your knowledge base is smaller than 200,000 tokens (about 500 pages of material), you can just include the entire knowledge base in the prompt that you give the model, with no need for RAG or similar methods.

### Some RAG context:

RAG solutions can more accurately retrieve the most applicable chunks by combining the embeddings and BM25 techniques using the following steps:

1. Break down the knowledge base (the "corpus" of documents) into smaller chunks of text, usually no more than a few hundred tokens;
2. Create TF-IDF encodings and semantic embeddings for these chunks;
3. Use BM25 to find top chunks based on exact matches;
4. Use embeddings to find top chunks based on semantic similarity;
5. Combine and deduplicate results from (3) and (4) using rank fusion techniques;
6. Add the top-K chunks to the prompt to generate the respon

![img](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F45603646e979c62349ce27744a940abf30200d57-3840x2160.png&w=3840&q=75)

### Traditional RAG limitations

In traditional RAG, documents are typically split into smaller chunks for efficient retrieval. While this approach works well for many applications, it can lead to problems when individual chunks lack sufficient context.

For example, imagine you had a collection of financial information (say, U.S. SEC filings) embedded in your knowledge base, and you received the following question: *What was the revenue growth for ACME Corp in Q2 2023?"*

A relevant chunk might contain the text: *"The company's revenue grew by 3% over the previous quarter."* However, this chunk on its own doesn't specify which company it's referring to or the relevant time period, making it difficult to retrieve the right information or use the information effectively.

## Introducing Contextual Retrieval

Contextual Retrieval solves this problem by prepending chunk-specific explanatory context to each chunk before embedding (“Contextual Embeddings”) and creating the BM25 index (“Contextual BM25”).

Let’s return to our SEC filings collection example. Here's an example of how a chunk might be transformed:

```
original_chunk = "The company's revenue grew by 3% over the previous quarter."

contextualized_chunk = "This chunk is from an SEC filing on ACME corp's performance in Q2 2023; the previous quarter's revenue was $314 million. The company's revenue grew by 3% over the previous quarter."
```

Use a model like Claude 3 Haiku with the following prompt:

`<document>`
{{WHOLE_DOCUMENT}}
`</document>`
Here is the chunk we want to situate within the whole document
`<chunk>`
{{CHUNK_CONTENT}}
`</chunk>`
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.

![img](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F2496e7c6fedd7ffaa043895c23a4089638b0c21b-3840x2160.png&w=3840&q=75)

## Using Prompt Caching to reduce the costs of Contextual Retrieval

Contextual Retrieval is uniquely possible at low cost with Claude, thanks to the special prompt caching feature we mentioned above. With prompt caching, you don’t need to pass in the reference document for every chunk. You simply load the document into the cache once and then reference the previously cached content. Assuming 800 token chunks, 8k token documents, 50 token context instructions, and 100 tokens of context per chunk, **the one-time cost to generate contextualized chunks is $1.02 per million document tokens** .

## Conclusions (from Article)

Here’s a summary of what we found:

1. Embeddings+BM25 is better than embeddings on their own;
2. Voyage and Gemini have the best embeddings of the ones we tested;
3. Passing the top-20 chunks to the model is more effective than just the top-10 or top-5;
4. Adding context to chunks improves retrieval accuracy a lot;
5. Reranking is better than no reranking;
6. **All these benefits stack**: to maximize performance improvements, we can combine contextual embeddings (from Voyage or Gemini) with contextual BM25, plus a reranking step, and adding the 20 chunks to the prompt.

## What is a Reranker?

A reranker is a crucial component in the information retrieval (IR) ecosystem that evaluates and reorders search results or passages to enhance their relevance to a specific query. In RAG, this tool builds on the primary vector Approximate Nearest Neighbor ([ANN](https://zilliz.com/glossary/anns)) search, improving search quality by more effectively determining the semantic relevance between documents and queries.

Rerankers fall into two main categories: **Score-based and Neural Network-based** .

**Score-based rerankers** work by aggregating multiple candidate lists from various sources, applying weighted scoring or Reciprocal Rank Fusion (RRF) to unify and reorder these candidates into a single, prioritized list based on their score or relative position in the original list. This type of reranker is known for its efficiency and is widely used in traditional search systems due to its lightweight nature.

On the other hand,**Neural Network-based reranker**s, often called cross-encoder rerankers, leverage a neural network to analyze the relevance between a query and a document. They are specifically designed to compute a similarity score that reflects the semantic proximity between the two, allowing for a refined rearrangement of results from single or multiple sources. This method ensures more semantic relatedness, thus providing useful search results and enhancing the overall effectiveness of the retrieval system.

* Cohere ReRank-3: [here](https://cohere.com/blog/rerank-3)

### Hybrid Search:

* Optimizing RAG with Hybrid Search & Reranking: [article](https://superlinked.com/vectorhub/articles/optimizing-rag-with-hybrid-search-reranking?utm_source=youtube&utm_medium=video&utm_campaign=vh-summary-video)
* Explanation from Microsoft Learn: [article and video](https://learn.microsoft.com/en-us/azure/search/hybrid-search-overview)


## RAG with Contextual Retrieval vs Other Similar Approaches

#### Using Summaries in Document Retrieval: [paper](https://aclanthology.org/W02-0405.pdf)

#### RAG with Hypothetical Document Embeddings (HyDE) : [article](https://zilliz.com/learn/improve-rag-and-information-retrieval-with-hyde-hypothetical-document-embeddings)
