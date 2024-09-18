# AI Training

![Understand AI, ML & Co in Contact Centers: Definitions & Explanations](https://lh7-us.googleusercontent.com/LzEWC6dsAER9egKvQWBSQ9Sr0ig2iAwpYcrq6XNOsmjAHp0K0X5_r9wgJOrwJTnH9squ5lPXTsia45ajT450JIBEKPmzAYw9Hk-wbyXiJXRlOqu9NfHimBW_AILVWQpO-_we1p4p3GaDbep07IS_-To)

1. **Artificial Intelligence (AI)**: AI is the field of computer science focused on creating systems that can perform tasks typically requiring human intelligence. This includes learning from experience, understanding natural language, recognizing patterns, and making decisions.
2. **Machine Learning (ML)**: ML is a subset of AI where algorithms learn from data to make predictions or decisions without being explicitly programmed. ML models improve their accuracy over time as they are exposed to more data.
3. **Deep Learning (DL)**: DL is a subset of ML that uses neural networks with many layers (hence "deep") to model complex patterns in data. DL is especially powerful in image and speech recognition, NLP, and LLMs.
4. **Natural Language Processing (NLP)**: NLP is the branch of AI that focuses on the interaction between computers and humans through natural language. It involves tasks like text classification, machine translation, and sentiment analysis.
5. **Large Language Models (LLMs)**: LLMs are a class of models (like GPT-3 and GPT-4) that use deep learning to process and generate human language. They are trained on massive datasets and have revolutionized NLP by enabling applications like chatbots, summarization, and question answering.



## Machine Learning

* [Course 1](https://www.coursera.org/learn/machine-learning/lecture/iYR2y/welcome-to-machine-learning): Introduction to Machine Learning, from Standford
* All ML algorithms explained: [here](https://scikit-learn.org/stable/supervised_learning.html)

In the context of **Machine Learning (ML)**, models are often categorized into **supervised**, **unsupervised**, and **semi-supervised** learning, based on the type of data and guidance available during training.

### 1. **Supervised Learning**:

- **Definition**: Supervised learning involves training a model on a labeled dataset, meaning that each input comes with an associated output (label). The goal is to learn a mapping from inputs to the correct outputs.
- **How it works**: The model learns by comparing its predictions to the true labels, minimizing the error, and adjusting the model parameters accordingly.
- **Example use cases**:
  - **Email classification**: Classifying emails as spam or not spam, where the labels (spam/not spam) are predefined.
  - **Image recognition**: Training a model to recognize objects (e.g., cats vs. dogs) where the correct label for each image is provided.

### 2. **Unsupervised Learning**:

- **Definition**: Unsupervised learning involves training a model on a dataset without labeled outputs. The model’s task is to uncover hidden patterns, structures, or relationships within the data.
- **How it works**: The model tries to find regularities in the data, such as clusters or associations, without any explicit guidance on what the correct output should be.
- **Example use cases**:
  - **Customer segmentation**: Grouping customers based on their behavior for targeted marketing.
  - **Anomaly detection**: Identifying unusual patterns in data, such as detecting fraud in financial transactions.

### 3. **Semi-Supervised Learning**:

- **Definition**: Semi-supervised learning lies between supervised and unsupervised learning. It uses a small amount of labeled data combined with a large amount of unlabeled data. The model can leverage the unlabeled data to improve learning when labeled data is scarce.
- **How it works**: The model initially learns from the labeled data but then generalizes further by using the unlabeled data to capture patterns. The assumption is that unlabeled data carries useful information that can help improve accuracy.
- **Example use cases**:
  - **Text classification**: Using a small set of labeled documents and a large set of unlabeled ones to classify emails.
  - **Medical image analysis**: Labeled medical data is often scarce, but combining it with unlabeled data can improve model performance.

### Summary:

| **Algorithm**                                                                             | **Type**  | **When to Use**                                                 | **Strengths**                                        | **Weaknesses**                                                 |
| ----------------------------------------------------------------------------------------------- | --------------- | --------------------------------------------------------------------- | ---------------------------------------------------------- | -------------------------------------------------------------------- |
| **Linear Regression**                                                                     | Supervised      | Continuous outcomes, when relationship between variables is linear    | Simple, interpretable, fast                                | Limited to linear relationships, sensitive to outliers               |
| **Logistic Regression**                                                                   | Supervised      | Binary classification problems                                        | Interpretable, works well on small datasets                | Assumes linear boundary, struggles with complex relationships        |
| **Decision Trees [video](https://www.youtube.com/watch?v=ZVR2Way4nwQ)**                     | Supervised      | Both classification and regression, interpretable models              | Easy to interpret, handles non-linear relationships        | Prone to overfitting, sensitive to small changes in data             |
| **Random Forest [video](https://www.youtube.com/watch?v=v6VJ2RO66Ag&t=192s)**               | Supervised      | When high accuracy is needed and interpretability is less important   | Reduces overfitting, works well with high-dimensional data | Slower, harder to interpret compared to decision trees               |
| **k-Nearest Neighbors**                                                                   | Supervised      | Classification, especially when class distribution is unknown         | Simple, effective for smaller datasets                     | Computationally expensive, sensitive to noisy data                   |
| **Support Vector Machines (SVM) [video](https://www.youtube.com/watch?v=_YPScrckx28)**      | Supervised      | High-dimensional feature spaces, binary or multi-class classification | Effective in high dimensions, clear margin of separation   | Memory-intensive, less effective on noisy data                       |
| **k-Means Clustering**                                                                    | Unsupervised    | Clustering similar data points, exploratory data analysis             | Simple, fast, works well with large datasets               | Sensitive to initial cluster placement, doesn't handle outliers well |
| **Principal Component Analysis (PCA) [video](https://www.youtube.com/watch?v=HMOI_lkzW08)** | Unsupervised    | Dimensionality reduction, exploratory data analysis                   | Reduces overfitting, useful for visualization              | Assumes linearity, may lose interpretability                         |
| **Naive Bayes**                                                                           | Supervised      | Text classification, spam filtering, naive assumptions                | Fast, works well with high-dimensional sparse data         | Strong assumptions about feature independence                        |
| **Reinforcement Learning**                                                                | Semi-supervised | Tasks requiring sequential decision-making (e.g., games, robotics)    | Learns optimal policies through exploration                | Requires large amounts of data and computational resources           |

- **Model evaluation techniques:**
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix
  - Cross-validation

#### Material:

* [scikit-learn](https://scikit-learn.org/stable/): most popular python library for ML
* **VIDEO:** Training AI to Play **Pokemon** with **Reinforcement Learning**: [video](https://www.youtube.com/watch?v=DcYLT37ImBY)
* **VIDEO:** Predicting the 3D Structure of proteins with **AlphaFold2 *(in spanish)*:** [link](https://www.youtube.com/watch?v=Uz7ucmqjZ08)
* **VIDEO:** Detecting Faces with Viola Jones Algorithm (how Snapchat and Instagram started applying face filters): [link](https://www.youtube.com/watch?v=uEJ71VlUmMQ)



## Deep Learning

* [Course 1](http://introtodeeplearning.com/): Introduction to Deep Learning, from MIT
* [Course 2](https://www.coursera.org/specializations/deep-learning): Deep Learning Specialization, from Deep Learning.AI

#### Summary:

- Key concepts:
  - Activation Functions (ReLU, Sigmoid, Tanh)
  - Forward Propagation and Backpropagation
  - Loss Functions (Mean Squared Error, Cross Entropy)
- Common Deep Learning Architectures:
  - Feedforward Neural Networks
  - Convolutional Neural Networks (CNNs)
  - Recurrent Neural Networks (RNNs): [video](https://www.youtube.com/watch?v=AsNTP8Kwu80)
  - Generative Adversarial Networks (GANs): [video](https://www.youtube.com/watch?v=_qB4B6ttXk8), [documentation](https://www.geeksforgeeks.org/generative-adversarial-network-gan/)
- Optimizers (SGD, Adam, RMSprop)
- Overfitting and Regularization (Dropout, L2 Regularization)

| **Architecture**                         | **When to Use**                                                      | **Strengths**                                                | **Weaknesses**                                              |
| ---------------------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------ | ----------------------------------------------------------------- |
| **Feedforward Neural Network (FNN)**     | Basic DL tasks, regression, and classification                             | Simple to implement, flexible                                      | Prone to overfitting, doesn’t handle sequential data well        |
| **Convolutional Neural Network (CNN)**   | Image recognition, video processing, spatial data tasks                    | Excellent for image data, fewer parameters than FNNs               | Requires large labeled datasets, not ideal for non-spatial data   |
| **Recurrent Neural Network (RNN)**       | Sequential data tasks like time series prediction, language modeling       | Handles sequential data, memory retention across inputs            | Vanishing gradient problem, struggles with long-term dependencies |
| **Long Short-Term Memory (LSTM)**        | Advanced sequential tasks, long-term dependency tasks (e.g., speech, text) | Solves vanishing gradient problem, captures long-term dependencies | Computationally expensive, slower training                        |
| **Generative Adversarial Network (GAN)** | Generating new data (images, music), creating deepfakes                    | Capable of generating high-quality data, unsupervised learning     | Difficult to train, sensitive to hyperparameters                  |
| **Variational Autoencoder (VAE)**        | Dimensionality reduction, anomaly detection, generative tasks              | Provides probabilistic approach to generation                      | Lower-quality generation compared to GANs                         |

#### Material:

* Python **notebooks** covering most of the basics of DL: [here](DeepLearning)
* **VIDEO:** What are deepfakes? [here](https://www.youtube.com/watch?v=pkF3m5wVUYI)



## NLP (Natural Language Processing)

- Key NLP tasks:
  - **Text Classification**: Assigning categories to text (e.g., spam detection).
  - **Named Entity Recognition (NER)**: Identifying proper nouns in text.
  - **Sentiment Analysis**: Classifying the sentiment of a given text.
  - **Machine Translation**: Translating text from one language to another.
- Key techniques:
  - Bag of Words (BoW) and TF-IDF
  - Word Embeddings (Word2Vec, GloVe)
  - Tokenization, Stemming, Lemmatization
- NLP libraries: spaCy, NLTK, Hugging Face Transformers

**The basics of language processing:** Language models process language by trying to understand grammatical structure (syntax) and meaning (semantics).

* **Parsing:** This technique analyzes the sentence structure, assigning parts of speech (noun, verb, adjective, etc.) to each word and identifying grammatical relationships.
* **Tokenization:** The model splits sentences into individual words (tokens), creating the building blocks for performing semantic analysis
* **Stemming:** This step reduces words to their root form (for example, "walking" becomes "walk"). This ensures the model treats similar words consistently.
* **Entity recognition and relationship extraction:** These techniques work together to identify and categorize specific entities (like people or places) within the text and uncover their relationships.
* **Word embeddings:** Finally, the model creates a numerical representation for each word (a vector), capturing its meaning and connections to other words. This allows the model to process the text and perform tasks like translation or summarization.



## LLMs (Large Language Models)

![Exploring business potential of AI, LLM, ML & DL | Inwedo](https://inwedo.com/app/uploads/2023/08/llm-explained-1024x550.png)

### **Transformers:** The origin of LLMs

**Transformers** are the foundation of LLMs, which led to significant advancements in NLP by allowing models to understand the context of words based on their relationships with other words. The **Transformer architecture** revolutionized tasks like machine translation and text generation.

**What is Attention?**

The **attention mechanism** was introduced in 2017 in the paper **Attention Is All You Need.** Unlike traditional methods that treat words in **isolation**, attention assigns weights to each word based on its **relevance to the current task**. This enables the model to capture long-range dependencies, analyze **both local and global contexts** simultaneously, and resolve ambiguities by attending to informative parts of the sentence.

* Original [paper](https://arxiv.org/abs/1706.03762), "Attention Is All You Need"
* [Audio summary](https://illuminate.google.com/home?pli=1&play=SKUdNc_PPLL8) of "Attention Is All You Need"
* Transformers explained: [video](https://www.youtube.com/watch?v=SZorAJ4I-sA)
* Transformers python library: [here](https://pypi.org/project/transformers/)
* Python notebook: [here](DeepLearning/11_lab_transformer_todo.ipynb)

### Word and Text embeddings

* **VIDEO:** Word Embeddings, word2vec: [link](https://www.youtube.com/watch?v=R3xHRSMCG5g)
* **VIDEO:** Text embeddings [explanation](https://www.youtube.com/watch?v=OATCgQtNX2o)

### LLM Tokenizers

LLMs process text by breaking it down into **tokens**. Tools like GPT-3 and ChatGPT rely heavily on tokenization to handle input efficiently. Understanding how **tokens** work is crucial for optimizing costs and performance in generative AI tasks.

* What are ChatGPT tokens? [link](https://www.youtube.com/watch?v=YM3dbgKnQgA)
* **VIDEO:** [Explanation](https://www.youtube.com/watch?v=hL4ZnAWSyuU)

### How to call API endpoints of LLMs like GPT-4?

When calling the API endpoint of a large language model like GPT-4 (via OpenAI’s API), you can typically pass several parameters to control the model's behavior and customize the response to suit your use case. Below are the common parameters you can pass:

1. **`model`**:

   - Specifies which model to use (e.g., `gpt-4`, `gpt-3.5-turbo`).
2. **`prompt`**:

   - The input text that you want the model to respond to or complete. In newer versions like GPT-4, this is referred to as `messages` for conversation-like interactions.
3. **`messages`**:

   - Used in chat models to pass a list of message objects between the user and the assistant. Example structure:
     ```json
     [
       {"role": "system", "content": "You are a helpful assistant."},
       {"role": "user", "content": "Tell me a joke."}
     ]
     ```
4. **`max_tokens`**:

   - The maximum number of tokens (words or word fragments) to generate in the completion. Higher values result in longer responses, but the model may cut off if it exceeds this limit.
5. **`temperature`**:

   - Controls the **randomness** of the output. A higher temperature (e.g., `1.0`) makes the model more creative and random, while a lower temperature (e.g., `0.2`) makes it more deterministic and focused.
6. **`top_p`**:

   - Also known as **nucleus sampling**, this parameter controls the diversity of the output. It restricts the model to considering only the top `p` percentage of the probability mass. For example, setting `top_p=0.9` means the model will consider only the top 90% of likely responses.
7. **`n`**:

   - Specifies the number of completions to generate for a given prompt. For example, `n=3` will return three different completions.
8. **`stop`**:

   - A list of tokens or strings where the model should stop generating further tokens. This is useful if you want the response to end at a specific point (e.g., after a certain phrase).
9. **`logprobs`**:

   - If set to a number (e.g., `5`), the API will return the log probabilities of the top `n` most likely tokens at each step. This is useful for analyzing model behavior or token choices.
10. **`presence_penalty`**:

    - A number between `-2.0` and `2.0`. Positive values encourage the model to talk about **new topics**, while negative values make it more likely to repeat itself.
11. **`frequency_penalty`**:

    - A number between `-2.0` and `2.0`. Positive values reduce the likelihood of the model repeating words or phrases within the same response.
12. **`stream`**:

    - If set to `true`, the API will return output tokens as they are generated, useful for applications requiring real-time interaction (e.g., streaming conversation).
13. **`best_of`**:

    - This parameter generates multiple completions server-side and returns only the best one based on the log probabilities. It differs from `n`, which returns all completions.
14. **`user`**:

    - A unique identifier that can be used to track API usage by different users of your application. This can help in organizing API calls by user in multi-user environments.
15. **`logit_bias`**:

    - A dictionary that allows you to influence the likelihood of certain tokens appearing in the model’s response. For example, you can increase or decrease the probability of specific words or phrases.

### Example API Call for GPT-4:

```json
{
  "model": "gpt-4",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the capital of France?"}
  ],
  "temperature": 0.5,
  "max_tokens": 100,
  "top_p": 1.0,
  "n": 1,
  "stop": ["\n"]
}
```

In this example, the model is set to respond with up to 100 tokens, using a temperature of 0.5 for balanced creativity and determinism, and will stop generating after a newline.

These parameters offer flexibility to fine-tune the behavior of GPT-4 for different use cases, including chatbots, content generation, programming assistance, and more.

#### Fine-tuning of LLMs: [link](../AI_Knowledge/LLMFineTuning.md)

#### Material:

* **VIDEO:** Train AI with your FACE (in spanish): [link](https://www.youtube.com/watch?v=rgKBjRLvjLs)
* **NOTES**: From a good LinkedIn friend: [link](https://docs.google.com/document/d/1K7ahLiopilE0TxpkRcrrzgUPTcjZq0aY_5J797_0o98/edit)


## RAG (Retrieval Augmented Generation)

![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXe0-JqD-TYU62yym64zlK8BnWZr3FFkjuEqhbwtrfcfH7bu4-zw7T7lafwOQV7RF3D5DB54fJYcibo2GFEl46u4yZDYIPwZVzB6a5upz0gE_OJjo6Q6Qu7S5SrKOZebVM8tQNgf_E773jxKo5do9UJk64Z-?key=9uyNk7FoZciHwUo0N8S5pQ)

**RAG (Retrieval-Augmented Generation)** is a technique that combines two AI systems: a **retrieval system** and a **generative model**, to enhance the ability of large language models (LLMs) to generate factually accurate and contextually relevant content. It is particularly useful in tasks where the model needs access to external knowledge, like open-domain question answering.

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

#### Advenced Techniques: [GraphRAG](../AI_Knowledge/GraphRAG.md)

## **Generative AI**

Generative AI focuses on creating new content, whether that be images, text, or audio. Key models include:

- **GANs** (Generative Adversarial Networks): Useful for generating realistic images or videos.
- **VAEs** (Variational Autoencoders): Good for generating less complex data like handwritten digits.
- **Diffusion Models**: State-of-the-art models for generating high-quality images (e.g., **DALL-E**, **Stable Diffusion**).

Applications range from image generation and speech synthesis to **prompt engineering** for text-to-image generation.

* Image Generation (e.g., DALL-E, Stable Diffusion)
* Text-to-Image (Prompt Engineering)
* Text Generation (e.g., GPT-3, GPT-4)
* Audio Generation (e.g., Speech Synthesis)
* Video and 3D Content Generation
