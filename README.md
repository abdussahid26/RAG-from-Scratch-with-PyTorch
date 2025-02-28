# RAG from Scratch with PyTorch

# 1. What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that enhances the performance of **Large Language Models (LLMs)** by combining them with an external retrieval mechanism. It involves retrieving relevant information from a knowledge source (such as a document database or vector store) to provide additional context before generating the final output. Since, RAG combines the power of **retrieval-based systems** and **generative models**, resulting in more accurate, dynamic, and efficient responses.

### **How RAG Works:**
1. **User Query**: A user provides an input prompt or query.
2. **Retrieval**: The system retrieves relevant documents or data from an external knowledge base (often through vector search).
3. **Augmentation**: The retrieved information is used to augment the input query, providing additional context.
4. **Generation**: The LLM processes the augmented input and generates a more accurate and contextually relevant response.

### **Key Components of RAG:**
- **Retriever**: A system to retrieve relevant information (e.g., vector search engines like FAISS, Pinecone).
- **Generator**: A large language model (e.g., GPT, Llama, BERT).
- **External Knowledge Base**: A repository of documents or data (e.g., a vector database or traditional database).

---

# 2. Why Should We Use RAG?

RAG enhances the capabilities of LLMs and mitigates some of their limitations, making it a valuable technique in many scenarios:

1. **Provides Up-to-Date Information**: 
   - LLMs may have outdated knowledge due to their training data. RAG allows the system to access real-time, dynamic information from external sources, ensuring that the model generates more relevant and current responses.

2. **Reduces Hallucinations**: 
   - LLMs sometimes generate responses that are factually incorrect or nonsensical, known as **hallucinations**. RAG reduces this by grounding the model's responses in external, factual data retrieved during the retrieval step.

3. **Improves Efficiency**:
   - Instead of re-training the LLM on vast amounts of data, RAG allows for **on-demand information retrieval**, making the system more efficient and reducing the need for large-scale fine-tuning.

4. **Supports Domain-Specific Knowledge**:
   - RAG enables LLMs to access **domain-specific or proprietary knowledge** that may not be part of the model's initial training data, allowing the system to be used in specialized fields like healthcare, law, or finance.

---

# 3. What Kind of Problems Can RAG Be Used For?

RAG is highly effective in scenarios that require accurate, context-aware, and data-intensive responses. Some common use cases include:

### 1. **Question Answering (QA)**:
   - **Example**: A medical assistant bot that retrieves the latest research papers and guidelines to answer a doctor's query.
   - **Benefit**: Provides precise, up-to-date answers from external sources, minimizing the risk of hallucinations.

### 2. **Document Summarization and Retrieval**:
   - **Example**: A legal document summarizer that retrieves relevant case laws from a legal database before generating a summary.
   - **Benefit**: Efficiently pulls in specific sections of documents, providing more accurate and relevant summaries.

### 3. **Enterprise Search & Knowledge Management**:
   - **Example**: An enterprise search tool that helps employees find specific company policies or documents by retrieving relevant content from an internal database.
   - **Benefit**: Users receive accurate answers by retrieving the right documents in response to specific queries.

### 4. **Personalized Recommendations**:
   - **Example**: An e-commerce platform suggesting products based on previous customer behavior, such as their past purchases or preferences.
   - **Benefit**: RAG allows the system to provide more personalized recommendations by retrieving relevant user preferences.

### 5. **Fraud and Anomaly Detection**:
   - **Example**: A financial monitoring system that detects unusual transactions by comparing them to patterns in a vectorized database of historical transactions.
   - **Benefit**: RAG can assist in identifying potential fraudulent activity by retrieving historical data to compare against new patterns.

---


# 4. Original RAG paper
* [Retrival-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401).

# 5. RAG Tutorial
* [Building RAG-based LLM Applications for Production](https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-1) by Goku Mohandas and Philipp Moritz.

# 6. Embedding Model
* [Text and Image Embedding Model] [Sentence Transformers](https://www.sbert.net/), [Massive Text Embedding (MTEB) Leaderboard](https://huggingface.co/spaces/mteb/leaderboard).

# 7. When a Vector Database Becomes Necessary?

A **vector database** is a specialized database designed to store, index, and search high-dimensional vector embeddings efficiently. It is needed when your dataset grows beyond what in-memory approaches (like NumPy or FAISS with a flat index) can handle efficiently. Below are the key reasons why:

### 1. Scalability  
- If you need to store and search through **millions or billions of embeddings**, a vector database provides optimized storage and retrieval.  
- Unlike NumPy arrays, vector databases efficiently **manage memory usage** and allow for **persistent storage** across sessions.  

### 2. Performance  
- Brute-force similarity search (e.g., using `np.dot`) works well for **small datasets** but becomes **slow** for large ones.  
- Vector databases use **specialized indexing** (like IVF, HNSW) to speed up searches, making queries much faster than scanning through raw NumPy arrays.  

### 3. Advanced Features  
- Many vector databases support **Approximate Nearest Neighbor (ANN) search**, which speeds up queries by sacrificing a small amount of accuracy.  
- **Hybrid search capabilities**: Some vector databases can **combine keyword search (BM25)** with **vector similarity** for better retrieval results.  
- **Metadata filtering**: Unlike NumPy, vector databases allow filtering based on additional metadata (e.g., “only retrieve documents from 2023”).  

### 4. Distributed Systems & Scaling  
- If your system needs **horizontal scaling**, vector databases can distribute data across multiple machines.  
- This is essential for **high-traffic applications** (e.g., real-time chatbots, recommendation systems).  

### When You Don’t Need a Vector Database  
- If your dataset has **less than 100K embeddings**, NumPy or FAISS (Flat Index) may be sufficient.  
- If **exact similarity search** is required (rather than approximate), brute-force approaches work well for smaller datasets.  
  
[INFO] If your RAG system is currently handling **100K+ embeddings**, it’s best to first try **NumPy or FAISS**. But if performance becomes a bottleneck, a **vector database like Pinecone, Weaviate, ChromaDB, or Milvus** will help scale efficiently.  

### Popular Vector Databases

Below are some of the most widely used vector databases, each optimized for different use cases.

#### 1. **FAISS (Facebook AI Similarity Search)**
🔹 **Best for**: High-performance, in-memory similarity search.  
🔹 **Features**:  
   - Supports Approximate Nearest Neighbor (ANN) search.  
   - Optimized for large-scale datasets.  
   - Can be used with GPUs for faster computations.  
🔹 **Repo**: [FAISS GitHub](https://github.com/facebookresearch/faiss)  

#### 2. **Pinecone**
🔹 **Best for**: Scalable, cloud-native vector search.  
🔹 **Features**:  
   - Fully managed, serverless infrastructure.  
   - Supports filtering, hybrid search (keyword + vectors).  
   - Integrates well with OpenAI, LangChain, etc.  
🔹 **Website**: [Pinecone.io](https://www.pinecone.io/)  

#### 3. **Weaviate**
🔹 **Best for**: Open-source, hybrid search with metadata filtering.  
🔹 **Features**:  
   - Combines BM25 keyword search with vector search.  
   - Supports GraphQL-based querying.  
   - Can run locally or in the cloud.  
🔹 **Repo**: [Weaviate GitHub](https://github.com/weaviate/weaviate)  

#### 4. **Milvus**
🔹 **Best for**: Distributed, high-speed vector search.  
🔹 **Features**:  
   - Scalable with support for billion-scale embeddings.  
   - Offers multiple indexing methods (HNSW, IVF, PQ, etc.).  
   - Supports multi-modal search (text, images, videos).  
🔹 **Repo**: [Milvus GitHub](https://github.com/milvus-io/milvus)  

#### 5. **ChromaDB**
🔹 **Best for**: Lightweight, easy-to-use vector database for RAG applications.  
🔹 **Features**:  
   - Simple API for fast integration with LLMs.  
   - Open-source and local-first by default.  
   - Optimized for small to mid-sized datasets.  
🔹 **Repo**: [ChromaDB GitHub](https://github.com/chroma-core/chroma)  

#### 6. **Qdrant**
🔹 **Best for**: High-performance, Rust-based vector search.  
🔹 **Features**:  
   - Fast, efficient indexing with HNSW.  
   - Supports filtering and payload metadata.  
   - Can be deployed on-premise or in the cloud.  
🔹 **Repo**: [Qdrant GitHub](https://github.com/qdrant/qdrant)  

---

##### 🔥 **Which One Should You Choose?**  
- **Small & fast local RAG** → Use **ChromaDB** or **FAISS**.  
- **Cloud-based, scalable search** → Try **Pinecone** or **Weaviate**.  
- **Hybrid search with metadata filtering** → Use **Weaviate** or **Qdrant**.  
- **Massive billion-scale datasets** → Go with **Milvus**.  

---

#### 🚀 **More Resources**  
- [Vector Database Benchmarks](https://ann-benchmarks.com/)  
