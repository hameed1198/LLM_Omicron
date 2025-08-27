# 📖 Technical Documentation - Omicron Sentiment Analysis

## 🔬 Deep Dive: Algorithm Implementation Details

### 1. Sentiment Analysis Algorithm Architecture

#### VADER (Valence Aware Dictionary and sEntiment Reasoner)

**Mathematical Foundation:**
- **Lexicon-based approach** with grammatical and syntactical heuristics
- **Compound Score Calculation:**
  ```python
  compound = normalize(sum_s, alpha=15)
  # Where sum_s is the sum of valence scores
  # Alpha parameter controls normalization intensity
  ```

**Implementation Details:**
```python
def analyze_sentiment_vader(self, text):
    scores = self.vader_analyzer.polarity_scores(text)
    # Returns: {'neg': 0.0, 'neu': 0.5, 'pos': 0.5, 'compound': 0.5}
    
    # Classification thresholds
    if scores['compound'] >= 0.05:
        label = 'positive'
    elif scores['compound'] <= -0.05:
        label = 'negative'
    else:
        label = 'neutral'
```

**Key Features:**
- **Punctuation Amplification**: Multiple exclamation marks increase intensity
- **Capitalization Boost**: ALL CAPS increases sentiment strength by 0.733
- **Negation Handling**: Detects and adjusts for negation words
- **Degree Modifiers**: Words like "very", "extremely" modify sentiment intensity

#### TextBlob Sentiment Analysis

**Algorithm Type:** Naive Bayes classifier trained on movie reviews
**Mathematical Model:**
```python
P(sentiment|text) = P(text|sentiment) * P(sentiment) / P(text)
```

**Implementation:**
```python
def analyze_sentiment_textblob(self, text):
    blob = TextBlob(text)
    return {
        'polarity': blob.sentiment.polarity,      # -1 to +1
        'subjectivity': blob.sentiment.subjectivity  # 0 to +1
    }
```

### 2. Vector Embedding Deep Dive

#### Sentence-BERT Model: all-MiniLM-L6-v2

**Architecture:**
- **Base Model**: Microsoft's MiniLM (Mini Language Model)
- **Training Objective**: Siamese and triplet networks
- **Embedding Dimension**: 384
- **Context Window**: 512 tokens
- **Performance**: 84.4% on STS benchmark

**Mathematical Process:**
1. **Tokenization**: BERT WordPiece tokenizer
2. **Contextualization**: 6-layer transformer with attention
3. **Pooling**: Mean pooling of token embeddings
4. **Normalization**: L2 normalization for cosine similarity

**Implementation:**
```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
```

### 3. FAISS Vector Search Algorithm

#### Index Type: Flat (Exact Search)
- **Algorithm**: Brute-force L2 distance computation
- **Complexity**: O(n*d) where n=documents, d=dimensions
- **Accuracy**: 100% recall (exact search)

**Distance Metric:**
```python
# L2 (Euclidean) Distance
distance = sqrt(sum((qi - pi)^2 for qi, pi in zip(query, point)))

# Converted to similarity score
similarity = 1 / (1 + distance)
```

**Index Creation Process:**
```python
# 1. Create document vectors
vectors = np.array([embedding_model.embed_query(doc) for doc in documents])

# 2. Build FAISS index
index = faiss.IndexFlatL2(384)  # 384-dimensional vectors
index.add(vectors.astype('float32'))

# 3. Perform similarity search
distances, indices = index.search(query_vector, k=5)
```

### 4. RAG Pipeline Architecture

#### Retrieval Component
**Algorithm**: Dense Passage Retrieval (DPR) variant
**Process Flow:**
```python
def retrieve_context(query, k=5):
    # 1. Encode query
    query_vector = embeddings.embed_query(query)
    
    # 2. Search vector store
    docs = vector_store.similarity_search_with_score(query, k=k)
    
    # 3. Filter by relevance threshold
    relevant_docs = [doc for doc, score in docs if score > threshold]
    
    return relevant_docs
```

#### Generation Component
**Model Integration Pattern:**
```python
def generate_response(query, context):
    # 1. Construct prompt with context
    prompt = format_prompt(query, context)
    
    # 2. Call LLM with structured input
    response = llm.invoke([
        HumanMessage(content=prompt)
    ])
    
    # 3. Post-process response
    return clean_response(response.content)
```

### 5. Data Processing Pipeline

#### Text Preprocessing Algorithm
```python
def clean_text(text):
    # 1. Lowercase conversion
    text = text.lower()
    
    # 2. URL removal (regex pattern)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 3. Mention removal
    text = re.sub(r'@\w+', '', text)
    
    # 4. Special character filtering
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # 5. Whitespace normalization
    text = ' '.join(text.split())
    
    return text.strip()
```

#### Hashtag Extraction Algorithm
```python
def extract_hashtags(text):
    # Regex pattern for hashtag detection
    pattern = r'#\w+'
    hashtags = re.findall(pattern, text, re.IGNORECASE)
    
    # Clean and normalize
    clean_hashtags = [tag.lower().replace('#', '') for tag in hashtags]
    
    return list(set(clean_hashtags))  # Remove duplicates
```

## 🔧 LLM Integration Patterns

### 1. Anthropic Claude Integration

**API Configuration:**
```python
llm = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    anthropic_api_key=api_key,
    temperature=0.1,           # Low temperature for consistency
    max_tokens=1000,          # Response length limit
    timeout=30,               # Request timeout
)
```

**Prompt Engineering Strategy:**
```python
system_prompt = """
You are an expert data analyst specializing in social media sentiment analysis.
Your responses should be:
1. Data-driven and specific
2. Based solely on provided context
3. Formatted with clear structure
4. Factual without speculation
"""

human_prompt = f"""
Context: {context}
Query: {query}
Provide analysis based on the tweet data above.
"""
```

### 2. OpenAI GPT Integration

**Configuration Differences:**
```python
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=api_key,
    temperature=0.1,
    max_tokens=1000,
    request_timeout=30,
)
```

### 3. Ollama Local Integration

**Setup Requirements:**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Download model
ollama pull llama2

# Start server
ollama serve
```

**Integration Code:**
```python
llm = Ollama(
    model="llama2",
    base_url="http://localhost:11434",
    temperature=0.1,
)
```

## 📊 Performance Optimization Strategies

### 1. Caching Implementation

**Streamlit Caching:**
```python
@st.cache_data
def load_analyzer():
    """Cache expensive initialization"""
    return OmicronSentimentRAG(csv_path, api_key)

@st.cache_data
def process_sentiment_data(_analyzer):
    """Cache sentiment analysis results"""
    return _analyzer.analyze_sentiment_distribution()
```

### 2. Memory Management

**Vector Store Optimization:**
```python
# Use float32 instead of float64 for embeddings
vectors = vectors.astype('float32')

# Implement batch processing for large datasets
def process_in_batches(data, batch_size=1000):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]
```

### 3. Query Optimization

**Retrieval Tuning:**
```python
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,                    # Top-k results
        "score_threshold": 0.7,    # Relevance threshold
    }
)
```

## 🛡️ Error Handling & Robustness

### 1. API Failure Handling

```python
def query_with_rag(self, query: str) -> str:
    try:
        if not self.retrieval_chain:
            return "RAG functionality requires an API key."
        
        response = self.retrieval_chain({"query": query})
        return response['result']
        
    except Exception as e:
        error_msg = f"Error processing RAG query: {str(e)}"
        
        # Log error details
        logging.error(f"RAG Error: {e}")
        
        # Provide fallback response
        return self._fallback_query_response(query)
```

### 2. Data Validation

```python
def validate_data(self):
    """Validate input data integrity"""
    required_columns = ['text', 'user_name', 'date']
    
    for col in required_columns:
        if col not in self.df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Check for empty text
    empty_texts = self.df['text'].isna().sum()
    if empty_texts > 0:
        logging.warning(f"Found {empty_texts} empty text entries")
```

### 3. Graceful Degradation

```python
def initialize_with_fallback(self):
    """Initialize with fallback options"""
    try:
        # Primary: Try Claude
        if self.anthropic_key:
            self.llm = ChatAnthropic(...)
    except:
        try:
            # Secondary: Try OpenAI
            if self.openai_key:
                self.llm = ChatOpenAI(...)
        except:
            # Tertiary: Try Ollama
            try:
                self.llm = Ollama(...)
            except:
                # Final: Disable RAG
                self.llm = None
                logging.warning("RAG disabled - no LLM available")
```

## 📈 Monitoring & Analytics

### 1. Performance Metrics

```python
def track_performance(func):
    """Decorator to track function performance"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logging.info(f"{func.__name__} took {end_time - start_time:.2f}s")
        return result
    return wrapper
```

### 2. Usage Statistics

```python
class AnalyticsTracker:
    def __init__(self):
        self.query_count = 0
        self.response_times = []
        self.error_count = 0
    
    def log_query(self, query, response_time, success=True):
        self.query_count += 1
        self.response_times.append(response_time)
        if not success:
            self.error_count += 1
```

## 🔮 Advanced Features & Extensions

### 1. Custom Embedding Models

```python
# Option to use domain-specific embeddings
def load_custom_embeddings(model_path):
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
```

### 2. Multi-modal Analysis

```python
# Future: Image analysis for tweets with media
def analyze_tweet_images(image_urls):
    # Implement CLIP or similar for image-text analysis
    pass
```

### 3. Real-time Streaming

```python
# Future: Real-time tweet analysis
def stream_twitter_analysis():
    # Implement Twitter API v2 streaming
    pass
```

---

This technical documentation provides the deep implementation details of all algorithms, libraries, and systems used in the Omicron Sentiment Analysis project. Each component is explained with mathematical foundations, code examples, and optimization strategies.
