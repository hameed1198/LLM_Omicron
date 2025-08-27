# 🚀 Deployment & Usage Guide

## Quick Start Guide

### 1. System Requirements

**Minimum Requirements:**
- Python 3.8+
- 4GB RAM
- 2GB free disk space
- Internet connection (for API-based features)

**Recommended:**
- Python 3.10+
- 8GB RAM
- 5GB free disk space
- GPU support (optional, for faster processing)

### 2. Installation Steps

```bash
# Step 1: Clone the repository
git clone https://github.com/hameed1198/LLM_Omicron.git
cd LLM_Omicron

# Step 2: Create virtual environment
python -m venv omicron
source omicron/bin/activate  # Linux/Mac
# OR
omicron\Scripts\activate     # Windows

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Configure environment (optional)
cp .env.example .env
# Edit .env with your API keys
```

### 3. Running the Application

#### Web Interface (Recommended)
```bash
cd omicron_analysis_project/web_app
streamlit run streamlit_app.py
```
Access at: http://localhost:8501

#### Command Line Interface
```bash
cd omicron_analysis_project/core
python demo_analysis.py
```

---

## 🔑 API Key Configuration

### Anthropic Claude (Recommended)

1. **Get API Key:**
   - Visit [Anthropic Console](https://console.anthropic.com/)
   - Create account and generate API key
   - Note: Requires payment method for API usage

2. **Configure:**
   ```bash
   # In .env file
   ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
   ```

3. **Pricing:** 
   - Claude Sonnet: ~$3 per million tokens
   - Typical query: 0.01-0.05 cents

### OpenAI GPT (Alternative)

1. **Get API Key:**
   - Visit [OpenAI Platform](https://platform.openai.com/)
   - Generate API key in your account settings

2. **Configure:**
   ```bash
   # In .env file
   OPENAI_API_KEY=sk-your-openai-key-here
   ```

### Ollama (Free Local Option)

1. **Install Ollama:**
   ```bash
   # Linux/Mac
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Windows: Download from ollama.ai
   ```

2. **Download Models:**
   ```bash
   ollama pull llama2        # 7B parameters, ~4GB
   ollama pull mistral       # Alternative model
   ollama pull codellama     # Code-focused model
   ```

3. **Start Server:**
   ```bash
   ollama serve
   ```

---

## 📋 Usage Examples

### Basic Analysis Workflow

```python
from core.omicron_sentiment_rag import OmicronSentimentRAG

# Initialize analyzer
analyzer = OmicronSentimentRAG(
    csv_path="data/omicron_2025.csv",
    anthropic_api_key="your-key-here"
)

# Basic sentiment analysis
sentiment_data = analyzer.analyze_sentiment_distribution()
print(f"Positive tweets: {sentiment_data['sentiment_distribution']['positive']}")

# Hashtag analysis
trending = analyzer.get_trending_hashtags(10)
for tag in trending:
    print(f"#{tag['hashtag']}: {tag['count']} tweets")

# User analysis
user_tweets = analyzer.get_user_tweets("john")
print(f"Found {len(user_tweets)} tweets from John")

# RAG-powered queries
response = analyzer.query_with_rag("What are the main concerns about Omicron?")
print(response)
```

### Advanced Query Examples

```python
# Sentiment-based queries
analyzer.query_with_rag("Show me the most negative tweets about vaccines")
analyzer.query_with_rag("Which users are most optimistic about recovery?")

# Topic-based queries  
analyzer.query_with_rag("What do healthcare workers say about Omicron?")
analyzer.query_with_rag("Find tweets about hospital capacity issues")

# Statistical queries
analyzer.query_with_rag("What's the sentiment trend over time?")
analyzer.query_with_rag("Which locations have the most negative sentiment?")

# User behavior queries
analyzer.query_with_rag("Who are the most influential users discussing Omicron?")
analyzer.query_with_rag("Find users with consistently negative sentiment")
```

---

## 🎛️ Customization Options

### 1. Data Source Customization

#### Using Different Datasets
```python
# Custom CSV format
analyzer = OmicronSentimentRAG("path/to/your/tweets.csv")

# Required columns:
# - text: Tweet content
# - user_name: Username
# - date: Tweet timestamp
# - hashtags: Hashtag list (JSON format)
```

#### Data Preprocessing Options
```python
# Custom text cleaning
def custom_clean_text(text):
    # Your custom cleaning logic
    return cleaned_text

# Override default cleaning
analyzer.clean_text = custom_clean_text
```

### 2. Model Configuration

#### Custom Embedding Models
```python
from langchain.embeddings import HuggingFaceEmbeddings

# Use different embedding model
custom_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Initialize with custom embeddings
analyzer = OmicronSentimentRAG(
    csv_path="data.csv",
    embeddings=custom_embeddings
)
```

#### LLM Provider Selection
```python
# Force specific LLM provider
analyzer = OmicronSentimentRAG(
    csv_path="data.csv",
    llm_provider="claude",  # or "openai", "ollama"
    anthropic_api_key="your-key"
)
```

### 3. Vector Store Tuning

```python
# Adjust retrieval parameters
retriever = analyzer.vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 10,                    # More documents
        "score_threshold": 0.8,     # Higher relevance threshold
    }
)
```

### 4. Sentiment Analysis Customization

```python
# Custom sentiment thresholds
def custom_sentiment_label(compound_score):
    if compound_score >= 0.3:
        return 'very_positive'
    elif compound_score >= 0.1:
        return 'positive'
    elif compound_score >= -0.1:
        return 'neutral'
    elif compound_score >= -0.3:
        return 'negative'
    else:
        return 'very_negative'

# Apply custom labeling
analyzer.custom_sentiment_label = custom_sentiment_label
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors
```bash
# Problem: ModuleNotFoundError
# Solution: Ensure virtual environment is activated
source omicron/bin/activate  # Linux/Mac
omicron\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

#### 2. Memory Issues
```bash
# Problem: Out of memory during vector store creation
# Solution: Reduce batch size or use smaller embedding model

# In code:
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,    # Reduced from 1000
    chunk_overlap=50   # Reduced from 200
)
```

#### 3. API Rate Limits
```python
# Problem: API rate limit exceeded
# Solution: Implement retry logic with exponential backoff

import time
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def query_with_retry(query):
    return analyzer.query_with_rag(query)
```

#### 4. Streamlit Issues
```bash
# Problem: Streamlit not starting
# Solution: Check port availability
streamlit run streamlit_app.py --server.port 8502

# Problem: Caching issues
# Solution: Clear Streamlit cache
streamlit cache clear
```

#### 5. FAISS Installation Issues
```bash
# Problem: FAISS installation fails
# Solution: Use conda or different FAISS version
conda install -c conda-forge faiss-cpu
# OR
pip install faiss-cpu==1.7.4
```

### Performance Optimization

#### 1. Faster Data Loading
```python
# Use pandas optimizations
df = pd.read_csv("data.csv", 
                 dtype={'user_followers': 'int32'},  # Smaller int type
                 parse_dates=['date'],               # Parse dates efficiently
                 nrows=10000)                        # Limit for testing
```

#### 2. Batch Processing
```python
# Process large datasets in chunks
def process_large_dataset(csv_path, chunk_size=5000):
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        yield analyze_chunk(chunk)
```

#### 3. GPU Acceleration (Optional)
```python
# Use GPU for embeddings if available
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
)
```

---

## 📊 Production Deployment

### 1. Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "web_app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
# Build and run
docker build -t omicron-analysis .
docker run -p 8501:8501 -e ANTHROPIC_API_KEY=your-key omicron-analysis
```

### 2. Cloud Deployment

#### Streamlit Cloud
1. Push code to GitHub
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy directly from repository
4. Add secrets in Streamlit Cloud dashboard

#### Heroku Deployment
```bash
# Install Heroku CLI and login
heroku create omicron-sentiment-app

# Set environment variables
heroku config:set ANTHROPIC_API_KEY=your-key

# Deploy
git push heroku main
```

#### AWS/Azure/GCP
- Use container services (ECS, Container Instances, Cloud Run)
- Configure environment variables in cloud console
- Set up load balancing for multiple instances

### 3. Production Configuration

```python
# config.py
import os

class ProductionConfig:
    DEBUG = False
    TESTING = False
    
    # API Configuration
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    # Performance Settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_RETRIEVAL_DOCS = 5
    
    # Security Settings
    ALLOWED_HOSTS = ['your-domain.com']
    RATE_LIMIT = 100  # requests per hour
```

---

## 📈 Monitoring and Maintenance

### 1. Logging Configuration

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('omicron_analysis.log'),
        logging.StreamHandler()
    ]
)

# Usage in application
logger = logging.getLogger(__name__)
logger.info("Starting sentiment analysis")
logger.error(f"API error: {error_message}")
```

### 2. Performance Monitoring

```python
# Simple performance tracker
class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'query_count': 0,
            'avg_response_time': 0,
            'error_rate': 0
        }
    
    def log_query(self, response_time, success=True):
        self.metrics['query_count'] += 1
        # Update average response time
        # Track error rate
```

### 3. Health Checks

```python
# Health check endpoint
def health_check():
    try:
        # Test database connection
        analyzer.df.shape
        
        # Test vector store
        analyzer.vector_store.similarity_search("test", k=1)
        
        # Test LLM (if available)
        if analyzer.llm:
            analyzer.llm.invoke("test")
        
        return {"status": "healthy", "timestamp": datetime.now()}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

---

## 🔄 Updates and Migration

### Updating Dependencies
```bash
# Check for updates
pip list --outdated

# Update specific packages
pip install --upgrade langchain streamlit

# Update all packages
pip install --upgrade -r requirements.txt
```

### Data Migration
```python
# Migrate to new data format
def migrate_data_format(old_csv, new_csv):
    df = pd.read_csv(old_csv)
    # Apply transformations
    df.to_csv(new_csv, index=False)
```

### Model Updates
```python
# Update embedding model
def update_embedding_model(new_model_name):
    # Backup existing vector store
    # Recreate with new embeddings
    # Validate performance
    pass
```

---

This comprehensive deployment and usage guide covers everything from basic setup to production deployment, providing practical solutions for common issues and optimization strategies.
