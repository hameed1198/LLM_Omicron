# 🔌 API Reference - Omicron Sentiment Analysis

## Core Classes and Methods

### `OmicronSentimentRAG` Class

The main analysis engine that combines sentiment analysis with RAG capabilities.

#### Constructor

```python
class OmicronSentimentRAG:
    def __init__(self, csv_path: str, 
                 anthropic_api_key: Optional[str] = None,
                 openai_api_key: Optional[str] = None, 
                 llm_provider: str = "auto")
```

**Parameters:**
- `csv_path` (str): Path to the CSV file containing tweet data
- `anthropic_api_key` (Optional[str]): Anthropic API key for Claude Sonnet
- `openai_api_key` (Optional[str]): OpenAI API key for GPT models
- `llm_provider` (str): LLM provider preference ("claude", "openai", "ollama", "auto")

**Returns:** OmicronSentimentRAG instance

---

## 📊 Data Analysis Methods

### `analyze_sentiment_distribution()`

Analyzes overall sentiment distribution across the dataset.

```python
def analyze_sentiment_distribution(self) -> Dict[str, Any]:
```

**Returns:**
```python
{
    'total_tweets': int,
    'sentiment_distribution': {
        'positive': int,
        'neutral': int, 
        'negative': int
    },
    'average_compound_score': float,
    'sentiment_breakdown': {
        'very_positive': int,    # compound > 0.5
        'positive': int,         # 0.05 < compound <= 0.5
        'neutral': int,          # -0.05 <= compound <= 0.05
        'negative': int,         # -0.5 <= compound < -0.05
        'very_negative': int     # compound < -0.5
    }
}
```

**Example:**
```python
analyzer = OmicronSentimentRAG("data/omicron_2025.csv")
sentiment_data = analyzer.analyze_sentiment_distribution()
print(f"Total tweets: {sentiment_data['total_tweets']}")
print(f"Positive: {sentiment_data['sentiment_distribution']['positive']}")
```

### `get_trending_hashtags(limit: int = 10)`

Identifies and ranks trending hashtags by frequency.

```python
def get_trending_hashtags(self, limit: int = 10) -> List[Dict[str, Union[str, int]]]:
```

**Parameters:**
- `limit` (int): Maximum number of hashtags to return (default: 10)

**Returns:**
```python
[
    {
        'hashtag': str,      # Hashtag text (without #)
        'count': int,        # Number of occurrences
        'percentage': float  # Percentage of total tweets
    }
]
```

**Example:**
```python
trending = analyzer.get_trending_hashtags(20)
for tag in trending[:5]:
    print(f"#{tag['hashtag']}: {tag['count']} tweets ({tag['percentage']:.1f}%)")
```

### `query_tweets_by_hashtag(hashtag: str)`

Retrieves all tweets containing a specific hashtag.

```python
def query_tweets_by_hashtag(self, hashtag: str) -> List[Dict[str, Any]]:
```

**Parameters:**
- `hashtag` (str): Hashtag to search for (with or without #)

**Returns:**
```python
[
    {
        'user_name': str,
        'user_location': str,
        'date': str,
        'tweet': str,
        'hashtags': List[str],
        'retweets': int,
        'favorites': int,
        'sentiment': str,        # 'positive', 'neutral', 'negative'
        'sentiment_score': float # VADER compound score
    }
]
```

**Example:**
```python
omicron_tweets = analyzer.query_tweets_by_hashtag("omicron")
print(f"Found {len(omicron_tweets)} tweets with #omicron")
```

### `get_user_tweets(username: str)`

Retrieves all tweets from a specific user or users matching the pattern.

```python
def get_user_tweets(self, username: str) -> List[Dict[str, Any]]:
```

**Parameters:**
- `username` (str): Username to search for (partial matching)

**Returns:** Same format as `query_tweets_by_hashtag()`

**Example:**
```python
user_tweets = analyzer.get_user_tweets("john")
print(f"Found {len(user_tweets)} tweets from users matching 'john'")
```

### `search_tweets_by_content(search_term: str, limit: int = 10)`

Searches tweet content for specific terms or phrases.

```python
def search_tweets_by_content(self, search_term: str, limit: int = 10) -> List[Dict[str, Any]]:
```

**Parameters:**
- `search_term` (str): Term or phrase to search for
- `limit` (int): Maximum number of results to return

**Returns:** Same format as `query_tweets_by_hashtag()`

**Example:**
```python
vaccine_tweets = analyzer.search_tweets_by_content("vaccine", 50)
print(f"Found {len(vaccine_tweets)} tweets mentioning 'vaccine'")
```

---

## 🤖 RAG and AI Methods

### `query_with_rag(query: str)`

Performs RAG-powered analysis using the configured LLM.

```python
def query_with_rag(self, query: str) -> str:
```

**Parameters:**
- `query` (str): Natural language question about the tweet data

**Returns:**
- `str`: AI-generated response based on relevant tweet context

**Example:**
```python
response = analyzer.query_with_rag("What are the main concerns about Omicron?")
print(response)
```

**Sample Queries:**
- "Which users are most worried about vaccine side effects?"
- "What's the sentiment towards CDC guidelines?"
- "Find tweets from healthcare workers about Omicron"
- "List the most retweeted negative tweets"

### `create_vector_store()`

Creates and initializes the FAISS vector store for semantic search.

```python
def create_vector_store(self) -> None:
```

**Process:**
1. Converts tweets to documents with metadata
2. Splits documents into chunks (1000 chars, 200 overlap)
3. Generates embeddings using sentence-transformers
4. Creates FAISS index for similarity search

**Called automatically during initialization.**

### `setup_retrieval_chain()`

Sets up the LangChain retrieval chain for RAG functionality.

```python
def setup_retrieval_chain(self) -> None:
```

**Components configured:**
- Custom prompt template for tweet analysis
- Vector store retriever (k=5 documents)
- LLM integration (Claude/GPT/Ollama)
- Response formatting and error handling

**Called automatically during initialization.**

---

## 🔧 Utility Methods

### `analyze_sentiment_vader(text: str)`

Performs VADER sentiment analysis on individual text.

```python
def analyze_sentiment_vader(self, text: str) -> Dict[str, Union[float, str]]:
```

**Parameters:**
- `text` (str): Text to analyze

**Returns:**
```python
{
    'negative': float,    # Negative sentiment score (0-1)
    'neutral': float,     # Neutral sentiment score (0-1)
    'positive': float,    # Positive sentiment score (0-1)
    'compound': float,    # Overall sentiment score (-1 to +1)
    'label': str         # 'positive', 'neutral', or 'negative'
}
```

### `analyze_sentiment_textblob(text: str)`

Performs TextBlob sentiment analysis on individual text.

```python
def analyze_sentiment_textblob(self, text: str) -> Dict[str, float]:
```

**Parameters:**
- `text` (str): Text to analyze

**Returns:**
```python
{
    'polarity': float,      # Sentiment polarity (-1 to +1)
    'subjectivity': float   # Subjectivity score (0 to +1)
}
```

### `clean_text(text: str)`

Cleans and preprocesses text for analysis.

```python
def clean_text(self, text: str) -> str:
```

**Parameters:**
- `text` (str): Raw text to clean

**Returns:**
- `str`: Cleaned text (lowercase, no URLs/mentions/special chars)

**Transformations applied:**
1. Convert to lowercase
2. Remove URLs (http/https links)
3. Remove user mentions (@username)
4. Remove special characters (keep alphanumeric + spaces)
5. Normalize whitespace

### `generate_report()`

Generates a comprehensive analysis report.

```python
def generate_report(self) -> Dict[str, Any]:
```

**Returns:**
```python
{
    'dataset_info': {
        'total_tweets': int,
        'date_range': str,
        'unique_users': int,
        'total_hashtags': int
    },
    'sentiment_analysis': Dict,      # From analyze_sentiment_distribution()
    'trending_hashtags': List,       # Top 10 hashtags
    'top_users': List,              # Most active users
    'engagement_stats': {
        'total_retweets': int,
        'total_favorites': int,
        'avg_engagement_per_tweet': float
    }
}
```

---

## 🌐 Streamlit Web Interface

### Main Application Function

```python
def main():
    """Main Streamlit application entry point"""
```

**Features:**
- Multi-page navigation
- Cached data loading
- Interactive visualizations
- Real-time RAG chat interface

### Page Functions

#### `show_overview(analyzer)`
Displays dataset overview with metrics and visualizations.

#### `show_interactive_query(analyzer)`
Provides natural language query interface with predefined examples.

#### `show_hashtag_analysis(analyzer)`
Shows hashtag trends with interactive exploration tools.

#### `show_user_analysis(analyzer)`
Analyzes user behavior and provides user search functionality.

#### `show_sentiment_analysis(analyzer)`
Deep dive into sentiment patterns with filtering and visualization.

#### `show_rag_chat(analyzer)`
AI-powered chat interface for conversational data analysis.

---

## 📝 Configuration Options

### Environment Variables

Create a `.env` file in the project root:

```bash
# Required for RAG functionality
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional alternatives
OPENAI_API_KEY=your_openai_api_key_here

# Optional configuration
LLM_PROVIDER=auto  # auto, claude, openai, ollama
```

### Streamlit Configuration

```python
st.set_page_config(
    page_title="Omicron Sentiment Analysis with RAG",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

### FAISS Index Configuration

```python
# Default settings for vector search
search_kwargs = {
    "k": 5,                    # Number of documents to retrieve
    "score_threshold": 0.0,    # Minimum similarity score
    "fetch_k": 20             # Number of documents to fetch initially
}
```

---

## 🚨 Error Handling

### Common Exceptions

#### `FileNotFoundError`
```python
# Raised when CSV file is not found
try:
    analyzer = OmicronSentimentRAG("invalid/path.csv")
except FileNotFoundError:
    print("CSV file not found. Please check the file path.")
```

#### `APIError`
```python
# Raised when API calls fail
try:
    response = analyzer.query_with_rag("test query")
except Exception as e:
    if "API" in str(e):
        print("API error occurred. Check your API key and internet connection.")
```

#### `VectorStoreError`
```python
# Raised when vector store operations fail
try:
    analyzer.create_vector_store()
except Exception as e:
    if "FAISS" in str(e):
        print("Vector store creation failed. Check data format and memory availability.")
```

### Graceful Degradation

The system is designed to work even when some components fail:

- **No API Key**: RAG features disabled, other analysis still works
- **No Internet**: Local analysis works, API-based features disabled  
- **Memory Issues**: Batch processing and reduced chunk sizes
- **Missing Data**: Skip problematic rows, continue with available data

---

## 📊 Performance Considerations

### Memory Usage
- **Base System**: ~200MB
- **With Embeddings**: ~500MB
- **With Large Vector Store**: ~1GB+

### Response Times
- **Local Analysis**: <1 second
- **RAG Queries**: 2-10 seconds (depending on LLM)
- **Data Loading**: 5-15 seconds (one-time)

### Optimization Tips

```python
# Use caching for repeated operations
@st.cache_data
def expensive_operation():
    return result

# Batch process large datasets
def process_in_batches(data, batch_size=1000):
    for batch in chunks(data, batch_size):
        yield process_batch(batch)

# Limit vector search scope
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
```

---

This API reference provides comprehensive documentation for all classes, methods, and functions in the Omicron Sentiment Analysis project, with practical examples and usage patterns.
