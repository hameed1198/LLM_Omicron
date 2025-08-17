# 🦠 Omicron Sentiment Analysis with RAG

A comprehensive sentiment analysis system for COVID-19 Omicron variant tweets using LangChain framework and Retrieval-Augmented Generation (RAG) with Claude Sonnet.

## 🚀 Features

- **Sentiment Analysis**: Advanced sentiment analysis using VADER and TextBlob
- **RAG Integration**: Retrieval-Augmented Generation using LangChain and Claude Sonnet
- **Interactive Queries**: Natural language queries about tweet data
- **Hashtag Analysis**: Find users and tweets by specific hashtags
- **User Analytics**: Analyze individual user behavior and engagement
- **Web Interface**: Beautiful Streamlit web application
- **Data Visualization**: Interactive charts and word clouds
- **Real-time Chat**: AI-powered chat interface for data exploration

## 📋 Requirements

- Python 3.8+
- Anthropic API key (optional, for RAG functionality)
- CSV file with tweet data

## 🛠️ Installation

1. **Clone or download the project files**

2. **Create a virtual environment** (recommended):
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

3. **Install dependencies**:

Choose one of the following options:

**Option A: Full installation (recommended)**
```bash
pip install -r requirements.txt
```

**Option B: Minimal installation (basic functionality only)**
```bash
pip install -r requirements-minimal.txt
```

**Option C: Development installation (includes testing tools)**
```bash
pip install -r requirements-dev.txt
```

4. **Set up environment variables** (optional for RAG):
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your Anthropic API key
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## 📊 Data Format

The system expects a CSV file with the following columns:
- `id`: Tweet ID
- `user_name`: Username
- `user_location`: User location
- `date`: Tweet date
- `text`: Tweet content
- `hashtags`: Hashtags (as string representation of list)
- `retweets`: Number of retweets
- `favorites`: Number of favorites
- Additional columns are preserved

## 🚀 Usage

### 1. Command Line Demo

Run the demo script to see basic functionality:

```bash
python demo.py
```

This will:
- Load and analyze the tweet data
- Show sentiment distribution
- Demonstrate hashtag queries
- Provide interactive query examples
- Work without requiring API keys

### 2. Streamlit Web Application

Launch the interactive web interface:

```bash
streamlit run streamlit_app.py
```

Features include:
- 📊 **Overview Dashboard**: Key metrics and visualizations
- 🔍 **Interactive Query**: Natural language queries
- #️⃣ **Hashtag Analysis**: Trending hashtags and exploration
- 👤 **User Analysis**: User behavior and engagement
- 😊 **Sentiment Deep Dive**: Detailed sentiment analysis
- 🤖 **RAG Chat**: AI-powered conversational interface

### 3. Python API

Use the system programmatically:

```python
from omicron_sentiment_rag import OmicronSentimentRAG

# Initialize (with or without API key)
analyzer = OmicronSentimentRAG("omicron_2025.csv", api_key="your_key_here")

# Query tweets by hashtag
omicron_tweets = analyzer.query_tweets_by_hashtag("omicron")
print(f"Found {len(omicron_tweets)} tweets with #omicron")

# Search by content
vaccine_tweets = analyzer.search_tweets_by_content("vaccine", limit=10)

# Get user tweets
user_tweets = analyzer.get_user_tweets("Nathan Joyner")

# Analyze sentiment distribution
sentiment_analysis = analyzer.analyze_sentiment_distribution()

# RAG query (requires API key)
response = analyzer.query_with_rag("What are the main concerns about Omicron?")
```

## 🔍 Example Queries

### Hashtag Queries
- "list users with hashtag omicron"
- "find tweets with hashtag CDC"
- "show users tweeting about vaccine"

### Content Search
- "find tweets mentioning hospital"
- "search for mild symptoms"
- "tweets about vaccine effectiveness"

### User Analysis
- "tweets by Nathan Joyner"
- "show user activity for pmc"
- "analyze user sentiment patterns"

### RAG Queries (with API key)
- "What are the main concerns about Omicron mentioned in tweets?"
- "Which users are most worried about vaccine effectiveness?"
- "Summarize the sentiment about Omicron severity"
- "What locations have the most negative sentiment?"

## 📈 System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CSV Data      │───▶│  Data Processing │───▶│  Sentiment      │
│   (Tweets)      │    │  & Cleaning      │    │  Analysis       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   RAG System    │◀───│  Vector Store    │◀───│  Embeddings     │
│   (Claude)      │    │  (FAISS)         │    │  Generation     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐    ┌──────────────────┐
│   Query Engine  │───▶│  Response        │
│   & Interface   │    │  Generation      │
└─────────────────┘    └──────────────────┘
```

## 🧪 Technical Components

### Sentiment Analysis
- **VADER**: Optimized for social media text
- **TextBlob**: General purpose sentiment analysis
- **Custom preprocessing**: URL removal, mention cleaning

### RAG Implementation
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS for efficient similarity search
- **LLM**: Claude Sonnet 3 for intelligent responses
- **Retrieval**: Top-k similarity search with context injection

### Data Processing
- **Text Cleaning**: URL removal, hashtag parsing
- **Feature Extraction**: Hashtag lists, engagement metrics
- **Preprocessing**: Normalization and standardization

## 🔧 Configuration

### Environment Variables (.env)
```bash
# Required for RAG functionality
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional configurations
CSV_FILE_PATH=omicron_2025.csv
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=claude-3-sonnet-20240229
LLM_TEMPERATURE=0.1
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=10
```

### Model Parameters
- **Chunk Size**: 1000 characters for document splitting
- **Overlap**: 200 characters for context preservation
- **Retrieval K**: Top 10 most relevant documents
- **Temperature**: 0.1 for consistent responses

## 📊 Output Examples

### Hashtag Query Response
```
Found 156 tweets with #omicron

Tweet 1:
  User: Wont_Back_Down
  Location: State 48
  Sentiment: neutral
  Tweet: Doctor Who Helped Discover #Omicron Says She Was Pressured Not to Reveal It's Mild
  Engagement: 0 retweets, 0 favorites
```

### Sentiment Analysis
```
Sentiment Distribution:
  Negative: 1,234 (35.6%)
  Neutral: 1,567 (45.2%)
  Positive: 666 (19.2%)

Average Sentiment Score: -0.127
```

### RAG Response
```
Query: "What are the main concerns about Omicron?"

Response: Based on the analyzed tweets, the main concerns about Omicron include:

1. **Hospital Capacity**: Many users express worry about healthcare system strain
2. **Vaccine Effectiveness**: Questions about booster efficacy against the variant
3. **Transmission Rates**: Concerns about rapid spread and infectivity
4. **Symptom Severity**: Mixed discussions about mild vs. severe outcomes
5. **Policy Response**: Debate about appropriate public health measures
```

## 🛡️ Limitations

- **API Dependencies**: RAG functionality requires Anthropic API access
- **Data Quality**: Results depend on tweet data quality and completeness
- **Language**: Optimized for English text analysis
- **Real-time**: Static analysis of provided dataset (not live tweets)

## 🔮 Future Enhancements

- **Live Twitter Integration**: Real-time tweet streaming
- **Multi-language Support**: Analysis in multiple languages
- **Advanced Visualizations**: Network analysis, geographic mapping
- **Model Fine-tuning**: Custom sentiment models for COVID-19 context
- **Export Features**: PDF reports, data exports
- **API Endpoint**: RESTful API for integration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is provided as-is for educational and research purposes.

## 🆘 Support

If you encounter issues:

1. **Check Dependencies**: Ensure all packages are installed
2. **Verify Data Format**: Confirm CSV structure matches requirements
3. **API Keys**: Verify Anthropic API key is valid (for RAG features)
4. **File Paths**: Ensure CSV file is in the correct location

## 📞 Contact

For questions or collaboration opportunities, please create an issue in the repository.

---

Built with ❤️ using LangChain, Claude Sonnet, and Streamlit
