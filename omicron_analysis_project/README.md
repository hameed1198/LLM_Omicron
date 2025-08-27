# 🦠 Omicron Sentiment Analysis with RAG

A comprehensive sentiment analysis system for COVID-19 Omicron variant discussions on Twitter, powered by LangChain, RAG (Retrieval-Augmented Generation), and **FREE AI models**.

> 🆓 **NEW**: Now supports multiple **FREE AI providers**! No need for expensive APIs. See `FREE_AI_SETUP_GUIDE.md` for complete setup instructions.

## 🆓 FREE AI OPTIONS

| Provider | Cost | Quality | Setup Difficulty | Recommended For |
|----------|------|---------|------------------|-----------------|
| **Google Gemini** ⭐ | FREE (1,500 requests/day) | Excellent | Easy | Most users |
| **Together AI** | FREE ($25 monthly) | High | Easy | Heavy usage |
| **Ollama** | 100% FREE | Good | Medium | Privacy-focused |
| **Cohere** | FREE tier | Professional | Easy | Developers |

## 📋 Table of Contents
- [FREE AI Options](#free-ai-options)
- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Large Language Models (LLMs)](#large-language-models-llms)
- [Algorithms & Analysis Methods](#algorithms--analysis-methods)
- [Libraries & Packages](#libraries--packages)
- [RAG Implementation](#rag-implementation)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

## 🎯 Project Overview

This project analyzes 17,046 Twitter tweets about the COVID-19 Omicron variant using advanced sentiment analysis techniques and RAG-powered AI chat functionality. The system provides comprehensive insights through multiple analysis methods and an interactive web dashboard.

### Key Capabilities:
- **Multi-method sentiment analysis** (VADER + TextBlob)
- **RAG-powered AI chat** with **FREE AI providers**
- **Interactive data exploration** with Streamlit dashboard
- **Hashtag and user analysis**
- **Real-time tweet querying and filtering**
- **Vector-based semantic search**

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Web Interface (Streamlit)                │
├─────────────────────────────────────────────────────────────┤
│                  Core Analysis Engine                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐│
│  │   Sentiment     │  │      RAG        │  │    Data      ││
│  │   Analysis      │  │     System      │  │  Processing  ││
│  │                 │  │                 │  │              ││
│  │ • VADER         │  │ • Vector Store  │  │ • CSV Parser ││
│  │ • TextBlob      │  │ • Embeddings    │  │ • Hashtag    ││
│  │ • Compound      │  │ • LLM Chain     │  │   Extraction ││
│  │   Scoring       │  │ • Retrieval     │  │ • Text Clean ││
│  └─────────────────┘  └─────────────────┘  └──────────────┘│
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐│
│  │   CSV Data      │  │  FAISS Vector   │  │   HuggingFace││
│  │   (17,046       │  │     Store       │  │   Embeddings ││
│  │    tweets)      │  │                 │  │              ││
│  └─────────────────┘  └─────────────────┘  └──────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## 🤖 Large Language Models (LLMs)

### 🆓 FREE AI OPTIONS AVAILABLE!

This system now supports **multiple FREE AI providers** so you don't need to pay for Claude Sonnet! See `FREE_AI_SETUP_GUIDE.md` for detailed setup instructions.

### Supported LLM Providers (Prioritized by Cost):

#### 🆓 **FREE OPTIONS**

1. **Google Gemini** (⭐ RECOMMENDED FREE)
   - **Cost**: Completely FREE up to 1,500 requests/day
   - **Quality**: Excellent performance, comparable to paid models
   - **Setup**: Simple API key from Google AI Studio
   - **Best for**: Most users wanting high-quality free AI

2. **Together AI** (FREE Tier)
   - **Cost**: $25 FREE credits monthly (very generous)
   - **Quality**: High-quality open-source models
   - **Setup**: Quick signup for free credits
   - **Best for**: Heavy usage needs

3. **Cohere** (FREE Tier)
   - **Cost**: FREE tier with good limits
   - **Quality**: Professional-grade language model
   - **Setup**: Free account registration

4. **Ollama** (Completely Free)
   - **Cost**: 100% FREE (runs locally)
   - **Quality**: Good performance with local models
   - **Setup**: Download and run locally
   - **Best for**: Privacy-focused users, unlimited usage

5. **HuggingFace Local Models** (Completely Free)
   - **Cost**: 100% FREE (runs locally)
   - **Quality**: Varies by model selection
   - **Setup**: Download models locally
   - **Best for**: Developers, customization needs

#### 💰 **PAID OPTIONS**

6. **Anthropic Claude Sonnet** (Original)
- **Model**: `claude-3-sonnet-20240229`
- **Purpose**: RAG-powered chat and query processing
- **Configuration**: Temperature 0.1 for consistent responses
- **API**: Anthropic API via LangChain ChatAnthropic
- **Use Case**: Advanced reasoning, context understanding, tweet analysis

#### 2. **OpenAI GPT** (Alternative)
- **Model**: `gpt-3.5-turbo` (configurable to GPT-4)
- **Purpose**: Backup LLM for RAG functionality
- **Configuration**: Temperature 0.1
- **API**: OpenAI API via LangChain ChatOpenAI
- **Use Case**: Natural language processing, query understanding

#### 3. **Ollama (Local)** (Optional)
- **Model**: `llama2` (configurable to other local models)
- **Purpose**: Privacy-focused local AI processing
- **Configuration**: Runs on localhost:11434
- **API**: Local Ollama server
- **Use Case**: Offline processing, data privacy compliance

### LLM Selection Logic:
```python
Auto-detection Priority:
1. Anthropic Claude (if API key available)
2. OpenAI GPT (if API key available)
3. Ollama (if server running locally)
4. Fallback: Disable RAG features
```

## 🧮 Algorithms & Analysis Methods

### 1. **Sentiment Analysis Algorithms**

#### VADER Sentiment Analysis
- **Algorithm**: Valence Aware Dictionary and sEntiment Reasoner
- **Implementation**: `vaderSentiment.vaderSentiment.SentimentIntensityAnalyzer`
- **Output Metrics**:
  - `positive`: Positive sentiment score (0-1)
  - `neutral`: Neutral sentiment score (0-1)
  - `negative`: Negative sentiment score (0-1)
  - `compound`: Overall sentiment (-1 to +1)
- **Classification Logic**:
  ```python
  if compound >= 0.05: label = 'positive'
  elif compound <= -0.05: label = 'negative'
  else: label = 'neutral'
  ```

#### TextBlob Sentiment Analysis
- **Algorithm**: Rule-based sentiment analysis with machine learning
- **Implementation**: `textblob.TextBlob.sentiment`
- **Output Metrics**:
  - `polarity`: Sentiment polarity (-1 to +1)
  - `subjectivity`: Objectivity vs subjectivity (0 to +1)

### 2. **Vector Embedding Algorithm**
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions**: 384-dimensional embeddings
- **Purpose**: Convert text to numerical vectors for semantic similarity
- **Implementation**: HuggingFace Transformers via LangChain

### 3. **Vector Search Algorithm**
- **Algorithm**: Facebook AI Similarity Search (FAISS)
- **Index Type**: Flat L2 (exact search)
- **Similarity Metric**: Cosine similarity
- **Implementation**: `langchain_community.vectorstores.FAISS`

### 4. **Text Processing Algorithms**

#### Text Cleaning Pipeline:
1. **Lowercasing**: Convert to lowercase
2. **URL Removal**: Remove HTTP/HTTPS links
3. **Mention Removal**: Remove @username mentions
4. **Special Character Filtering**: Keep alphanumeric and spaces
5. **Whitespace Normalization**: Remove extra spaces

#### Hashtag Extraction:
- **Regex Pattern**: `#\w+`
- **Processing**: Extract, clean, and normalize hashtags
- **Storage**: JSON array format

### 5. **Retrieval-Augmented Generation (RAG) Algorithm**

#### RAG Pipeline:
1. **Query Processing**: User input preprocessing
2. **Vector Retrieval**: Similarity search in FAISS
3. **Context Assembly**: Combine relevant documents
4. **Prompt Engineering**: Structure context + query
5. **LLM Generation**: Generate response using context
6. **Response Processing**: Format and return result

## 📚 Libraries & Packages

### Core Dependencies:

#### **Data Processing & Analysis**
```python
pandas==2.3.1              # Data manipulation and analysis
numpy==2.3.2               # Numerical computing
```

#### **Sentiment Analysis**
```python
vaderSentiment==3.3.2       # VADER sentiment analyzer
textblob==0.17.1            # TextBlob NLP library
```

#### **Machine Learning & AI**
```python
langchain==0.3.27           # LLM application framework
langchain-community==0.3.27 # Community integrations
langchain-anthropic==0.2.7  # Anthropic Claude integration
langchain-openai==0.2.11    # OpenAI GPT integration
langchain-ollama==0.2.2     # Ollama local LLM integration
```

#### **Vector Database & Embeddings**
```python
faiss-cpu==1.9.1            # Facebook AI Similarity Search
sentence-transformers==3.4.1 # Sentence embeddings
transformers==4.47.1        # HuggingFace transformers
torch==2.6.1                # PyTorch for model inference
```

#### **Web Interface & Visualization**
```python
streamlit==1.48.1           # Web application framework
plotly==5.24.1              # Interactive plotting
matplotlib==3.8.4           # Static plotting
wordcloud==1.9.4            # Word cloud generation
```

#### **Utilities**
```python
python-dotenv==1.0.1        # Environment variable management
requests==2.32.4            # HTTP requests
```

### Package Usage Details:

#### **LangChain Ecosystem**
- **Purpose**: Orchestrates the entire RAG pipeline
- **Components Used**:
  - `ChatAnthropic`: Claude Sonnet integration
  - `ChatOpenAI`: GPT integration  
  - `Ollama`: Local LLM integration
  - `HuggingFaceEmbeddings`: Text embeddings
  - `FAISS`: Vector store
  - `PromptTemplate`: Query templating
  - `RetrievalQA`: RAG chain implementation

#### **Streamlit Components**
- **st.cache_data**: Caches data loading and processing
- **st.plotly_chart**: Interactive visualizations
- **st.chat_message**: Chat interface for RAG
- **st.sidebar**: Navigation menu
- **st.columns**: Layout management

## 🔗 RAG Implementation

### Vector Store Creation Process:

#### 1. **Document Preparation**
```python
# Create comprehensive document content for each tweet
doc_content = f"""
User: {row['user_name']}
Location: {row['user_location']}
Date: {row['date']}
Tweet: {row['text']}
Hashtags: {row['hashtags']}
Retweets: {row['retweets']}
Favorites: {row['favorites']}
Sentiment: {row['vader_sentiment']['label']}
"""
```

#### 2. **Text Chunking**
- **Strategy**: CharacterTextSplitter
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Purpose**: Optimize retrieval granularity

#### 3. **Vector Store Setup**
```python
# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS vector store
vector_store = FAISS.from_documents(split_docs, embeddings)
```

### RAG Query Processing:

#### 1. **Prompt Template**
```python
prompt_template = """
You are an expert data analyst specializing in social media sentiment analysis.
Use the following context from Omicron-related tweets to answer the user's question.

Context:
{context}

Question: {question}

Instructions:
- Provide specific, data-driven answers
- Include relevant user names, tweet content, and statistics
- For hashtag queries, list users and their tweets
- For sentiment queries, provide analysis with evidence
- Format responses with bullet points or lists
- Keep responses concise but informative

Answer:
"""
```

#### 2. **Retrieval Chain**
```python
retrieval_chain = RetrievalQA.from_chain_type(
    llm=self.llm,
    chain_type="stuff",
    retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)
```

### RAG Usage Locations:

#### **Primary Interface**: Streamlit Web App
- **File**: `web_app/streamlit_app.py`
- **Function**: `show_rag_chat(analyzer)`
- **Features**:
  - Chat interface with message history
  - Real-time query processing
  - Source document display
  - Error handling for API failures

#### **Core Implementation**: Analysis Engine
- **File**: `core/omicron_sentiment_rag.py`
- **Method**: `query_with_rag(query: str)`
- **Process**:
  1. Validate LLM availability
  2. Process user query
  3. Retrieve relevant documents
  4. Generate contextualized response
  5. Return formatted answer

### Example RAG Queries:
```python
# Hashtag analysis
"What are the main concerns about Omicron?"

# User analysis  
"Which users are most worried about vaccines?"

# Sentiment analysis
"Show me negative tweets about hospital capacity"

# Content search
"Find tweets mentioning CDC guidelines"
```

## ✨ Features

### **Dashboard Pages:**

1. **Overview**: Dataset statistics, sentiment distribution, timeline analysis, word clouds
2. **Interactive Query**: Natural language search interface
3. **Hashtag Analysis**: Trending hashtags, hashtag-specific tweet exploration
4. **User Analysis**: Most active users, user-specific analytics
5. **Sentiment Deep Dive**: Detailed sentiment metrics and filtering
6. **RAG Chat**: AI-powered conversational analysis

### **Analysis Capabilities:**

- **Real-time Sentiment Scoring**: VADER + TextBlob dual analysis
- **Semantic Search**: Vector-based tweet retrieval
- **Trend Analysis**: Temporal sentiment patterns
- **User Profiling**: Individual user behavior analysis
- **Hashtag Tracking**: Popular hashtag identification
- **Engagement Metrics**: Retweets, favorites, reach analysis

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/hameed1198/LLM_Omicron.git
cd LLM_Omicron

# Create virtual environment
python -m venv omicron
source omicron/bin/activate  # Linux/Mac
# or
omicron\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# 🆓 Configure FREE AI (RECOMMENDED)
# See FREE_AI_SETUP_GUIDE.md for detailed instructions
# Quick setup for Google Gemini (FREE):
# 1. Visit https://aistudio.google.com/app/apikey
# 2. Create free API key
# 3. Add to .env file:
# GOOGLE_API_KEY=your_free_google_key_here

# 💰 Or configure paid AI (optional)
# ANTHROPIC_API_KEY=your_anthropic_key_here
# OPENAI_API_KEY=your_openai_key_here
```

## 📈 Usage

### **Start the Application:**
```bash
cd omicron_analysis_project/web_app
streamlit run streamlit_app.py
```

### **Access the Dashboard:**
- Open browser to `http://localhost:8501`
- Navigate through different analysis pages
- Use RAG chat with your chosen AI provider (FREE options available!)

### **🆓 FREE AI Setup:**
- **Google Gemini**: Best free option - follow `FREE_AI_SETUP_GUIDE.md`
- **Ollama**: 100% free local AI - no API key needed
- **Together AI**: $25 free monthly credits

### **Command Line Analysis:**
```bash
cd omicron_analysis_project/core
python demo_analysis.py
```

## 📁 Project Structure

```
omicron_analysis_project/
├── core/                           # Core analysis engine
│   ├── omicron_sentiment_rag.py    # Main analysis class with RAG
│   └── demo_analysis.py            # Command-line demo
├── web_app/                        # Streamlit web interface
│   └── streamlit_app.py            # Main web application
├── data/                           # Dataset storage
│   └── omicron_2025.csv           # Twitter dataset (17,046 tweets)
├── .env                           # Environment variables (API keys)
├── requirements.txt               # Python dependencies
└── README.md                     # This documentation
```

### **Data Schema:**
```python
Columns: [
    'id', 'user_name', 'user_location', 'user_description', 
    'user_created', 'user_followers', 'user_friends', 
    'user_favourites', 'user_verified', 'date', 'text', 
    'hashtags', 'source', 'retweets', 'favorites', 
    'is_retweet', 'hashtags_parsed', 'clean_text', 
    'textblob_sentiment', 'vader_sentiment'
]
```

---

## 📊 Performance Metrics

- **Dataset Size**: 17,046 tweets
- **Vector Store**: 17,046 document chunks
- **Embedding Dimensions**: 384
- **Average Processing Time**: <2 seconds per query
- **Memory Usage**: ~500MB (including models)
- **Supported Concurrent Users**: 10-50 (depending on hardware)

## 🔮 Future Enhancements

- **Additional LLMs**: Gemini, Llama 3, Mistral integration
- **Advanced Analytics**: Topic modeling, entity recognition
- **Real-time Data**: Twitter API integration for live analysis
- **Export Features**: PDF reports, CSV exports
- **Advanced Visualizations**: Network graphs, geographic mapping
- **Multi-language Support**: Non-English tweet analysis

---

**Built with ❤️ by Hameed1198**  
**Powered by LangChain, Streamlit, and Anthropic Claude**
