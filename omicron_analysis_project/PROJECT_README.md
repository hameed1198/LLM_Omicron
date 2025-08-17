# Omicron Sentiment Analysis Project

## 📁 Project Structure

This project has been organized into a clean, modular structure for better maintainability and understanding.

```
omicron_analysis_project/
├── core/                          # Core analysis modules
│   ├── omicron_sentiment_rag.py   # Main RAG-based sentiment analysis system
│   └── copilot_analysis.py        # GitHub Copilot enhanced analysis
├── data/                          # Data files
│   └── omicron_2025.csv          # Main dataset (17,046 omicron tweets)
├── analysis_scripts/             # Analysis and research scripts
│   ├── demo.py                   # Interactive demo script
│   ├── detailed_analysis.py      # Comprehensive data analysis
│   ├── influential_users_analysis.py  # Top influencers analysis
│   ├── omicron_hashtag_analysis.py   # Hashtag-focused analysis
│   └── simple_omicron_hashtag.py     # Simple hashtag user analysis
├── web_app/                      # Web interface
│   └── streamlit_app.py          # Interactive Streamlit dashboard
├── config/                       # Configuration files
│   ├── requirements.txt          # Core dependencies
│   ├── requirements-minimal.txt  # Minimal dependencies
│   ├── requirements-dev.txt      # Development dependencies
│   ├── .env.example             # Environment configuration template
│   └── setup.py                 # Package setup configuration
└── documentation/               # Project documentation
    ├── README.md                # Main project documentation
    ├── COPILOT_GUIDE.md        # GitHub Copilot integration guide
    ├── COPILOT_INTEGRATION.md  # Detailed Copilot usage
    └── INSTALL.md              # Installation instructions
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
cd omicron_analysis_project
pip install -r config/requirements.txt
```

### 2. Run Basic Analysis
```bash
python analysis_scripts/demo.py
```

### 3. Launch Web Dashboard
```bash
streamlit run web_app/streamlit_app.py
```

### 4. Run Specific Analysis
```bash
# Find top influencers
python analysis_scripts/influential_users_analysis.py

# Analyze hashtag usage
python analysis_scripts/simple_omicron_hashtag.py

# Comprehensive analysis
python analysis_scripts/detailed_analysis.py
```

## 📊 Key Features

### Core Analysis (`core/`)
- **RAG-based Sentiment Analysis**: LangChain + FAISS vector store
- **Multi-LLM Support**: Anthropic Claude, OpenAI, GitHub Copilot
- **Advanced Sentiment Analysis**: VADER + TextBlob integration
- **Hashtag Query System**: Find users by hashtags efficiently

### Analysis Scripts (`analysis_scripts/`)
- **Influential Users**: Identify top 20 most influential users
- **Hashtag Analysis**: Deep dive into #omicron hashtag usage  
- **Detailed Analytics**: Comprehensive dataset exploration
- **Interactive Demo**: User-friendly analysis interface

### Web Interface (`web_app/`)
- **Multi-page Dashboard**: Overview, hashtag analysis, user analysis
- **Interactive Visualizations**: Charts and graphs
- **Real-time RAG Chat**: Ask questions about the dataset
- **Export Capabilities**: Download analysis results

## 📈 Dataset Overview

- **Total Tweets**: 17,046
- **Date Range**: February 2022 omicron discussions
- **Tweets with #omicron**: 6,965 (40.9% of dataset)
- **Top Influencer**: @Eric Feigl-Ding (12,441 influence score)
- **Most Active #omicron User**: @bron druider (156 tweets)

## 🔧 Configuration

### Environment Variables (`.env.example`)
```bash
# API Keys (optional - system works without them)
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here

# Data Configuration
CSV_FILE_PATH=data/omicron_2025.csv
VECTOR_STORE_PATH=data/vector_store
```

### Dependencies
- **Core**: pandas, numpy, scikit-learn, streamlit
- **NLP**: vaderSentiment, textblob, sentence-transformers
- **RAG**: langchain, faiss-cpu, openai, anthropic
- **Visualization**: matplotlib, seaborn, plotly

## 🎯 Analysis Results Summary

### Top Insights
1. **Most Influential User**: Eric Feigl-Ding (27,599 total engagement)
2. **Most #omicron Tweets**: @bron druider (156 tweets, 100% omicron-focused)
3. **Sentiment Distribution**: 90% of top influencers use neutral sentiment
4. **Engagement Pattern**: Positive tweets get 13.5x higher engagement than negative

### Geographic Distribution
- **Top Locations**: New York City, USA, India, Australia
- **Health Organizations**: Major presence from official health accounts
- **International Reach**: Global discussion spanning multiple continents

## 🔍 Usage Examples

### Find Users by Hashtag
```python
from core.copilot_analysis import CopilotEnhancedOmicronAnalysis

analyzer = CopilotEnhancedOmicronAnalysis("data/omicron_2025.csv")
omicron_tweets = analyzer.find_hashtag_tweets("omicron")
print(f"Found {len(omicron_tweets)} tweets with #omicron")
```

### RAG Query System
```python
from core.omicron_sentiment_rag import OmicronSentimentRAG

analyzer = OmicronSentimentRAG("data/omicron_2025.csv")
response = analyzer.query_with_rag("What are the main concerns about Omicron?")
print(response)
```

### Web Dashboard
```bash
streamlit run web_app/streamlit_app.py
# Navigate to: http://localhost:8501
```

## 📚 Documentation

- **Installation Guide**: `documentation/INSTALL.md`
- **Copilot Integration**: `documentation/COPILOT_GUIDE.md`
- **API Reference**: `documentation/COPILOT_INTEGRATION.md`

## 🤖 GitHub Copilot Integration

This project is optimized for GitHub Copilot Chat. Try these prompts:

```
@workspace Analyze sentiment patterns in the omicron dataset
@workspace Find users who tweeted with hashtag "omicron"
@workspace What are the main concerns about Omicron in the tweets?
@workspace Create a visualization of sentiment over time
```

## 🔄 Project Evolution

This organized structure supports:
- **Scalability**: Easy to add new analysis modules
- **Maintainability**: Clear separation of concerns
- **Collaboration**: Well-documented and modular code
- **Deployment**: Ready for production with proper configuration

## 📞 Support

For questions or issues:
1. Check the documentation in `documentation/`
2. Review the demo scripts in `analysis_scripts/`
3. Use GitHub Copilot Chat with the prompts above
4. Examine the web dashboard for interactive exploration

---

**Note**: This project can run with or without API keys. Core functionality works offline using local sentiment analysis models.
