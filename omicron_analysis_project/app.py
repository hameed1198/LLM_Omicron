"""
Omicron Sentiment Analysis - Streamlit Application
A comprehensive sentiment analysis tool for COVID-19 Omicron variant tweets
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import re
from collections import Counter
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="Omicron Sentiment Analysis",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / 'core'))

# Import dependencies with error handling
DEPENDENCIES = {
    'plotly': False,
    'wordcloud': False,
    'textblob': False,
    'vaderSentiment': False,
    'scikit-learn': False
}

try:
    import plotly.express as px
    import plotly.graph_objects as go
    DEPENDENCIES['plotly'] = True
except ImportError:
    pass

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    DEPENDENCIES['wordcloud'] = True
except ImportError:
    pass

try:
    from textblob import TextBlob
    DEPENDENCIES['textblob'] = True
except ImportError:
    pass

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    DEPENDENCIES['vaderSentiment'] = True
except ImportError:
    pass

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    DEPENDENCIES['scikit-learn'] = True
except ImportError:
    pass

# Import custom modules
MODULES = {
    'simple_analyzer': False,
    'rag_system': False
}

try:
    from core.simple_sentiment_analyzer import SimpleSentimentAnalyzer
    MODULES['simple_analyzer'] = True
except ImportError:
    try:
        from simple_sentiment_analyzer import SimpleSentimentAnalyzer
        MODULES['simple_analyzer'] = True
    except ImportError:
        pass

try:
    from core.omicron_sentiment_rag import OmicronSentimentRAG
    MODULES['rag_system'] = True
except ImportError:
    try:
        from omicron_sentiment_rag import OmicronSentimentRAG
        MODULES['rag_system'] = True
    except ImportError:
        pass

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None

def find_data_file():
    """Find the omicron dataset file"""
    possible_paths = [
        'data/omicron_2025.csv',
        'omicron_2025.csv',
        'web_app/omicron_2025.csv',
        current_dir / 'data' / 'omicron_2025.csv',
        current_dir / 'omicron_2025.csv'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return str(path)
    return None

def load_data():
    """Load the omicron dataset"""
    try:
        csv_path = find_data_file()
        if not csv_path:
            st.error("❌ Data file 'omicron_2025.csv' not found")
            return None
        
        # Try to use custom analyzer first
        if MODULES['simple_analyzer']:
            analyzer = SimpleSentimentAnalyzer(csv_path)
            st.session_state.analyzer = analyzer
            st.session_state.df = analyzer.df
            st.session_state.data_loaded = True
            return analyzer.df
        else:
            # Fallback to pandas
            df = pd.read_csv(csv_path)
            st.session_state.df = df
            st.session_state.data_loaded = True
            return df
            
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

def basic_sentiment_analysis(text):
    """Basic sentiment analysis fallback"""
    if pd.isna(text):
        return 'neutral'
    
    text = str(text).lower()
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'happy', 'positive', 'best', 'awesome']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'sad', 'negative', 'worst', 'disgusting', 'pathetic']
    
    positive_count = sum(1 for word in positive_words if word in text)
    negative_count = sum(1 for word in negative_words if word in text)
    
    if positive_count > negative_count:
        return 'positive'
    elif negative_count > positive_count:
        return 'negative'
    else:
        return 'neutral'

def extract_hashtags(text):
    """Extract hashtags from text"""
    if pd.isna(text):
        return []
    return re.findall(r'#\w+', str(text))

def overview_page():
    """Main overview page"""
    st.title("🦠 Omicron Sentiment Analysis")
    st.markdown("### Analyzing COVID-19 Omicron variant discussions on Twitter")
    
    if st.session_state.data_loaded and st.session_state.df is not None:
        df = st.session_state.df
        analyzer = st.session_state.analyzer
        
        st.success("✅ Data loaded successfully!")
        
        # Dataset metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Tweets", f"{len(df):,}")
        
        with col2:
            unique_users = df['user_name'].nunique() if 'user_name' in df.columns else "N/A"
            st.metric("Unique Users", unique_users)
        
        with col3:
            if 'date' in df.columns:
                unique_dates = pd.to_datetime(df['date'], errors='coerce').dt.date.nunique()
                st.metric("Date Range", f"{unique_dates} days")
            else:
                st.metric("Columns", len(df.columns))
        
        with col4:
            st.metric("Data Points", f"{len(df):,}")
        
        # Sentiment analysis
        if analyzer and hasattr(analyzer, 'analyze_sentiment_distribution'):
            st.subheader("😊 Sentiment Analysis")
            sentiment_dist = analyzer.analyze_sentiment_distribution()
            
            if 'sentiment_distribution' in sentiment_dist:
                sent_data = sentiment_dist['sentiment_distribution']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    positive_count = sent_data.get('positive', 0)
                    positive_pct = (positive_count / len(df) * 100) if len(df) > 0 else 0
                    st.metric("Positive", positive_count, f"{positive_pct:.1f}%")
                
                with col2:
                    neutral_count = sent_data.get('neutral', 0)
                    neutral_pct = (neutral_count / len(df) * 100) if len(df) > 0 else 0
                    st.metric("Neutral", neutral_count, f"{neutral_pct:.1f}%")
                
                with col3:
                    negative_count = sent_data.get('negative', 0)
                    negative_pct = (negative_count / len(df) * 100) if len(df) > 0 else 0
                    st.metric("Negative", negative_count, f"{negative_pct:.1f}%")
                
                # Sentiment chart
                if DEPENDENCIES['plotly']:
                    fig = px.pie(
                        values=list(sent_data.values()), 
                        names=list(sent_data.keys()),
                        title="Sentiment Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.bar_chart(pd.Series(sent_data))
        
        # Sample data
        st.subheader("📝 Sample Data")
        display_columns = ['user_name', 'text', 'date', 'retweets', 'favorites']
        available_columns = [col for col in display_columns if col in df.columns]
        if available_columns:
            st.dataframe(df[available_columns].head(10), use_container_width=True)
        else:
            st.dataframe(df.head(10), use_container_width=True)
        
    else:
        st.warning("⚠️ No data loaded. Click 'Load Data' in the sidebar.")
        
        # Show demo information
        st.info("📊 **Demo Dataset Information**")
        st.markdown("""
        - **Total Tweets**: 17,046
        - **Unique Users**: 8,523  
        - **Date Range**: 30 days
        - **Sentiment Methods**: VADER + TextBlob
        """)

def interactive_query_page():
    """Interactive query and search"""
    st.title("🔍 Interactive Query")
    st.markdown("Search and filter tweets based on your criteria")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first from the Overview page")
        return
    
    df = st.session_state.df
    
    # Search functionality
    search_term = st.text_input("🔍 Search tweets:", placeholder="Enter keywords to search...")
    
    if search_term:
        text_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['text', 'tweet', 'content'])]
        if text_columns:
            mask = df[text_columns[0]].str.contains(search_term, case=False, na=False)
            filtered_df = df[mask]
            
            st.write(f"**Found {len(filtered_df):,} tweets containing '{search_term}'**")
            
            if len(filtered_df) > 0:
                display_columns = ['user_name', 'text', 'date', 'retweets', 'favorites']
                available_columns = [col for col in display_columns if col in filtered_df.columns]
                
                if available_columns:
                    st.dataframe(filtered_df[available_columns].head(50), use_container_width=True)
                else:
                    st.dataframe(filtered_df.head(50), use_container_width=True)
            else:
                st.info("No results found for your search term.")
        else:
            st.warning("No text columns found for searching")

def hashtag_analysis_page():
    """Hashtag analysis and trending topics"""
    st.title("🏷️ Hashtag Analysis")
    st.markdown("Discover trending hashtags in omicron discussions")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first from the Overview page")
        return
    
    df = st.session_state.df
    analyzer = st.session_state.analyzer
    
    if analyzer and hasattr(analyzer, 'get_trending_hashtags'):
        # Use advanced analyzer
        st.subheader("📈 Trending Hashtags")
        trending_hashtags = analyzer.get_trending_hashtags(20)
        
        if trending_hashtags:
            hashtag_df = pd.DataFrame(trending_hashtags)
            
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(hashtag_df, use_container_width=True)
            
            with col2:
                if DEPENDENCIES['plotly']:
                    fig = px.bar(
                        hashtag_df.head(10), 
                        x='count', 
                        y='hashtag', 
                        orientation='h',
                        title="Top 10 Hashtags"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.bar_chart(hashtag_df.head(10).set_index('hashtag')['count'])
        else:
            st.info("No hashtags found in the dataset")
    else:
        # Basic hashtag analysis
        text_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['text', 'tweet', 'content'])]
        
        if text_columns:
            text_col = text_columns[0]
            
            # Extract hashtags
            all_hashtags = []
            for text in df[text_col].dropna():
                hashtags = extract_hashtags(str(text))
                all_hashtags.extend(hashtags)
            
            if all_hashtags:
                hashtag_counts = Counter(all_hashtags)
                top_hashtags = hashtag_counts.most_common(20)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📈 Top Hashtags")
                    hashtag_df = pd.DataFrame(top_hashtags, columns=['Hashtag', 'Count'])
                    st.dataframe(hashtag_df, use_container_width=True)
                
                with col2:
                    st.subheader("📊 Hashtag Distribution")
                    if DEPENDENCIES['plotly']:
                        fig = px.bar(hashtag_df.head(10), x='Count', y='Hashtag', orientation='h')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.bar_chart(hashtag_df.head(10).set_index('Hashtag'))
            else:
                st.info("No hashtags found in the dataset")

def system_status_page():
    """System status and diagnostics"""
    st.title("🔧 System Status")
    st.markdown("System diagnostics and dependency status")
    
    # Dependencies status
    st.subheader("📦 Dependencies")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Visualization Libraries:**")
        st.markdown(f"- Plotly: {'✅' if DEPENDENCIES['plotly'] else '❌'}")
        st.markdown(f"- WordCloud: {'✅' if DEPENDENCIES['wordcloud'] else '❌'}")
        
    with col2:
        st.markdown("**Analysis Libraries:**")
        st.markdown(f"- TextBlob: {'✅' if DEPENDENCIES['textblob'] else '❌'}")
        st.markdown(f"- VADER Sentiment: {'✅' if DEPENDENCIES['vaderSentiment'] else '❌'}")
        st.markdown(f"- Scikit-learn: {'✅' if DEPENDENCIES['scikit-learn'] else '❌'}")
    
    # Custom modules status
    st.subheader("🔧 Custom Modules")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"- Simple Analyzer: {'✅' if MODULES['simple_analyzer'] else '❌'}")
        
    with col2:
        st.markdown(f"- RAG System: {'✅' if MODULES['rag_system'] else '❌'}")
    
    # Data status
    st.subheader("📊 Data Status")
    if st.session_state.data_loaded:
        st.success("✅ Dataset loaded successfully")
        st.markdown(f"- Rows: {len(st.session_state.df):,}")
        st.markdown(f"- Columns: {len(st.session_state.df.columns)}")
        st.markdown(f"- Analyzer: {'✅ Active' if st.session_state.analyzer else '❌ Not available'}")
    else:
        st.warning("❌ No dataset loaded")
    
    # File paths
    st.subheader("📁 File Paths")
    data_file = find_data_file()
    if data_file:
        st.success(f"✅ Data file found: {data_file}")
    else:
        st.error("❌ Data file not found")
    
    st.markdown(f"**Current directory:** {current_dir}")
    st.markdown(f"**Python path entries:** {len(sys.path)}")

def main():
    """Main application"""
    # Sidebar navigation
    st.sidebar.title("🦠 Navigation")
    
    # Load data button
    if st.sidebar.button("🔄 Load Data", type="primary"):
        with st.spinner("Loading data..."):
            result = load_data()
        if st.session_state.data_loaded:
            st.sidebar.success("✅ Data loaded!")
            st.rerun()
        else:
            st.sidebar.error("❌ Failed to load data")
    
    # Navigation pages
    pages = {
        "Overview": overview_page,
        "Interactive Query": interactive_query_page, 
        "Hashtag Analysis": hashtag_analysis_page,
        "System Status": system_status_page
    }
    
    selected_page = st.sidebar.selectbox("Choose a page:", list(pages.keys()))
    
    # Data status in sidebar
    if st.session_state.data_loaded:
        st.sidebar.success(f"📊 {len(st.session_state.df):,} tweets loaded")
    else:
        st.sidebar.warning("📊 No data loaded")
    
    # System info
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🔧 System Info:**")
    dependencies_ok = sum(DEPENDENCIES.values())
    modules_ok = sum(MODULES.values())
    st.sidebar.markdown(f"- Dependencies: {dependencies_ok}/5")
    st.sidebar.markdown(f"- Modules: {modules_ok}/2")
    
    # Run selected page
    pages[selected_page]()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Built with Streamlit & Python**")

if __name__ == "__main__":
    main()
