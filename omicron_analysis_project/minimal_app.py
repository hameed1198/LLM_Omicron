import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from collections import Counter

# Page config
st.set_page_config(
    page_title="Omicron Sentiment Analysis",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None

def load_data():
    """Load the omicron dataset"""
    try:
        # Look for data file in multiple locations
        possible_paths = [
            os.path.join(os.path.dirname(__file__), 'data', 'omicron_2025.csv'),
            os.path.join(os.path.dirname(__file__), 'omicron_2025.csv'),
            'omicron_2025.csv',
            'data/omicron_2025.csv'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                st.session_state.df = df
                st.session_state.data_loaded = True
                return df
        
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def basic_sentiment_analysis(text):
    """Basic sentiment analysis without external libraries"""
    if pd.isna(text):
        return 'neutral'
    
    text = str(text).lower()
    
    # Simple positive/negative word lists
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'happy', 'positive', 'best', 'awesome', 'perfect', 'brilliant', 'outstanding', 'superb']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'sad', 'negative', 'worst', 'disgusting', 'pathetic', 'useless', 'stupid', 'annoying', 'frustrating']
    
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

def extract_mentions(text):
    """Extract mentions from text"""
    if pd.isna(text):
        return []
    return re.findall(r'@\w+', str(text))

def overview_page():
    """Overview page content"""
    st.title("🦠 Omicron Tweets Sentiment Analysis")
    st.markdown("### Analyzing COVID-19 Omicron variant discussions on Twitter")
    
    if st.session_state.data_loaded and st.session_state.df is not None:
        df = st.session_state.df
        
        st.success("✅ Data file found!")
        
        # Dataset overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Tweets", f"{len(df):,}")
        
        with col2:
            unique_users = df['user_name'].nunique() if 'user_name' in df.columns else "N/A"
            st.metric("Unique Users", unique_users)
        
        with col3:
            if 'created_at' in df.columns:
                date_range = f"{len(pd.to_datetime(df['created_at'], errors='coerce').dt.date.unique())} days"
            else:
                date_range = "N/A"
            st.metric("Date Range", date_range)
        
        with col4:
            st.metric("Columns", len(df.columns))
        
        # Show sample data
        st.subheader("📝 Sample Tweets")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Basic stats
        st.subheader("📊 Dataset Information")
        st.write(f"**Columns**: {', '.join(df.columns)}")
        
    else:
        st.warning("⚠️ Data file not found. Showing demo information.")
        
        # Demo metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Tweets", "17,046")
        with col2:
            st.metric("Unique Users", "8,523")
        with col3:
            st.metric("Date Range", "30 days")
        with col4:
            st.metric("Sentiment Methods", "3")

def interactive_query_page():
    """Interactive query page"""
    st.title("🔍 Interactive Query")
    st.markdown("Search and filter tweets based on your criteria")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first from the Overview page")
        return
    
    df = st.session_state.df
    
    # Search functionality
    search_term = st.text_input("🔍 Search tweets:", placeholder="Enter keywords to search...")
    
    if search_term:
        # Search in text columns
        text_columns = [col for col in df.columns if 'text' in col.lower() or 'tweet' in col.lower()]
        if text_columns:
            mask = df[text_columns[0]].str.contains(search_term, case=False, na=False)
            filtered_df = df[mask]
            
            st.write(f"**Found {len(filtered_df)} tweets containing '{search_term}'**")
            st.dataframe(filtered_df, use_container_width=True)
        else:
            st.warning("No text columns found for searching")
    else:
        st.info("Enter a search term above to filter tweets")

def hashtag_analysis_page():
    """Hashtag analysis page"""
    st.title("# Hashtag Analysis")
    st.markdown("Discover trending hashtags in omicron discussions")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first from the Overview page")
        return
    
    df = st.session_state.df
    
    # Find text column
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
                try:
                    import plotly.express as px
                    fig = px.bar(hashtag_df.head(10), x='Count', y='Hashtag', orientation='h',
                               title="Top 10 Hashtags")
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.bar_chart(hashtag_df.head(10).set_index('Hashtag'))
        else:
            st.info("No hashtags found in the dataset")
    else:
        st.warning("No text column found for hashtag analysis")

def user_analysis_page():
    """User analysis page"""
    st.title("👥 User Analysis")
    st.markdown("Analyze user activity and engagement patterns")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first from the Overview page")
        return
    
    df = st.session_state.df
    
    # User activity analysis
    if 'user_name' in df.columns:
        user_counts = df['user_name'].value_counts().head(20)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Most Active Users")
            user_df = pd.DataFrame({
                'User': user_counts.index,
                'Tweet Count': user_counts.values
            })
            st.dataframe(user_df, use_container_width=True)
        
        with col2:
            st.subheader("📊 User Activity Distribution")
            st.bar_chart(user_counts.head(10))
        
        # User engagement metrics
        engagement_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['like', 'retweet', 'reply', 'favorite'])]
        
        if engagement_cols:
            st.subheader("💫 User Engagement Metrics")
            for col in engagement_cols[:3]:  # Show top 3 engagement metrics
                if df[col].dtype in ['int64', 'float64']:
                    avg_engagement = df.groupby('user_name')[col].mean().sort_values(ascending=False).head(10)
                    st.write(f"**Top users by average {col}:**")
                    st.bar_chart(avg_engagement)
    else:
        st.warning("No user information found in the dataset")

def sentiment_deep_dive_page():
    """Sentiment analysis deep dive"""
    st.title("😊 Sentiment Deep Dive")
    st.markdown("Comprehensive sentiment analysis of omicron tweets")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first from the Overview page")
        return
    
    df = st.session_state.df
    
    # Find text column
    text_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['text', 'tweet', 'content'])]
    
    if text_columns:
        text_col = text_columns[0]
        
        # Perform basic sentiment analysis
        if 'sentiment_basic' not in df.columns:
            with st.spinner("Analyzing sentiment..."):
                df['sentiment_basic'] = df[text_col].apply(basic_sentiment_analysis)
        
        # Sentiment distribution
        sentiment_counts = df['sentiment_basic'].value_counts()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Positive Tweets", sentiment_counts.get('positive', 0), 
                     delta=f"{sentiment_counts.get('positive', 0)/len(df)*100:.1f}%")
        
        with col2:
            st.metric("Neutral Tweets", sentiment_counts.get('neutral', 0),
                     delta=f"{sentiment_counts.get('neutral', 0)/len(df)*100:.1f}%")
        
        with col3:
            st.metric("Negative Tweets", sentiment_counts.get('negative', 0),
                     delta=f"{sentiment_counts.get('negative', 0)/len(df)*100:.1f}%")
        
        # Sentiment visualization
        st.subheader("📊 Sentiment Distribution")
        try:
            import plotly.express as px
            fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                        title="Overall Sentiment Distribution")
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.bar_chart(sentiment_counts)
        
        # Sample tweets by sentiment
        st.subheader("📝 Sample Tweets by Sentiment")
        
        sentiment_filter = st.selectbox("Select sentiment:", ['positive', 'negative', 'neutral'])
        
        filtered_tweets = df[df['sentiment_basic'] == sentiment_filter][text_col].head(5)
        
        for i, tweet in enumerate(filtered_tweets, 1):
            st.write(f"**{i}.** {tweet}")
            st.write("---")
    
    else:
        st.warning("No text column found for sentiment analysis")

def rag_chat_page():
    """RAG Chat interface"""
    st.title("🤖 RAG Chat")
    st.markdown("Chat with your omicron dataset using AI")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first from the Overview page")
        return
    
    st.info("🚧 **RAG Chat Coming Soon!**")
    st.markdown("""
    This feature will allow you to:
    - Ask questions about the omicron dataset
    - Get AI-powered insights
    - Retrieve relevant tweets based on queries
    - Generate summaries and reports
    
    **Available AI Models:**
    - Google Gemini ✅
    - Claude Sonnet
    - OpenAI GPT
    - Hugging Face Models
    """)
    
    # Demo chat interface
    st.subheader("💬 Chat Interface (Demo)")
    
    user_question = st.text_input("Ask a question about the omicron tweets:")
    
    if user_question:
        st.markdown("**🤖 AI Response:**")
        st.info(f"Based on the analysis of 17,046 omicron tweets, here's what I found regarding: '{user_question}'. This is a demo response - full AI integration coming soon!")

def main():
    # Sidebar navigation
    st.sidebar.title("🦠 Navigation")
    
    # Load data button
    if st.sidebar.button("🔄 Load Data", type="primary"):
        with st.spinner("Loading data..."):
            load_data()
        if st.session_state.data_loaded:
            st.sidebar.success("✅ Data loaded successfully!")
        else:
            st.sidebar.error("❌ Failed to load data")
    
    # Navigation options
    pages = {
        "Overview": overview_page,
        "Interactive Query": interactive_query_page,
        "Hashtag Analysis": hashtag_analysis_page,
        "User Analysis": user_analysis_page,
        "Sentiment Deep Dive": sentiment_deep_dive_page,
        "RAG Chat": rag_chat_page
    }
    
    # Page selection
    selected_page = st.sidebar.selectbox("Choose a page:", list(pages.keys()))
    
    # Data status
    if st.session_state.data_loaded:
        st.sidebar.success(f"📊 Dataset: {len(st.session_state.df)} tweets loaded")
    else:
        st.sidebar.warning("📊 No data loaded")
    
    # Run selected page
    pages[selected_page]()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🔧 Built with:**")
    st.sidebar.markdown("- Streamlit")
    st.sidebar.markdown("- Pandas") 
    st.sidebar.markdown("- Python 3.13")

if __name__ == "__main__":
    main()
