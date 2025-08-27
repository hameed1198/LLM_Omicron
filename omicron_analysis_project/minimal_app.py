import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import sys
from collections import Counter
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'core'))
sys.path.append(str(project_root / 'analysis_scripts'))

# Try to import your existing analysis modules
try:
    from core.simple_sentiment_analyzer import SimpleSentimentAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False
    st.warning("⚠️ Analysis modules not found. Using basic functionality.")

try:
    from core.omicron_sentiment_rag import OmicronSentimentRAG
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

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
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None

def load_data():
    """Load the omicron dataset using your existing analyzer"""
    try:
        # Look for data file in multiple locations
        possible_paths = [
            'omicron_2025.csv',
            'data/omicron_2025.csv',
            os.path.join('data', 'omicron_2025.csv'),
            os.path.join(os.path.dirname(__file__), 'omicron_2025.csv'),
            os.path.join(os.path.dirname(__file__), 'data', 'omicron_2025.csv')
        ]
        
        csv_path = None
        for path in possible_paths:
            if os.path.exists(path):
                csv_path = path
                break
        
        if csv_path and ANALYZER_AVAILABLE:
            # Use your existing analyzer
            analyzer = SimpleSentimentAnalyzer(csv_path)
            st.session_state.analyzer = analyzer
            st.session_state.df = analyzer.df
            st.session_state.data_loaded = True
            return analyzer.df
        elif csv_path:
            # Fallback to basic pandas loading
            df = pd.read_csv(csv_path)
            st.session_state.df = df
            st.session_state.data_loaded = True
            return df
        
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def initialize_rag_system():
    """Initialize the RAG system if available"""
    if not RAG_AVAILABLE:
        return None
    
    try:
        # Look for API keys
        google_api_key = os.getenv('GOOGLE_API_KEY', 'AIzaSyC9WVZri_Gas_scMlkk-OeveNCkR5LMLCc')
        
        # Find CSV path
        csv_path = None
        possible_paths = [
            'omicron_2025.csv',
            'data/omicron_2025.csv',
            os.path.join('data', 'omicron_2025.csv')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                csv_path = path
                break
        
        if csv_path:
            rag_system = OmicronSentimentRAG(
                csv_path=csv_path,
                google_api_key=google_api_key,
                llm_provider="google"
            )
            st.session_state.rag_system = rag_system
            return rag_system
    except Exception as e:
        st.warning(f"RAG system initialization failed: {e}")
        return None

def overview_page():
    """Overview page with real data"""
    st.title("🦠 Omicron Tweets Sentiment Analysis")
    st.markdown("### Analyzing COVID-19 Omicron variant discussions on Twitter")
    
    if st.session_state.data_loaded and st.session_state.df is not None:
        df = st.session_state.df
        analyzer = st.session_state.analyzer
        
        st.success("✅ Data file found!")
        
        # Real dataset metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Tweets", f"{len(df):,}")
        
        with col2:
            unique_users = df['user_name'].nunique() if 'user_name' in df.columns else len(df['user_name'].unique()) if 'user_name' in df.columns else "N/A"
            st.metric("Unique Users", unique_users)
        
        with col3:
            if 'date' in df.columns:
                unique_dates = df['date'].nunique()
                st.metric("Date Range", f"{unique_dates} days")
            else:
                st.metric("Columns", len(df.columns))
        
        with col4:
            if analyzer and hasattr(analyzer, 'analyze_sentiment_distribution'):
                sentiment_dist = analyzer.analyze_sentiment_distribution()
                total_analyzed = sentiment_dist.get('total_tweets', len(df))
                st.metric("Analyzed Tweets", f"{total_analyzed:,}")
            else:
                st.metric("Data Points", f"{len(df):,}")
        
        # Real sentiment analysis if available
        if analyzer:
            st.subheader("😊 Sentiment Analysis Results")
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
                try:
                    import plotly.express as px
                    fig = px.pie(
                        values=list(sent_data.values()), 
                        names=list(sent_data.keys()),
                        title="Sentiment Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.bar_chart(pd.Series(sent_data))
        
        # Show sample data
        st.subheader("📝 Sample Tweets")
        display_columns = ['user_name', 'text', 'date', 'retweets', 'favorites']
        available_columns = [col for col in display_columns if col in df.columns]
        if available_columns:
            st.dataframe(df[available_columns].head(10), use_container_width=True)
        else:
            st.dataframe(df.head(10), use_container_width=True)
        
    else:
        st.warning("⚠️ Data not loaded. Click 'Load Data' in the sidebar.")

def interactive_query_page():
    """Interactive query page with real search functionality"""
    st.title("🔍 Interactive Query")
    st.markdown("Search and filter tweets based on your criteria")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first from the Overview page")
        return
    
    df = st.session_state.df
    
    # Search functionality
    search_term = st.text_input("🔍 Search tweets:", placeholder="Enter keywords to search...")
    
    # Advanced filters
    col1, col2 = st.columns(2)
    
    with col1:
        sentiment_filter = st.selectbox("Filter by sentiment:", ['All', 'positive', 'negative', 'neutral'])
    
    with col2:
        if 'user_name' in df.columns:
            users = ['All'] + sorted(df['user_name'].unique().tolist())
            user_filter = st.selectbox("Filter by user:", users[:100])  # Limit for performance
        else:
            user_filter = 'All'
    
    # Apply filters
    filtered_df = df.copy()
    
    if search_term:
        text_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['text', 'tweet', 'content'])]
        if text_columns:
            mask = filtered_df[text_columns[0]].str.contains(search_term, case=False, na=False)
            filtered_df = filtered_df[mask]
    
    if sentiment_filter != 'All' and 'sentiment_label' in df.columns:
        filtered_df = filtered_df[filtered_df['sentiment_label'] == sentiment_filter]
    
    if user_filter != 'All':
        filtered_df = filtered_df[filtered_df['user_name'] == user_filter]
    
    # Results
    st.write(f"**Found {len(filtered_df):,} tweets matching your criteria**")
    
    if len(filtered_df) > 0:
        # Show results with engagement metrics
        display_columns = ['user_name', 'text', 'date', 'retweets', 'favorites', 'sentiment_label']
        available_columns = [col for col in display_columns if col in filtered_df.columns]
        
        if available_columns:
            st.dataframe(filtered_df[available_columns].head(50), use_container_width=True)
        else:
            st.dataframe(filtered_df.head(50), use_container_width=True)
    else:
        st.info("No tweets found matching your criteria. Try different search terms or filters.")

def hashtag_analysis_page():
    """Real hashtag analysis using your existing analyzer"""
    st.title("🏷️ Hashtag Analysis")
    st.markdown("Discover trending hashtags in omicron discussions")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first from the Overview page")
        return
    
    analyzer = st.session_state.analyzer
    df = st.session_state.df
    
    if analyzer and hasattr(analyzer, 'get_trending_hashtags'):
        # Use your existing hashtag analysis
        st.subheader("📈 Trending Hashtags")
        
        # Get trending hashtags
        trending_hashtags = analyzer.get_trending_hashtags(20)
        
        if trending_hashtags:
            col1, col2 = st.columns(2)
            
            with col1:
                hashtag_df = pd.DataFrame(trending_hashtags)
                st.dataframe(hashtag_df, use_container_width=True)
            
            with col2:
                # Visualization
                try:
                    import plotly.express as px
                    fig = px.bar(
                        hashtag_df.head(10), 
                        x='count', 
                        y='hashtag', 
                        orientation='h',
                        title="Top 10 Hashtags"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.bar_chart(hashtag_df.head(10).set_index('hashtag')['count'])
            
            # Hashtag search
            st.subheader("🔍 Search by Hashtag")
            selected_hashtag = st.selectbox("Select a hashtag to explore:", [h['hashtag'] for h in trending_hashtags[:10]])
            
            if selected_hashtag and analyzer and hasattr(analyzer, 'query_tweets_by_hashtag'):
                hashtag_tweets = analyzer.query_tweets_by_hashtag(selected_hashtag)
                
                if hashtag_tweets:
                    st.write(f"**Found {len(hashtag_tweets)} tweets with #{selected_hashtag}**")
                    
                    # Convert to DataFrame for display
                    hashtag_df = pd.DataFrame(hashtag_tweets)
                    display_columns = ['user_name', 'text', 'retweets', 'favorites']
                    available_columns = [col for col in display_columns if col in hashtag_df.columns]
                    
                    if available_columns:
                        st.dataframe(hashtag_df[available_columns].head(20), use_container_width=True)
                    else:
                        st.dataframe(hashtag_df.head(20), use_container_width=True)
        else:
            st.info("No hashtags found in the dataset")
    else:
        st.warning("Hashtag analysis not available. Please ensure the analyzer is properly loaded.")

def user_analysis_page():
    """Real user analysis"""
    st.title("👥 User Analysis")
    st.markdown("Analyze user activity and engagement patterns")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first from the Overview page")
        return
    
    df = st.session_state.df
    
    if 'user_name' in df.columns:
        # Most active users
        user_counts = df['user_name'].value_counts().head(20)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Most Active Users")
            user_activity_df = pd.DataFrame({
                'User': user_counts.index,
                'Tweet Count': user_counts.values
            })
            st.dataframe(user_activity_df, use_container_width=True)
        
        with col2:
            st.subheader("📊 User Activity Distribution")
            st.bar_chart(user_counts.head(10))
        
        # User engagement analysis
        engagement_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['retweet', 'favorite', 'like'])]
        
        if engagement_cols:
            st.subheader("💫 User Engagement Analysis")
            
            # Calculate total engagement
            if 'retweets' in df.columns and 'favorites' in df.columns:
                df['total_engagement'] = df['retweets'] + df['favorites']
                
                user_engagement = df.groupby('user_name').agg({
                    'total_engagement': ['sum', 'mean'],
                    'retweets': 'sum',
                    'favorites': 'sum',
                    'text': 'count'
                }).round(2)
                
                user_engagement.columns = ['total_engagement', 'avg_engagement', 'total_retweets', 'total_favorites', 'tweet_count']
                top_engaged_users = user_engagement.sort_values('total_engagement', ascending=False).head(10)
                
                st.write("**Top 10 Users by Total Engagement:**")
                st.dataframe(top_engaged_users, use_container_width=True)
        
        # User selection for detailed analysis
        st.subheader("🔍 Individual User Analysis")
        selected_user = st.selectbox("Select a user for detailed analysis:", [''] + user_counts.head(20).index.tolist())
        
        if selected_user:
            user_tweets = df[df['user_name'] == selected_user]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Tweets", len(user_tweets))
            with col2:
                if 'retweets' in df.columns:
                    st.metric("Total Retweets", user_tweets['retweets'].sum())
            with col3:
                if 'favorites' in df.columns:
                    st.metric("Total Favorites", user_tweets['favorites'].sum())
            
            # Show user's tweets
            st.write(f"**Recent tweets from @{selected_user}:**")
            display_columns = ['text', 'date', 'retweets', 'favorites']
            available_columns = [col for col in display_columns if col in user_tweets.columns]
            
            if available_columns:
                st.dataframe(user_tweets[available_columns].head(10), use_container_width=True)
            else:
                st.dataframe(user_tweets.head(10), use_container_width=True)
    else:
        st.warning("No user information found in the dataset")

def sentiment_deep_dive_page():
    """Real sentiment analysis using your existing analyzer"""
    st.title("😊 Sentiment Deep Dive")
    st.markdown("Comprehensive sentiment analysis of omicron tweets")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first from the Overview page")
        return
    
    analyzer = st.session_state.analyzer
    df = st.session_state.df
    
    if analyzer:
        # Get real sentiment analysis
        sentiment_dist = analyzer.analyze_sentiment_distribution()
        
        if 'sentiment_distribution' in sentiment_dist:
            sent_data = sentiment_dist['sentiment_distribution']
            total_tweets = sentiment_dist.get('total_tweets', len(df))
            
            # Sentiment metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                positive_count = sent_data.get('positive', 0)
                positive_pct = (positive_count / total_tweets * 100) if total_tweets > 0 else 0
                st.metric("Positive Tweets", f"{positive_count:,}", f"{positive_pct:.1f}%")
            
            with col2:
                neutral_count = sent_data.get('neutral', 0)
                neutral_pct = (neutral_count / total_tweets * 100) if total_tweets > 0 else 0
                st.metric("Neutral Tweets", f"{neutral_count:,}", f"{neutral_pct:.1f}%")
            
            with col3:
                negative_count = sent_data.get('negative', 0)
                negative_pct = (negative_count / total_tweets * 100) if total_tweets > 0 else 0
                st.metric("Negative Tweets", f"{negative_count:,}", f"{negative_pct:.1f}%")
            
            # Sentiment visualization
            st.subheader("📊 Sentiment Distribution")
            try:
                import plotly.express as px
                fig = px.pie(values=list(sent_data.values()), names=list(sent_data.keys()),
                           title="Overall Sentiment Distribution")
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.bar_chart(pd.Series(sent_data))
            
            # Sample tweets by sentiment
            st.subheader("📝 Sample Tweets by Sentiment")
            
            sentiment_filter = st.selectbox("Select sentiment:", ['positive', 'negative', 'neutral'])
            
            if 'sentiment_label' in df.columns:
                filtered_tweets = df[df['sentiment_label'] == sentiment_filter]
                
                if len(filtered_tweets) > 0:
                    st.write(f"**Sample {sentiment_filter} tweets:**")
                    
                    # Show top engaging tweets for this sentiment
                    if 'total_engagement' in filtered_tweets.columns:
                        filtered_tweets = filtered_tweets.nlargest(10, 'total_engagement')
                    else:
                        filtered_tweets = filtered_tweets.head(10)
                    
                    for i, (_, tweet) in enumerate(filtered_tweets.iterrows(), 1):
                        with st.expander(f"Tweet {i} - @{tweet.get('user_name', 'Unknown')}"):
                            st.write(tweet.get('text', ''))
                            if 'retweets' in tweet and 'favorites' in tweet:
                                st.write(f"📊 {tweet['retweets']} retweets, {tweet['favorites']} favorites")
                else:
                    st.info(f"No {sentiment_filter} tweets found")
            else:
                st.warning("Sentiment labels not available in the dataset")
    else:
        st.warning("Sentiment analyzer not available. Please ensure data is properly loaded.")

def rag_chat_page():
    """RAG Chat interface with real functionality"""
    st.title("🤖 RAG Chat")
    st.markdown("Chat with your omicron dataset using AI")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first from the Overview page")
        return
    
    # Initialize RAG system if not already done
    if st.session_state.rag_system is None and RAG_AVAILABLE:
        with st.spinner("Initializing RAG system..."):
            st.session_state.rag_system = initialize_rag_system()
    
    if st.session_state.rag_system:
        st.success("✅ RAG system initialized and ready!")
        
        # Chat interface
        st.subheader("💬 Ask Questions About Your Data")
        
        # Predefined example questions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("What are the top hashtags?"):
                st.session_state.user_question = "What are the top trending hashtags in the omicron tweets?"
        
        with col2:
            if st.button("Show negative sentiment analysis"):
                st.session_state.user_question = "Analyze the negative sentiment tweets about omicron. What are people concerned about?"
        
        # User input
        user_question = st.text_input(
            "Ask a question about the omicron tweets:",
            value=st.session_state.get('user_question', ''),
            placeholder="e.g., Which users are most influential? What are people saying about vaccines?"
        )
        
        if user_question:
            with st.spinner("Analyzing your question..."):
                try:
                    # Use your RAG system
                    response = st.session_state.rag_system.query_with_rag(user_question)
                    
                    st.markdown("**🤖 AI Response:**")
                    st.markdown(response)
                    
                except Exception as e:
                    st.error(f"Error processing question: {e}")
                    
                    # Fallback to basic analysis
                    analyzer = st.session_state.analyzer
                    if analyzer:
                        fallback_response = analyzer.query_rag(user_question)
                        st.markdown("**📊 Basic Analysis:**")
                        st.markdown(fallback_response['answer'])
    
    elif RAG_AVAILABLE:
        st.warning("🔄 RAG system not initialized. Click the button below to set it up.")
        if st.button("Initialize RAG System"):
            with st.spinner("Setting up RAG system..."):
                st.session_state.rag_system = initialize_rag_system()
            st.rerun()
    
    else:
        st.info("🚧 **RAG Chat Not Available**")
        st.markdown("""
        The RAG (Retrieval Augmented Generation) system requires additional dependencies.
        
        **Available AI Models:**
        - Google Gemini ✅ (API key provided)
        - Claude Sonnet
        - OpenAI GPT
        - Hugging Face Models
        
        **Basic Analysis Available:**
        - Use other tabs for detailed analysis
        - Sentiment analysis
        - Hashtag trending
        - User analysis
        """)

def main():
    # Sidebar navigation
    st.sidebar.title("🦠 Navigation")
    
    # Load data button
    if st.sidebar.button("🔄 Load Data", type="primary"):
        with st.spinner("Loading data..."):
            result = load_data()
        if st.session_state.data_loaded:
            st.sidebar.success("✅ Data loaded successfully!")
            st.rerun()
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
        st.sidebar.success(f"📊 Dataset: {len(st.session_state.df):,} tweets loaded")
        if st.session_state.analyzer:
            st.sidebar.success("🔍 Analyzer: Ready")
        if st.session_state.rag_system:
            st.sidebar.success("🤖 RAG: Active")
    else:
        st.sidebar.warning("📊 No data loaded")
    
    # System status
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🔧 System Status:**")
    st.sidebar.markdown(f"- Analyzer: {'✅' if ANALYZER_AVAILABLE else '❌'}")
    st.sidebar.markdown(f"- RAG System: {'✅' if RAG_AVAILABLE else '❌'}")
    
    # Run selected page
    pages[selected_page]()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🔧 Built with:**")
    st.sidebar.markdown("- Streamlit")
    st.sidebar.markdown("- Your Analysis Modules") 
    st.sidebar.markdown("- Python 3.13")

if __name__ == "__main__":
    main()

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
