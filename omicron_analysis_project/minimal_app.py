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

# Check for additional LLM providers
try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI  
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_AVAILABLE = True  
except ImportError:
    GOOGLE_AVAILABLE = False

try:
    from langchain_community.llms import Ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from langchain_community.llms import Cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

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
    """Initialize the RAG system with default Google Gemini model"""
    return initialize_rag_system_with_model("google", "AIzaSyC9WVZri_Gas_scMlkk-OeveNCkR5LMLCc")

def overview_page():
    """Overview page with real data including timeline and word cloud"""
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
        
        # Timeline Analysis
        st.subheader("📈 Timeline Analysis")
        if 'date' in df.columns:
            try:
                # Convert date column to datetime
                df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
                
                # Create timeline of tweets
                timeline_data = df.groupby(df['date_parsed'].dt.date).size().reset_index()
                timeline_data.columns = ['Date', 'Tweet Count']
                
                if len(timeline_data) > 1:
                    import plotly.express as px
                    fig = px.line(timeline_data, x='Date', y='Tweet Count', 
                                title="Daily Tweet Volume")
                    fig.update_traces(line_color='#1f77b4', line_width=3)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Sentiment over time
                    if 'sentiment_label' in df.columns:
                        st.subheader("📊 Sentiment Over Time")
                        sentiment_timeline = df.groupby([df['date_parsed'].dt.date, 'sentiment_label']).size().unstack(fill_value=0)
                        
                        if not sentiment_timeline.empty:
                            fig = px.area(sentiment_timeline, 
                                        title="Sentiment Trends Over Time",
                                        color_discrete_map={
                                            'positive': '#2ecc71',
                                            'neutral': '#95a5a6', 
                                            'negative': '#e74c3c'
                                        })
                            st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Could not create timeline visualization: {e}")
                # Fallback - show basic timeline
                if 'date' in df.columns:
                    date_counts = df['date'].value_counts().sort_index()
                    st.line_chart(date_counts)
        else:
            st.info("Date information not available for timeline analysis")
        
        # Word Cloud
        st.subheader("☁️ Word Cloud")
        try:
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt
            
            # Get text data
            if 'text' in df.columns:
                text_data = ' '.join(df['text'].fillna('').astype(str))
                
                # Clean text for word cloud
                import re
                # Remove URLs, mentions, hashtags for cleaner word cloud
                clean_text = re.sub(r'http\S+|www\S+|https\S+', '', text_data, flags=re.MULTILINE)
                clean_text = re.sub(r'@\w+|#\w+', '', clean_text)
                clean_text = re.sub(r'[^a-zA-Z\s]', '', clean_text)
                
                if clean_text.strip():
                    # Create word cloud
                    wordcloud = WordCloud(
                        width=800, 
                        height=400, 
                        background_color='white',
                        colormap='viridis',
                        max_words=100,
                        relative_scaling=0.5,
                        stopwords=['omicron', 'covid', 'coronavirus', 'pandemic', 'virus']
                    ).generate(clean_text)
                    
                    # Display word cloud
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                else:
                    st.info("Not enough text data to generate word cloud")
            else:
                st.warning("Text data not available for word cloud generation")
                
        except ImportError:
            st.warning("Word cloud library not available. Install wordcloud package for visualization.")
        except Exception as e:
            st.warning(f"Could not generate word cloud: {e}")
        
        # Most Engaging Tweets
        st.subheader("🔥 Most Viral Tweets")
        if 'retweets' in df.columns and 'favorites' in df.columns:
            df['total_engagement'] = df['retweets'] + df['favorites']
            top_tweets = df.nlargest(5, 'total_engagement')
            
            for i, (_, tweet) in enumerate(top_tweets.iterrows(), 1):
                with st.expander(f"#{i} - {int(tweet['total_engagement']):,} total engagement"):
                    st.write(f"**@{tweet.get('user_name', 'Unknown')}**")
                    st.write(tweet.get('text', ''))
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Retweets", int(tweet.get('retweets', 0)))
                    with col2:
                        st.metric("Favorites", int(tweet.get('favorites', 0)))
                    with col3:
                        if 'sentiment_label' in tweet:
                            sentiment_emoji = {'positive': '😊', 'negative': '😟', 'neutral': '😐'}
                            emoji = sentiment_emoji.get(tweet['sentiment_label'], '😐')
                            st.metric("Sentiment", f"{emoji} {tweet['sentiment_label']}")
        
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
    search_term = st.text_input("🔍 Search tweets:", placeholder="Enter keywords to search...", key="interactive_search_input")
    
    # Advanced filters
    col1, col2 = st.columns(2)
    
    with col1:
        sentiment_filter = st.selectbox("Filter by sentiment:", ['All', 'positive', 'negative', 'neutral'], key="sentiment_filter_select")
    
    with col2:
        if 'user_name' in df.columns:
            users = ['All'] + sorted(df['user_name'].unique().tolist())
            user_filter = st.selectbox("Filter by user:", users[:100], key="user_filter_select")  # Limit for performance
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
            selected_hashtag = st.selectbox("Select a hashtag to explore:", [h['hashtag'] for h in trending_hashtags[:10]], key="hashtag_explore_select")
            
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
        selected_user = st.selectbox("Select a user for detailed analysis:", [''] + user_counts.head(20).index.tolist(), key="user_analysis_select")
        
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
            
            sentiment_filter = st.selectbox("Select sentiment:", ['positive', 'negative', 'neutral'], key="sentiment_sample_select")
            
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
    """RAG Chat interface with comprehensive model selection"""
    st.title("🤖 RAG Chat - AI-Powered Tweet Analysis")
    st.markdown("Chat with your omicron dataset using state-of-the-art AI models")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first from the Overview page")
        return
    
    # Model selection and configuration
    st.subheader("🔧 AI Model Configuration")
    
    # Available models with status
    available_models = {
        "Google Gemini": {
            "id": "google",
            "status": "✅ FREE - Generous limits", 
            "description": "Google's latest multimodal AI - Best for comprehensive analysis",
            "api_key": "AIzaSyC9WVZri_Gas_scMlkk-OeveNCkR5LMLCc",
            "available": RAG_AVAILABLE and GOOGLE_AVAILABLE
        },
        "Claude Sonnet": {
            "id": "claude", 
            "status": "💳 Paid - High quality",
            "description": "Anthropic's reasoning model - Excellent for nuanced analysis",
            "api_key": None,
            "available": RAG_AVAILABLE and ANTHROPIC_AVAILABLE
        },
        "OpenAI GPT": {
            "id": "openai",
            "status": "💳 Paid - Popular choice", 
            "description": "OpenAI's ChatGPT - Versatile and well-rounded",
            "api_key": None,
            "available": RAG_AVAILABLE and OPENAI_AVAILABLE
        },
        "Ollama Local": {
            "id": "ollama",
            "status": "🔒 Local - Privacy focused",
            "description": "Run models locally - No internet required",
            "api_key": None,
            "available": RAG_AVAILABLE and OLLAMA_AVAILABLE
        },
        "Cohere": {
            "id": "cohere",
            "status": "🆓 FREE tier available",
            "description": "Cohere's language model - Good for text analysis",
            "api_key": None,
            "available": RAG_AVAILABLE and COHERE_AVAILABLE
        }
    }
    
    # Display model options
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_model = st.selectbox(
            "Choose AI Model:",
            options=list(available_models.keys()),
            format_func=lambda x: f"{x} - {available_models[x]['status']}",
            key="rag_model_select"
        )
    
    with col2:
        if st.button("🔄 Initialize Model", type="primary", key="init_selected_model_btn"):
            model_config = available_models[selected_model]
            if model_config['available']:
                with st.spinner(f"Initializing {selected_model}..."):
                    st.session_state.rag_system = initialize_rag_system_with_model(
                        model_config['id'], 
                        model_config['api_key']
                    )
                if st.session_state.rag_system:
                    st.success(f"✅ {selected_model} initialized successfully!")
                    st.rerun()
                else:
                    st.error(f"❌ Failed to initialize {selected_model}")
            else:
                st.error(f"❌ {selected_model} not available")
    
    # Model details
    with st.expander(f"📋 About {selected_model}", expanded=False):
        model_info = available_models[selected_model]
        st.markdown(f"**Description:** {model_info['description']}")
        st.markdown(f"**Status:** {model_info['status']}")
        st.markdown(f"**Available:** {'✅ Yes' if model_info['available'] else '❌ No'}")
        
        if selected_model == "Google Gemini":
            st.markdown("**Features:**")
            st.markdown("- 🆓 Free with generous usage limits")
            st.markdown("- 🚀 Fast response times")
            st.markdown("- 🎯 Excellent for sentiment analysis")
            st.markdown("- 📊 Good at data interpretation")
            
        elif selected_model == "Claude Sonnet":
            st.markdown("**Features:**")
            st.markdown("- 🧠 Superior reasoning capabilities")
            st.markdown("- 📝 Excellent for complex analysis")
            st.markdown("- 🎯 Nuanced understanding")
            st.markdown("- 💡 Creative insights")
            
        elif selected_model == "OpenAI GPT":
            st.markdown("**Features:**")
            st.markdown("- 🌟 Most popular AI model")
            st.markdown("- 🔄 Versatile and reliable")
            st.markdown("- 🎯 Good general performance")
            st.markdown("- 📚 Extensive training data")
    
    # RAG System Status
    st.subheader("🔍 RAG System Status")
    
    if st.session_state.rag_system:
        st.success(f"✅ RAG system active with {selected_model}")
        
        # System capabilities
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Vector Store", "✅ Ready")
        with col2:
            st.metric("Embeddings", "✅ HuggingFace")
        with col3:
            st.metric("Model", f"✅ {selected_model}")
        
        # Chat Interface
        st.subheader("💬 Interactive Chat")
        
        # Quick action buttons
        st.markdown("**🚀 Quick Questions:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📈 Trending Topics", key="trending_topics_btn"):
                st.session_state.user_question = "What are the main topics and trending hashtags in the omicron tweets? Provide insights on what people are discussing most."
        
        with col2:
            if st.button("😷 Sentiment Analysis", key="sentiment_analysis_btn"):
                st.session_state.user_question = "Analyze the overall sentiment towards omicron. What are people's main concerns and positive aspects mentioned?"
        
        with col3:
            if st.button("👥 User Insights", key="user_insights_btn"):
                st.session_state.user_question = "Who are the most influential users discussing omicron? What patterns do you see in user behavior and engagement?"
        
        # Advanced questions
        with st.expander("🔬 Advanced Analysis Questions", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🌍 Geographic Patterns", key="geo_patterns_btn"):
                    st.session_state.user_question = "Are there any geographic or regional patterns in how omicron is being discussed?"
                if st.button("📅 Timeline Analysis", key="timeline_analysis_btn"):
                    st.session_state.user_question = "How has the conversation about omicron evolved over time? What changes do you notice?"
            
            with col2:
                if st.button("🔗 Network Analysis", key="network_analysis_btn"):
                    st.session_state.user_question = "Analyze the retweet and mention patterns. Who are the key influencers and how is information spreading?"
                if st.button("📊 Engagement Metrics", key="engagement_metrics_btn"):
                    st.session_state.user_question = "Which types of omicron-related content get the most engagement? What drives virality?"
        
        # Custom question input
        user_question = st.text_input(
            "💭 Ask your own question:",
            value=st.session_state.get('user_question', ''),
            placeholder="e.g., What are the main misconceptions about omicron in the tweets?",
            key="rag_custom_question_input"
        )
        
        # Process question
        if user_question:
            st.markdown("---")
            st.markdown(f"**❓ Your Question:** {user_question}")
            
            with st.spinner(f"🤖 {selected_model} is analyzing your data..."):
                try:
                    # Use the RAG system
                    response = st.session_state.rag_system.query_with_rag(user_question)
                    
                    st.markdown(f"**🤖 {selected_model} Response:**")
                    st.markdown(response)
                    
                    # Add follow-up suggestions
                    st.markdown("**🔄 Follow-up suggestions:**")
                    suggestions = [
                        "Can you provide more specific examples?",
                        "What are the implications of these findings?",
                        "How does this compare to other topics in the dataset?",
                        "Can you quantify these insights with numbers?"
                    ]
                    
                    suggestion_cols = st.columns(2)
                    for i, suggestion in enumerate(suggestions):
                        with suggestion_cols[i % 2]:
                            if st.button(suggestion, key=f"followup_{i}"):
                                st.session_state.user_question = f"{user_question} {suggestion}"
                                st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error processing question: {e}")
                    
                    # Fallback analysis
                    if st.session_state.analyzer:
                        st.markdown("**📊 Fallback Analysis:**")
                        with st.spinner("Using basic analysis..."):
                            try:
                                fallback_response = st.session_state.analyzer.query_rag(user_question)
                                st.markdown(fallback_response.get('answer', 'Analysis could not be completed.'))
                            except Exception as fallback_error:
                                st.error(f"Fallback analysis also failed: {fallback_error}")
    
    elif RAG_AVAILABLE:
        st.warning("🔄 RAG system not initialized. Select a model above and click 'Initialize Model'.")
        
        # Show what's needed for each model
        st.subheader("🔑 Model Requirements")
        
        for model_name, config in available_models.items():
            with st.expander(f"Setup {model_name}"):
                if config['api_key']:
                    st.success(f"✅ API key available for {model_name}")
                else:
                    st.warning(f"⚠️ API key needed for {model_name}")
                
                if model_name == "Google Gemini":
                    st.markdown("**How to get Google Gemini API key:**")
                    st.markdown("1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)")
                    st.markdown("2. Sign in with Google account")
                    st.markdown("3. Create new API key")
                    st.markdown("4. Copy and paste in the configuration")
                
                elif model_name == "Claude Sonnet":
                    st.markdown("**How to get Claude API key:**")
                    st.markdown("1. Visit [Anthropic Console](https://console.anthropic.com/)")
                    st.markdown("2. Create account and verify")
                    st.markdown("3. Go to API Keys section")
                    st.markdown("4. Generate new key")
                
                elif model_name == "OpenAI GPT":
                    st.markdown("**How to get OpenAI API key:**")
                    st.markdown("1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)")
                    st.markdown("2. Sign in or create account")
                    st.markdown("3. Navigate to API keys")
                    st.markdown("4. Create new secret key")
    
    else:
        st.error("🚧 **RAG System Unavailable**")
        st.markdown("""
        The RAG (Retrieval Augmented Generation) system requires additional dependencies.
        
        **Missing Components:**
        - LangChain framework
        - Vector database (FAISS)
        - Embeddings model
        - LLM providers
        
        **Alternative Analysis:**
        Use other tabs for comprehensive analysis without AI chat.
        """)

def initialize_rag_system_with_model(model_id: str, api_key: str = None):
    """Initialize RAG system with specific model"""
    if not RAG_AVAILABLE:
        return None
    
    try:
        # Use the provided API key or fallback
        if model_id == "google":
            google_api_key = api_key or os.getenv('GOOGLE_API_KEY', 'AIzaSyC9WVZri_Gas_scMlkk-OeveNCkR5LMLCc')
        else:
            google_api_key = None
        
        # Find CSV path
        csv_path = None
        possible_paths = [
            'omicron_2025.csv',
            'data/omicron_2025.csv',
            os.path.join('data', 'omicron_2025.csv'),
            os.path.join(os.path.dirname(__file__), 'omicron_2025.csv')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                csv_path = path
                break
        
        if csv_path:
            rag_system = OmicronSentimentRAG(
                csv_path=csv_path,
                google_api_key=google_api_key,
                llm_provider=model_id
            )
            return rag_system
    except Exception as e:
        st.error(f"RAG system initialization failed: {e}")
        return None

def main():
    # Sidebar navigation
    st.sidebar.title("🦠 Navigation")
    
    # Load data button
    if st.sidebar.button("🔄 Load Data", type="primary", key="load_data_btn"):
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
    
    # RAG Models Status
    if RAG_AVAILABLE:
        st.sidebar.markdown("**🤖 Available AI Models:**")
        st.sidebar.markdown(f"- Google Gemini: {'✅' if GOOGLE_AVAILABLE else '❌'}")
        st.sidebar.markdown(f"- Claude Sonnet: {'✅' if ANTHROPIC_AVAILABLE else '❌'}")
        st.sidebar.markdown(f"- OpenAI GPT: {'✅' if OPENAI_AVAILABLE else '❌'}")
        st.sidebar.markdown(f"- Cohere: {'✅' if COHERE_AVAILABLE else '❌'}")
        st.sidebar.markdown(f"- Ollama: {'✅' if OLLAMA_AVAILABLE else '❌'}")
        
        if st.session_state.rag_system:
            st.sidebar.success("🤖 RAG Active")
        else:
            st.sidebar.warning("🤖 RAG Inactive")
    
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
    selected_page = st.sidebar.selectbox("Choose a page:", list(pages.keys()), key="page_navigation_select")
    
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
