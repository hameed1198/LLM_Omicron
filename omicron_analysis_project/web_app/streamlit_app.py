import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

from omicron_sentiment_rag import OmicronSentimentRAG
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="Omicron Sentiment Analysis with RAG",
    page_icon="🦠",
    layout="wide"
)

@st.cache_data
def load_analyzer():
    """Load the sentiment analyzer with caching."""
    api_key = os.getenv('ANTHROPIC_API_KEY')
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'omicron_2025.csv')
    return OmicronSentimentRAG(csv_path, anthropic_api_key=api_key)

def main():
    st.title("🦠 Omicron Tweets Sentiment Analysis with RAG")
    st.markdown("### Analyzing COVID-19 Omicron variant discussions on Twitter using LangChain and Claude Sonnet")
    
    # Load analyzer
    with st.spinner("Loading data and initializing AI models..."):
        analyzer = load_analyzer()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Overview", "Interactive Query", "Hashtag Analysis", "User Analysis", "Sentiment Deep Dive", "RAG Chat"]
    )
    
    if page == "Overview":
        show_overview(analyzer)
    elif page == "Interactive Query":
        show_interactive_query(analyzer)
    elif page == "Hashtag Analysis":
        show_hashtag_analysis(analyzer)
    elif page == "User Analysis":
        show_user_analysis(analyzer)
    elif page == "Sentiment Deep Dive":
        show_sentiment_analysis(analyzer)
    elif page == "RAG Chat":
        show_rag_chat(analyzer)

def show_overview(analyzer):
    """Show overview dashboard."""
    st.header("📊 Overview Dashboard")
    
    # Generate report
    report = analyzer.generate_report()
    
    # Basic stats in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tweets", len(analyzer.df))
    
    with col2:
        st.metric("Unique Users", analyzer.df['user_name'].nunique())
    
    with col3:
        sentiment_dist = analyzer.analyze_sentiment_distribution()
        avg_sentiment = sentiment_dist['average_compound_score']
        st.metric("Avg Sentiment", f"{avg_sentiment:.3f}")
    
    with col4:
        total_engagement = analyzer.df['retweets'].sum() + analyzer.df['favorites'].sum()
        st.metric("Total Engagement", f"{total_engagement:,}")
    
    # Sentiment distribution chart
    st.subheader("Sentiment Distribution")
    sentiment_data = analyzer.analyze_sentiment_distribution()
    
    fig_pie = px.pie(
        values=list(sentiment_data['sentiment_distribution'].values()),
        names=list(sentiment_data['sentiment_distribution'].keys()),
        title="Tweet Sentiment Distribution"
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Timeline analysis
    st.subheader("Tweet Timeline")
    # Fix date parsing - dates are in DD-MM-YYYY HH:MM format
    analyzer.df['date'] = pd.to_datetime(analyzer.df['date'], format='%d-%m-%Y %H:%M', errors='coerce')
    timeline_data = analyzer.df.groupby(analyzer.df['date'].dt.date).size()
    
    fig_timeline = px.line(
        x=timeline_data.index,
        y=timeline_data.values,
        title="Tweets Over Time",
        labels={'x': 'Date', 'y': 'Number of Tweets'}
    )
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Word cloud
    st.subheader("Word Cloud")
    if st.button("Generate Word Cloud"):
        all_text = ' '.join(analyzer.df['clean_text'].dropna())
        if all_text.strip():
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.warning("No text data available for word cloud generation.")

def show_interactive_query(analyzer):
    """Show interactive query interface."""
    st.header("🔍 Interactive Query Interface")
    
    st.markdown("### Query Examples:")
    st.markdown("""
    - **Hashtag queries**: "list users with hashtag omicron"
    - **Content search**: "find tweets mentioning vaccine"
    - **Sentiment filter**: "show positive tweets about omicron"
    - **User search**: "tweets by Nathan Joyner"
    """)
    
    # Query input
    query = st.text_input("Enter your query:", placeholder="e.g., list users with hashtag omicron")
    
    if query:
        with st.spinner("Processing query..."):
            if 'hashtag' in query.lower():
                # Extract hashtag
                words = query.split()
                hashtag = None
                for word in words:
                    if word.startswith('#'):
                        hashtag = word[1:]
                        break
                    elif word.lower() in ['omicron', 'covid', 'vaccine', 'hospital', 'cdc']:
                        hashtag = word.lower()
                        break
                
                if hashtag:
                    results = analyzer.query_tweets_by_hashtag(hashtag)
                    st.success(f"Found {len(results)} tweets with hashtag '{hashtag}'")
                    
                    if results:
                        df_results = pd.DataFrame(results)
                        st.dataframe(df_results)
                        
                        # Show some sample tweets
                        st.subheader("Sample Tweets:")
                        for i, tweet in enumerate(results[:3]):
                            with st.expander(f"Tweet {i+1} by {tweet['user_name']}"):
                                st.write(f"**Location:** {tweet['user_location']}")
                                st.write(f"**Date:** {tweet['date']}")
                                st.write(f"**Sentiment:** {tweet['sentiment']}")
                                st.write(f"**Tweet:** {tweet['tweet']}")
                                st.write(f"**Engagement:** {tweet['retweets']} retweets, {tweet['favorites']} favorites")
                else:
                    st.error("Please specify a hashtag to search for.")
            
            elif 'user' in query.lower() or 'by' in query.lower():
                # Extract username
                words = query.split()
                username = None
                for i, word in enumerate(words):
                    if word.lower() in ['by', 'user'] and i + 1 < len(words):
                        username = words[i + 1]
                        break
                
                if username:
                    results = analyzer.get_user_tweets(username)
                    st.success(f"Found {len(results)} tweets by users matching '{username}'")
                    
                    if results:
                        df_results = pd.DataFrame(results)
                        st.dataframe(df_results)
                else:
                    st.error("Please specify a username to search for.")
            
            elif any(word in query.lower() for word in ['find', 'search', 'mention']):
                # Content search
                search_terms = ['hospital', 'vaccine', 'covid', 'death', 'mild', 'severe', 'symptom']
                search_term = None
                for term in search_terms:
                    if term in query.lower():
                        search_term = term
                        break
                
                if search_term:
                    results = analyzer.search_tweets_by_content(search_term, 10)
                    st.success(f"Found {len(results)} tweets mentioning '{search_term}'")
                    
                    if results:
                        df_results = pd.DataFrame(results)
                        st.dataframe(df_results)
                else:
                    st.error("Please specify what to search for.")
            
            else:
                st.info("Try a more specific query like 'hashtag omicron' or 'find vaccine mentions'")

def show_hashtag_analysis(analyzer):
    """Show hashtag analysis."""
    st.header("#️⃣ Hashtag Analysis")
    
    # Get trending hashtags
    trending = analyzer.get_trending_hashtags(20)
    
    if trending:
        # Create DataFrame for easier manipulation
        df_trending = pd.DataFrame(trending)
        
        # Bar chart
        fig_bar = px.bar(
            df_trending.head(10),
            x='hashtag',
            y='count',
            title="Top 10 Hashtags by Frequency",
            labels={'count': 'Number of Tweets', 'hashtag': 'Hashtag'}
        )
        fig_bar.update_xaxes(tickangle=45)
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Full table
        st.subheader("All Trending Hashtags")
        st.dataframe(df_trending)
        
        # Hashtag selector
        st.subheader("Explore Specific Hashtag")
        selected_hashtag = st.selectbox("Select a hashtag to explore:", [h['hashtag'] for h in trending])
        
        if selected_hashtag:
            tweets = analyzer.query_tweets_by_hashtag(selected_hashtag)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Tweets", len(tweets))
            with col2:
                if tweets:
                    sentiments = [t['sentiment'] for t in tweets]
                    sentiment_counts = pd.Series(sentiments).value_counts()
                    most_common_sentiment = sentiment_counts.index[0]
                    st.metric("Dominant Sentiment", most_common_sentiment)
            
            # Show sample tweets
            if tweets:
                st.subheader(f"Sample Tweets with #{selected_hashtag}")
                for i, tweet in enumerate(tweets[:5]):
                    with st.expander(f"Tweet {i+1}"):
                        st.write(f"**User:** {tweet['user_name']}")
                        st.write(f"**Sentiment:** {tweet['sentiment']}")
                        st.write(f"**Tweet:** {tweet['tweet']}")

def show_user_analysis(analyzer):
    """Show user analysis."""
    st.header("👤 User Analysis")
    
    # Top users by tweet count
    user_counts = analyzer.df['user_name'].value_counts().head(10)
    
    fig_users = px.bar(
        x=user_counts.values,
        y=user_counts.index,
        orientation='h',
        title="Top 10 Most Active Users",
        labels={'x': 'Number of Tweets', 'y': 'Username'}
    )
    st.plotly_chart(fig_users, use_container_width=True)
    
    # User search
    st.subheader("Search for Specific User")
    username_search = st.text_input("Enter username to search:")
    
    if username_search:
        user_tweets = analyzer.get_user_tweets(username_search)
        
        if user_tweets:
            st.success(f"Found {len(user_tweets)} tweets")
            
            # User stats
            df_user = pd.DataFrame(user_tweets)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Tweets", len(user_tweets))
            with col2:
                total_retweets = df_user['retweets'].sum()
                st.metric("Total Retweets", total_retweets)
            with col3:
                total_favorites = df_user['favorites'].sum()
                st.metric("Total Favorites", total_favorites)
            
            # Sentiment distribution for this user
            sentiment_dist = df_user['sentiment'].value_counts()
            fig_user_sentiment = px.pie(
                values=sentiment_dist.values,
                names=sentiment_dist.index,
                title=f"Sentiment Distribution for {username_search}"
            )
            st.plotly_chart(fig_user_sentiment, use_container_width=True)
            
            # Show tweets
            st.subheader("All Tweets")
            st.dataframe(df_user)
        else:
            st.warning(f"No tweets found for user '{username_search}'")

def show_sentiment_analysis(analyzer):
    """Show detailed sentiment analysis."""
    st.header("😊😐😢 Sentiment Deep Dive")
    
    # Overall sentiment metrics
    sentiment_data = analyzer.analyze_sentiment_distribution()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        positive_pct = (sentiment_data['sentiment_distribution'].get('positive', 0) / sentiment_data['total_tweets']) * 100
        st.metric("Positive Tweets", f"{positive_pct:.1f}%")
    
    with col2:
        neutral_pct = (sentiment_data['sentiment_distribution'].get('neutral', 0) / sentiment_data['total_tweets']) * 100
        st.metric("Neutral Tweets", f"{neutral_pct:.1f}%")
    
    with col3:
        negative_pct = (sentiment_data['sentiment_distribution'].get('negative', 0) / sentiment_data['total_tweets']) * 100
        st.metric("Negative Tweets", f"{negative_pct:.1f}%")
    
    # Sentiment over time
    # Ensure dates are properly parsed (if not already done)
    if not pd.api.types.is_datetime64_any_dtype(analyzer.df['date']):
        analyzer.df['date'] = pd.to_datetime(analyzer.df['date'], format='%d-%m-%Y %H:%M', errors='coerce')
    analyzer.df['sentiment_label'] = analyzer.df['vader_sentiment'].apply(lambda x: x['label'])
    
    daily_sentiment = analyzer.df.groupby([
        analyzer.df['date'].dt.date,
        'sentiment_label'
    ]).size().unstack(fill_value=0)
    
    fig_timeline_sentiment = px.line(
        daily_sentiment,
        title="Sentiment Over Time",
        labels={'index': 'Date', 'value': 'Number of Tweets'}
    )
    st.plotly_chart(fig_timeline_sentiment, use_container_width=True)
    
    # Sentiment filter
    st.subheader("Filter Tweets by Sentiment")
    sentiment_filter = st.selectbox("Select sentiment:", ['positive', 'neutral', 'negative'])
    
    filtered_tweets = analyzer.df[analyzer.df['sentiment_label'] == sentiment_filter]
    
    st.write(f"Showing {len(filtered_tweets)} {sentiment_filter} tweets:")
    
    # Show sample tweets
    for i, (_, row) in enumerate(filtered_tweets.head(5).iterrows()):
        with st.expander(f"{sentiment_filter.title()} Tweet {i+1}"):
            st.write(f"**User:** {row['user_name']}")
            st.write(f"**Date:** {row['date']}")
            st.write(f"**Tweet:** {row['text']}")
            compound_score = row['vader_sentiment']['compound']
            st.write(f"**Sentiment Score:** {compound_score:.3f}")

def show_rag_chat(analyzer):
    """Show RAG-powered chat interface."""
    st.header("🤖 RAG-Powered Chat")
    
    if not analyzer.retrieval_chain:
        st.warning("RAG functionality requires an Anthropic API key. Please set ANTHROPIC_API_KEY in your .env file.")
        st.info("You can still use the other features of the application without the API key.")
        return
    
    st.markdown("### Ask questions about the Omicron tweets data using AI!")
    st.markdown("Examples: 'What are the main concerns about Omicron?', 'Which users are most worried about vaccines?'")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the Omicron tweets..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = analyzer.query_with_rag(prompt)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
