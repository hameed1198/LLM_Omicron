"""
Demo script for Omicron Sentiment Analysis with RAG
This script demonstrates the key functionality without requiring API keys.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

from omicron_sentiment_rag import OmicronSentimentRAG
import json

def demo_basic_functionality():
    """Demonstrate basic functionality without requiring API keys."""
    print("🦠 OMICRON SENTIMENT ANALYSIS DEMO")
    print("=" * 50)
    
    # Initialize analyzer (no API key needed for basic features)
    print("Loading data and initializing system...")
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'omicron_2025.csv')
    analyzer = OmicronSentimentRAG(data_path, anthropic_api_key=None)
    
    print("\n✅ System initialized successfully!")
    print(f"📊 Loaded {len(analyzer.df)} tweets")
    
    # Generate overall report
    print("\n" + "=" * 50)
    print("📈 OVERALL ANALYSIS REPORT")
    print("=" * 50)
    report = analyzer.generate_report()
    print(report)
    
    # Demo 1: Hashtag analysis
    print("\n" + "=" * 50)
    print("🏷️  HASHTAG ANALYSIS DEMO")
    print("=" * 50)
    
    # Query tweets with #omicron hashtag
    print("1. Finding users who tweeted with hashtag 'omicron'...")
    omicron_tweets = analyzer.query_tweets_by_hashtag("omicron")
    print(f"Found {len(omicron_tweets)} tweets with #omicron hashtag")
    
    print("\nSample users and their tweets:")
    for i, tweet in enumerate(omicron_tweets[:5]):
        print(f"\n📱 Tweet {i+1}:")
        print(f"   👤 User: {tweet['user_name']}")
        print(f"   📍 Location: {tweet['user_location']}")
        print(f"   😊 Sentiment: {tweet['sentiment']}")
        print(f"   📝 Tweet: {tweet['tweet'][:100]}...")
        print(f"   📊 Engagement: {tweet['retweets']} RT, {tweet['favorites']} ❤️")
    
    # Demo 2: Content search
    print("\n" + "=" * 50)
    print("🔍 CONTENT SEARCH DEMO")
    print("=" * 50)
    
    search_terms = ["vaccine", "hospital", "mild", "severe"]
    for term in search_terms:
        print(f"\n🔎 Searching for tweets mentioning '{term}'...")
        results = analyzer.search_tweets_by_content(term, 3)
        print(f"Found {len(results)} tweets")
        
        for i, tweet in enumerate(results):
            print(f"   {i+1}. @{tweet['user_name']}: {tweet['tweet'][:80]}...")
    
    # Demo 3: User analysis
    print("\n" + "=" * 50)
    print("👥 USER ANALYSIS DEMO")
    print("=" * 50)
    
    # Find most active users
    user_counts = analyzer.df['user_name'].value_counts()
    print("Most active users:")
    for i, (user, count) in enumerate(user_counts.head(5).items()):
        print(f"   {i+1}. {user}: {count} tweets")
    
    # Demo 4: Sentiment analysis
    print("\n" + "=" * 50)
    print("😊 SENTIMENT ANALYSIS DEMO")
    print("=" * 50)
    
    sentiment_analysis = analyzer.analyze_sentiment_distribution()
    print("Sentiment breakdown:")
    for sentiment, count in sentiment_analysis['sentiment_distribution'].items():
        percentage = (count / sentiment_analysis['total_tweets']) * 100
        print(f"   {sentiment.title()}: {count} tweets ({percentage:.1f}%)")
    
    print(f"\nAverage sentiment score: {sentiment_analysis['average_compound_score']:.3f}")
    print("(Range: -1.0 = very negative, 0 = neutral, +1.0 = very positive)")
    
    # Demo 5: Trending hashtags
    print("\n" + "=" * 50)
    print("📈 TRENDING HASHTAGS")
    print("=" * 50)
    
    trending = analyzer.get_trending_hashtags(10)
    print("Top trending hashtags:")
    for i, hashtag_info in enumerate(trending):
        print(f"   {i+1}. #{hashtag_info['hashtag']}: {hashtag_info['count']} tweets ({hashtag_info['percentage']:.1f}%)")
    
    return analyzer

def demo_interactive_queries(analyzer):
    """Demonstrate interactive query functionality."""
    print("\n" + "=" * 50)
    print("🔮 INTERACTIVE QUERY DEMO")
    print("=" * 50)
    
    # Example queries
    example_queries = [
        "list users with hashtag CDC",
        "find tweets mentioning vaccine",
        "show tweets about hospital"
    ]
    
    for query in example_queries:
        print(f"\n❓ Query: '{query}'")
        print("Response:")
        
        if 'hashtag' in query.lower():
            words = query.split()
            hashtag = None
            for word in words:
                if word.upper() in ['CDC', 'OMICRON', 'COVID', 'VACCINE']:
                    hashtag = word.lower()
                    break
            
            if hashtag:
                results = analyzer.query_tweets_by_hashtag(hashtag)
                print(f"   Found {len(results)} tweets with #{hashtag}")
                if results:
                    for i, tweet in enumerate(results[:2]):
                        print(f"   • {tweet['user_name']}: {tweet['tweet'][:60]}...")
        
        elif 'find' in query.lower() or 'mention' in query.lower():
            if 'vaccine' in query.lower():
                results = analyzer.search_tweets_by_content('vaccine', 2)
                print(f"   Found {len(results)} tweets mentioning 'vaccine'")
                for i, tweet in enumerate(results):
                    print(f"   • {tweet['user_name']}: {tweet['tweet'][:60]}...")
            elif 'hospital' in query.lower():
                results = analyzer.search_tweets_by_content('hospital', 2)
                print(f"   Found {len(results)} tweets mentioning 'hospital'")
                for i, tweet in enumerate(results):
                    print(f"   • {tweet['user_name']}: {tweet['tweet'][:60]}...")

def demo_advanced_features(analyzer):
    """Demonstrate advanced features that would work with API keys."""
    print("\n" + "=" * 50)
    print("🚀 ADVANCED FEATURES (Requires API Key)")
    print("=" * 50)
    
    if analyzer.retrieval_chain:
        print("✅ RAG functionality is available!")
        
        # Demo RAG queries
        sample_questions = [
            "What are the main concerns about Omicron mentioned in the tweets?",
            "Which users are discussing vaccine effectiveness?",
            "What is the general sentiment about Omicron severity?"
        ]
        
        for question in sample_questions:
            print(f"\n❓ Question: {question}")
            response = analyzer.query_with_rag(question)
            print(f"🤖 RAG Response: {response}")
    
    else:
        print("❌ RAG functionality not available (no API key provided)")
        print("\nTo enable RAG features:")
        print("1. Get an Anthropic API key")
        print("2. Set ANTHROPIC_API_KEY in your .env file")
        print("3. Restart the application")
        
        print("\n🔮 Simulated RAG responses:")
        print("Q: What are the main concerns about Omicron?")
        print("A: Based on the tweets, main concerns include hospital capacity,")
        print("   vaccine effectiveness, and variant transmission rates.")

def main():
    """Main demo function."""
    try:
        # Run basic functionality demo
        analyzer = demo_basic_functionality()
        
        # Interactive queries demo
        demo_interactive_queries(analyzer)
        
        # Advanced features demo
        demo_advanced_features(analyzer)
        
        print("\n" + "=" * 50)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
        print("\nNext steps:")
        print("1. Run the Streamlit app: streamlit run streamlit_app.py")
        print("2. Add your API key to .env for RAG functionality")
        print("3. Explore the interactive web interface")
        
        # Simple interactive mode
        print("\n" + "=" * 50)
        print("💬 SIMPLE INTERACTIVE MODE")
        print("=" * 50)
        print("Try asking questions! (Type 'quit' to exit)")
        print("Examples:")
        print("- hashtag omicron")
        print("- search vaccine")
        print("- user Nathan")
        
        while True:
            try:
                query = input("\n🤔 Your question: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not query:
                    continue
                
                if query.lower().startswith('hashtag'):
                    hashtag = query.split()[-1]
                    results = analyzer.query_tweets_by_hashtag(hashtag)
                    print(f"📊 Found {len(results)} tweets with #{hashtag}")
                    for i, tweet in enumerate(results[:3]):
                        print(f"   {i+1}. @{tweet['user_name']}: {tweet['tweet'][:70]}...")
                
                elif query.lower().startswith('search'):
                    term = query.split()[-1]
                    results = analyzer.search_tweets_by_content(term, 3)
                    print(f"🔍 Found {len(results)} tweets mentioning '{term}'")
                    for i, tweet in enumerate(results):
                        print(f"   {i+1}. @{tweet['user_name']}: {tweet['tweet'][:70]}...")
                
                elif query.lower().startswith('user'):
                    username = query.split()[-1]
                    results = analyzer.get_user_tweets(username)
                    print(f"👤 Found {len(results)} tweets by users matching '{username}'")
                    for i, tweet in enumerate(results[:3]):
                        print(f"   {i+1}. {tweet['date']}: {tweet['tweet'][:70]}...")
                
                else:
                    print("💡 Try: 'hashtag omicron', 'search vaccine', or 'user Nathan'")
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        print("Make sure you're running from the analysis_scripts directory and the data file exists in ../data/omicron_2025.csv")

if __name__ == "__main__":
    main()
