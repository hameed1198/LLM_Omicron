"""
Simplified test script to check basic functionality
"""

import pandas as pd
import numpy as np

def test_basic_functionality():
    """Test basic CSV loading and processing."""
    print("🦠 TESTING BASIC FUNCTIONALITY")
    print("=" * 50)
    
    try:
        # Load CSV
        print("Loading CSV file...")
        df = pd.read_csv("omicron_2025.csv")
        print(f"✅ Successfully loaded {len(df)} tweets")
        print(f"📊 Columns: {df.columns.tolist()}")
        
        # Basic data info
        print(f"\n📈 Data Overview:")
        print(f"   Total tweets: {len(df)}")
        print(f"   Unique users: {df['user_name'].nunique()}")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Sample data
        print(f"\n📝 Sample tweets:")
        for i, (_, row) in enumerate(df.head(3).iterrows()):
            print(f"   {i+1}. @{row['user_name']}: {row['text'][:80]}...")
        
        # Test hashtag parsing
        print(f"\n🏷️ Testing hashtag parsing:")
        import ast
        
        def parse_hashtags(hashtag_str):
            try:
                if hashtag_str == '[]' or hashtag_str == '':
                    return []
                return ast.literal_eval(hashtag_str)
            except:
                if isinstance(hashtag_str, str):
                    return [tag.strip("'\"") for tag in hashtag_str.strip('[]').split(',') if tag.strip()]
                return []
        
        df['hashtags_parsed'] = df['hashtags'].fillna('[]').apply(parse_hashtags)
        
        # Find tweets with specific hashtags
        omicron_tweets = []
        for idx, row in df.iterrows():
            hashtags = row['hashtags_parsed']
            if any('omicron' in tag.lower() for tag in hashtags):
                omicron_tweets.append({
                    'user_name': row['user_name'],
                    'tweet': row['text'],
                    'hashtags': hashtags
                })
        
        print(f"   Found {len(omicron_tweets)} tweets with 'omicron' hashtag")
        
        for i, tweet in enumerate(omicron_tweets[:3]):
            print(f"   {i+1}. @{tweet['user_name']}: {tweet['tweet'][:60]}...")
        
        print(f"\n✅ Basic functionality test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_simple_sentiment():
    """Test simple sentiment analysis without external libraries."""
    print(f"\n😊 TESTING SIMPLE SENTIMENT ANALYSIS")
    print("=" * 50)
    
    try:
        df = pd.read_csv("omicron_2025.csv")
        
        # Simple keyword-based sentiment
        positive_words = ['good', 'great', 'excellent', 'positive', 'mild', 'recovery', 'better']
        negative_words = ['bad', 'terrible', 'negative', 'severe', 'death', 'hospital', 'worse']
        
        def simple_sentiment(text):
            if not isinstance(text, str):
                return 'neutral'
            
            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count > neg_count:
                return 'positive'
            elif neg_count > pos_count:
                return 'negative'
            else:
                return 'neutral'
        
        df['simple_sentiment'] = df['text'].apply(simple_sentiment)
        
        sentiment_counts = df['simple_sentiment'].value_counts()
        print(f"📊 Sentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {sentiment.title()}: {count} ({percentage:.1f}%)")
        
        # Show examples of each sentiment
        for sentiment in ['positive', 'negative', 'neutral']:
            sentiment_tweets = df[df['simple_sentiment'] == sentiment]
            if len(sentiment_tweets) > 0:
                print(f"\n{sentiment.title()} example:")
                sample = sentiment_tweets.iloc[0]
                print(f"   @{sample['user_name']}: {sample['text'][:80]}...")
        
        print(f"\n✅ Simple sentiment analysis test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def interactive_demo():
    """Simple interactive demo."""
    print(f"\n🔍 INTERACTIVE DEMO")
    print("=" * 50)
    
    try:
        df = pd.read_csv("omicron_2025.csv")
        
        # Parse hashtags
        import ast
        def parse_hashtags(hashtag_str):
            try:
                if hashtag_str == '[]' or hashtag_str == '':
                    return []
                return ast.literal_eval(hashtag_str)
            except:
                if isinstance(hashtag_str, str):
                    return [tag.strip("'\"") for tag in hashtag_str.strip('[]').split(',') if tag.strip()]
                return []
        
        df['hashtags_parsed'] = df['hashtags'].fillna('[]').apply(parse_hashtags)
        
        print("Available queries:")
        print("1. hashtag [name] - find tweets with specific hashtag")
        print("2. user [name] - find tweets by specific user")
        print("3. search [term] - find tweets containing term")
        print("4. stats - show basic statistics")
        print("Type 'quit' to exit")
        
        while True:
            try:
                query = input("\n💬 Enter query: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if query.startswith('hashtag '):
                    hashtag = query[8:].lower()
                    results = []
                    for idx, row in df.iterrows():
                        hashtags = row['hashtags_parsed']
                        if any(hashtag in tag.lower() for tag in hashtags):
                            results.append(row)
                    
                    print(f"\n📊 Found {len(results)} tweets with '{hashtag}' hashtag:")
                    for i, row in enumerate(results[:5]):
                        print(f"   {i+1}. @{row['user_name']}: {row['text'][:70]}...")
                
                elif query.startswith('user '):
                    username = query[5:]
                    user_tweets = df[df['user_name'].str.contains(username, case=False, na=False)]
                    print(f"\n👤 Found {len(user_tweets)} tweets by users matching '{username}':")
                    for i, (_, row) in enumerate(user_tweets.head(5).iterrows()):
                        print(f"   {i+1}. {row['date']}: {row['text'][:70]}...")
                
                elif query.startswith('search '):
                    term = query[7:]
                    search_results = df[df['text'].str.contains(term, case=False, na=False)]
                    print(f"\n🔎 Found {len(search_results)} tweets mentioning '{term}':")
                    for i, (_, row) in enumerate(search_results.head(5).iterrows()):
                        print(f"   {i+1}. @{row['user_name']}: {row['text'][:70]}...")
                
                elif query == 'stats':
                    print(f"\n📈 Basic Statistics:")
                    print(f"   Total tweets: {len(df)}")
                    print(f"   Unique users: {df['user_name'].nunique()}")
                    print(f"   Total retweets: {df['retweets'].sum()}")
                    print(f"   Total favorites: {df['favorites'].sum()}")
                    
                    # Top users
                    top_users = df['user_name'].value_counts().head(3)
                    print(f"\n👥 Most active users:")
                    for user, count in top_users.items():
                        print(f"   {user}: {count} tweets")
                
                else:
                    print("❓ Unknown query. Try: 'hashtag omicron', 'user Nathan', 'search vaccine', or 'stats'")
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 SIMPLIFIED OMICRON ANALYSIS TEST")
    print("=" * 60)
    
    # Test basic functionality
    if not test_basic_functionality():
        print("❌ Basic test failed. Check if omicron_2025.csv exists.")
        return
    
    # Test sentiment analysis
    if not test_simple_sentiment():
        print("❌ Sentiment test failed.")
        return
    
    # Interactive demo
    print(f"\n🎉 All tests passed! Starting interactive demo...")
    interactive_demo()
    
    print(f"\n✅ TESTING COMPLETED!")
    print("=" * 60)
    print("Next steps:")
    print("1. Install missing packages for full functionality")
    print("2. Add Anthropic API key for RAG features")
    print("3. Run: streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()
