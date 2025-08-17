"""
Copilot-Enhanced Omicron Sentiment Analysis
This version is designed to work with GitHub Copilot Chat for AI assistance
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import ast
import re

class CopilotEnhancedOmicronAnalysis:
    """
    Omicron sentiment analysis designed to work with GitHub Copilot Chat.
    No API keys required - use Copilot Chat for AI insights!
    """
    
    def __init__(self, csv_path: str):
        """Initialize the analysis system."""
        self.csv_path = csv_path
        self.df = None
        self.load_and_preprocess_data()
    
    def load_and_preprocess_data(self):
        """Load and preprocess the CSV data."""
        print("Loading omicron tweets data...")
        
        # Load CSV
        self.df = pd.read_csv(self.csv_path)
        
        # Clean and preprocess
        self.df['text'] = self.df['text'].fillna('')
        self.df['hashtags'] = self.df['hashtags'].fillna('[]')
        
        # Parse hashtags
        def parse_hashtags(hashtag_str):
            try:
                if hashtag_str == '[]' or hashtag_str == '':
                    return []
                return ast.literal_eval(hashtag_str)
            except:
                if isinstance(hashtag_str, str):
                    return [tag.strip("'\"") for tag in hashtag_str.strip('[]').split(',') if tag.strip()]
                return []
        
        self.df['hashtags_parsed'] = self.df['hashtags'].apply(parse_hashtags)
        
        # Simple sentiment analysis
        self.df['sentiment'] = self.df['text'].apply(self.simple_sentiment)
        
        print(f"✅ Loaded {len(self.df)} tweets successfully!")
        print(f"📊 Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        print(f"👥 Unique users: {self.df['user_name'].nunique()}")
    
    def simple_sentiment(self, text: str) -> str:
        """Simple keyword-based sentiment analysis."""
        if not isinstance(text, str):
            return 'neutral'
        
        text_lower = text.lower()
        
        # Positive indicators
        positive_words = ['good', 'great', 'mild', 'recovery', 'better', 'positive', 'safe', 'effective']
        
        # Negative indicators  
        negative_words = ['bad', 'terrible', 'severe', 'death', 'hospital', 'worse', 'fear', 'dangerous']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        else:
            return 'neutral'
    
    def find_hashtag_tweets(self, hashtag: str) -> pd.DataFrame:
        """Find tweets containing specific hashtag."""
        hashtag_lower = hashtag.lower().replace('#', '')
        
        mask = self.df['hashtags_parsed'].apply(
            lambda tags: any(hashtag_lower in tag.lower() for tag in tags)
        )
        
        return self.df[mask].copy()
    
    def analyze_user_activity(self, username: str) -> Dict[str, Any]:
        """Analyze activity for a specific user."""
        user_tweets = self.df[self.df['user_name'].str.contains(username, case=False, na=False)]
        
        if len(user_tweets) == 0:
            return {"error": f"No tweets found for user: {username}"}
        
        return {
            "total_tweets": len(user_tweets),
            "sentiment_distribution": user_tweets['sentiment'].value_counts().to_dict(),
            "total_retweets": user_tweets['retweets'].sum(),
            "total_favorites": user_tweets['favorites'].sum(),
            "most_retweeted": user_tweets.loc[user_tweets['retweets'].idxmax()]['text'],
            "sample_tweets": user_tweets['text'].head(3).tolist()
        }
    
    def get_sentiment_summary(self) -> Dict[str, Any]:
        """Get overall sentiment summary."""
        sentiment_counts = self.df['sentiment'].value_counts()
        
        return {
            "total_tweets": len(self.df),
            "sentiment_distribution": sentiment_counts.to_dict(),
            "sentiment_percentages": (sentiment_counts / len(self.df) * 100).round(1).to_dict(),
            "most_active_users": self.df['user_name'].value_counts().head(5).to_dict(),
            "engagement_stats": {
                "total_retweets": self.df['retweets'].sum(),
                "total_favorites": self.df['favorites'].sum(),
                "avg_retweets": self.df['retweets'].mean(),
                "avg_favorites": self.df['favorites'].mean()
            }
        }
    
    def search_content(self, search_term: str, sentiment_filter: str = None) -> pd.DataFrame:
        """Search tweets by content with optional sentiment filter."""
        # Content search
        mask = self.df['text'].str.contains(search_term, case=False, na=False)
        
        # Apply sentiment filter if specified
        if sentiment_filter and sentiment_filter in ['positive', 'negative', 'neutral']:
            mask = mask & (self.df['sentiment'] == sentiment_filter)
        
        return self.df[mask].copy()
    
    def get_trending_hashtags(self, top_n: int = 10) -> List[Dict]:
        """Get most popular hashtags."""
        all_hashtags = []
        for hashtags in self.df['hashtags_parsed']:
            all_hashtags.extend(hashtags)
        
        from collections import Counter
        hashtag_counts = Counter(all_hashtags)
        
        trending = []
        for hashtag, count in hashtag_counts.most_common(top_n):
            trending.append({
                'hashtag': hashtag,
                'count': count,
                'percentage': round((count / len(self.df)) * 100, 1)
            })
        
        return trending
    
    def copilot_analysis_suggestions(self):
        """
        Suggestions for GitHub Copilot Chat analysis.
        Copy these prompts into Copilot Chat for deeper insights!
        """
        suggestions = [
            "@workspace Analyze the sentiment patterns in this omicron dataset over time",
            "@workspace Find correlations between user location and sentiment about omicron",
            "@workspace Identify the most influential users (high retweets/favorites) and their sentiment",
            "@workspace Create a word cloud of the most common terms in negative sentiment tweets",
            "@workspace Compare sentiment between tweets with many hashtags vs few hashtags",
            "@workspace Find tweets that mention both 'vaccine' and 'omicron' and analyze their sentiment",
            "@workspace Identify geographical patterns in omicron discussions",
            "@workspace Analyze the relationship between tweet length and sentiment",
            "@workspace Find the most polarizing topics (tweets with extreme positive/negative responses)",
            "@workspace Create a timeline analysis showing how sentiment changed over the data period"
        ]
        
        print("🤖 COPILOT CHAT ANALYSIS SUGGESTIONS")
        print("=" * 50)
        print("Copy these prompts into GitHub Copilot Chat for AI-powered insights:")
        print()
        
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i:2d}. {suggestion}")
        
        print("\n💡 Pro tip: Use @workspace to give Copilot context about your entire project!")
        return suggestions

def demo_copilot_enhanced_analysis():
    """Demo the Copilot-enhanced analysis."""
    print("🦠 COPILOT-ENHANCED OMICRON ANALYSIS DEMO")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = CopilotEnhancedOmicronAnalysis("omicron_2025.csv")
    
    # Overall sentiment summary
    print("\n📊 SENTIMENT SUMMARY")
    print("-" * 30)
    summary = analyzer.get_sentiment_summary()
    print(f"Total tweets: {summary['total_tweets']:,}")
    print("Sentiment distribution:")
    for sentiment, percentage in summary['sentiment_percentages'].items():
        print(f"  {sentiment.title()}: {percentage}%")
    
    # Hashtag analysis
    print("\n🏷️ TRENDING HASHTAGS")
    print("-" * 30)
    trending = analyzer.get_trending_hashtags(5)
    for item in trending:
        print(f"#{item['hashtag']}: {item['count']} tweets ({item['percentage']}%)")
    
    # Omicron-specific analysis
    print("\n🔍 OMICRON HASHTAG ANALYSIS")
    print("-" * 30)
    omicron_tweets = analyzer.find_hashtag_tweets("omicron")
    print(f"Found {len(omicron_tweets)} tweets with #omicron")
    
    if len(omicron_tweets) > 0:
        omicron_sentiment = omicron_tweets['sentiment'].value_counts()
        print("Omicron sentiment breakdown:")
        for sentiment, count in omicron_sentiment.items():
            percentage = (count / len(omicron_tweets)) * 100
            print(f"  {sentiment.title()}: {count} ({percentage:.1f}%)")
    
    # Sample analysis
    print("\n📝 SAMPLE ANALYSIS QUERIES")
    print("-" * 30)
    
    # Vaccine mentions
    vaccine_tweets = analyzer.search_content("vaccine")
    print(f"Tweets mentioning 'vaccine': {len(vaccine_tweets)}")
    
    # Hospital mentions
    hospital_tweets = analyzer.search_content("hospital")
    print(f"Tweets mentioning 'hospital': {len(hospital_tweets)}")
    
    # Negative vaccine sentiment
    negative_vaccine = analyzer.search_content("vaccine", "negative")
    print(f"Negative sentiment about vaccines: {len(negative_vaccine)}")
    
    # Show Copilot suggestions
    print("\n" + "=" * 60)
    analyzer.copilot_analysis_suggestions()
    
    # Interactive mode
    print("\n🔮 INTERACTIVE MODE")
    print("=" * 60)
    print("Available commands:")
    print("1. hashtag [name] - Find tweets with hashtag")
    print("2. user [name] - Analyze user activity") 
    print("3. search [term] - Search tweet content")
    print("4. sentiment [positive/negative/neutral] - Filter by sentiment")
    print("5. copilot - Show Copilot Chat suggestions")
    print("6. quit - Exit")
    
    while True:
        try:
            command = input("\n💬 Enter command: ").strip().lower()
            
            if command in ['quit', 'exit', 'q']:
                break
            elif command.startswith('hashtag '):
                hashtag = command[8:]
                results = analyzer.find_hashtag_tweets(hashtag)
                print(f"\n📊 Found {len(results)} tweets with #{hashtag}")
                if len(results) > 0:
                    print("Sample tweets:")
                    for i, (_, row) in enumerate(results.head(3).iterrows()):
                        print(f"  {i+1}. @{row['user_name']}: {row['text'][:80]}...")
            
            elif command.startswith('user '):
                username = command[5:]
                analysis = analyzer.analyze_user_activity(username)
                if "error" in analysis:
                    print(f"❌ {analysis['error']}")
                else:
                    print(f"\n👤 Analysis for '{username}':")
                    print(f"  Total tweets: {analysis['total_tweets']}")
                    print(f"  Sentiment: {analysis['sentiment_distribution']}")
                    print(f"  Engagement: {analysis['total_retweets']} RT, {analysis['total_favorites']} ❤️")
            
            elif command.startswith('search '):
                term = command[7:]
                results = analyzer.search_content(term)
                print(f"\n🔍 Found {len(results)} tweets mentioning '{term}'")
                if len(results) > 0:
                    sentiment_breakdown = results['sentiment'].value_counts()
                    print(f"Sentiment: {sentiment_breakdown.to_dict()}")
            
            elif command.startswith('sentiment '):
                sentiment = command[10:]
                if sentiment in ['positive', 'negative', 'neutral']:
                    filtered = analyzer.df[analyzer.df['sentiment'] == sentiment]
                    print(f"\n😊 Found {len(filtered)} {sentiment} tweets")
                    if len(filtered) > 0:
                        print("Sample tweets:")
                        for i, (_, row) in enumerate(filtered.head(3).iterrows()):
                            print(f"  {i+1}. @{row['user_name']}: {row['text'][:80]}...")
                else:
                    print("❌ Use: positive, negative, or neutral")
            
            elif command == 'copilot':
                analyzer.copilot_analysis_suggestions()
            
            else:
                print("❓ Unknown command. Try: hashtag, user, search, sentiment, copilot, or quit")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    demo_copilot_enhanced_analysis()
