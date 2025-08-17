"""
Analysis to find users with highest tweets containing #omicron hashtag
"""

import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

from copilot_analysis import CopilotEnhancedOmicronAnalysis

def find_highest_omicron_hashtag_users():
    """Find users with the most tweets containing #omicron hashtag."""
    
    print("🏷️ USERS WITH HIGHEST #OMICRON HASHTAG TWEETS")
    print("=" * 60)
    
    # Initialize analyzer
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'omicron_2025.csv')
    analyzer = CopilotEnhancedOmicronAnalysis(data_path)
    
    # Find all tweets with omicron hashtag (case insensitive)
    omicron_tweets = analyzer.find_hashtag_tweets("omicron")
    
    print(f"📊 Total tweets with #omicron hashtag: {len(omicron_tweets):,}")
    
    if len(omicron_tweets) == 0:
        print("❌ No tweets found with #omicron hashtag")
        return
    
    # Count tweets per user
    user_omicron_counts = omicron_tweets['user_name'].value_counts()
    
    print(f"\n👑 TOP 20 USERS BY #OMICRON HASHTAG TWEETS")
    print("-" * 60)
    
    for i, (username, tweet_count) in enumerate(user_omicron_counts.head(20).items(), 1):
        # Get user's omicron tweets for analysis
        user_omicron_tweets = omicron_tweets[omicron_tweets['user_name'] == username]
        
        # Calculate engagement stats
        total_engagement = user_omicron_tweets['total_engagement'].sum()
        avg_engagement = user_omicron_tweets['total_engagement'].mean()
        total_retweets = user_omicron_tweets['retweets'].sum()
        total_favorites = user_omicron_tweets['favorites'].sum()
        
        # Sentiment analysis
        sentiment_dist = user_omicron_tweets['sentiment'].value_counts().to_dict()
        dominant_sentiment = user_omicron_tweets['sentiment'].mode().iloc[0] if len(user_omicron_tweets) > 0 else 'unknown'
        
        # User location
        user_location = user_omicron_tweets['user_location'].iloc[0] if len(user_omicron_tweets) > 0 else 'Unknown'
        
        print(f"\n{i:2d}. @{username}")
        print(f"    📝 #Omicron tweets: {tweet_count}")
        print(f"    📍 Location: {user_location}")
        print(f"    📊 Total engagement: {int(total_engagement):,} (avg: {avg_engagement:.1f})")
        print(f"    🔄 Retweets: {int(total_retweets):,}")
        print(f"    ❤️  Favorites: {int(total_favorites):,}")
        print(f"    😊 Dominant sentiment: {dominant_sentiment}")
        print(f"    📈 Sentiment breakdown: {sentiment_dist}")
    
    # Detailed analysis of top user
    print(f"\n🔍 DETAILED ANALYSIS: TOP #OMICRON HASHTAG USER")
    print("=" * 60)
    
    top_user = user_omicron_counts.index[0]
    top_user_tweets = omicron_tweets[omicron_tweets['user_name'] == top_user]
    
    print(f"👤 User: @{top_user}")
    print(f"📝 Total #omicron tweets: {len(top_user_tweets)}")
    print(f"📊 Total engagement on #omicron tweets: {top_user_tweets['total_engagement'].sum():,}")
    print(f"📈 Average engagement per tweet: {top_user_tweets['total_engagement'].mean():.1f}")
    print(f"🏆 Most viral #omicron tweet: {top_user_tweets['total_engagement'].max()} engagement")
    
    # Show sentiment breakdown
    sentiment_breakdown = top_user_tweets['sentiment'].value_counts()
    print(f"\n😊 Sentiment breakdown for @{top_user}'s #omicron tweets:")
    for sentiment, count in sentiment_breakdown.items():
        percentage = (count / len(top_user_tweets)) * 100
        print(f"   {sentiment.title()}: {count} tweets ({percentage:.1f}%)")
    
    # Show sample tweets from top user
    print(f"\n📝 SAMPLE #OMICRON TWEETS FROM @{top_user}:")
    print("-" * 60)
    
    # Show top 5 most engaging tweets
    top_engaging_tweets = top_user_tweets.nlargest(5, 'total_engagement')
    
    for i, (_, tweet) in enumerate(top_engaging_tweets.iterrows(), 1):
        print(f"\n{i}. Engagement: {int(tweet['total_engagement']):,} ({int(tweet['retweets'])} RT, {int(tweet['favorites'])} ❤️)")
        print(f"   Sentiment: {tweet['sentiment']}")
        print(f"   Date: {tweet['date']}")
        print(f"   Tweet: {tweet['text'][:150]}...")
    
    # Compare with overall dataset
    print(f"\n📊 COMPARISON WITH OVERALL DATASET")
    print("=" * 60)
    
    total_tweets_in_dataset = len(analyzer.df)
    total_omicron_tweets = len(omicron_tweets)
    top_user_percentage = (len(top_user_tweets) / total_omicron_tweets) * 100
    
    print(f"📈 Dataset Overview:")
    print(f"   Total tweets in dataset: {total_tweets_in_dataset:,}")
    print(f"   Total tweets with #omicron: {total_omicron_tweets:,} ({(total_omicron_tweets/total_tweets_in_dataset)*100:.1f}%)")
    print(f"   @{top_user}'s share of #omicron tweets: {top_user_percentage:.1f}%")
    print(f"   @{top_user}'s total tweets in dataset: {len(analyzer.df[analyzer.df['user_name'] == top_user])}")
    
    # Find what percentage of top user's tweets contain #omicron
    user_all_tweets = analyzer.df[analyzer.df['user_name'] == top_user]
    omicron_percentage = (len(top_user_tweets) / len(user_all_tweets)) * 100
    print(f"   Percentage of @{top_user}'s tweets with #omicron: {omicron_percentage:.1f}%")
    
    # Summary insights
    print(f"\n💡 KEY INSIGHTS")
    print("=" * 60)
    
    print(f"🏆 Highest #omicron hashtag user: @{top_user}")
    print(f"📝 They posted {len(top_user_tweets)} tweets with #omicron hashtag")
    print(f"📊 This represents {top_user_percentage:.1f}% of all #omicron hashtag tweets")
    print(f"🎯 {omicron_percentage:.1f}% of their tweets contain #omicron hashtag")
    
    if len(top_user_tweets) > 0:
        avg_engagement = top_user_tweets['total_engagement'].mean()
        dataset_avg = analyzer.df['total_engagement'].mean()
        if avg_engagement > dataset_avg:
            print(f"📈 Their #omicron tweets perform {(avg_engagement/dataset_avg):.1f}x better than average")
        else:
            print(f"📉 Their #omicron tweets perform {(dataset_avg/avg_engagement):.1f}x worse than average")
    
    return user_omicron_counts

if __name__ == "__main__":
    result = find_highest_omicron_hashtag_users()
