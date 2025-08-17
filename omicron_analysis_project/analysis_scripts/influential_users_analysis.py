"""
Analysis of Most Influential Users and Their Sentiment
Identifying key opinion leaders in the Omicron discussion
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

from copilot_analysis import CopilotEnhancedOmicronAnalysis
from collections import Counter

def analyze_influential_users():
    """Identify and analyze the most influential users in the Omicron dataset."""
    
    print("👑 MOST INFLUENTIAL USERS ANALYSIS")
    print("=" * 60)
    
    # Initialize analyzer
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'omicron_2025.csv')
    analyzer = CopilotEnhancedOmicronAnalysis(data_path)
    df = analyzer.df.copy()
    
    # Calculate influence metrics
    print("📊 Calculating influence metrics...")
    
    # Add engagement metrics
    df['total_engagement'] = df['retweets'] + df['favorites']
    df['engagement_score'] = df['retweets'] * 2 + df['favorites']  # Weight retweets higher
    
    # User-level analysis
    print("\n🔍 Analyzing user-level influence...")
    
    user_stats = df.groupby('user_name').agg({
        'text': 'count',  # Number of tweets
        'retweets': ['sum', 'mean', 'max'],
        'favorites': ['sum', 'mean', 'max'],
        'total_engagement': ['sum', 'mean', 'max'],
        'engagement_score': ['sum', 'mean'],
        'sentiment': lambda x: x.value_counts().to_dict()
    }).round(2)
    
    # Flatten column names
    user_stats.columns = [
        'tweet_count', 'total_retweets', 'avg_retweets', 'max_retweets',
        'total_favorites', 'avg_favorites', 'max_favorites',
        'total_engagement', 'avg_engagement', 'max_engagement',
        'total_score', 'avg_score', 'sentiment_dist'
    ]
    
    # Calculate influence ranks
    user_stats['influence_rank'] = (
        user_stats['total_engagement'] * 0.4 +  # Total reach
        user_stats['avg_engagement'] * 0.3 +    # Consistency
        user_stats['tweet_count'] * 0.2 +       # Activity
        user_stats['max_engagement'] * 0.1      # Viral potential
    )
    
    # Sort by influence
    top_influencers = user_stats.sort_values('influence_rank', ascending=False).head(20)
    
    print(f"\n👑 TOP 20 MOST INFLUENTIAL USERS")
    print("-" * 60)
    
    for i, (username, stats) in enumerate(top_influencers.iterrows(), 1):
        sentiment_dist = stats['sentiment_dist']
        dominant_sentiment = max(sentiment_dist.keys(), key=lambda x: sentiment_dist[x]) if sentiment_dist else 'unknown'
        sentiment_count = sentiment_dist.get(dominant_sentiment, 0) if sentiment_dist else 0
        sentiment_percentage = (sentiment_count / stats['tweet_count'] * 100) if stats['tweet_count'] > 0 else 0
        
        print(f"\n{i:2d}. @{username}")
        print(f"    📊 Influence Score: {stats['influence_rank']:.1f}")
        print(f"    📝 Tweets: {int(stats['tweet_count'])}")
        print(f"    📈 Total Engagement: {int(stats['total_engagement']):,} (avg: {stats['avg_engagement']:.1f})")
        print(f"    🔄 Retweets: {int(stats['total_retweets']):,} (avg: {stats['avg_retweets']:.1f})")
        print(f"    ❤️  Favorites: {int(stats['total_favorites']):,} (avg: {stats['avg_favorites']:.1f})")
        print(f"    😊 Dominant Sentiment: {dominant_sentiment} ({sentiment_percentage:.1f}%)")
        if sentiment_dist:
            sentiment_breakdown = ", ".join([f"{k}: {v}" for k, v in sentiment_dist.items()])
            print(f"    📋 Sentiment Breakdown: {sentiment_breakdown}")
    
    # Analyze by sentiment category
    print(f"\n🎭 INFLUENTIAL USERS BY SENTIMENT CATEGORY")
    print("-" * 60)
    
    # Most influential positive users
    positive_users = df[df['sentiment'] == 'positive'].groupby('user_name').agg({
        'total_engagement': 'sum',
        'text': 'count'
    }).sort_values('total_engagement', ascending=False).head(5)
    
    print(f"\n😊 Most Influential POSITIVE Users:")
    for i, (username, stats) in enumerate(positive_users.iterrows(), 1):
        user_tweets = df[(df['user_name'] == username) & (df['sentiment'] == 'positive')]
        avg_engagement = stats['total_engagement'] / stats['text']
        print(f"   {i}. @{username}: {int(stats['total_engagement']):,} engagement, {int(stats['text'])} tweets (avg: {avg_engagement:.1f})")
    
    # Most influential negative users
    negative_users = df[df['sentiment'] == 'negative'].groupby('user_name').agg({
        'total_engagement': 'sum',
        'text': 'count'
    }).sort_values('total_engagement', ascending=False).head(5)
    
    print(f"\n😟 Most Influential NEGATIVE Users:")
    for i, (username, stats) in enumerate(negative_users.iterrows(), 1):
        user_tweets = df[(df['user_name'] == username) & (df['sentiment'] == 'negative')]
        avg_engagement = stats['total_engagement'] / stats['text']
        print(f"   {i}. @{username}: {int(stats['total_engagement']):,} engagement, {int(stats['text'])} tweets (avg: {avg_engagement:.1f})")
    
    # Most influential neutral users
    neutral_users = df[df['sentiment'] == 'neutral'].groupby('user_name').agg({
        'total_engagement': 'sum',
        'text': 'count'
    }).sort_values('total_engagement', ascending=False).head(5)
    
    print(f"\n😐 Most Influential NEUTRAL Users:")
    for i, (username, stats) in enumerate(neutral_users.iterrows(), 1):
        user_tweets = df[(df['user_name'] == username) & (df['sentiment'] == 'neutral')]
        avg_engagement = stats['total_engagement'] / stats['text']
        print(f"   {i}. @{username}: {int(stats['total_engagement']):,} engagement, {int(stats['text'])} tweets (avg: {avg_engagement:.1f})")
    
    # Analyze most viral individual tweets
    print(f"\n🚀 MOST VIRAL INDIVIDUAL TWEETS")
    print("-" * 60)
    
    most_viral = df.nlargest(10, 'total_engagement')
    
    for i, (_, tweet) in enumerate(most_viral.iterrows(), 1):
        print(f"\n{i:2d}. @{tweet['user_name']} | {tweet['sentiment']} sentiment")
        print(f"    📊 {int(tweet['total_engagement']):,} total engagement ({int(tweet['retweets'])} RT, {int(tweet['favorites'])} ❤️)")
        print(f"    📍 {tweet['user_location']}")
        print(f"    📝 {tweet['text'][:150]}...")
    
    # Geographic influence analysis
    print(f"\n🌍 INFLUENCE BY LOCATION")
    print("-" * 60)
    
    location_influence = df.groupby('user_location').agg({
        'total_engagement': 'sum',
        'text': 'count',
        'user_name': 'nunique'
    }).sort_values('total_engagement', ascending=False).head(10)
    
    location_influence.columns = ['total_engagement', 'tweet_count', 'unique_users']
    
    print(f"Top 10 Most Influential Locations:")
    for i, (location, stats) in enumerate(location_influence.iterrows(), 1):
        if pd.notna(location) and location.strip():
            avg_engagement = stats['total_engagement'] / stats['unique_users']
            print(f"   {i:2d}. {location}")
            print(f"       📊 {int(stats['total_engagement']):,} total engagement")
            print(f"       👥 {int(stats['unique_users'])} users, {int(stats['tweet_count'])} tweets")
            print(f"       📈 {avg_engagement:.1f} avg engagement per user")
    
    # Sentiment influence correlation
    print(f"\n📈 SENTIMENT & INFLUENCE CORRELATION")
    print("-" * 60)
    
    sentiment_engagement = df.groupby('sentiment').agg({
        'total_engagement': ['sum', 'mean', 'median'],
        'retweets': ['sum', 'mean'],
        'favorites': ['sum', 'mean'],
        'text': 'count'
    }).round(2)
    
    print(f"Engagement by Sentiment:")
    for sentiment in ['positive', 'neutral', 'negative']:
        if sentiment in sentiment_engagement.index:
            stats = sentiment_engagement.loc[sentiment]
            total_eng = stats[('total_engagement', 'sum')]
            avg_eng = stats[('total_engagement', 'mean')]
            tweet_count = stats[('text', 'count')]
            
            print(f"   {sentiment.title()}:")
            print(f"     Total: {int(total_eng):,} engagement from {int(tweet_count):,} tweets")
            print(f"     Average: {avg_eng:.1f} engagement per tweet")
    
    # Key insights summary
    print(f"\n💡 KEY INSIGHTS")
    print("=" * 60)
    
    top_user = top_influencers.index[0]
    top_stats = top_influencers.iloc[0]
    
    print(f"🏆 Most Influential User: @{top_user}")
    print(f"   📊 Influence Score: {top_stats['influence_rank']:.1f}")
    print(f"   📝 {int(top_stats['tweet_count'])} tweets with {int(top_stats['total_engagement']):,} total engagement")
    
    # Sentiment distribution among top influencers
    top_10_sentiments = []
    for username in top_influencers.head(10).index:
        user_sentiment = top_influencers.loc[username, 'sentiment_dist']
        if user_sentiment:
            dominant = max(user_sentiment.keys(), key=lambda x: user_sentiment[x])
            top_10_sentiments.append(dominant)
    
    sentiment_counter = Counter(top_10_sentiments)
    print(f"\n📊 Top 10 Influencers Sentiment Distribution:")
    for sentiment, count in sentiment_counter.items():
        print(f"   {sentiment.title()}: {count} users ({count/10*100:.1f}%)")
    
    print(f"\n🎯 Strategic Recommendations:")
    print(f"   • Monitor top 5 influencers for emerging narratives")
    print(f"   • Engage with neutral influencers to share factual information")
    print(f"   • Address concerns from negative influencers proactively")
    print(f"   • Leverage positive influencers for health messaging")
    
    return top_influencers

if __name__ == "__main__":
    influential_users = analyze_influential_users()
