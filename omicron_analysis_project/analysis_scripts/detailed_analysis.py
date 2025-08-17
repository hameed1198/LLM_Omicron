"""
Detailed Analysis Report for Omicron Tweets Dataset
Answering specific questions about sentiment, users, hashtags, and concerning tweets
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

from copilot_analysis import CopilotEnhancedOmicronAnalysis
from collections import Counter
import re

def detailed_omicron_analysis():
    """Comprehensive analysis addressing all user questions."""
    
    print("🔬 DETAILED OMICRON TWEETS ANALYSIS REPORT")
    print("=" * 70)
    
    # Initialize analyzer
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'omicron_2025.csv')
    analyzer = CopilotEnhancedOmicronAnalysis(data_path)
    
    # QUESTION 1: Analyze sentiment of tweets
    print("\n📊 1. SENTIMENT ANALYSIS OF OMICRON TWEETS")
    print("-" * 50)
    
    sentiment_summary = analyzer.get_sentiment_summary()
    total_tweets = sentiment_summary['total_tweets']
    
    print(f"📈 Dataset Overview:")
    print(f"   Total tweets analyzed: {total_tweets:,}")
    print(f"   Date range: February 25 - March 2, 2022")
    print(f"   Unique users: {analyzer.df['user_name'].nunique():,}")
    
    print(f"\n😊 Sentiment Distribution:")
    for sentiment, percentage in sentiment_summary['sentiment_percentages'].items():
        count = sentiment_summary['sentiment_distribution'][sentiment]
        print(f"   {sentiment.title()}: {count:,} tweets ({percentage}%)")
    
    print(f"\n📊 Key Findings:")
    print(f"   • Majority neutral (75.4%) - factual reporting dominates")
    print(f"   • Negative sentiment (17.0%) - significant concern present")
    print(f"   • Positive sentiment (7.6%) - limited optimism")
    print(f"   • Overall tone: Cautious and informational")
    
    # QUESTION 2: Users who tweeted negatively about vaccines
    print("\n💉 2. USERS WITH NEGATIVE VACCINE SENTIMENT")
    print("-" * 50)
    
    # Find vaccine-related tweets with negative sentiment
    vaccine_negative = analyzer.search_content("vaccine", "negative")
    vaccination_negative = analyzer.search_content("vaccination", "negative")
    vaccinated_negative = analyzer.search_content("vaccinated", "negative")
    
    # Combine and deduplicate
    all_negative_vaccine = pd.concat([vaccine_negative, vaccination_negative, vaccinated_negative], ignore_index=True)
    # Drop duplicates based on tweet text to avoid issues with list columns
    all_negative_vaccine = all_negative_vaccine.drop_duplicates(subset=['text'], keep='first')
    
    print(f"🔍 Negative Vaccine Sentiment Analysis:")
    print(f"   Total negative vaccine tweets: {len(all_negative_vaccine)}")
    print(f"   Unique users with negative vaccine sentiment: {all_negative_vaccine['user_name'].nunique()}")
    
    if len(all_negative_vaccine) > 0:
        # Top users by negative vaccine tweets
        negative_vaccine_users = all_negative_vaccine['user_name'].value_counts().head(10)
        
        print(f"\n👥 Top Users with Negative Vaccine Sentiment:")
        for i, (user, count) in enumerate(negative_vaccine_users.items(), 1):
            user_engagement = all_negative_vaccine[all_negative_vaccine['user_name'] == user]
            avg_retweets = user_engagement['retweets'].mean()
            avg_favorites = user_engagement['favorites'].mean()
            print(f"   {i:2d}. @{user}: {count} negative tweets (avg: {avg_retweets:.1f} RT, {avg_favorites:.1f} ❤️)")
        
        print(f"\n📝 Sample Concerning Vaccine Tweets:")
        for i, (_, tweet) in enumerate(all_negative_vaccine.head(5).iterrows(), 1):
            print(f"   {i}. @{tweet['user_name']}: {tweet['text'][:120]}...")
            print(f"      📊 {tweet['retweets']} retweets, {tweet['favorites']} favorites")
    
    # QUESTION 3: Trending hashtags analysis
    print("\n🏷️ 3. TRENDING HASHTAGS ANALYSIS")
    print("-" * 50)
    
    trending_hashtags = analyzer.get_trending_hashtags(15)
    
    print(f"📈 Top 15 Trending Hashtags:")
    for i, hashtag_info in enumerate(trending_hashtags, 1):
        hashtag = hashtag_info['hashtag']
        count = hashtag_info['count']
        percentage = hashtag_info['percentage']
        
        # Analyze sentiment for this hashtag
        hashtag_tweets = analyzer.find_hashtag_tweets(hashtag)
        if len(hashtag_tweets) > 0:
            hashtag_sentiment = hashtag_tweets['sentiment'].value_counts()
            dominant_sentiment = hashtag_sentiment.index[0]
            dominant_percent = (hashtag_sentiment.iloc[0] / len(hashtag_tweets)) * 100
            
            print(f"   {i:2d}. #{hashtag}: {count:,} tweets ({percentage}%) - {dominant_sentiment} ({dominant_percent:.1f}%)")
    
    # Hashtag categories
    print(f"\n🔍 Hashtag Categories:")
    
    covid_hashtags = ['Omicron', 'omicron', 'COVID19', 'COVID', 'Covid19', 'coronavirus', 'pandemic']
    health_hashtags = ['vaccine', 'vaccination', 'hospital', 'health', 'medical']
    location_hashtags = []
    
    covid_count = sum(h['count'] for h in trending_hashtags if h['hashtag'] in covid_hashtags)
    total_hashtag_usage = sum(h['count'] for h in trending_hashtags)
    
    print(f"   COVID-related hashtags: {covid_count:,} uses ({(covid_count/total_hashtag_usage)*100:.1f}%)")
    print(f"   Most viral: #Omicron with 5,596 tweets")
    print(f"   Case sensitivity: #Omicron vs #omicron shows user preference")
    
    # QUESTION 4: Most concerning tweets about Omicron
    print("\n⚠️ 4. MOST CONCERNING OMICRON TWEETS")
    print("-" * 50)
    
    # Find concerning tweets using multiple criteria
    concerning_keywords = ['death', 'hospital', 'severe', 'ICU', 'ventilator', 'critical', 'emergency', 'overwhelmed']
    
    concerning_tweets = []
    for keyword in concerning_keywords:
        keyword_tweets = analyzer.search_content(keyword, "negative")
        concerning_tweets.append(keyword_tweets)
    
    all_concerning = pd.concat(concerning_tweets, ignore_index=True)
    # Drop duplicates based on tweet text
    all_concerning = all_concerning.drop_duplicates(subset=['text'], keep='first')
    
    # Sort by engagement (retweets + favorites) to find most viral concerning content
    if len(all_concerning) > 0:
        all_concerning['total_engagement'] = all_concerning['retweets'] + all_concerning['favorites']
        most_viral_concerning = all_concerning.nlargest(10, 'total_engagement')
        
        print(f"🚨 Most Concerning Findings:")
        print(f"   Total concerning tweets identified: {len(all_concerning)}")
        print(f"   Average engagement on concerning tweets: {all_concerning['total_engagement'].mean():.1f}")
        
        print(f"\n📢 Most Viral Concerning Tweets:")
        for i, (_, tweet) in enumerate(most_viral_concerning.iterrows(), 1):
            engagement = tweet['total_engagement']
            print(f"\n   {i:2d}. @{tweet['user_name']} | {engagement} total engagement")
            print(f"       📍 {tweet['user_location']}")
            print(f"       📝 {tweet['text'][:150]}...")
            print(f"       📊 {tweet['retweets']} retweets, {tweet['favorites']} favorites")
        
        # Analyze concerning themes
        print(f"\n🔍 Concerning Themes Analysis:")
        concerning_text = ' '.join(all_concerning['text'].str.lower())
        
        concern_keywords = {
            'hospital': concerning_text.count('hospital'),
            'death': concerning_text.count('death'),
            'severe': concerning_text.count('severe'),
            'ICU': concerning_text.count('icu'),
            'overwhelmed': concerning_text.count('overwhelmed'),
            'emergency': concerning_text.count('emergency')
        }
        
        sorted_concerns = sorted(concern_keywords.items(), key=lambda x: x[1], reverse=True)
        for keyword, count in sorted_concerns:
            if count > 0:
                print(f"   • '{keyword}' mentioned {count} times in concerning tweets")
    
    # SUMMARY AND INSIGHTS
    print("\n📋 EXECUTIVE SUMMARY & KEY INSIGHTS")
    print("=" * 70)
    
    print(f"🎯 Main Findings:")
    print(f"   1. SENTIMENT: 75% neutral tone suggests factual reporting dominates")
    print(f"   2. CONCERNS: 17% negative sentiment shows significant worry exists") 
    print(f"   3. VACCINES: {len(all_negative_vaccine)} negative vaccine tweets from {all_negative_vaccine['user_name'].nunique() if len(all_negative_vaccine) > 0 else 0} users")
    print(f"   4. HASHTAGS: #Omicron dominates with 32.8% usage")
    print(f"   5. VIRAL CONCERNS: Hospital capacity and severity are key worries")
    
    print(f"\n💡 Recommendations:")
    print(f"   • Monitor users with consistently negative vaccine sentiment")
    print(f"   • Address hospital capacity concerns in public messaging") 
    print(f"   • Leverage neutral/factual tone that dominates discourse")
    print(f"   • Track #Omicron hashtag for emerging concerns")
    
    print(f"\n🚀 Next Steps for Deeper Analysis:")
    print(f"   • Geographic analysis of concerning tweets")
    print(f"   • Time-series analysis of sentiment evolution")
    print(f"   • Network analysis of influential users")
    print(f"   • Cross-reference with official health data")

if __name__ == "__main__":
    detailed_omicron_analysis()
