"""
Simple analysis to find users with highest #omicron hashtag tweets
"""

import pandas as pd
import re
import os

def find_omicron_hashtag_users():
    """Find users with the most tweets containing #omicron hashtag."""
    
    print("🔍 FINDING USERS WITH HIGHEST #OMICRON HASHTAG TWEETS")
    print("=" * 60)
    
    # Load the data
    try:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'omicron_2025.csv')
        df = pd.read_csv(data_path)
        print(f"✅ Loaded {len(df):,} tweets from omicron_2025.csv")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Find tweets containing #omicron hashtag (case insensitive)
    omicron_pattern = r'#omicron\b'
    omicron_mask = df['text'].str.contains(omicron_pattern, case=False, na=False, regex=True)
    omicron_tweets = df[omicron_mask].copy()
    
    print(f"📊 Found {len(omicron_tweets):,} tweets containing #omicron hashtag")
    print(f"📈 This represents {(len(omicron_tweets)/len(df))*100:.1f}% of all tweets")
    
    if len(omicron_tweets) == 0:
        print("❌ No tweets found with #omicron hashtag")
        return
    
    # Count tweets per user
    user_counts = omicron_tweets['user_name'].value_counts()
    
    print(f"\n🏆 TOP 15 USERS WITH MOST #OMICRON HASHTAG TWEETS")
    print("-" * 60)
    
    for i, (username, count) in enumerate(user_counts.head(15).items(), 1):
        # Get user's details from their omicron tweets
        user_tweets = omicron_tweets[omicron_tweets['user_name'] == username]
        
        # Calculate engagement stats
        total_engagement = user_tweets['retweets'].sum() + user_tweets['favorites'].sum()
        avg_engagement = total_engagement / len(user_tweets)
        
        # Get user location (from first tweet)
        location = user_tweets['user_location'].iloc[0] if not user_tweets['user_location'].isna().all() else 'Unknown'
        
        # Calculate percentage of all omicron tweets
        percentage = (count / len(omicron_tweets)) * 100
        
        print(f"{i:2d}. @{username}")
        print(f"    📝 #Omicron tweets: {count} ({percentage:.1f}% of all #omicron tweets)")
        print(f"    📍 Location: {location}")
        print(f"    📊 Total engagement: {int(total_engagement):,} (avg: {avg_engagement:.1f} per tweet)")
        print(f"    🔄 Total retweets: {user_tweets['retweets'].sum():,}")
        print(f"    ❤️  Total favorites: {user_tweets['favorites'].sum():,}")
        print()
    
    # Detailed analysis of the top user
    top_user = user_counts.index[0]
    top_count = user_counts.iloc[0]
    
    print(f"🎯 WINNER: @{top_user}")
    print("=" * 60)
    print(f"👑 User with highest #omicron hashtag tweets: @{top_user}")
    print(f"📝 Total #omicron tweets: {top_count}")
    print(f"📊 Percentage of all #omicron tweets: {(top_count/len(omicron_tweets))*100:.1f}%")
    
    # Get all tweets from this user to see their overall activity
    all_user_tweets = df[df['user_name'] == top_user]
    omicron_percentage = (top_count / len(all_user_tweets)) * 100
    
    print(f"📈 Total tweets by @{top_user} in dataset: {len(all_user_tweets)}")
    print(f"🎯 Percentage of their tweets with #omicron: {omicron_percentage:.1f}%")
    
    # Show sample tweets
    print(f"\n📝 SAMPLE #OMICRON TWEETS FROM @{top_user}:")
    print("-" * 60)
    
    top_user_omicron_tweets = omicron_tweets[omicron_tweets['user_name'] == top_user]
    
    # Show 3 most engaging tweets
    sample_tweets = top_user_omicron_tweets.nlargest(3, ['retweets', 'favorites'])
    
    for i, (_, tweet) in enumerate(sample_tweets.iterrows(), 1):
        engagement = tweet['retweets'] + tweet['favorites']
        print(f"\n{i}. Engagement: {int(engagement):,} ({int(tweet['retweets'])} RT + {int(tweet['favorites'])} ❤️)")
        print(f"   Date: {tweet['date']}")
        print(f"   Tweet: {tweet['text'][:200]}...")
    
    return user_counts

if __name__ == "__main__":
    result = find_omicron_hashtag_users()
