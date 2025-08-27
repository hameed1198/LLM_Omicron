import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import ast
import re
from pathlib import Path
import os

# Sentiment analysis libraries
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class SimpleSentimentAnalyzer:
    """
    Simplified sentiment analyzer that works without LangChain.
    Provides basic sentiment analysis and data exploration functionality.
    """
    
    def __init__(self, csv_path: str):
        """Initialize the simple sentiment analyzer."""
        self.csv_path = csv_path
        self.df = None
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.load_and_preprocess_data()
    
    def load_and_preprocess_data(self):
        """Load and preprocess the CSV data."""
        print("Loading and preprocessing data...")
        
        # Load the CSV file
        self.df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.df)} tweets successfully!")
        
        # Clean and preprocess text
        if 'text' in self.df.columns:
            self.df['clean_text'] = self.df['text'].apply(self.clean_text)
        
        # Perform sentiment analysis
        self.analyze_sentiment()
        
        print("Data preprocessing complete!")
    
    def clean_text(self, text):
        """Clean tweet text."""
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.lower().strip()
    
    def analyze_sentiment(self):
        """Perform sentiment analysis using VADER and TextBlob."""
        print("Performing sentiment analysis...")
        
        # VADER sentiment analysis
        vader_scores = []
        textblob_scores = []
        
        for text in self.df['text'].fillna(''):
            # VADER
            vader_score = self.vader_analyzer.polarity_scores(str(text))
            vader_scores.append(vader_score)
            
            # TextBlob
            blob = TextBlob(str(text))
            textblob_scores.append({
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            })
        
        # Add to dataframe
        self.df['vader_sentiment'] = vader_scores
        self.df['textblob_sentiment'] = textblob_scores
        
        # Create simple sentiment labels
        self.df['sentiment_label'] = self.df['vader_sentiment'].apply(
            lambda x: 'positive' if x['compound'] > 0.05 
            else 'negative' if x['compound'] < -0.05 
            else 'neutral'
        )
    
    def analyze_sentiment_distribution(self):
        """Analyze sentiment distribution."""
        sentiment_counts = self.df['sentiment_label'].value_counts().to_dict()
        
        return {
            'sentiment_distribution': sentiment_counts,
            'total_tweets': len(self.df),
            'average_compound_score': self.df['vader_sentiment'].apply(lambda x: x['compound']).mean()
        }
    
    def get_trending_hashtags(self, limit: int = 20):
        """Get trending hashtags."""
        if 'hashtags' not in self.df.columns:
            return []
        
        hashtag_counts = {}
        
        for hashtags in self.df['hashtags'].fillna(''):
            if hashtags and hashtags != '[]':
                try:
                    # Try to parse as list
                    if isinstance(hashtags, str):
                        hashtag_list = ast.literal_eval(hashtags)
                    else:
                        hashtag_list = hashtags
                    
                    if isinstance(hashtag_list, list):
                        for hashtag in hashtag_list:
                            hashtag = str(hashtag).lower().strip()
                            if hashtag:
                                hashtag_counts[hashtag] = hashtag_counts.get(hashtag, 0) + 1
                except:
                    # If parsing fails, treat as single hashtag
                    hashtag = str(hashtags).lower().strip()
                    if hashtag and hashtag != '[]':
                        hashtag_counts[hashtag] = hashtag_counts.get(hashtag, 0) + 1
        
        # Sort by count
        sorted_hashtags = sorted(hashtag_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [{'hashtag': tag, 'count': count} for tag, count in sorted_hashtags[:limit]]
    
    def query_tweets_by_hashtag(self, hashtag: str):
        """Query tweets by hashtag."""
        if 'hashtags' not in self.df.columns:
            return []
        
        hashtag = hashtag.lower().strip()
        matching_tweets = []
        
        for idx, row in self.df.iterrows():
            hashtags = row.get('hashtags', '')
            if hashtags and hashtags != '[]':
                try:
                    if isinstance(hashtags, str):
                        hashtag_list = ast.literal_eval(hashtags)
                    else:
                        hashtag_list = hashtags
                    
                    if isinstance(hashtag_list, list):
                        hashtag_list = [str(h).lower().strip() for h in hashtag_list]
                        if hashtag in hashtag_list:
                            matching_tweets.append(row.to_dict())
                except:
                    # If parsing fails, check if hashtag is in the string
                    if hashtag in str(hashtags).lower():
                        matching_tweets.append(row.to_dict())
        
        return matching_tweets
    
    def query_rag(self, query: str):
        """Simple text search since RAG is not available."""
        return {
            'answer': f"RAG functionality is not available. However, I can tell you that your dataset contains {len(self.df)} tweets about Omicron. You can explore the data using the dashboard features like sentiment analysis and hashtag trends.",
            'sources': []
        }
