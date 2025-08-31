"""
Test script to verify all imports work correctly
"""

import sys
import os
from pathlib import Path

# Add core to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / 'core'))

print("🔍 Testing imports...")

# Test basic imports
try:
    import pandas as pd
    print("✅ pandas")
except ImportError as e:
    print(f"❌ pandas: {e}")

try:
    import numpy as np
    print("✅ numpy")
except ImportError as e:
    print(f"❌ numpy: {e}")

try:
    import streamlit as st
    print("✅ streamlit")
except ImportError as e:
    print(f"❌ streamlit: {e}")

# Test visualization
try:
    import plotly.express as px
    print("✅ plotly")
except ImportError as e:
    print(f"❌ plotly: {e}")

try:
    from wordcloud import WordCloud
    print("✅ wordcloud")
except ImportError as e:
    print(f"❌ wordcloud: {e}")

# Test sentiment analysis
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    print("✅ vaderSentiment")
except ImportError as e:
    print(f"❌ vaderSentiment: {e}")

try:
    from textblob import TextBlob
    print("✅ textblob")
except ImportError as e:
    print(f"❌ textblob: {e}")

# Test custom modules
try:
    from simple_sentiment_analyzer import SimpleSentimentAnalyzer
    print("✅ SimpleSentimentAnalyzer")
except ImportError as e:
    print(f"❌ SimpleSentimentAnalyzer: {e}")

# Test data file
data_paths = [
    'data/omicron_2025.csv',
    'omicron_2025.csv'
]

data_found = False
for path in data_paths:
    if os.path.exists(path):
        print(f"✅ Data file found: {path}")
        data_found = True
        break

if not data_found:
    print("❌ Data file not found")

print("\n🏁 Test complete!")
