# Using GitHub Copilot with Omicron Sentiment Analysis

## 🤖 GitHub Copilot Integration Options

### Option 1: GitHub Copilot Chat (Recommended for Interactive Use)
If you have GitHub Copilot access in VS Code, you can use it directly:

1. **Open the project in VS Code**
2. **Use Copilot Chat** to ask questions about your data
3. **Example prompts for Copilot Chat:**
   ```
   Analyze the sentiment of tweets in omicron_2025.csv file
   Show me users who tweeted negatively about vaccines
   What are the trending hashtags in this dataset?
   Create a summary of the most concerning tweets about Omicron
   ```

### Option 2: OpenAI API (GitHub Copilot Compatible)
GitHub Copilot uses OpenAI models. You can use OpenAI API key with our system:

1. **Get OpenAI API Key:**
   - Go to [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
   - Create an API key
   - Add it to your `.env` file

2. **Set up your environment:**
   ```bash
   # Create .env file
   cp .env.example .env
   
   # Edit .env and add:
   OPENAI_API_KEY=your_openai_api_key_here
   LLM_PROVIDER=openai
   ```

3. **Install OpenAI dependencies:**
   ```bash
   pip install openai langchain-openai
   ```

### Option 3: Use Copilot as Code Assistant
Use GitHub Copilot to help you write custom analysis code:

```python
# Example: Ask Copilot to help generate analysis code
# Prompt: "Create a function to analyze tweet sentiment by location"

def analyze_sentiment_by_location(df):
    """Analyze sentiment distribution by user location"""
    # Copilot will help generate this code
    location_sentiment = df.groupby('user_location')['vader_sentiment'].apply(
        lambda x: x.apply(lambda s: s['label']).value_counts()
    )
    return location_sentiment

# Prompt: "Create a function to find most influential users"
def find_influential_users(df, top_n=10):
    """Find users with highest engagement"""
    df['total_engagement'] = df['retweets'] + df['favorites']
    return df.nlargest(top_n, 'total_engagement')[['user_name', 'total_engagement']]
```

## 🚀 Quick Setup for Copilot Integration

### Method 1: Using OpenAI API
```python
from omicron_sentiment_rag import OmicronSentimentRAG

# Initialize with OpenAI API (compatible with Copilot models)
analyzer = OmicronSentimentRAG(
    csv_path="omicron_2025.csv",
    openai_api_key="your_openai_api_key",
    llm_provider="openai"
)

# Now you can use AI-powered queries
response = analyzer.query_with_rag("What are the main concerns about Omicron?")
print(response)
```

### Method 2: Direct Copilot Chat Integration
```python
# Use this template with Copilot Chat in VS Code
"""
Context: I have a CSV file with Omicron tweets containing columns:
- user_name, user_location, date, text, hashtags, retweets, favorites

Task: Help me analyze the sentiment and find insights about:
1. Most negative sentiment tweets
2. Users most concerned about vaccines
3. Geographic distribution of concerns
4. Trending topics and hashtags

Data file: omicron_2025.csv
"""
```

## 📊 Copilot-Assisted Analysis Examples

### 1. Sentiment Analysis by Location
```python
# Ask Copilot: "Analyze sentiment by location for this tweet dataset"
import pandas as pd

df = pd.read_csv('omicron_2025.csv')

# Copilot-generated analysis
def analyze_location_sentiment(df):
    # Parse sentiment labels
    df['sentiment'] = df['vader_sentiment'].apply(lambda x: eval(x)['label'] if isinstance(x, str) else x['label'])
    
    # Group by location and sentiment
    location_analysis = df.groupby(['user_location', 'sentiment']).size().unstack(fill_value=0)
    
    # Calculate percentages
    location_percentages = location_analysis.div(location_analysis.sum(axis=1), axis=0) * 100
    
    return location_percentages

result = analyze_location_sentiment(df)
print(result)
```

### 2. Find Most Concerning Tweets
```python
# Ask Copilot: "Find tweets with most negative sentiment about vaccines"
def find_concerning_vaccine_tweets(df, limit=10):
    # Filter for vaccine-related tweets
    vaccine_tweets = df[df['text'].str.contains('vaccine|vaccination|vaccinated', case=False, na=False)]
    
    # Get sentiment scores
    vaccine_tweets['compound_score'] = vaccine_tweets['vader_sentiment'].apply(
        lambda x: eval(x)['compound'] if isinstance(x, str) else x['compound']
    )
    
    # Sort by most negative
    most_negative = vaccine_tweets.nsmallest(limit, 'compound_score')
    
    return most_negative[['user_name', 'text', 'compound_score', 'retweets', 'favorites']]
```

### 3. Hashtag Trend Analysis
```python
# Ask Copilot: "Analyze hashtag trends over time"
def analyze_hashtag_trends(df):
    # Parse hashtags and dates
    df['date'] = pd.to_datetime(df['date'])
    df['hashtags_list'] = df['hashtags'].apply(eval)
    
    # Expand hashtags to individual rows
    hashtag_data = []
    for _, row in df.iterrows():
        for hashtag in row['hashtags_list']:
            hashtag_data.append({
                'date': row['date'],
                'hashtag': hashtag,
                'user': row['user_name']
            })
    
    hashtag_df = pd.DataFrame(hashtag_data)
    
    # Daily hashtag counts
    daily_trends = hashtag_df.groupby(['date', 'hashtag']).size().unstack(fill_value=0)
    
    return daily_trends
```

## 💡 Pro Tips for Using Copilot

1. **Be Specific in Prompts:**
   ```
   ❌ "Analyze data"
   ✅ "Analyze sentiment distribution by user location for negative tweets about vaccines"
   ```

2. **Provide Context:**
   ```
   "I have a Twitter dataset about Omicron with sentiment analysis. 
   Help me find users who are most worried about hospital capacity."
   ```

3. **Ask for Code Explanations:**
   ```
   "Explain this sentiment analysis code and suggest improvements"
   ```

4. **Request Multiple Solutions:**
   ```
   "Show me 3 different ways to visualize tweet sentiment over time"
   ```

## 🔧 Integration Code Template

```python
# Template for using Copilot with your analysis
import pandas as pd
from omicron_sentiment_rag import OmicronSentimentRAG

# Load your analyzer
analyzer = OmicronSentimentRAG("omicron_2025.csv")

# Use Copilot to help with custom analysis
def copilot_assisted_analysis():
    """
    Use this function as a template for Copilot-assisted analysis.
    Ask Copilot to help you fill in the analysis logic.
    """
    
    # Example: Ask Copilot to help analyze user engagement patterns
    # Prompt: "Analyze which users have highest engagement on negative vs positive tweets"
    
    # Your Copilot-generated code here
    pass

# Interactive Copilot session
def interactive_analysis():
    """
    Use this for interactive analysis with Copilot Chat.
    Copy and paste this into Copilot Chat with your specific questions.
    """
    
    # Data overview
    print(f"Dataset: {len(analyzer.df)} tweets")
    print(f"Columns: {analyzer.df.columns.tolist()}")
    
    # Ask Copilot for specific analysis based on your needs
    # Example questions:
    # - "What are the most retweeted negative sentiment tweets?"
    # - "Which locations have the most vaccine hesitancy?"
    # - "What are the trending concerns about Omicron severity?"
    
    return analyzer.df

# Run the analysis
if __name__ == "__main__":
    interactive_analysis()
```

This way, you can leverage GitHub Copilot's capabilities while still using the structured analysis framework we've built!
