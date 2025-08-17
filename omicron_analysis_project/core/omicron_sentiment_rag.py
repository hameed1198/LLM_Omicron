import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import ast
import re
from pathlib import Path

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic

# Additional LLM providers
try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain_community.llms import Ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Sentiment analysis imports
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings('ignore')

class OmicronSentimentRAG:
    """
    A comprehensive sentiment analysis system for Omicron tweets using LangChain and RAG.
    """
    
    def __init__(self, csv_path: str, anthropic_api_key: Optional[str] = None, 
                 openai_api_key: Optional[str] = None, llm_provider: str = "auto"):
        """
        Initialize the sentiment analysis system.
        
        Args:
            csv_path: Path to the CSV file containing tweet data
            anthropic_api_key: Anthropic API key for Claude Sonnet
            openai_api_key: OpenAI API key (for GPT models, GitHub Copilot, etc.)
            llm_provider: LLM provider to use ("claude", "openai", "ollama", "auto")
        """
        self.csv_path = csv_path
        self.df = None
        self.vector_store = None
        self.embeddings = None
        self.llm = None
        self.retrieval_chain = None
        self.llm_provider = llm_provider
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize LLM based on available API keys and provider preference
        self.llm = self._initialize_llm(anthropic_api_key, openai_api_key, llm_provider)
        
        # Load and preprocess data
        self.load_and_preprocess_data()
        self.create_vector_store()
        self.setup_retrieval_chain()
    
    def _initialize_llm(self, anthropic_api_key: Optional[str], openai_api_key: Optional[str], 
                       llm_provider: str):
        """Initialize the appropriate LLM based on available keys and preferences."""
        
        if llm_provider == "claude" and anthropic_api_key:
            print("🤖 Initializing Claude Sonnet...")
            return ChatAnthropic(
                model="claude-3-sonnet-20240229",
                anthropic_api_key=anthropic_api_key,
                temperature=0.1
            )
        
        elif llm_provider == "openai" and openai_api_key and OPENAI_AVAILABLE:
            print("🤖 Initializing OpenAI GPT...")
            return ChatOpenAI(
                model="gpt-3.5-turbo",  # You can change to "gpt-4" if you have access
                openai_api_key=openai_api_key,
                temperature=0.1
            )
        
        elif llm_provider == "ollama" and OLLAMA_AVAILABLE:
            print("🤖 Initializing Ollama (local)...")
            try:
                return Ollama(model="llama2")  # You can change to other local models
            except Exception as e:
                print(f"⚠️ Failed to initialize Ollama: {e}")
                return None
        
        elif llm_provider == "auto":
            # Auto-detect best available option
            if anthropic_api_key:
                print("🤖 Auto-detected: Using Claude Sonnet...")
                return ChatAnthropic(
                    model="claude-3-sonnet-20240229",
                    anthropic_api_key=anthropic_api_key,
                    temperature=0.1
                )
            elif openai_api_key and OPENAI_AVAILABLE:
                print("🤖 Auto-detected: Using OpenAI GPT...")
                return ChatOpenAI(
                    model="gpt-3.5-turbo",
                    openai_api_key=openai_api_key,
                    temperature=0.1
                )
            elif OLLAMA_AVAILABLE:
                print("🤖 Auto-detected: Trying Ollama (local)...")
                try:
                    return Ollama(model="llama2")
                except Exception as e:
                    print(f"⚠️ Ollama not available: {e}")
                    return None
            else:
                print("⚠️ No LLM provider available. RAG features will be disabled.")
                return None
        
        else:
            print(f"⚠️ LLM provider '{llm_provider}' not available or no API key provided.")
            return None
    
    def load_and_preprocess_data(self):
        """Load and preprocess the CSV data."""
        print("Loading and preprocessing data...")
        
        # Load CSV
        self.df = pd.read_csv(self.csv_path)
        
        # Clean and preprocess text
        self.df['text'] = self.df['text'].fillna('')
        self.df['hashtags'] = self.df['hashtags'].fillna('[]')
        
        # Parse hashtags from string representation of list
        def parse_hashtags(hashtag_str):
            try:
                if hashtag_str == '[]' or hashtag_str == '':
                    return []
                # Try to evaluate as literal list
                return ast.literal_eval(hashtag_str)
            except:
                # If that fails, try to extract hashtags from the string
                if isinstance(hashtag_str, str):
                    return [tag.strip("'\"") for tag in hashtag_str.strip('[]').split(',') if tag.strip()]
                return []
        
        self.df['hashtags_parsed'] = self.df['hashtags'].apply(parse_hashtags)
        
        # Clean text for analysis
        self.df['clean_text'] = self.df['text'].apply(self.clean_text)
        
        # Perform sentiment analysis
        self.df['textblob_sentiment'] = self.df['clean_text'].apply(self.get_textblob_sentiment)
        self.df['vader_sentiment'] = self.df['clean_text'].apply(self.get_vader_sentiment)
        
        print(f"Loaded {len(self.df)} tweets successfully!")
        print(f"Columns: {self.df.columns.tolist()}")
    
    def clean_text(self, text: str) -> str:
        """Clean tweet text for analysis."""
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove user mentions and hashtags for sentiment analysis
        text = re.sub(r'@\w+|#\w+', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def get_textblob_sentiment(self, text: str) -> Dict[str, float]:
        """Get sentiment using TextBlob."""
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity,
            'label': 'positive' if blob.sentiment.polarity > 0.1 else 'negative' if blob.sentiment.polarity < -0.1 else 'neutral'
        }
    
    def get_vader_sentiment(self, text: str) -> Dict[str, float]:
        """Get sentiment using VADER."""
        scores = self.vader_analyzer.polarity_scores(text)
        # Determine overall sentiment
        if scores['compound'] >= 0.05:
            label = 'positive'
        elif scores['compound'] <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        
        scores['label'] = label
        return scores
    
    def create_vector_store(self):
        """Create FAISS vector store from tweet data."""
        print("Creating vector store...")
        
        # Create documents for vector store
        documents = []
        for idx, row in self.df.iterrows():
            # Create comprehensive document content
            doc_content = f"""
            User: {row['user_name']}
            Location: {row['user_location']}
            Date: {row['date']}
            Tweet: {row['text']}
            Hashtags: {row['hashtags']}
            Retweets: {row['retweets']}
            Favorites: {row['favorites']}
            Sentiment: {row['vader_sentiment']['label']}
            """
            
            metadata = {
                'id': str(row['id']),
                'user_name': row['user_name'],
                'user_location': row['user_location'],
                'date': row['date'],
                'hashtags': row['hashtags_parsed'],
                'retweets': row['retweets'],
                'favorites': row['favorites'],
                'sentiment': row['vader_sentiment']['label'],
                'text': row['text']
            }
            
            documents.append(Document(page_content=doc_content, metadata=metadata))
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        split_docs = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vector_store = FAISS.from_documents(split_docs, self.embeddings)
        
        print(f"Created vector store with {len(split_docs)} document chunks!")
    
    def setup_retrieval_chain(self):
        """Setup the retrieval chain for RAG."""
        if not self.llm:
            print("Warning: No LLM configured. Some features will be limited.")
            return
        
        # Create custom prompt template
        prompt_template = """
        You are an expert data analyst specializing in social media sentiment analysis and tweet data.
        Use the following context from Omicron-related tweets to answer the user's question.
        
        Context:
        {context}
        
        Question: {question}
        
        Instructions:
        - Provide specific, data-driven answers based on the tweet context
        - Include relevant user names, tweet content, and statistics when applicable
        - For hashtag queries, list users and their tweets that contain the specified hashtags
        - For sentiment queries, provide sentiment analysis with supporting evidence
        - Format your response clearly with bullet points or lists when appropriate
        - Keep responses concise but informative
        
        Answer:
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retrieval chain
        self.retrieval_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 10}),
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        print("RAG retrieval chain setup complete!")
    
    def query_tweets_by_hashtag(self, hashtag: str) -> List[Dict]:
        """Query tweets containing a specific hashtag."""
        hashtag_lower = hashtag.lower().replace('#', '')
        
        matching_tweets = []
        for idx, row in self.df.iterrows():
            hashtags = row['hashtags_parsed']
            if any(hashtag_lower in tag.lower() for tag in hashtags):
                matching_tweets.append({
                    'user_name': row['user_name'],
                    'user_location': row['user_location'],
                    'date': row['date'],
                    'tweet': row['text'],
                    'hashtags': hashtags,
                    'sentiment': row['vader_sentiment']['label'],
                    'retweets': row['retweets'],
                    'favorites': row['favorites']
                })
        
        return matching_tweets
    
    def get_user_tweets(self, username: str) -> List[Dict]:
        """Get all tweets from a specific user."""
        user_tweets = self.df[self.df['user_name'].str.contains(username, case=False, na=False)]
        
        tweets = []
        for idx, row in user_tweets.iterrows():
            tweets.append({
                'user_name': row['user_name'],
                'date': row['date'],
                'tweet': row['text'],
                'hashtags': row['hashtags_parsed'],
                'sentiment': row['vader_sentiment']['label'],
                'retweets': row['retweets'],
                'favorites': row['favorites']
            })
        
        return tweets
    
    def analyze_sentiment_distribution(self) -> Dict[str, Any]:
        """Analyze overall sentiment distribution."""
        sentiment_counts = self.df['vader_sentiment'].apply(lambda x: x['label']).value_counts()
        
        return {
            'sentiment_distribution': sentiment_counts.to_dict(),
            'total_tweets': len(self.df),
            'average_compound_score': self.df['vader_sentiment'].apply(lambda x: x['compound']).mean(),
            'most_positive_tweet': self.df.loc[self.df['vader_sentiment'].apply(lambda x: x['compound']).idxmax()]['text'],
            'most_negative_tweet': self.df.loc[self.df['vader_sentiment'].apply(lambda x: x['compound']).idxmin()]['text']
        }
    
    def search_tweets_by_content(self, search_term: str, limit: int = 10) -> List[Dict]:
        """Search tweets by content using similarity."""
        # Simple text matching for now
        matching_tweets = self.df[self.df['text'].str.contains(search_term, case=False, na=False)]
        
        results = []
        for idx, row in matching_tweets.head(limit).iterrows():
            results.append({
                'user_name': row['user_name'],
                'date': row['date'],
                'tweet': row['text'],
                'hashtags': row['hashtags_parsed'],
                'sentiment': row['vader_sentiment']['label'],
                'retweets': row['retweets'],
                'favorites': row['favorites']
            })
        
        return results
    
    def query_with_rag(self, question: str) -> str:
        """Query using RAG if LLM is available."""
        if not self.retrieval_chain:
            return "RAG querying is not available. Please provide an Anthropic API key to enable this feature."
        
        try:
            response = self.retrieval_chain.run(question)
            return response
        except Exception as e:
            return f"Error processing RAG query: {str(e)}"
    
    def get_trending_hashtags(self, top_n: int = 10) -> List[Dict]:
        """Get most frequently used hashtags."""
        all_hashtags = []
        for hashtags in self.df['hashtags_parsed']:
            all_hashtags.extend(hashtags)
        
        # Count hashtags
        from collections import Counter
        hashtag_counts = Counter(all_hashtags)
        
        trending = []
        for hashtag, count in hashtag_counts.most_common(top_n):
            trending.append({
                'hashtag': hashtag,
                'count': count,
                'percentage': (count / len(self.df)) * 100
            })
        
        return trending
    
    def generate_report(self) -> str:
        """Generate a comprehensive analysis report."""
        report = []
        report.append("=== OMICRON TWEETS SENTIMENT ANALYSIS REPORT ===\n")
        
        # Basic statistics
        report.append(f"Total Tweets Analyzed: {len(self.df)}")
        report.append(f"Unique Users: {self.df['user_name'].nunique()}")
        report.append(f"Date Range: {self.df['date'].min()} to {self.df['date'].max()}\n")
        
        # Sentiment distribution
        sentiment_analysis = self.analyze_sentiment_distribution()
        report.append("Sentiment Distribution:")
        for sentiment, count in sentiment_analysis['sentiment_distribution'].items():
            percentage = (count / sentiment_analysis['total_tweets']) * 100
            report.append(f"  {sentiment.title()}: {count} ({percentage:.1f}%)")
        
        report.append(f"\nAverage Sentiment Score: {sentiment_analysis['average_compound_score']:.3f}")
        
        # Trending hashtags
        trending = self.get_trending_hashtags(5)
        report.append("\nTop 5 Hashtags:")
        for item in trending:
            report.append(f"  #{item['hashtag']}: {item['count']} tweets ({item['percentage']:.1f}%)")
        
        return "\n".join(report)

def main():
    """Main function to demonstrate the system."""
    # Initialize the system
    csv_path = "omicron_2025.csv"
    
    # Load environment variables
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    # Multiple API key options
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')  # GitHub Copilot can use OpenAI API
    
    # You can also set them directly for testing (not recommended for production)
    # anthropic_key = "your_anthropic_key_here"
    # openai_key = "your_openai_key_here"
    
    # Choose LLM provider: "claude", "openai", "ollama", or "auto"
    llm_provider = "auto"  # Let the system auto-detect the best option
    
    print("🦠 OMICRON SENTIMENT ANALYSIS WITH MULTI-LLM SUPPORT")
    print("=" * 60)
    print(f"Available API keys:")
    print(f"  - Anthropic (Claude): {'✅' if anthropic_key else '❌'}")
    print(f"  - OpenAI (GPT/Copilot): {'✅' if openai_key else '❌'}")
    print(f"  - Local Ollama: {'✅' if OLLAMA_AVAILABLE else '❌'}")
    print(f"Selected provider: {llm_provider}")
    
    try:
        analyzer = OmicronSentimentRAG(
            csv_path=csv_path, 
            anthropic_api_key=anthropic_key,
            openai_api_key=openai_key,
            llm_provider=llm_provider
        )
        
        print("\n" + "="*60)
        print("OMICRON SENTIMENT ANALYSIS SYSTEM INITIALIZED")
        print("="*60)
        
        # Generate and display report
        report = analyzer.generate_report()
        print(report)
        
        # Example queries
        print("\n" + "="*60)
        print("EXAMPLE QUERIES")
        print("="*60)
        
        # Query 1: Users who tweeted with hashtag omicron
        print("\n1. Users who tweeted with hashtag 'omicron':")
        omicron_tweets = analyzer.query_tweets_by_hashtag("omicron")
        print(f"Found {len(omicron_tweets)} tweets with #omicron")
        
        for i, tweet in enumerate(omicron_tweets[:5]):  # Show first 5
            print(f"\n  Tweet {i+1}:")
            print(f"    User: {tweet['user_name']}")
            print(f"    Location: {tweet['user_location']}")
            print(f"    Sentiment: {tweet['sentiment']}")
            print(f"    Tweet: {tweet['tweet'][:100]}...")
        
        # Query 2: Search for specific content
        print(f"\n2. Tweets mentioning 'vaccine':")
        vaccine_tweets = analyzer.search_tweets_by_content("vaccine", 3)
        for i, tweet in enumerate(vaccine_tweets):
            print(f"\n  Tweet {i+1}:")
            print(f"    User: {tweet['user_name']}")
            print(f"    Sentiment: {tweet['sentiment']}")
            print(f"    Tweet: {tweet['tweet'][:100]}...")
        
        # RAG demonstration
        if analyzer.llm:
            print(f"\n3. AI-Powered Analysis (using {analyzer.llm.__class__.__name__}):")
            sample_questions = [
                "What are the main concerns about Omicron in these tweets?",
                "Which users are most active in discussing vaccines?",
                "What is the overall sentiment about hospital capacity?"
            ]
            
            for question in sample_questions[:1]:  # Just show one example
                print(f"\n   Question: {question}")
                try:
                    response = analyzer.query_with_rag(question)
                    print(f"   AI Response: {response[:200]}...")
                except Exception as e:
                    print(f"   Error: {e}")
        
        # Interactive mode
        print("\n" + "="*60)
        print("INTERACTIVE MODE")
        print("="*60)
        print("You can now ask questions about the data!")
        print("Examples:")
        print("- 'list users with hashtag omicron'")
        print("- 'show negative sentiment tweets'")
        print("- 'find tweets mentioning hospital'")
        if analyzer.llm:
            print("- 'What are the main themes in negative tweets?' (AI-powered)")
        print("Type 'quit' to exit\n")
        
        while True:
            try:
                question = input("Ask a question: ").strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not question:
                    continue
                
                # Process the question
                if 'hashtag' in question.lower():
                    # Extract hashtag from question
                    words = question.split()
                    hashtag = None
                    for word in words:
                        if word.startswith('#') or word in ['omicron', 'covid', 'vaccine', 'hospital']:
                            hashtag = word.replace('#', '')
                            break
                    
                    if hashtag:
                        results = analyzer.query_tweets_by_hashtag(hashtag)
                        print(f"\nFound {len(results)} tweets with hashtag '{hashtag}':")
                        for i, tweet in enumerate(results[:3]):
                            print(f"\n  {i+1}. {tweet['user_name']}: {tweet['tweet'][:100]}...")
                    else:
                        print("Please specify a hashtag to search for.")
                
                elif 'sentiment' in question.lower():
                    sentiment_type = 'positive' if 'positive' in question.lower() else 'negative' if 'negative' in question.lower() else 'neutral'
                    sentiment_tweets = analyzer.df[analyzer.df['vader_sentiment'].apply(lambda x: x['label']) == sentiment_type]
                    print(f"\nShowing {sentiment_type} sentiment tweets:")
                    for i, (_, row) in enumerate(sentiment_tweets.head(3).iterrows()):
                        print(f"\n  {i+1}. {row['user_name']}: {row['text'][:100]}...")
                
                elif any(word in question.lower() for word in ['find', 'search', 'mention']):
                    # Extract search term
                    words = question.lower().split()
                    search_terms = ['hospital', 'vaccine', 'covid', 'death', 'mild', 'severe']
                    search_term = None
                    for term in search_terms:
                        if term in question.lower():
                            search_term = term
                            break
                    
                    if search_term:
                        results = analyzer.search_tweets_by_content(search_term, 3)
                        print(f"\nTweets mentioning '{search_term}':")
                        for i, tweet in enumerate(results):
                            print(f"\n  {i+1}. {tweet['user_name']}: {tweet['tweet'][:100]}...")
                    else:
                        print("Please specify what to search for (e.g., hospital, vaccine, etc.)")
                
                else:
                    # Try RAG if available
                    if analyzer.retrieval_chain:
                        response = analyzer.query_with_rag(question)
                        print(f"\nRAG Response: {response}")
                    else:
                        print("For advanced queries, please provide an Anthropic API key.")
                        print("Available simple queries: hashtag searches, sentiment analysis, content search")
            
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    except Exception as e:
        print(f"Error initializing system: {e}")

if __name__ == "__main__":
    main()
