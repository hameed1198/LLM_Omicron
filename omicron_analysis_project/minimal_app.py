import streamlit as st
import pandas as pd
import os

# Page config
st.set_page_config(
    page_title="Omicron Sentiment Analysis",
    page_icon="🦠",
    layout="wide"
)

def main():
    st.title("🦠 Omicron Tweets Sentiment Analysis")
    st.markdown("### Analyzing COVID-19 Omicron variant discussions on Twitter")
    
    # Check for data file
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'omicron_2025.csv')
    
    if os.path.exists(csv_path):
        st.success("✅ Data file found!")
        try:
            # Load data
            df = pd.read_csv(csv_path)
            st.write(f"📊 **Dataset loaded successfully!**")
            st.write(f"- **Total tweets**: {len(df):,}")
            st.write(f"- **Columns**: {', '.join(df.columns)}")
            
            # Show sample data
            st.subheader("📝 Sample Tweets")
            st.dataframe(df.head(10))
            
            # Basic analysis
            if 'sentiment' in df.columns:
                st.subheader("📈 Sentiment Distribution")
                sentiment_counts = df['sentiment'].value_counts()
                st.bar_chart(sentiment_counts)
            
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
    else:
        st.warning("⚠️ Data file not found. Showing demo information.")
        
        # Demo information
        st.info("""
        ## 📊 Project Overview
        
        This application analyzes **17,046 Twitter tweets** about the COVID-19 Omicron variant.
        
        **Features:**
        - 🤖 AI-Powered sentiment analysis
        - 📊 Interactive visualizations  
        - 📈 Hashtag trend analysis
        - 🔍 Tweet search and filtering
        
        **Technologies:**
        - Streamlit for web interface
        - Pandas for data processing
        - VADER & TextBlob for sentiment analysis
        - Google Gemini AI integration
        """)
        
        # Sample metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Tweets", "17,046")
        with col2:
            st.metric("Sentiment Methods", "2")
        with col3:
            st.metric("AI Providers", "5+")

if __name__ == "__main__":
    main()
