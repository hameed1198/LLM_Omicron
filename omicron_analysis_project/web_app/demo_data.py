import streamlit as st
import pandas as pd
import numpy as np

def show_demo_data():
    """Show demo data if actual data is not available"""
    st.header("📊 Demo Data")
    st.info("Using demo data for demonstration purposes")
    
    # Create demo data
    demo_data = {
        'text': [
            "The omicron variant seems less severe than previous variants. #omicron #covid19",
            "Worried about omicron spreading so fast. Need more vaccines. #vaccination",
            "Omicron symptoms are mild compared to delta variant. Good news! #health",
            "The new variant is concerning but we'll get through this together. #community",
            "Omicron breakthrough infections are common but mild. #breakthrough"
        ],
        'sentiment': ['positive', 'negative', 'positive', 'neutral', 'positive'],
        'date': pd.date_range(start='2022-01-01', periods=5, freq='D')
    }
    
    df = pd.DataFrame(demo_data)
    
    st.subheader("Sample Tweets")
    st.dataframe(df)
    
    # Simple sentiment distribution
    sentiment_counts = df['sentiment'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sentiment Distribution")
        st.bar_chart(sentiment_counts)
    
    with col2:
        st.subheader("Statistics")
        st.metric("Total Tweets", len(df))
        st.metric("Positive %", f"{(sentiment_counts.get('positive', 0) / len(df) * 100):.1f}%")
        st.metric("Negative %", f"{(sentiment_counts.get('negative', 0) / len(df) * 100):.1f}%")
    
    st.info("This is demo data. In the full version, you would see analysis of 17,046 real tweets about the Omicron variant.")
    
    return df
