# 🆓 FREE AI API Setup Guide

## Overview of FREE LLM Options

Your Omicron Sentiment Analysis project now supports multiple **FREE** AI providers as alternatives to the paid Claude Sonnet API. Here's a comprehensive guide to get you started with zero-cost AI capabilities.

---

## 🥇 **Option 1: Google Gemini (RECOMMENDED FREE)**

### ✅ **Why Choose Google Gemini:**
- **Completely FREE** with generous limits
- **High quality** responses (comparable to GPT-3.5)
- **60 requests per minute, 1500 requests per day**
- **Easy setup** with Google account

### 🔧 **Setup Instructions:**

1. **Get Your FREE API Key:**
   - Visit: [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Sign in with your Google account
   - Click "Create API Key"
   - Copy the generated key

2. **Configure in your .env file:**
   ```bash
   # Uncomment and add your key:
   GOOGLE_API_KEY=your_google_gemini_api_key_here
   ```

3. **Install required dependency:**
   ```bash
   pip install langchain-google-genai
   ```

4. **Test your setup:**
   ```python
   # The system will auto-detect and use Google Gemini
   analyzer = OmicronSentimentRAG("data/omicron_2025.csv")
   response = analyzer.query_with_rag("What are the main concerns about Omicron?")
   ```

---

## 🥈 **Option 2: Together AI (GREAT FREE TIER)**

### ✅ **Why Choose Together AI:**
- **$25 in FREE credits monthly**
- **Access to Llama 2 and other top models**
- **Fast inference**
- **Good for high-volume usage**

### 🔧 **Setup Instructions:**

1. **Get Your FREE API Key:**
   - Visit: [Together AI](https://api.together.xyz/)
   - Sign up for a free account
   - Go to your dashboard and copy your API key

2. **Configure in your .env file:**
   ```bash
   # Uncomment and add your key:
   TOGETHER_API_KEY=your_together_ai_api_key_here
   ```

3. **Install required dependency:**
   ```bash
   pip install together
   ```

---

## 🥉 **Option 3: Cohere (LIMITED FREE TIER)**

### ✅ **Why Choose Cohere:**
- **FREE tier available**
- **Good for basic text generation**
- **Easy integration**

### 🔧 **Setup Instructions:**

1. **Get Your FREE API Key:**
   - Visit: [Cohere Dashboard](https://dashboard.cohere.ai/)
   - Sign up for a free account
   - Generate your API key

2. **Configure in your .env file:**
   ```bash
   # Uncomment and add your key:
   COHERE_API_KEY=your_cohere_api_key_here
   ```

3. **Install required dependency:**
   ```bash
   pip install cohere
   ```

---

## 🏠 **Option 4: Local Models (100% FREE)**

### A. **Ollama (Recommended Local Option)**

#### ✅ **Advantages:**
- **Completely FREE** - no API keys needed
- **Privacy-focused** - runs on your computer
- **No internet required** after setup
- **Multiple model options**

#### 🔧 **Setup Instructions:**

1. **Install Ollama:**
   ```bash
   # Windows: Download from https://ollama.ai
   # Linux/Mac:
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **Download a model:**
   ```bash
   ollama pull llama2        # 7B model (~4GB)
   ollama pull mistral       # Alternative model
   ollama pull codellama     # Code-focused model
   ```

3. **Start the server:**
   ```bash
   ollama serve
   ```

4. **No .env configuration needed** - works automatically!

### B. **HuggingFace Local Models**

#### ✅ **Advantages:**
- **Completely FREE**
- **No separate installation**
- **Automatic model download**

#### 🔧 **Setup:**
- Set `LLM_PROVIDER=huggingface` in your .env file
- Models download automatically on first use

---

## ⚙️ **Configuration & Usage**

### **1. Set Your Preferred Provider**

Edit your `.env` file:
```bash
# Options: "auto", "google", "together", "cohere", "huggingface", "ollama"
LLM_PROVIDER=google  # Use Google Gemini specifically

# OR use auto-detection (tries FREE options first)
LLM_PROVIDER=auto
```

### **2. Priority Order (Auto Mode)**

When `LLM_PROVIDER=auto`, the system tries providers in this order:
1. 🆓 **Google Gemini** (if API key available)
2. 🆓 **Together AI** (if API key available)  
3. 🆓 **Cohere** (if API key available)
4. 🆓 **HuggingFace Local** (always available)
5. 🆓 **Ollama Local** (if installed and running)
6. 💰 **Claude Sonnet** (if API key available)
7. 💰 **OpenAI GPT** (if API key available)

### **3. Test Your Setup**

```python
from core.omicron_sentiment_rag import OmicronSentimentRAG

# Initialize with your preferred setup
analyzer = OmicronSentimentRAG("data/omicron_2025.csv")

# Test RAG functionality
response = analyzer.query_with_rag("What are people saying about vaccines?")
print(response)
```

---

## 📊 **Comparison of FREE Options**

| Provider | Cost | Quality | Speed | Limits | Setup Difficulty |
|----------|------|---------|-------|--------|------------------|
| **Google Gemini** | 🆓 FREE | ⭐⭐⭐⭐⭐ | ⚡⚡⚡⚡ | 1500/day | ⭐ Easy |
| **Together AI** | 🆓 $25/month | ⭐⭐⭐⭐ | ⚡⚡⚡⚡⚡ | High | ⭐ Easy |
| **Cohere** | 🆓 LIMITED | ⭐⭐⭐ | ⚡⚡⚡ | 100/month | ⭐ Easy |
| **Ollama Local** | 🆓 FREE | ⭐⭐⭐⭐ | ⚡⚡ | Unlimited | ⭐⭐ Medium |
| **HuggingFace** | 🆓 FREE | ⭐⭐ | ⚡ | Unlimited | ⭐ Easy |

---

## 🚀 **Quick Start with Google Gemini (Recommended)**

### **Step-by-Step:**

1. **Get FREE API Key:** Visit [Google AI Studio](https://makersuite.google.com/app/apikey)

2. **Install dependencies:**
   ```bash
   pip install langchain-google-genai
   ```

3. **Update your .env file:**
   ```bash
   GOOGLE_API_KEY=your_actual_key_here
   LLM_PROVIDER=google
   ```

4. **Run your Streamlit app:**
   ```bash
   cd omicron_analysis_project/web_app
   streamlit run streamlit_app.py
   ```

5. **Go to RAG Chat page** and ask: "What are the main concerns about Omicron?"

---

## 🔧 **Troubleshooting**

### **Common Issues:**

#### **"No LLM provider available"**
- Check that you've added an API key to your .env file
- Ensure you've installed the required dependencies
- Try setting `LLM_PROVIDER=auto` to test all options

#### **"Failed to initialize [Provider]"**
- Verify your API key is correct and active
- Check your internet connection
- Try a different provider as backup

#### **Rate limits exceeded**
- Google Gemini: Wait for daily reset or use multiple API keys
- Together AI: Monitor your $25 monthly credit usage
- Switch to local models (Ollama/HuggingFace) for unlimited usage

### **Performance Tips:**

- **For high usage:** Use Ollama locally
- **For best quality:** Use Google Gemini
- **For speed:** Use Together AI
- **For privacy:** Use local models only

---

## 🎯 **Recommended Setup for Different Users**

### **🆓 Cost-Conscious Users:**
```bash
GOOGLE_API_KEY=your_key_here
LLM_PROVIDER=auto  # Falls back to free options
```

### **🔒 Privacy-Focused Users:**
```bash
LLM_PROVIDER=ollama  # Everything runs locally
```

### **⚡ Performance Users:**
```bash
TOGETHER_API_KEY=your_key_here
LLM_PROVIDER=together
```

### **🔧 Developers/Testers:**
```bash
# Set multiple keys for testing
GOOGLE_API_KEY=your_key_here
TOGETHER_API_KEY=your_key_here
COHERE_API_KEY=your_key_here
LLM_PROVIDER=auto  # Auto-fallback between providers
```

---

Now you have multiple **FREE** alternatives to Claude Sonnet! Start with Google Gemini for the best free experience, and you can always add other providers as backups. 🚀
