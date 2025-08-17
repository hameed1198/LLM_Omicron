# Quick Installation Guide

## 🚀 Quick Start

### Method 1: Automatic Setup (Recommended)
```bash
python setup.py
```
Follow the prompts to choose your installation type.

### Method 2: Manual Installation

1. **Create virtual environment:**
```bash
python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate

# Activate on macOS/Linux:
source .venv/bin/activate
```

2. **Install packages:**
```bash
# Full installation (recommended):
pip install -r requirements.txt

# OR minimal installation:
pip install -r requirements-minimal.txt
```

3. **Set up environment (optional):**
```bash
cp .env.example .env
# Edit .env and add your Anthropic API key
```

## 📋 Requirements Files

- **requirements.txt** - Full installation with all features
- **requirements-minimal.txt** - Basic functionality only
- **requirements-dev.txt** - Development tools included

## 🧪 Test Installation

```bash
# Test basic functionality:
python simple_test.py

# Run full demo:
python demo.py

# Launch web interface:
streamlit run streamlit_app.py
```

## 🔧 Troubleshooting

### Common Issues:

1. **Module not found errors:**
   - Make sure virtual environment is activated
   - Reinstall requirements: `pip install -r requirements.txt`

2. **LangChain deprecation warnings:**
   - These are warnings, not errors - the code will still work
   - Future versions will update to newer imports

3. **NLTK data missing:**
   - Run: `python -c "import nltk; nltk.download('all')"`

4. **API key issues:**
   - RAG features require Anthropic API key in .env file
   - Basic features work without API key

## 📦 Package Versions

Key package versions used:
- pandas==2.3.1
- langchain==0.3.27
- streamlit==1.48.1
- anthropic==0.64.0
- textblob==0.19.0
- vaderSentiment==3.3.2

## 🆘 Support

If you encounter issues:
1. Check that Python 3.8+ is installed
2. Ensure virtual environment is activated
3. Try reinstalling requirements
4. Check README.md for detailed documentation
