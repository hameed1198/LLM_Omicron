import re

def remove_verbose_prints():
    with open('core/omicron_sentiment_rag.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace verbose print statements with comments
    replacements = [
        ('print("🤖 Initializing Together AI (FREE tier)...")', '# Initializing Together AI'),
        ('print("🤖 Initializing Cohere (FREE tier)...")', '# Initializing Cohere'),
        ('print("🤖 Initializing HuggingFace Local Model (FREE)...")', '# Initializing HuggingFace'),
        ('print("🤖 Initializing OpenAI GPT...")', '# Initializing OpenAI GPT'),
        ('print("🤖 Initializing Ollama (local)...")', '# Initializing Ollama'),
        ('print("🔍 Auto-detecting available LLM providers...")', '# Auto-detecting providers'),
        ('print("🤖 Auto-detected: Using Google Gemini (FREE)...")', '# Using Google Gemini'),
        ('print("🤖 Auto-detected: Using Together AI (FREE tier)...")', '# Using Together AI'),
    ]
    
    for old, new in replacements:
        content = content.replace(old, new)
    
    # Remove print statements with f-strings
    content = re.sub(r'print\(f"⚠️ [^"]*"\)', '# Initialization failed', content)
    content = re.sub(r'print\(f"[^"]*"\)', '# Output suppressed', content)
    
    with open('core/omicron_sentiment_rag.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Cleaned up verbose output from RAG system")

if __name__ == "__main__":
    remove_verbose_prints()
