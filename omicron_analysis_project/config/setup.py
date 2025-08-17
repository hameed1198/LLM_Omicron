#!/usr/bin/env python3
"""
Setup script for Omicron Sentiment Analysis with RAG
This script helps set up the environment and install dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. Current version: {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible!")
    return True

def setup_virtual_environment():
    """Set up virtual environment if it doesn't exist."""
    venv_path = Path(".venv")
    if venv_path.exists():
        print("✅ Virtual environment already exists!")
        return True
    
    return run_command(f"{sys.executable} -m venv .venv", "Creating virtual environment")

def install_requirements(requirements_file="requirements.txt"):
    """Install requirements from specified file."""
    if not Path(requirements_file).exists():
        print(f"❌ {requirements_file} not found!")
        return False
    
    # Determine pip command based on OS
    if os.name == 'nt':  # Windows
        pip_cmd = ".venv\\Scripts\\pip.exe"
    else:  # macOS/Linux
        pip_cmd = ".venv/bin/pip"
    
    return run_command(f"{pip_cmd} install -r {requirements_file}", f"Installing packages from {requirements_file}")

def download_nltk_data():
    """Download required NLTK data."""
    print("📚 Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to download NLTK data: {e}")
        return False

def create_env_file():
    """Create .env file from template if it doesn't exist."""
    env_path = Path(".env")
    env_example_path = Path(".env.example")
    
    if env_path.exists():
        print("✅ .env file already exists!")
        return True
    
    if env_example_path.exists():
        try:
            with open(env_example_path, 'r') as src, open(env_path, 'w') as dst:
                dst.write(src.read())
            print("✅ Created .env file from template!")
            print("📝 Please edit .env file to add your API keys.")
            return True
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            return False
    else:
        print("⚠️ No .env.example template found. Skipping .env creation.")
        return True

def verify_installation():
    """Verify that key packages can be imported."""
    print("🔍 Verifying installation...")
    
    test_imports = [
        ("pandas", "Data processing"),
        ("numpy", "Numerical computing"),
        ("langchain", "LangChain framework"),
        ("textblob", "Text processing"),
        ("vadersentiment", "Sentiment analysis"),
        ("streamlit", "Web interface"),
        ("plotly", "Visualization")
    ]
    
    failed_imports = []
    
    for package, description in test_imports:
        try:
            __import__(package)
            print(f"✅ {description} ({package}) - OK")
        except ImportError:
            print(f"❌ {description} ({package}) - FAILED")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Installation verification failed for: {', '.join(failed_imports)}")
        return False
    
    print("\n✅ All packages imported successfully!")
    return True

def main():
    """Main setup function."""
    print("🦠 OMICRON SENTIMENT ANALYSIS SETUP")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Set up virtual environment
    if not setup_virtual_environment():
        print("❌ Failed to set up virtual environment!")
        sys.exit(1)
    
    # Choose installation type
    print("\n📦 Choose installation type:")
    print("1. Full installation (recommended)")
    print("2. Minimal installation (basic functionality)")
    print("3. Development installation (includes testing tools)")
    
    while True:
        choice = input("\nEnter choice (1-3): ").strip()
        if choice == "1":
            requirements_file = "requirements.txt"
            break
        elif choice == "2":
            requirements_file = "requirements-minimal.txt"
            break
        elif choice == "3":
            requirements_file = "requirements-dev.txt"
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")
    
    # Install requirements
    if not install_requirements(requirements_file):
        print("❌ Failed to install requirements!")
        sys.exit(1)
    
    # Download NLTK data
    download_nltk_data()
    
    # Create .env file
    create_env_file()
    
    # Verify installation
    if not verify_installation():
        print("❌ Installation verification failed!")
        sys.exit(1)
    
    # Success message
    print("\n🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Edit .env file to add your Anthropic API key (optional)")
    print("2. Test basic functionality: python simple_test.py")
    print("3. Run the demo: python demo.py")
    print("4. Launch web interface: streamlit run streamlit_app.py")
    
    print("\n📚 For more information, see README.md")

if __name__ == "__main__":
    main()
