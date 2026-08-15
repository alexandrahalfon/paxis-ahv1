#!/usr/bin/env python3
"""
Validate that the repository is correctly set up for integrated processing and ingestion.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

def validate_setup():
    """Validate the complete setup."""
    print("🔍 Validating Paxis Knowledge Base Setup...")
    print("="*60)
    
    issues = []
    warnings = []
    
    # Check .env file
    env_file = Path(".env")
    if not env_file.exists():
        issues.append("❌ .env file not found")
    else:
        print("✅ .env file found")
        load_dotenv()
        
        # Check required API keys
        required_keys = [
            "OPENAI_API_KEY",
            "MISTRAL_API_KEY", 
            "QDRANT_URL"
        ]
        
        for key in required_keys:
            value = os.getenv(key)
            if not value or value == "your_api_key_here":
                issues.append(f"❌ {key} not set in .env")
            else:
                print(f"✅ {key} configured")
        
        # Check optional keys
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        if not qdrant_api_key or qdrant_api_key == "your_qdrant_api_key_here":
            warnings.append("⚠️ QDRANT_API_KEY not set (may be required for cloud Qdrant)")
    
    # Check backend structure
    src_dir = Path("src")
    if not src_dir.exists():
        issues.append("❌ src directory not found")
    else:
        print("✅ Backend structure found")
        
        # Check key modules
        key_modules = [
            "src/ingestion/colab_pipeline.py",
            "src/core/config.py",
            "data/keywords/extractor_keywords.json"
        ]
        
        for module in key_modules:
            if not Path(module).exists():
                issues.append(f"❌ {module} not found")
            else:
                print(f"✅ {Path(module).name} found")
    
    # Check dependencies
    try:
        import openai
        print("✅ openai package available")
    except ImportError:
        issues.append("❌ openai package not installed")
    
    try:
        import mistralai
        print("✅ mistralai package available")
    except ImportError:
        issues.append("❌ mistralai package not installed")
    
    try:
        from qdrant_client import QdrantClient
        print("✅ qdrant-client package available")
    except ImportError:
        issues.append("❌ qdrant-client package not installed")
    
    try:
        import tiktoken
        print("✅ tiktoken package available")
    except ImportError:
        issues.append("❌ tiktoken package not installed")
    
    # Test imports
    try:
        sys.path.insert(0, str(src_dir))
        from ingestion.colab_pipeline import ColabIngestionPipeline
        from core.config import get_settings
        print("✅ Backend modules can be imported")
    except ImportError as e:
        issues.append(f"❌ Backend import failed: {e}")
    
    # Check main processing file
    main_processor = Path("process_document_complete.py")
    if not main_processor.exists():
        issues.append("❌ process_document_complete.py not found")
    else:
        print("✅ Main processor found")
    
    # Print results
    print("\n" + "="*60)
    
    if issues:
        print("❌ SETUP ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
    
    if warnings:
        print("\n⚠️ WARNINGS:")
        for warning in warnings:
            print(f"  {warning}")
    
    if not issues:
        print("✅ SETUP VALIDATION SUCCESSFUL!")
        print("\n🎉 Your repository is correctly configured!")
        print("\nYou can now run:")
        print("  python process_document_complete.py path/to/document.pdf")
        print("\nThis will:")
        print("  1. Process the PDF document")
        print("  2. Extract all content and metadata")
        print("  3. Automatically ingest into vector database")
        
        auto_ingest = os.getenv('AUTO_INGEST', 'true').lower()
        if auto_ingest in ['true', '1', 'yes']:
            print("\n🚀 Auto-ingestion is ENABLED")
        else:
            print("\n💡 Auto-ingestion is DISABLED")
            print("   Set AUTO_INGEST=true in .env to enable")
    else:
        print(f"\n❌ Found {len(issues)} issues that need to be resolved.")
        print("\nTo fix:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Configure .env file with your API keys")
        print("3. Ensure backend structure is complete")
    
    return len(issues) == 0

if __name__ == "__main__":
    success = validate_setup()
    sys.exit(0 if success else 1)