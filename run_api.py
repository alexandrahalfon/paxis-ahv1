#!/usr/bin/env python3
"""
Paxis Phase 3 API Launcher

Quick launcher for the enhanced RAG API.
"""

import uvicorn
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
