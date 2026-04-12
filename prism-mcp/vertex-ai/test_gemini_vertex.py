#!/usr/bin/env python3
"""
Test script for Gemini via Google Cloud Vertex AI
Model: gemini-1.5-flash-001
"""

import vertexai
from vertexai.generative_models import GenerativeModel
import os
import sys

# Configuration
# This will use the active gcloud config
# Uses the active gcloud configuration (project + credentials)

def test_gemini():
    print(f"Testing Gemini on Project: {os.environ.get('GOOGLE_CLOUD_PROJECT', 'Default from gcloud')}")
    
    try:
        # Initialize Vertex AI
        # We don't hardcode project here to let it pick up from gcloud
        vertexai.init()
        
        model = GenerativeModel("gemini-1.0-pro")
        
        print("Sending request to Gemini...", end=" ")
        response = model.generate_content("Say 'Gemini works!'")
        
        print("✅ SUCCESS!")
        print(f"Response: {response.text}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_gemini()
    sys.exit(0 if success else 1)
