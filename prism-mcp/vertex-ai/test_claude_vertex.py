#!/usr/bin/env python3
"""
Test script for Claude Sonnet 4.5 via Google Cloud Vertex AI

Model: claude-sonnet-4-5@20250929
Project: Set via GOOGLE_CLOUD_PROJECT env var
"""

import anthropic
import sys
import os

# Configuration
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "<your-gcp-project-id>")

# Models to try
MODELS = [
    "claude-3-5-haiku@20241022",
    "claude-3-haiku@20240307",
    "claude-3-5-sonnet-v2@20241022",
    "claude-3-5-sonnet@20240620"
]

# Regions to try
REGIONS = [
    "us-east5",
    "us-central1",
    "europe-west1",
    "us-east1"
]

# Regions to try
REGIONS = ["us-east5", "us-central1", "europe-west1", "us-east1"]

def test_claude_vertex():
    """Test Claude Sonnet 4.5 via Vertex AI"""
    
    for model in MODELS:
        print(f"Testing Model: {model}")
        for region in REGIONS:
            print(f"  Trying region={region}...", end=" ")
            try:
                client = anthropic.AnthropicVertex(
                    project_id=PROJECT_ID,
                    region=region,
                )
                
                message = client.messages.create(
                    model=model,
                    max_tokens=256,
                    messages=[{"role": "user", "content": "Hello! Say 'Claude on Vertex AI works!'."}]
                )
                
                print("✅ SUCCESS!")
                print("=" * 60)
                print(f"Model: {message.model}")
                print(f"Region: {region}")
                print(f"Response: {message.content[0].text}")
                print("=" * 60)
                return True
                
            except anthropic.NotFoundError:
                print(f"❌ Not found")
            except anthropic.PermissionDeniedError as e:
                print(f"❌ Forbidden (403): {e.body.get('error', {}).get('message')}")
            except Exception as e:
                print(f"❌ Error: {str(e)[:200]}")
    
    print("\n⚠️ No models accessible in any region.")
    return False
    
    print("\n⚠️ Model not accessible in any region.")
    print("Check that Claude Sonnet 4.5 is fully enabled in Vertex AI Model Garden.")
    return False

if __name__ == "__main__":
    success = test_claude_vertex()
    sys.exit(0 if success else 1)
