#!/bin/bash
set -euo pipefail

: "${BRAVE_API_KEY:?BRAVE_API_KEY is required}"
: "${GOOGLE_API_KEY:?GOOGLE_API_KEY is required}"

echo "Starting Brave-Gemini Research MCP Server..."
# Run the server
node dist/server.js
