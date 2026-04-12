import subprocess
import json
import time
from pathlib import Path

def call_tool(process, tool_name, arguments={}):
    # Call tool
    call_request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        }
    }
    process.stdin.write(json.dumps(call_request) + "\n")
    process.stdin.flush()

    # Read result
    result = None
    while True:
        line = process.stdout.readline()
        if not line: break
        resp = json.loads(line)
        if resp.get("id") == 2:
            result = resp.get("result")
            break
    
    return result

def setup_mcp(command):
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # MCP Initialize
    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "python-client", "version": "1.0"}
        }
    }
    process.stdin.write(json.dumps(init_request) + "\n")
    process.stdin.flush()
    
    # Read responses until initialized
    while True:
        line = process.stdout.readline()
        if not line: break
        resp = json.loads(line)
        if resp.get("id") == 1:
            # Send initialized notification
            initialized_notif = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }
            process.stdin.write(json.dumps(initialized_notif) + "\n")
            process.stdin.flush()
            break
    
    return process

if __name__ == "__main__":
    command = ["npx", "-y", "chrome-devtools-mcp"]
    process = setup_mcp(command)
    
    # Navigate to Gmail
    print("Navigating to Gmail...")
    nav = call_tool(process, "navigate_page", {"url": "https://workspace.google.com/intl/en-US/gmail/"})
    print("Navigation complete.")
    
    time.sleep(2) # Wait for page load
    
    # Take snapshot
    print("Taking snapshot...")
    snapshot = call_tool(process, "take_snapshot")
    
    if snapshot:
        # Save snapshot for transcription
        output_path = Path.home() / "gmail_snapshot.json"
        with open(output_path, "w") as f:
            json.dump(snapshot, f, indent=2)
        print(f"Snapshot saved to {output_path}")
    else:
        print("Failed to take snapshot.")
        
    process.terminate()
