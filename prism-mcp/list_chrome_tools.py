import subprocess
import json
import time

def setup_mcp(command):
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

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
    
    while True:
        line = process.stdout.readline()
        if not line: break
        resp = json.loads(line)
        if resp.get("id") == 1:
            initialized_notif = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }
            process.stdin.write(json.dumps(initialized_notif) + "\n")
            process.stdin.flush()
            break
    
    return process

def list_tools(process):
    call_request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    }
    process.stdin.write(json.dumps(call_request) + "\n")
    process.stdin.flush()

    result = None
    while True:
        line = process.stdout.readline()
        if not line: break
        resp = json.loads(line)
        if resp.get("id") == 2:
            result = resp.get("result")
            break
    
    return result

if __name__ == "__main__":
    command = ["/usr/local/bin/npx", "-y", "chrome-devtools-mcp"]
    process = setup_mcp(command)
    
    tools = list_tools(process)
    print(json.dumps(tools, indent=2))
        
    process.terminate()
