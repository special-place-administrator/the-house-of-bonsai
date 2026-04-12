import os
import sys
import site
import re
from pathlib import Path

def patch_cgc():
    try:
        # Find codegraphcontext installation
        import codegraphcontext
        base_path = Path(codegraphcontext.__file__).parent
        
        main_py = base_path / "cli" / "main.py"
        server_py = base_path / "server.py"

        print(f"Patching CodeGraphContext at {base_path}...")

        # 1. Patch main.py to remove stdout pollution
        if main_py.exists():
            with open(main_py, 'r') as f:
                content = f.read()
            
            if 'console.print("[bold green]Starting CodeGraphContext Server...[/bold green]")' in content:
                content = content.replace(
                    'console.print("[bold green]Starting CodeGraphContext Server...[/bold green]")',
                    'import sys\n    print("Starting CodeGraphContext Server...", file=sys.stderr, flush=True)'
                )
                content = content.replace(
                    'console.print(f"[bold red]Configuration Error:[/bold red] {e}")',
                    'print(f"Configuration Error: {e}", file=sys.stderr, flush=True)'
                )
                content = content.replace(
                    'console.print("Please run `cgc neo4j setup` or use FalkorDB (default).")',
                    'print("Please run `cgc neo4j setup` or use FalkorDB (default).", file=sys.stderr, flush=True)'
                )
                content = content.replace(
                    'console.print("\\n[bold yellow]Server stopped by user.[/bold yellow]")',
                    'print("\\nServer stopped by user.", file=sys.stderr, flush=True)'
                )
                with open(main_py, 'w') as f:
                    f.write(content)
                print("✅ Patched cli/main.py (stdout pollution fixed)")
            else:
                print("⏭️  cli/main.py already patched")

        # 2. Patch server.py to handle empty lines and binary unbuffered stdin
        if server_py.exists():
            with open(server_py, 'r') as f:
                content = f.read()

            if 'line = await loop.run_in_executor(None, sys.stdin.readline)' in content:
                replacement = """                # Read a request from the standard input (unbuffered binary to avoid deadlocks).
                loop_func = getattr(sys.stdin.buffer, 'readline', sys.stdin.readline)
                line_bytes = await loop.run_in_executor(None, loop_func)
                
                if not line_bytes:
                    debug_logger("Client disconnected (EOF received). Shutting down.")
                    break
                
                # Decode bytes to string, or just use it directly if it's already a string (fallback)
                line = line_bytes.decode('utf-8') if isinstance(line_bytes, bytes) else line_bytes
                
                stripped_line = line.strip()
                if not stripped_line:
                    continue  # Skip empty lines"""
                
                content = content.replace(
                    '                line = await loop.run_in_executor(None, sys.stdin.readline)\n                if not line:\n                    debug_logger("Client disconnected (EOF received). Shutting down.")\n                    break',
                    replacement
                )
                # replace the second json.loads to use stripped_line
                content = content.replace(
                    'request = json.loads(line.strip())',
                    'request = json.loads(stripped_line)'
                )

                with open(server_py, 'w') as f:
                    f.write(content)
                print("✅ Patched server.py (unbuffered stdin + empty line fix)")
            else:
                print("⏭️  server.py already patched")
                
        print("Done!")

    except Exception as e:
        print(f"Error patching: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    patch_cgc()
