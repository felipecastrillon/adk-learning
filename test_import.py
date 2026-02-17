import sys
import os
# Add the current directory to sys.path so we can import hello_agent
sys.path.append(os.getcwd())

print("Importing hello_agent...")
try:
    import hello_agent.agent
    print("Import successful")
except Exception as e:
    print(f"Error during import: {e}")
    import traceback
    traceback.print_exc()
