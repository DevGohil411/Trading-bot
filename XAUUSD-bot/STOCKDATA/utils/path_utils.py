import os
import sys

def setup_project_paths():
    """Setup Python path to include project root."""
    # Get the absolute path to the project root (XAUUSD-bot folder)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Add project root to Python path if not already there
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Also add STOCKDATA to path for absolute imports
    stockdata_path = os.path.join(project_root, 'STOCKDATA')
    if stockdata_path not in sys.path:
        sys.path.insert(0, stockdata_path)
    
    return project_root, stockdata_path

# Call this function when the module is imported
PROJECT_ROOT, STOCKDATA_PATH = setup_project_paths()
