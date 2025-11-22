import os
import sys

def setup_project_paths():
    # Get the directory of the current file (path_utils.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory (XAUUSD-bot/)
    project_root = os.path.dirname(current_dir)

    # Add the project root to sys.path if not already there
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"Added project root to sys.path: {project_root}")
    
    # Also add STOCKDATA directory if it's considered a primary working directory
    # Although when running with -m, STOCKDATA is implicitly handled,
    # adding it explicitly can prevent some relative import issues.
    stockdata_dir = os.path.join(project_root, 'STOCKDATA')
    if stockdata_dir not in sys.path:
        sys.path.insert(0, stockdata_dir)
        print(f"Added STOCKDATA to sys.path: {stockdata_dir}")

    # You might also want to add the modules directory if modules are directly imported
    modules_dir = os.path.join(stockdata_dir, 'modules')
    if modules_dir not in sys.path:
        sys.path.insert(0, modules_dir)
        print(f"Added modules to sys.path: {modules_dir}")

    print(f"Current sys.path: {sys.path}")

# Note: You generally don't call setup_project_paths() directly within path_utils.py.
# It's intended to be imported and called by main entry points.