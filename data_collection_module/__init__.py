# Verify __init__.py files exist in the package hierarchy
module_path = 'data_collection_module'
init_path = os.path.join(module_path, '__init__.py')
print(f"__init__.py exists: {os.path.exists(init_path)}")

# If missing, create it
if not os.path.exists(init_path):
    try:
        with open(init_path, 'w') as f:
            pass  # Create empty file
        print(f"Created missing {init_path}")
    except Exception as e:
        print(f"Error creating __init__.py: {e}")