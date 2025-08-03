"""
remove .py files after cythonized
"""
import os
import yaml


with open("cython_modules.yaml", "r") as f:
    data = yaml.safe_load(f)
    all_modules = data["python_files"]    # ex: src/already_cythonized.py

for path in all_modules:
    try:
        # remove .py modules
        os.remove(path)

        # remove temporary .c files
        c_code_generated = path[:-3] + '.c' if str(path).endswith('.py') else ""
        if os.path.exists(c_code_generated):
            os.remove(c_code_generated)     # remove .c files (gen by Cython)

    except Exception as e:
        print(f"Failed to delete {path}: {e}")
