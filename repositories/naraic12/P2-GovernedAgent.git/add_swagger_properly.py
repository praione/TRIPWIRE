#!/usr/bin/env python3
import sys

# Read the file
with open('mission_control/mission_control.py', 'r') as f:
    lines = f.readlines()

# Find the line with Flask import and add Swagger after it
new_lines = []
swagger_added = False

for i, line in enumerate(lines):
    new_lines.append(line)
    
    # Add after the Flask imports
    if 'from flask import' in line and not swagger_added:
        new_lines.append('from flasgger import Swagger\n')
        swagger_added = True
    
    # Add Swagger initialization after app = Flask(__name__)
    if 'app = Flask(__name__)' in line and swagger_added:
        new_lines.append('\n# Initialize Swagger\n')
        new_lines.append('swagger = Swagger(app)\n\n')

# Write back
with open('mission_control/mission_control.py', 'w') as f:
    f.writelines(new_lines)

print(f"✅ Swagger {'added' if swagger_added else 'already present'}")
