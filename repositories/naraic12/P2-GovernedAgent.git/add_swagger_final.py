#!/usr/bin/env python3

# Read the file
with open('mission_control/mission_control.py', 'r') as f:
    lines = f.readlines()

# Add imports after the datetime import (around line 20)
new_lines = []
swagger_added = False
app_initialized = False

for i, line in enumerate(lines):
    new_lines.append(line)
    
    # Add Swagger import after datetime imports
    if 'from datetime import' in line and not swagger_added:
        new_lines.append('from flask import Flask, request, jsonify, render_template, session, Response\n')
        new_lines.append('from flasgger import Swagger\n')
        swagger_added = True
    
    # Add Swagger initialization after app = Flask
    if 'app = Flask' in line and swagger_added and not app_initialized:
        new_lines.append('\n# Initialize Swagger for API documentation\n')
        new_lines.append('swagger = Swagger(app)\n\n')
        app_initialized = True

# Write back
with open('mission_control/mission_control.py', 'w') as f:
    f.writelines(new_lines)

print(f"✅ Swagger added: {swagger_added}")
print(f"✅ App initialized: {app_initialized}")
