#!/usr/bin/env python3
"""
Add Swagger documentation to mission_control.py
"""

import sys

# Read the existing mission_control.py
with open('mission_control/mission_control.py', 'r') as f:
    lines = f.readlines()

# Find where Flask is imported and add Swagger import
new_lines = []
flask_import_found = False
app_created = False

for i, line in enumerate(lines):
    # Add Swagger import after Flask import
    if 'from flask import' in line and not flask_import_found:
        new_lines.append(line)
        new_lines.append('from flasgger import Swagger\n')
        flask_import_found = True
    # Add Swagger initialization after app = Flask(__name__)
    elif 'app = Flask(__name__)' in line and not app_created:
        new_lines.append(line)
        new_lines.append('\n# Swagger configuration\n')
        new_lines.append('swagger_config = {\n')
        new_lines.append('    "headers": [],\n')
        new_lines.append('    "specs": [{\n')
        new_lines.append('        "endpoint": "apispec",\n')
        new_lines.append('        "route": "/apispec.json",\n')
        new_lines.append('        "rule_filter": lambda rule: True,\n')
        new_lines.append('    }],\n')
        new_lines.append('    "static_url_path": "/flasgger_static",\n')
        new_lines.append('    "swagger_ui": True,\n')
        new_lines.append('    "specs_route": "/api/docs/"\n')
        new_lines.append('}\n\n')
        new_lines.append('swagger_template = {\n')
        new_lines.append('    "info": {\n')
        new_lines.append('        "title": "Project Resilience API",\n')
        new_lines.append('        "description": "Intent-to-Institution Engine with Constitutional Governance",\n')
        new_lines.append('        "version": "0.85.0",\n')
        new_lines.append('        "contact": {\n')
        new_lines.append('            "name": "Ciarán Doyle",\n')
        new_lines.append('            "email": "support@ciarandoyle.com"\n')
        new_lines.append('        }\n')
        new_lines.append('    },\n')
        new_lines.append('    "tags": [\n')
        new_lines.append('        {"name": "Workflow", "description": "Agent workflow operations"},\n')
        new_lines.append('        {"name": "Governance", "description": "Constitutional governance endpoints"},\n')
        new_lines.append('        {"name": "Health", "description": "System health and monitoring"},\n')
        new_lines.append('        {"name": "Registry", "description": "Agent registry operations"}\n')
        new_lines.append('    ]\n')
        new_lines.append('}\n\n')
        new_lines.append('swagger = Swagger(app, config=swagger_config, template=swagger_template)\n\n')
        app_created = True
    else:
        new_lines.append(line)

# Write the updated file
with open('mission_control/mission_control.py', 'w') as f:
    f.writelines(new_lines)

print("✅ Swagger configuration added to mission_control.py")
print("📝 Now adding docstrings to endpoints...")
