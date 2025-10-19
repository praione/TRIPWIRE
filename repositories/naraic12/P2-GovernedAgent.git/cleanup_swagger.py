#!/usr/bin/env python3

# Read the file
with open('mission_control/mission_control.py', 'r') as f:
    lines = f.readlines()

new_lines = []
flasgger_added = False
swagger_initialized = False

for line in lines:
    # Skip duplicate flasgger imports
    if 'from flasgger import Swagger' in line:
        if not flasgger_added:
            new_lines.append(line)
            flasgger_added = True
        continue
    
    # Skip duplicate swagger initializations
    if 'swagger = Swagger(app)' in line:
        if not swagger_initialized:
            new_lines.append(line)
            swagger_initialized = True
        continue
    
    # Keep all other lines
    new_lines.append(line)

# Write back
with open('mission_control/mission_control.py', 'w') as f:
    f.writelines(new_lines)

print(f"Cleaned up - kept 1 import and 1 initialization")
