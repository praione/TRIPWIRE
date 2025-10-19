#!/usr/bin/env python3
"""
Add Swagger docstring to run_workflow endpoint
"""

# Read the file
with open('mission_control/mission_control.py', 'r') as f:
    lines = f.readlines()

# Find run_workflow and add docstring
new_lines = []
for i, line in enumerate(lines):
    new_lines.append(line)
    if i == 672 and 'def run_workflow():' in line:  # Line 673 is the function def
        # Add the docstring right after the function definition
        docstring = '''    """
    Execute an agent workflow from natural language intent
    ---
    tags:
      - Workflow
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            prompt:
              type: string
              example: "Create a customer support system"
              description: Natural language description of desired institution
    responses:
      200:
        description: Workflow initiated successfully
        schema:
          type: object
          properties:
            trace_id:
              type: string
              example: "trace_123e4567"
            status:
              type: string
              example: "initiated"
            team_id:
              type: string
              example: "team_de56ea1d"
      500:
        description: Workflow initiation failed
    """
'''
        new_lines.append(docstring)

# Write back
with open('mission_control/mission_control.py', 'w') as f:
    f.writelines(new_lines)

print("✅ Docstring added to run_workflow endpoint")
