"""
Update Registry Script - Adds the monitoring system agents to registry
"""
import json
from datetime import datetime

# Your actual monitoring system agents from today
monitoring_agents = {
    "agent_fc6244cf": {
        "role": "intake_coordinator_data_collection_setup",
        "status": "active",
        "input_queue": "pubsub://project/topics/agent_fc6244cf_input",
        "output_queue": "pubsub://project/topics/agent_fc6244cf_output",
        "constitutional_dna": "dna_2074ab6758b6486f"
    },
    "agent_38f81900": {
        "role": "analysis_specialist_anomaly_detection",
        "status": "active", 
        "input_queue": "pubsub://project/topics/agent_38f81900_input",
        "output_queue": "pubsub://project/topics/agent_38f81900_output",
        "constitutional_dna": "dna_ddb52116b15a4533"
    },
    "agent_a7e6533e": {
        "role": "notification_dispatcher_alert",
        "status": "active",
        "input_queue": "pubsub://project/topics/agent_a7e6533e_input",
        "output_queue": "pubsub://project/topics/agent_a7e6533e_output",
        "constitutional_dna": "dna_5b2c20b276574d1b"
    },
    "agent_c070b0a1": {
        "role": "analysis_specialist_dashboard",
        "status": "active",
        "input_queue": "pubsub://project/topics/agent_c070b0a1_input",
        "output_queue": "pubsub://project/topics/agent_c070b0a1_output",
        "constitutional_dna": "dna_ac78095889fb40c1"
    },
    "agent_37435175": {
        "role": "analysis_specialist_incident_response",
        "status": "active",
        "input_queue": "pubsub://project/topics/agent_37435175_input",
        "output_queue": "pubsub://project/topics/agent_37435175_output",
        "constitutional_dna": "dna_0aecbb1aa2df4f9e"
    }
}

# Create new registry structure
new_registry = monitoring_agents

# Save to file
with open('deployed_agents/agent_registry.json', 'w') as f:
    json.dump(new_registry, f, indent=2)

print("Registry updated with monitoring system agents:")
for agent_id in monitoring_agents:
    print(f"  - {agent_id}")

print("\nNow you can run:")
print("  python agent_runtime.py agent_fc6244cf")
print("  python agent_runtime.py agent_38f81900")
print("  etc...")