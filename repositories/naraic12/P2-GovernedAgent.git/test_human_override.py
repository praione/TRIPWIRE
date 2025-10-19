import json
from google.cloud import pubsub_v1

# Setup
project_id = "project-resilience-ai-one"
topic_id = "governance-topic"
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(project_id, topic_id)

# The test message - simulates human pressing emergency stop
message = {
    "command": "HUMAN_OVERRIDE_STOP",
    "trace_id": "test_override_001",
    "agent": "governed_agent"
}

# Send it
message_json = json.dumps(message)
message_bytes = message_json.encode('utf-8')
future = publisher.publish(topic_path, message_bytes)
print(f"Sent human override command: {future.result()}")
print("Check dispatcher output to see Tier 1 dissonance trigger!")