# --- client.py ---
# V1.2: Temporarily disabled ordering_key to match topic configuration.
# Publishes a task to the configured Pub/Sub topic.
# Supports passing a trace_id and an ordering_key for idempotency and ordering.

import os
import sys
import json
import argparse
from uuid import uuid4
from google.cloud import pubsub_v1

# --- Configuration ---
PROJECT_ID     = os.environ.get("GCP_PROJECT_ID", "project-resilience-ai-one")
TASKS_TOPIC_ID = os.environ.get("TASKS_TOPIC", "tasks-topic")

def publish_task(prompt: str, trace_id: str = None, ordering_key: str = None, is_risky: bool = False):
    """
    Publishes a single task message to the Pub/Sub topic with optional attributes.
    """
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(PROJECT_ID, TASKS_TOPIC_ID)

    # Generate a trace_id if one isn't provided
    trace_id = trace_id or str(uuid4())
    
    # NOTE: The ordering_key is received but not used in the publish call below
    # to avoid errors with topics that do not have message ordering enabled.
    ordering_key = ordering_key or trace_id

    payload = {
        "prompt": prompt,
        "trace_id": trace_id,
    }
    data = json.dumps(payload).encode("utf-8")

    attributes = {
        "trace_id": trace_id,
        "is_risky": str(is_risky).lower()
    }

    try:
        # The ordering_key parameter is removed from the call below.
        future = publisher.publish(
            topic_path,
            data,
            **attributes 
        )
        message_id = future.result()
        print(f"Published task with Trace ID: {trace_id}")
        print(f"  - Risky Flag: {is_risky}")
        print(f"  - Pub/Sub Message ID: {message_id}")
    except Exception as e:
        print(f"Error publishing task: {e}")
        sys.exit(1)

def main():
    """
    Parses command-line arguments and publishes the task.
    """
    parser = argparse.ArgumentParser(description="Publish a task to the Project Resilience dispatcher.")
    parser.add_argument("prompt", help="The user prompt for the agent colony.")
    parser.add_argument("--trace-id", help="Optional trace ID for idempotency.")
    parser.add_argument("--risky", action="store_true", help="Flag the task as risky to trigger the Guardian agent.")
    args = parser.parse_args()

    publish_task(
        prompt=args.prompt,
        trace_id=args.trace_id,
        is_risky=args.risky
    )

if __name__ == "__main__":
    main()

