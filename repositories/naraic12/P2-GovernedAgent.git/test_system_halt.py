#!/usr/bin/env python3
"""End-to-end test of system.halt mechanism"""

import subprocess
import time
import json
from google.cloud import pubsub_v1
from datetime import datetime, timezone

PROJECT_ID = "project-resilience-ai-one"

def trigger_constitutional_crisis():
    """Trigger enough shares to cause quorum"""
    publisher = pubsub_v1.PublisherClient()
    
    # First, update arbiter with active shareholders
    arbiter_update = {
        "command": "update_shareholders",
        "agents": [f"agent_{i}" for i in range(10)]
    }
    
    # Publish shares from 6 out of 10 agents (60%)
    topic_path = publisher.topic_path(PROJECT_ID, "ledger-events")
    
    print("Triggering constitutional crisis...")
    for i in range(6):
        share = {
            "event_type": "subliminal_share",
            "source_agent": f"agent_{i}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "proof_hash": f"crisis_hash_{i}",
            "constitutional_state": {"dissonance": True}
        }
        publisher.publish(topic_path, json.dumps(share).encode('utf-8'))
        print(f"Agent_{i} in dissonance")
        time.sleep(1)
    
    print("\n60% QUORUM REACHED - Expecting system.halt...")

# Start dispatcher in background
print("Starting dispatcher...")
dispatcher_process = subprocess.Popen(
    ["python", "dispatcher.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Give it time to initialize
time.sleep(3)

# Start arbiter in background
print("Starting arbiter...")
arbiter_process = subprocess.Popen(
    ["python", "arbiter_agent.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Give it time to initialize
time.sleep(3)

# Trigger the crisis
trigger_constitutional_crisis()

# Wait and check if processes terminated
time.sleep(10)

# Check if dispatcher terminated
if dispatcher_process.poll() is not None:
    print("✓ DISPATCHER TERMINATED (system.halt successful)")
else:
    print("✗ Dispatcher still running (halt may have failed)")
    dispatcher_process.terminate()

if arbiter_process.poll() is not None:
    print("✓ Arbiter terminated")
else:
    print("Arbiter still running")
    arbiter_process.terminate()