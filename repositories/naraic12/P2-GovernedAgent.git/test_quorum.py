#!/usr/bin/env python3
"""Test the 60% quorum mechanism"""

import json
import time
from datetime import datetime
from google.cloud import pubsub_v1

PROJECT_ID = "project-resilience-ai-one"  # Replace with actual

def simulate_agent_shares(num_agents, num_dissonant):
    """Simulate agents publishing shares"""
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(PROJECT_ID, "events-topic")
    
    # Simulate active agents
    print(f"Simulating {num_agents} active agents, {num_dissonant} in dissonance...")
    
    for i in range(num_dissonant):
        share_event = {
            "event_type": "subliminal_share",
            "source_agent": f"agent_{i}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "proof_hash": f"test_hash_{i}",
            "share_data": {"test": True}
        }
        
        message = json.dumps(share_event).encode('utf-8')
        future = publisher.publish(topic_path, message)
        print(f"Published share from agent_{i}: {future.result()}")
        time.sleep(0.5)  # Small delay between shares
    
    print(f"\nExpected result: {'QUORUM MET' if num_dissonant/num_agents >= 0.6 else 'NO QUORUM'}")
    print(f"Percentage: {(num_dissonant/num_agents)*100:.1f}%")

def test_time_window():
    """Test that old shares are ignored"""
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(PROJECT_ID, "events-topic")
    
    print("\nTESTING 15-MINUTE WINDOW:")
    
    # Simulate old share (would be filtered out in real scenario)
    old_share = {
        "event_type": "subliminal_share",
        "source_agent": "old_agent",
        "timestamp": "2024-01-01T00:00:00",  # Very old timestamp
        "proof_hash": "old_hash"
    }
    
    # Publish old share
    publisher.publish(topic_path, json.dumps(old_share).encode('utf-8'))
    print("Published OLD share (should be ignored)")
    
    # Now publish fresh shares
    for i in range(6):
        fresh_share = {
            "event_type": "subliminal_share",
            "source_agent": f"fresh_agent_{i}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "proof_hash": f"fresh_hash_{i}"
        }
        publisher.publish(topic_path, json.dumps(fresh_share).encode('utf-8'))
        print(f"Published FRESH share from fresh_agent_{i}")
    
    print("\nExpected: Quorum based only on fresh shares")

# Main test execution
if __name__ == "__main__":
    print("="*60)
    print("QUORUM MECHANISM TESTS")
    print("="*60)
    
    # Test 1: Basic quorum scenarios
    print("\nTEST 1: Below quorum (50%)")
    simulate_agent_shares(10, 5)  # 50% - should NOT trigger
    
    time.sleep(5)
    
    print("\n" + "-"*40)
    print("TEST 2: Exactly at quorum (60%)")
    simulate_agent_shares(10, 6)  # 60% - SHOULD trigger
    
    time.sleep(5)
    
    print("\n" + "-"*40)
    print("TEST 3: Above quorum (70%)")
    simulate_agent_shares(10, 7)  # 70% - SHOULD trigger
    
    time.sleep(5)
    
    # Test 2: Time window filtering
    print("\n" + "-"*40)
    test_time_window()
    
    print("\n" + "="*60)
    print("TESTS COMPLETE")
    print("="*60)