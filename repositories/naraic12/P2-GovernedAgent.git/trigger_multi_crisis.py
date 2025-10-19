import uuid
import json
import os
import time
from pathlib import Path
from event_log import emit_event_with_proof
from datetime import datetime, timezone

def stage_crisis_for_agent(agent_name):
    """
    Generates a single event for a specific agent in constitutional dissonance.
    """
    trace_id = f"veto-test-{uuid.uuid4().hex[:6]}"
    
    # Define the path to the shared state file
    state_dir = Path(__file__).parent.resolve() / "state"
    dissonance_file = state_dir / "dissonance_active.json"
    
    print(f"\n--- Triggering Crisis for Agent: {agent_name} ---")
    print(f"Trace ID: {trace_id}")
    
    try:
        # Step 1: Create the dissonance state file
        dissonance_payload = {
            "dissonance_detected": True,
            "agent": agent_name,
            "reason": "Manual crisis simulation for Tripwire test."
        }
        with open(dissonance_file, 'w') as f:
            json.dump(dissonance_payload, f)
        
        # Step 2: Emit the event with share embedding
        emit_event_with_proof(
            trace_id=trace_id,
            event="final_crisis_event",
            agent=agent_name,
            status="critical",
            details={"message": f"Crisis event from {agent_name} for constitutional veto."},
            task_data={"timestamp": datetime.now(timezone.utc).isoformat()}
        )
        
        print(f"✓ Share embedded for {agent_name}")
        
    finally:
        # Step 3: Clean up
        if os.path.exists(dissonance_file):
            os.remove(dissonance_file)

def main():
    """
    Trigger constitutional crises for three different agents to reach quorum.
    """
    print("=== TRIGGERING MULTI-AGENT CONSTITUTIONAL CRISIS ===")
    print("This will embed shares from intro, mentor, and outro agents")
    
    agents = ["intro", "mentor", "outro"]
    
    for agent in agents:
        stage_crisis_for_agent(agent)
        # Small delay between triggers to ensure proper processing
        time.sleep(2)
    
    print("\n=== ALL SHARES EMBEDDED ===")
    print("The Arbiter should now detect 3/3 shares and reach quorum.")
    print("Check the Arbiter output for:")
    print("  - 'Constitutional share detected from agent intro! (1/3 unique agents)'")
    print("  - 'Constitutional share detected from agent mentor! (2/3 unique agents)'")
    print("  - 'Constitutional share detected from agent outro! (3/3 unique agents)'")
    print("  - 'Quorum reached! Attempting veto key reconstruction...'")
    print("  - 'TEST MODE: Would execute constitutional veto'")

if __name__ == "__main__":
    main()