import json
from pathlib import Path
from datetime import datetime, timezone

# Check what's actually in the ledger
ledger_file = Path("state") / f"ledger_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.ndjson"

print(f"Checking {ledger_file}\n")

if ledger_file.exists():
    with open(ledger_file, 'r') as f:
        lines = f.readlines()
        print(f"Total events in ledger: {len(lines)}")
        
        # Show last 5 events
        print("\nLast 5 events:")
        for line in lines[-5:]:
            event = json.loads(line)
            print(f"Agent: {event.get('agent')}, Event: {event.get('event')}, Proof: {event.get('details', {}).get('proof', 'NO PROOF')[:20]}...")
        
        # Check if any have the word "proof" in details
        proof_count = sum(1 for line in lines if '"proof"' in line)
        print(f"\nEvents with proof field: {proof_count}")
else:
    print("Ledger file doesn't exist!")

# Check if subliminal_proof is working
print("\n--- Testing subliminal_proof system ---")
try:
    from subliminal_proof import get_subliminal_system
    system = get_subliminal_system()
    print("Subliminal system initialized successfully")
    
    # Try to generate a test proof
    test_proof = system.generate_proof_hash("test_agent", {"test": True}, "test_trace")
    print(f"Generated test proof: {test_proof[:20]}...")
    
    # Try to verify it
    result = system.verify_subliminal_share(
        proof_hash=test_proof,
        agent="test_agent",
        trace_id="test_trace",
        timestamp=datetime.now(timezone.utc).isoformat()
    )
    print(f"Verification result: {result}")
    
except Exception as e:
    print(f"Error with subliminal system: {e}")