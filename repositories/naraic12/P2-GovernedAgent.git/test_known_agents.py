from event_log import emit_event_with_proof
from pathlib import Path
import json
from datetime import datetime, timezone
import time

# Clear the ledger first
ledger_file = Path("state") / f"ledger_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.ndjson"
if ledger_file.exists():
    ledger_file.unlink()

Path("state").mkdir(exist_ok=True)
known_agents = ['intro', 'mentor', 'outro']

# Wait a moment to ensure we're writing future events
time.sleep(2)

for agent in known_agents:
    # Generate timestamp NOW for this specific event
    event_timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    
    # Create dissonance file
    dissonance_file = Path("state") / "dissonance_active.json"
    with open(dissonance_file, 'w') as f:
        json.dump({
            "dissonance_detected": True,
            "agent": agent,
            "timestamp": event_timestamp  # Use same timestamp
        }, f)
    
    # Pass the timestamp explicitly
    emit_event_with_proof(
        trace_id=f"crisis_{agent}",
        event="constitutional_crisis",
        agent=agent,
        status="dissonance",
        details={"crisis": True},
        task_data={
            "constitutional_state": {"dissonance_detected": True},
            "timestamp": event_timestamp  # Pass exact timestamp
        }
    )
    
    print(f"Emitted share for {agent} at {event_timestamp}")
    time.sleep(0.5)  # Small delay between events

dissonance_file.unlink(missing_ok=True)