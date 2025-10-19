import json
from datetime import datetime, timezone
from pathlib import Path

# Ensure state directory exists
Path("state").mkdir(exist_ok=True)

# Clear any old test data by starting fresh
ledger_file = Path("state") / f"ledger_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.ndjson"

print(f"Writing to {ledger_file}")

# Simulate 10 agents, write shares for 6 (60%)
total_agents = 10
dissonant_agents = 6

for i in range(dissonant_agents):
    event = {
        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "trace_id": f"test_{i}",
        "event": "test_share",
        "agent": f"test_agent_{i}",  # Different names to avoid mixing with old data
        "status": "dissonance",
        "details": {"proof": f"test_proof_{i}"}
    }
    
    with open(ledger_file, "a") as f:
        f.write(json.dumps(event) + "\n")
    
    print(f"Wrote event for test_agent_{i}")

print(f"\nWrote {dissonant_agents}/{total_agents} agents (60%) - should trigger quorum")