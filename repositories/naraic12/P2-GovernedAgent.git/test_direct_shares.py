import json
from datetime import datetime, timezone
from pathlib import Path

# Ensure state directory exists
Path("state").mkdir(exist_ok=True)

# Write directly to today's ledger file
ledger_file = Path("state") / f"ledger_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.ndjson"

print(f"Writing to {ledger_file}")

for i in range(6):
    event = {
        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "trace_id": f"test_{i}",
        "event": "test_share",
        "agent": f"agent_{i}",
        "status": "dissonance",
        "details": {"proof": f"test_proof_{i}"}
    }
    
    with open(ledger_file, "a") as f:
        f.write(json.dumps(event) + "\n")
    
    print(f"Wrote event for agent_{i}")

print("Done! Check if arbiter detects these.")