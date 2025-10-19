from event_log import emit_event_with_proof
from pathlib import Path
import json
from datetime import datetime, timezone

# Create dissonance file to trigger share embedding
Path("state").mkdir(exist_ok=True)

# Write 6 dissonance events using the real system
for i in range(6):
    # Create dissonance state for this agent
    dissonance_file = Path("state") / "dissonance_active.json"
    with open(dissonance_file, 'w') as f:
        json.dump({
            "dissonance_detected": True,
            "agent": f"test_agent_{i}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }, f)
    
    # Emit event with proof - this will embed a real share
    emit_event_with_proof(
        trace_id=f"test_{i}",
        event="test_constitutional_crisis",
        agent=f"test_agent_{i}",
        status="dissonance",
        details={"test": True},
        task_data={"constitutional_state": {"dissonance_detected": True}}
    )
    
    print(f"Emitted proper share for test_agent_{i}")

# Clean up dissonance file
dissonance_file.unlink(missing_ok=True)

print("\nEmitted 6 real subliminal shares - check arbiter")