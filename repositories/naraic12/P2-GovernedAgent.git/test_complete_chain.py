from event_log import emit_event_with_proof
from datetime import datetime, timezone

# With dissonance file present, this should embed a share
timestamp = datetime.now(timezone.utc).isoformat()
emit_event_with_proof(
    trace_id="chain_test_001",
    event="testing_share_embed",
    agent="governed_agent",  # Matches agent in dissonance file
    status="testing",
    task_data={"timestamp": timestamp}
)
print("Event emitted - check ledger for subliminal share!")