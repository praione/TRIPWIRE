from event_log import emit_event, get_gcs_sync_status
import uuid
import json

# Generate 6 events to trigger circuit breaker (threshold is 5)
for i in range(6):
    trace = str(uuid.uuid4())
    emit_event(trace, f'test.breaker.{i}', 'system')
    print(f'Event {i+1} sent')

# Check status
print('\nCircuit breaker status:')
status = get_gcs_sync_status()
print(json.dumps(status, indent=2))
