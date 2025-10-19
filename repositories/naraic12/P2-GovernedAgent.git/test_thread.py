from event_log import emit_event
import uuid

# This will fail to upload to GCS and should start the background thread
trace = str(uuid.uuid4())
emit_event(trace, 'test.start.thread', 'system')
print(f'Event sent: {trace}')
print('This should trigger the background sync thread on GCS failure')
