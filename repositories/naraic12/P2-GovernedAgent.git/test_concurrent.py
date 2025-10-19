import threading
from event_log import emit_event

def write_events(thread_id):
    for i in range(5):
        emit_event(f'THREAD{thread_id}', f'concurrent.test.{i}', f'agent_{thread_id}', 'ok')
    print(f'Thread {thread_id} complete')

# Create multiple threads
threads = []
for i in range(3):
    t = threading.Thread(target=write_events, args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print('All threads complete - check ledger for proper ordering')
