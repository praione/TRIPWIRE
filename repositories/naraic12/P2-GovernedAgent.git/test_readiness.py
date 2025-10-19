# Test dispatcher readiness
import json

# Test the readiness function
def _check_halt_readiness():
    return True, 'Ready for shutdown'

can_halt, reason = _check_halt_readiness()
print(f'Can halt: {can_halt}')
print(f'Reason: {reason}')

# Test payload
payload = {
    'command': 'SYSTEM_HALT',
    'authority': 'ARBITER',
    'trace_id': 'TEST_001',
    'reason': 'Test halt'
}

print(f'Test payload ready: {json.dumps(payload, indent=2)}')
