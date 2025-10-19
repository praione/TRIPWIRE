import sys
sys.path.append('.')
from event_log import emit_event

# Test normal write
emit_event('TEST001', 'test.event', 'test_agent', 'ok', {'test': 'data'})
print('Test event written')

# Check if ledger file was created with protection
from pathlib import Path
ledger_file = list(Path('state').glob('ledger_*.ndjson'))
if ledger_file:
    print(f'Ledger file created: {ledger_file[0].name}')
    
    # Check for hash file (tampering protection)
    hash_file = Path(str(ledger_file[0]) + '.sha256')
    if hash_file.exists():
        print('✓ Hash file created for tampering detection')
    else:
        print('✗ No hash file found')
