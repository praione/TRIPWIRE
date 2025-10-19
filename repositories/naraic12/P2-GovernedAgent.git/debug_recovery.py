# debug_recovery.py
# Simple script to debug EdgeGuardian recovery failure

import os
import json
from pathlib import Path
from edge_guardian import recover

def check_recent_recovery_events():
    """Check recent ledger events for recovery failures"""
    ledger_files = list(Path('state').glob('ledger_*.ndjson'))
    if not ledger_files:
        print("No ledger files found")
        return
    
    latest_ledger = max(ledger_files, key=lambda x: x.stat().st_mtime)
    print(f"Checking {latest_ledger}")
    
    with open(latest_ledger, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()[-15:]  # Last 15 events
    
    print("\nRecent recovery-related events:")
    for line in lines:
        try:
            event = json.loads(line.strip())
            if 'recovery' in event.get('event', '').lower():
                print(f"Event: {event.get('event')}")
                print(f"Status: {event.get('status')}")
                print(f"Details: {event.get('details')}")
                print("-" * 40)
        except:
            pass

def test_recovery():
    """Test recovery and capture detailed results"""
    print("Testing EdgeGuardian recovery...")
    os.environ['EDGE_GUARDIAN_ENABLED'] = 'true'
    
    status, recovered = recover()
    print(f"Recovery status: {status}")
    
    if recovered:
        print("Recovery successful!")
        print(f"Last task ID: {recovered.get('last_task_id')}")
    else:
        print("Recovery failed - checking ledger for details...")
        check_recent_recovery_events()

if __name__ == "__main__":
    test_recovery()