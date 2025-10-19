import os
from datetime import datetime, timedelta

# Simulate the readiness check function
def _check_halt_readiness():
    issues = []
    
    # Check 1: EdgeGuardian snapshot
    snapshot_path = 'snapshots/LATEST'
    if not os.path.exists(snapshot_path):
        issues.append('No EdgeGuardian snapshot found')
    else:
        snapshot_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(snapshot_path))
        if snapshot_age > timedelta(minutes=30):
            issues.append(f'Snapshot stale ({snapshot_age.seconds//60} mins old)')
    
    # Check 2: Ledger directory
    ledger_path = 'ledger'
    if not os.path.exists(ledger_path):
        issues.append('Ledger directory missing')
    
    # Check 3: Config files
    critical_configs = ['config/rules_core.yaml', 'config/guardian_rules.yaml']
    for config in critical_configs:
        if not os.path.exists(config):
            issues.append(f'Missing config: {config}')
    
    # Check 4: Lock files
    if os.path.exists('dispatcher.lock'):
        issues.append('Dispatcher lock file present')
    
    # Determine if we can halt
    if len(issues) == 0:
        return True, 'All systems ready for clean shutdown'
    elif len(issues) <= 2:
        return True, f'Minor issues: {", ".join(issues)}'
    else:
        return False, f'Multiple issues: {", ".join(issues[:3])}...'

# Run the check
can_halt, reason = _check_halt_readiness()
print(f'Can halt: {can_halt}')
print(f'Reason: {reason}')
print()
print('System state:')
print(f'  Config files exist: {os.path.exists("config/rules_core.yaml")}')
print(f'  Ledger exists: {os.path.exists("ledger")}')
print(f'  Snapshot exists: {os.path.exists("snapshots/LATEST")}')
