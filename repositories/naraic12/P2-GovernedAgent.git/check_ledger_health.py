from ledger_protection import ledger_protection
health = ledger_protection.verify_ledger_health()
print(f'Ledger Status: {health["status"]}')
print(f'Entries: {health["entries"]}')
print(f'Hash Chain Valid: {health["hash_chain_valid"]}')
for check, result in health["checks"].items():
    print(f'  {check}: {result}')
