from sal import resilience_manager
import json

# Get the health status which shows all breakers
health = resilience_manager.get_health_status()

print('Circuit Breakers Configured:')
print('=' * 40)
for name, state in health['circuit_breaker_states'].items():
    print(f'{name}: {state["state"]}')

print(f'\nOverall Health: {health["overall_health"]}')
print(f'Total Breakers: {health["total_circuit_breakers"]}')
