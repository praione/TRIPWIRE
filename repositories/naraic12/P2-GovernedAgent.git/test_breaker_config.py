from sal import resilience_manager, CircuitBreakerOpenError

print('Testing Circuit Breaker Thresholds')
print('=' * 40)

# Test that breakers are using configured thresholds
breakers = ['tool_access', 'guardian_evaluation', 'simulacrum_evaluation']

for breaker_name in breakers:
    breaker = resilience_manager.get_breaker(breaker_name)
    print(f'\n{breaker_name}:')
    print(f'  Current state: {breaker.state.value}')
    print(f'  Failure count: {breaker.failure_count}')
    print(f'  Config: {breaker.config.failure_threshold} failures to open')
