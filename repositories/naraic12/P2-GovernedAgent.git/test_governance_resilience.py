# test_governance_resilience.py
"""Test governance resilience configuration"""

from governance_resilience_loader import get_resilience_config
import json

print("Universal Governance Middleware Resilience Configuration")
print("=" * 60)

config = get_resilience_config()

# Display component configurations
for component in ["tool_access", "guardian_evaluation", "simulacrum_evaluation"]:
    comp_config = config.get_component_config(component)
    print(f"\n{component.upper()}:")
    print(f"  Failure Threshold: {comp_config.failure_threshold} failures before opening")
    print(f"  Timeout: {comp_config.timeout_seconds} seconds")
    print(f"  Recovery Timeout: {comp_config.recovery_timeout} seconds")
    print(f"  Success Threshold: {comp_config.success_threshold} successes to close")

# Display middleware config
middleware_config = config.get_middleware_config()
print(f"\nMIDDLEWARE SETTINGS:")
print(f"  Cascade Protection: {middleware_config.get('cascade_protection')}")
print(f"  Degrade Gracefully: {middleware_config.get('degrade_gracefully')}")
print(f"  Alert Threshold: {middleware_config.get('alert_threshold')} open breakers")
print(f"  Emergency Shutdown: {middleware_config.get('emergency_shutdown_threshold')} open breakers")

# Test trigger logic
print(f"\nTRIGGER LOGIC:")
for i in range(4):
    alert = config.should_alert(i)
    dissonance = config.should_trigger_dissonance(i)
    print(f"  {i} breakers open: Alert={alert}, Dissonance={dissonance}")

print(f"\n✅ Configuration loaded successfully from: {config.config_path}")
