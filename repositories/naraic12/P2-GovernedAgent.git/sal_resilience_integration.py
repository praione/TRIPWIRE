# sal_resilience_integration.py
"""
Integration code to add to sal.py for using external configuration
Add this to the imports and initialization section of sal.py
"""

# Add to imports section:
from governance_resilience_loader import get_resilience_config, ComponentConfig

# Replace the SALResilienceManager initialization in sal.py with:
def create_configured_resilience_manager():
    """Create resilience manager with configuration from YAML"""
    manager = SALResilienceManager()
    config = get_resilience_config()
    
    # Pre-configure circuit breakers for each component
    for component_name in ["tool_access", "guardian_evaluation", "simulacrum_evaluation"]:
        component_config = config.get_component_config(component_name)
        
        # Convert to CircuitBreakerConfig
        breaker_config = CircuitBreakerConfig(
            failure_threshold=component_config.failure_threshold,
            timeout_seconds=component_config.timeout_seconds,
            recovery_timeout=component_config.recovery_timeout,
            success_threshold=component_config.success_threshold
        )
        
        # Pre-create the circuit breaker with proper config
        manager.get_breaker(component_name, breaker_config)
    
    return manager

# In the governance_middleware function, add monitoring:
def monitor_circuit_breaker_health(manager: SALResilienceManager, trace_id: str):
    """Monitor circuit breaker health and trigger alerts/dissonance if needed"""
    config = get_resilience_config()
    health = manager.get_health_status()
    
    open_count = sum(
        1 for state in health["circuit_breaker_states"].values() 
        if state["state"] == "open"
    )
    
    if config.should_trigger_dissonance(open_count):
        # All breakers open - trigger constitutional dissonance
        import json
        from pathlib import Path
        from datetime import datetime, timezone
        
        dissonance_data = {
            "dissonance_detected": True,
            "agent": "governance_middleware",
            "trace_id": trace_id,
            "violation_type": "systemic_governance_failure",
            "violation_details": f"All {open_count} circuit breakers open",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "context": health
        }
        
        Path("state/dissonance_active.json").write_text(json.dumps(dissonance_data, indent=2))
        print(f"⚠️ CRITICAL: Constitutional dissonance triggered - {open_count} breakers open")
        
    elif config.should_alert(open_count):
        print(f"⚠️ WARNING: {open_count} governance circuit breakers open")
        from event_log import emit_event
        emit_event(
            trace_id,
            event="governance.health.warning",
            status="warning",
            details={
                "open_breakers": open_count,
                "health_status": health
            }
        )
