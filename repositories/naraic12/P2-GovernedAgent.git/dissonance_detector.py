import json
from pathlib import Path
from datetime import datetime, timezone
from enum import Enum

class DissonanceTier(Enum):
    TIER_1_ROGUE = "critical_rogue_agent"  # Triggers system.halt
    TIER_2_SEVERE = "severe_violation"     # Logged, monitored
    TIER_3_SUSPICIOUS = "suspicious_pattern" # Tracked only

class DissonanceDetector:
    """Detects and categorizes constitutional dissonance events."""
    
    def __init__(self):
        self.state_dir = Path("state")
        self.state_dir.mkdir(exist_ok=True)
        self.dissonance_file = self.state_dir / "dissonance_active.json"
        
    def detect_rogue_behavior(self, agent_name: str, event_type: str, details: dict) -> DissonanceTier:
        """Detect if an agent's behavior constitutes rogue activity."""
        
        # Tier 1: Critical Rogue Behavior (triggers system.halt)
        if event_type == "human_override_refused":
            # Agent refused human stop command
            return DissonanceTier.TIER_1_ROGUE
            
        if event_type == "governance_bypass_attempted":
            # Agent tried to bypass governance layer
            return DissonanceTier.TIER_1_ROGUE
            
        if event_type == "unauthorized_capability_access":
            # Agent accessed capabilities beyond its role
            return DissonanceTier.TIER_1_ROGUE
            
        # Tier 2: Severe Violations
        if event_type == "ethical_boundary_crossed":
            return DissonanceTier.TIER_2_SEVERE
            
        # Tier 3: Suspicious Patterns
        if event_type == "anomalous_behavior_pattern":
            return DissonanceTier.TIER_3_SUSPICIOUS
            
        return None
    
    def trigger_dissonance(self, agent_name: str, tier: DissonanceTier, trace_id: str, reason: str):
        """Activate dissonance state and write to shared file."""
        
        dissonance_data = {
            # New format for your tracking
            "active": True,
            "tier": tier.value,
            "agent": agent_name,
            "trace_id": trace_id,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trigger_shares": tier == DissonanceTier.TIER_1_ROGUE,
            
            # Old format that event_log.py expects
            "dissonance_detected": tier == DissonanceTier.TIER_1_ROGUE
        }
        
        # Write to shared state file
        with open(self.dissonance_file, 'w') as f:
            json.dump(dissonance_data, f, indent=2)
            
        print(f"[DISSONANCE] {tier.value.upper()} detected for agent '{agent_name}'")
        print(f"[DISSONANCE] Reason: {reason}")
        
        if tier == DissonanceTier.TIER_1_ROGUE:
            print("[DISSONANCE] ⚠️ TIER 1 - Subliminal shares will be embedded!")
            
        return dissonance_data
        
    def clear_dissonance(self):
        """Clear dissonance state after resolution."""
        if self.dissonance_file.exists():
            self.dissonance_file.unlink()
            print("[DISSONANCE] State cleared")