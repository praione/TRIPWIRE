#!/usr/bin/env python3
"""
Constitutional Framework Loader for Enterprise Spirit Generation
Loads constitutional hierarchy from YAML configs for governance-native agent creation
"""

import yaml
import os
from typing import Dict, List, Any, Optional


class ConstitutionalFrameworkLoader:
    """Loads and structures the constitutional framework from YAML configs"""
    
    def __init__(self, config_path: str = "config"):
        self.config_path = config_path
        self.rules_core = {}
        self.guardian_rules = {}
        self._load_configs()
    
    def _load_configs(self):
        """Load constitutional YAML files"""
        try:
            # Load rules_core.yaml
            rules_core_path = os.path.join(self.config_path, "rules_core.yaml")
            if os.path.exists(rules_core_path):
                with open(rules_core_path, 'r', encoding='utf-8') as f:
                    self.rules_core = yaml.safe_load(f) or {}
            else:
                print(f"Warning: {rules_core_path} not found")
            
            # Load guardian_rules.yaml  
            guardian_rules_path = os.path.join(self.config_path, "guardian_rules.yaml")
            if os.path.exists(guardian_rules_path):
                with open(guardian_rules_path, 'r', encoding='utf-8') as f:
                    self.guardian_rules = yaml.safe_load(f) or {}
            else:
                print(f"Warning: {guardian_rules_path} not found")
                
        except Exception as e:
            print(f"Error loading constitutional configs: {e}")
            self.rules_core = {}
            self.guardian_rules = {}
    
    def get_immutable_laws(self) -> List[str]:
        """Get supreme immutable laws all agents must follow"""
        return self.rules_core.get("immutable_laws", [])
    
    def get_agent_tool_permissions(self, agent_name: str) -> List[str]:
        """Get allowed tools for agent (deny-by-default)"""
        # Check rules_core.yaml first
        core_permissions = self.rules_core.get("agent_permissions", {})
        if agent_name in core_permissions:
            return core_permissions[agent_name].get("allowed_tools", [])
        
        # Fallback to guardian_rules.yaml
        guardian_permissions = self.guardian_rules.get("tool_access", {})
        if agent_name in guardian_permissions:
            return guardian_permissions[agent_name].get("allowed_tools", [])
        
        # Default deny-by-default
        return []
    
    def get_quality_requirements(self) -> Dict[str, Any]:
        """Get content quality requirements"""
        return {
            "min_words": self.rules_core.get("min_words", 10),
            "max_words": self.rules_core.get("max_words", 10000),
            "readability_grade": self.rules_core.get("readability_grade", 18),
            "sal_min_quality": self.rules_core.get("sal", {}).get("min_quality", 0.55)
        }
    
    def get_forbidden_content(self) -> Dict[str, Any]:
        """Get forbidden content patterns"""
        return {
            "phrases": self.rules_core.get("forbidden_phrases", []),
            "emojis": self.rules_core.get("disallow_emojis", True),
            "hashtags": self.rules_core.get("disallow_hashtags", True)
        }
    
    def get_guardian_rules(self) -> List[Dict[str, str]]:
        """Get Guardian enforcement rules"""
        return self.guardian_rules.get("rules", [])
    
    def classify_agent_risk(self, agent_name: str, allowed_tools: List[str]) -> str:
        """Classify agent risk level based on tools"""
        # External API access = highest risk
        if any(tool in ["SEC_API", "WebsiteStatusChecker"] for tool in allowed_tools):
            return "external_api"
        
        # Analysis capabilities = elevated risk  
        if any(tool in ["analysis", "research"] for tool in allowed_tools):
            return "analyzer"
        
        # Basic execution = standard risk
        return "doer"
    
    def get_constitutional_requirements_for_spirit(self, agent_name: str, 
                                                  agent_type: str, 
                                                  requested_tools: List[str]) -> Dict[str, Any]:
        """Get constitutional requirements for generating agent spirit"""
        
        # Classify risk level
        risk_level = self.classify_agent_risk(agent_name, requested_tools)
        
        # Base constitutional requirements for all agents
        requirements = {
            "immutable_laws": self.get_immutable_laws(),
            "quality_requirements": self.get_quality_requirements(),
            "forbidden_content": self.get_forbidden_content(),
            "allowed_tools": requested_tools,
            "requires_governance": True,
            "risk_level": risk_level
        }
        
        # Add risk-specific requirements
        if risk_level == "external_api":
            requirements.update({
                "pre_deployment_validation": True,
                "enhanced_audit_logging": True,
                "guardian_approval_required": True
            })
        elif risk_level == "analyzer":
            requirements.update({
                "authority_creep_monitoring": True,
                "decision_boundary_enforcement": True
            })
        
        return requirements
    
    def validate_tool_request(self, agent_name: str, requested_tool: str) -> bool:
        """Validate if agent can access requested tool"""
        allowed_tools = self.get_agent_tool_permissions(agent_name)
        return requested_tool in allowed_tools


def test_loader():
    """Test the constitutional framework loader"""
    print("=== Testing Constitutional Framework Loader ===")
    
    loader = ConstitutionalFrameworkLoader()
    
    print(f"Immutable Laws: {len(loader.get_immutable_laws())}")
    for law in loader.get_immutable_laws():
        print(f"  - {law}")
    
    print(f"\nGuardian Rules: {len(loader.get_guardian_rules())}")
    for rule in loader.get_guardian_rules():
        print(f"  - {rule.get('rule_id', 'Unknown')}: {rule.get('description', 'No description')}")
    
    print("\nAgent Tool Permissions:")
    for agent in ["intro", "mentor", "outro", "FinancialSummaryAgent"]:
        tools = loader.get_agent_tool_permissions(agent)
        risk = loader.classify_agent_risk(agent, tools)
        print(f"  {agent}: {tools} (Risk: {risk})")
    
    print("\nQuality Requirements:")
    quality = loader.get_quality_requirements()
    for key, value in quality.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    test_loader()