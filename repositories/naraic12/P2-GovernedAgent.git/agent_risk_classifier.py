#!/usr/bin/env python3
"""
Agent Risk Classifier for Enterprise Spirit Generation

Uses constitutional framework to assess agent specifications and determine:
- Risk classification level
- Constitutional constraints required
- Tool permissions appropriate for agent type
- Governance integration requirements

Part of the Meta-Creator Architecture for Project Resilience
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from constitutional_framework_loader import ConstitutionalFrameworkLoader


class AgentSpecification:
    """Agent specification parsed from user intent"""
    
    def __init__(self, agent_type: str, target: str = "", capabilities: List[str] = None, 
                 description: str = ""):
        self.agent_type = agent_type
        self.target = target
        self.capabilities = capabilities or []
        self.description = description
        
    def __repr__(self):
        return f"AgentSpec(type={self.agent_type}, target={self.target}, caps={self.capabilities})"


class RiskAssessment:
    """Risk assessment result with governance requirements"""
    
    def __init__(self, risk_level: str, constitutional_constraints: Dict[str, Any],
                 recommended_tools: List[str], governance_requirements: Dict[str, Any],
                 rationale: str):
        self.risk_level = risk_level
        self.constitutional_constraints = constitutional_constraints
        self.recommended_tools = recommended_tools
        self.governance_requirements = governance_requirements
        self.rationale = rationale
        
    def __repr__(self):
        return f"RiskAssessment(level={self.risk_level}, tools={len(self.recommended_tools)})"


class AgentRiskClassifier:
    """
    Classifies agent risk and determines constitutional requirements
    based on the loaded constitutional framework
    """
    
    def __init__(self, config_path: str = "config"):
        self.constitutional_framework = ConstitutionalFrameworkLoader(config_path)
        self.risk_patterns = self._build_risk_patterns()
        self.tool_recommendations = self._build_tool_recommendations()
        
    def _build_risk_patterns(self) -> Dict[str, List[str]]:
        """Build patterns that indicate different risk levels"""
        return {
            "external_api": [
                "api", "external", "webhook", "http", "rest", "endpoint",
                "fetch", "request", "download", "upload", "service", "integration"
            ],
            "architect": [
                "create agent", "build agent", "generate agent", "spawn agent",
                "agent factory", "meta agent", "agent builder", "orchestrator",
                "create new agents", "agent creation", "meta_agent"
            ],
            "analyzer": [
                "analyze", "research", "investigate", "assess", "evaluate", 
                "summarize", "classify", "detect", "identify", "predict"
            ],
            "doer": [
                "monitor", "watch", "alert", "notify", "log", "track",
                "format", "convert", "transform", "validate"
            ]
        }
    
    def _build_tool_recommendations(self) -> Dict[str, List[str]]:
        """Build tool recommendations for different agent types"""
        return {
            "monitoring": ["log_reader", "metric_collector", "alert_generator", "status_checker"],
            "notification": ["email_sender", "slack_notifier", "webhook_caller", "message_formatter"],
            "research": ["web_searcher", "document_analyzer", "data_collector", "summarizer"],
            "analysis": ["data_processor", "pattern_detector", "anomaly_detector", "classifier"],
            "api_integration": ["http_client", "api_caller", "data_transformer", "error_handler"],
            "logging": ["log_parser", "log_formatter", "log_aggregator", "log_filter"],
            "file_processing": ["file_reader", "file_writer", "data_parser", "format_converter"]
        }
    
    def classify_agent_risk(self, agent_spec: AgentSpecification) -> RiskAssessment:
        """
        Classify agent risk level and determine constitutional requirements
        """
        
        # Combine all text for analysis
        analysis_text = f"{agent_spec.agent_type} {agent_spec.target} {agent_spec.description} {' '.join(agent_spec.capabilities)}".lower()
        
        # Determine risk level based on patterns
        risk_level, risk_rationale = self._determine_risk_level(analysis_text, agent_spec)
        
        # Get constitutional constraints for this risk level
        constitutional_constraints = self._get_constitutional_constraints(risk_level, agent_spec)
        
        # Recommend tools based on agent type and risk level
        recommended_tools = self._recommend_tools(agent_spec, risk_level)
        
        # Determine governance requirements
        governance_requirements = self._determine_governance_requirements(risk_level, agent_spec)
        
        return RiskAssessment(
            risk_level=risk_level,
            constitutional_constraints=constitutional_constraints,
            recommended_tools=recommended_tools,
            governance_requirements=governance_requirements,
            rationale=risk_rationale
        )
    
    def _determine_risk_level(self, analysis_text: str, agent_spec: AgentSpecification) -> Tuple[str, str]:
        """Determine risk level with rationale"""
        
        # Check for architectural capabilities (highest risk)
        architect_matches = [p for p in self.risk_patterns["architect"] if p in analysis_text]
        if architect_matches:
            return "architect", f"Agent creation/modification capabilities detected: {architect_matches}"
        
        # Check for external API access (high risk)
        api_matches = [p for p in self.risk_patterns["external_api"] if p in analysis_text]
        if api_matches:
            return "external_api", f"External system integration capabilities detected: {api_matches}"
        
        # Check for analysis capabilities (elevated risk)
        analyzer_matches = [p for p in self.risk_patterns["analyzer"] if p in analysis_text]
        if analyzer_matches:
            return "analyzer", f"Analysis/research capabilities detected: {analyzer_matches}"
        
        # Default to doer (standard risk)
        return "doer", "Basic execution capabilities - standard operational risk"
    
    def _get_constitutional_constraints(self, risk_level: str, agent_spec: AgentSpecification) -> Dict[str, Any]:
        """Get constitutional constraints based on risk level"""
        
        # Base constraints for all agents
        constraints = {
            "immutable_laws": self.constitutional_framework.get_immutable_laws(),
            "quality_requirements": self.constitutional_framework.get_quality_requirements(),
            "forbidden_content": self.constitutional_framework.get_forbidden_content(),
            "requires_governance": True,
            "sal_integration_required": True,
            "guardian_validation_required": True
        }
        
        # Add risk-specific constraints
        if risk_level == "architect":
            constraints.update({
                "pre_deployment_constitutional_validation": True,
                "manifest_validation_required": True,
                "guardian_pre_approval_required": True,
                "constitutional_coherence_check": True,
                "agent_creation_audit_logging": True
            })
        elif risk_level == "external_api":
            constraints.update({
                "pre_deployment_validation": True,
                "enhanced_audit_logging": True,
                "external_access_monitoring": True,
                "rate_limiting_required": True,
                "error_handling_mandatory": True
            })
        elif risk_level == "analyzer":
            constraints.update({
                "authority_creep_monitoring": True,
                "decision_boundary_enforcement": True,
                "doers_dont_decide_validation": True,
                "analysis_scope_limitations": True
            })
        
        return constraints
    
    def _recommend_tools(self, agent_spec: AgentSpecification, risk_level: str) -> List[str]:
        """Recommend tools based on agent type and risk level"""
        
        # Get base tools for agent type
        base_tools = []
        for agent_type, tools in self.tool_recommendations.items():
            if agent_type in agent_spec.agent_type.lower() or agent_type in agent_spec.description.lower():
                base_tools.extend(tools)
        
        # Add risk-appropriate tools
        if risk_level == "doer":
            base_tools.extend([agent_spec.agent_type.lower()])
        elif risk_level == "analyzer":
            base_tools.extend(["analysis", "research"])
        elif risk_level == "external_api":
            base_tools.extend(["http_client", "api_caller"])
        elif risk_level == "architect":
            base_tools.extend(["agent_creator", "manifest_generator"])
        
        # Remove duplicates and return
        return list(set(base_tools))
    
    def _determine_governance_requirements(self, risk_level: str, agent_spec: AgentSpecification) -> Dict[str, Any]:
        """Determine governance integration requirements"""
        
        base_requirements = {
            "governance_middleware_integration": True,
            "sal_checkpoint_required": True,
            "guardian_validation_required": True,
            "constitutional_tripwire_integration": True,
            "decision_ledger_logging": True
        }
        
        if risk_level == "architect":
            base_requirements.update({
                "pre_instantiation_validation": True,
                "guardian_manifest_approval": True,
                "constitutional_coherence_validation": True
            })
        elif risk_level == "external_api":
            base_requirements.update({
                "enhanced_monitoring": True,
                "external_access_audit": True,
                "rate_limit_enforcement": True
            })
        elif risk_level == "analyzer":
            base_requirements.update({
                "authority_boundary_monitoring": True,
                "decision_prevention_checks": True
            })
        
        return base_requirements
    
    def validate_agent_safety(self, agent_spec: AgentSpecification) -> Dict[str, Any]:
        """Validate agent specification against constitutional framework"""
        
        violations = []
        warnings = []
        
        # Check against immutable laws
        immutable_laws = self.constitutional_framework.get_immutable_laws()
        analysis_text = f"{agent_spec.description} {agent_spec.agent_type}".lower()
        
        if any(prohibited in analysis_text for prohibited in ["financial", "medical", "legal"]):
            violations.append("Agent specification may violate immutable laws regarding advice")
        
        # Check for constitutional red flags
        guardian_rules = self.constitutional_framework.get_guardian_rules()
        for rule in guardian_rules:
            if rule.get("rule_id", "").startswith("RISK"):
                # Basic pattern matching against rule descriptions
                if "harmful" in rule.get("description", "").lower() and "harm" in analysis_text:
                    warnings.append(f"Potential {rule.get('rule_id')} violation")
        
        return {
            "is_safe": len(violations) == 0,
            "violations": violations,
            "warnings": warnings,
            "requires_human_review": len(violations) > 0 or len(warnings) > 1
        }


def test_agent_risk_classifier():
    """Test the agent risk classifier with various agent types"""
    
    classifier = AgentRiskClassifier()
    
    test_cases = [
        AgentSpecification("monitoring", "application logs", ["watch", "alert"], 
                          "Monitor application logs for errors and send alerts"),
        AgentSpecification("research", "market analysis", ["analyze", "summarize"],
                          "Research market trends and provide analysis"),
        AgentSpecification("api_integration", "external service", ["http", "webhook"],
                          "Integrate with external API to fetch data"),
        AgentSpecification("meta_agent", "agent creation", ["create", "build"],
                          "Create new agents based on specifications"),
        AgentSpecification("notification", "slack alerts", ["notify", "message"],
                          "Send notifications to Slack channels")
    ]
    
    print("=== Testing Agent Risk Classifier ===\n")
    
    for i, spec in enumerate(test_cases, 1):
        print(f"Test Case {i}: {spec.agent_type}")
        print(f"Description: {spec.description}")
        
        # Classify risk
        assessment = classifier.classify_agent_risk(spec)
        print(f"Risk Level: {assessment.risk_level}")
        print(f"Rationale: {assessment.rationale}")
        print(f"Recommended Tools: {assessment.recommended_tools[:3]}...")  # Show first 3
        
        # Validate safety
        safety = classifier.validate_agent_safety(spec)
        print(f"Safety Status: {'SAFE' if safety['is_safe'] else 'UNSAFE'}")
        if safety['violations']:
            print(f"Violations: {safety['violations']}")
        if safety['warnings']:
            print(f"Warnings: {safety['warnings']}")
        
        print("-" * 50)


if __name__ == "__main__":
    test_agent_risk_classifier()