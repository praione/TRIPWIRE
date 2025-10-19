#!/usr/bin/env python3
"""
Auto-Provisioning System for Agent Team Infrastructure

Handles GCP resource creation, configuration, and lifecycle management for live agent teams.
Integrates with Dynamic Agent Factory to provision infrastructure on-demand.

Capabilities:
- GCP resource provisioning (Pub/Sub, Storage, IAM)
- Secure communication channel setup
- Monitoring and audit infrastructure
- Resource cleanup and lifecycle management
- Constitutional governance infrastructure integration

Part of Week 3 Agent Instantiation & Auto-Wiring for Project Resilience
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from dynamic_agent_factory import LiveAgent, LiveTeam, AgentDeploymentPlan


@dataclass
class ProvisionedResource:
    """Represents a provisioned GCP resource"""
    resource_id: str
    resource_type: str  # pubsub_topic, storage_bucket, iam_role
    resource_name: str
    resource_config: Dict[str, Any]
    provisioning_status: str  # pending, active, failed, cleanup
    creation_timestamp: str
    owner_agent_id: str
    owner_team_id: str


@dataclass
class ProvisioningPlan:
    """Plan for provisioning resources for an agent team"""
    plan_id: str
    team_id: str
    agent_resources: Dict[str, List[ProvisionedResource]]
    team_resources: List[ProvisionedResource]
    provisioning_order: List[str]
    cleanup_procedures: Dict[str, Any]
    estimated_cost: Dict[str, float]


class AutoProvisioningSystem:
    """
    Handles automated provisioning of GCP infrastructure for agent teams.
    Manages resource lifecycle and integrates with governance infrastructure.
    """
    
    def __init__(self, config_path: str = "config", provisioning_path: str = "provisioned_resources"):
        self.config_path = config_path
        self.provisioning_path = Path(provisioning_path)
        self.provisioning_path.mkdir(exist_ok=True)
        
        # Resource registry for tracking provisioned resources
        self.resource_registry = {}
        self.provisioning_plans = {}
        
        # GCP configuration (mock implementation)
        self.gcp_config = self._load_gcp_config()
        
        # Load existing provisioned resources
        self._load_existing_resources()
    
    def _load_gcp_config(self) -> Dict[str, Any]:
        """Load GCP configuration for resource provisioning"""
        
        # Mock GCP configuration - replace with actual GCP client setup
        return {
            "project_id": "project-resilience-agents",
            "default_region": "us-central1",
            "service_account": "agent-provisioner@project-resilience-agents.iam.gserviceaccount.com",
            "resource_labels": {
                "system": "project-resilience",
                "component": "agent-infrastructure",
                "managed_by": "auto-provisioning-system"
            }
        }
    
    def _load_existing_resources(self):
        """Load previously provisioned resources from persistent storage"""
        
        registry_file = self.provisioning_path / "resource_registry.json"
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                data = json.load(f)
                self.resource_registry = data.get("resources", {})
                self.provisioning_plans = data.get("plans", {})
    
    def _save_resources(self):
        """Save current resource state to persistent storage"""
        
        registry_file = self.provisioning_path / "resource_registry.json"
        with open(registry_file, 'w') as f:
            json.dump({
                "resources": self.resource_registry,
                "plans": self.provisioning_plans,
                "last_updated": datetime.now().isoformat()
            }, f, indent=2)
    
    def provision_team_infrastructure(self, live_team: LiveTeam) -> ProvisioningPlan:
        """
        Main method: Provision complete infrastructure for a live agent team
        """
        
        print(f"PROVISIONER: Creating infrastructure for team {live_team.team_id}")
        
        # Create provisioning plan
        provisioning_plan = self._create_provisioning_plan(live_team)
        
        # Validate provisioning plan
        validated_plan = self._validate_provisioning_plan(provisioning_plan)
        
        # Execute provisioning in order
        provisioned_plan = self._execute_provisioning_plan(validated_plan)
        
        # Setup team communication infrastructure
        self._setup_team_communication(provisioned_plan, live_team)
        
        # Setup monitoring and audit infrastructure
        self._setup_monitoring_infrastructure(provisioned_plan, live_team)
        
        # Register provisioned resources
        self.provisioning_plans[provisioned_plan.plan_id] = asdict(provisioned_plan)
        self._save_resources()
        
        print(f"PROVISIONER: Infrastructure provisioning complete for team {live_team.team_id}")
        
        return provisioned_plan
    
    def _create_provisioning_plan(self, live_team: LiveTeam) -> ProvisioningPlan:
        """Create comprehensive provisioning plan for team"""
        
        plan_id = f"plan_{live_team.team_id}_{int(time.time())}"
        agent_resources = {}
        team_resources = []
        
        # Create resources for each agent
        for agent_id, agent in live_team.live_agents.items():
            agent_resources[agent_id] = self._plan_agent_resources(agent, live_team.team_id)
        
        # Create team-level resources
        team_resources = self._plan_team_resources(live_team)
        
        # Determine provisioning order
        provisioning_order = self._calculate_provisioning_order(agent_resources, team_resources)
        
        # Calculate cost estimates
        estimated_cost = self._estimate_provisioning_costs(agent_resources, team_resources)
        
        return ProvisioningPlan(
            plan_id=plan_id,
            team_id=live_team.team_id,
            agent_resources=agent_resources,
            team_resources=team_resources,
            provisioning_order=provisioning_order,
            cleanup_procedures=self._generate_cleanup_procedures(agent_resources, team_resources),
            estimated_cost=estimated_cost
        )
    
    def _plan_agent_resources(self, agent: LiveAgent, team_id: str) -> List[ProvisionedResource]:
        """Plan resources needed for individual agent"""
        
        resources = []
        
        # Pub/Sub topics for agent communication
        input_topic = ProvisionedResource(
            resource_id=f"{agent.agent_id}_input_topic",
            resource_type="pubsub_topic",
            resource_name=f"agent-{agent.agent_id}-input",
            resource_config={
                "message_retention_duration": "1h",
                "message_storage_policy": {"allowed_persistence_regions": ["us-central1"]},
                "labels": self.gcp_config["resource_labels"]
            },
            provisioning_status="pending",
            creation_timestamp=datetime.now().isoformat(),
            owner_agent_id=agent.agent_id,
            owner_team_id=team_id
        )
        resources.append(input_topic)
        
        output_topic = ProvisionedResource(
            resource_id=f"{agent.agent_id}_output_topic",
            resource_type="pubsub_topic", 
            resource_name=f"agent-{agent.agent_id}-output",
            resource_config={
                "message_retention_duration": "1h",
                "message_storage_policy": {"allowed_persistence_regions": ["us-central1"]},
                "labels": self.gcp_config["resource_labels"]
            },
            provisioning_status="pending",
            creation_timestamp=datetime.now().isoformat(),
            owner_agent_id=agent.agent_id,
            owner_team_id=team_id
        )
        resources.append(output_topic)
        
        # Storage bucket for agent data
        storage_bucket = ProvisionedResource(
            resource_id=f"{agent.agent_id}_storage",
            resource_type="storage_bucket",
            resource_name=f"agent-{agent.agent_id}-data",
            resource_config={
                "location": "us-central1",
                "storage_class": "STANDARD",
                "uniform_bucket_level_access": True,
                "versioning": {"enabled": True},
                "labels": self.gcp_config["resource_labels"]
            },
            provisioning_status="pending",
            creation_timestamp=datetime.now().isoformat(),
            owner_agent_id=agent.agent_id,
            owner_team_id=team_id
        )
        resources.append(storage_bucket)
        
        # IAM role for agent
        iam_role = ProvisionedResource(
            resource_id=f"{agent.agent_id}_iam_role",
            resource_type="iam_role",
            resource_name=f"agent-{agent.agent_id}-role",
            resource_config={
                "title": f"Agent {agent.agent_id} Service Role",
                "description": f"Service role for agent {agent.agent_id} in team {team_id}",
                "included_permissions": [
                    "pubsub.messages.publish",
                    "pubsub.messages.consume", 
                    "storage.objects.create",
                    "storage.objects.get",
                    "logging.logEntries.create"
                ],
                "stage": "GA"
            },
            provisioning_status="pending",
            creation_timestamp=datetime.now().isoformat(),
            owner_agent_id=agent.agent_id,
            owner_team_id=team_id
        )
        resources.append(iam_role)
        
        return resources
    
    def _plan_team_resources(self, live_team: LiveTeam) -> List[ProvisionedResource]:
        """Plan team-level shared resources"""
        
        resources = []
        
        # Team coordination topic
        coordination_topic = ProvisionedResource(
            resource_id=f"{live_team.team_id}_coordination",
            resource_type="pubsub_topic",
            resource_name=f"team-{live_team.team_id}-coordination",
            resource_config={
                "message_retention_duration": "24h",
                "message_storage_policy": {"allowed_persistence_regions": ["us-central1"]},
                "labels": self.gcp_config["resource_labels"]
            },
            provisioning_status="pending",
            creation_timestamp=datetime.now().isoformat(),
            owner_agent_id="team_shared",
            owner_team_id=live_team.team_id
        )
        resources.append(coordination_topic)
        
        # Team status monitoring topic
        status_topic = ProvisionedResource(
            resource_id=f"{live_team.team_id}_status",
            resource_type="pubsub_topic",
            resource_name=f"team-{live_team.team_id}-status",
            resource_config={
                "message_retention_duration": "7d",
                "message_storage_policy": {"allowed_persistence_regions": ["us-central1"]},
                "labels": self.gcp_config["resource_labels"]
            },
            provisioning_status="pending",
            creation_timestamp=datetime.now().isoformat(),
            owner_agent_id="team_shared",
            owner_team_id=live_team.team_id
        )
        resources.append(status_topic)
        
        # Team audit log storage
        audit_storage = ProvisionedResource(
            resource_id=f"{live_team.team_id}_audit_storage",
            resource_type="storage_bucket",
            resource_name=f"team-{live_team.team_id}-audit-logs",
            resource_config={
                "location": "us-central1",
                "storage_class": "NEARLINE",  # Cost-effective for audit logs
                "uniform_bucket_level_access": True,
                "versioning": {"enabled": True},
                "retention_policy": {"retention_period": "2592000"},  # 30 days
                "labels": self.gcp_config["resource_labels"]
            },
            provisioning_status="pending",
            creation_timestamp=datetime.now().isoformat(),
            owner_agent_id="team_shared",
            owner_team_id=live_team.team_id
        )
        resources.append(audit_storage)
        
        return resources
    
    def _calculate_provisioning_order(self, agent_resources: Dict[str, List[ProvisionedResource]],
                                    team_resources: List[ProvisionedResource]) -> List[str]:
        """Calculate optimal order for resource provisioning"""
        
        # IAM roles first (dependencies for other resources)
        # Storage buckets next (no dependencies)  
        # Pub/Sub topics last (may depend on IAM)
        
        order = []
        
        # Phase 1: IAM roles
        for agent_id, resources in agent_resources.items():
            for resource in resources:
                if resource.resource_type == "iam_role":
                    order.append(resource.resource_id)
        
        # Phase 2: Storage buckets
        for agent_id, resources in agent_resources.items():
            for resource in resources:
                if resource.resource_type == "storage_bucket":
                    order.append(resource.resource_id)
        
        for resource in team_resources:
            if resource.resource_type == "storage_bucket":
                order.append(resource.resource_id)
        
        # Phase 3: Pub/Sub topics
        for agent_id, resources in agent_resources.items():
            for resource in resources:
                if resource.resource_type == "pubsub_topic":
                    order.append(resource.resource_id)
        
        for resource in team_resources:
            if resource.resource_type == "pubsub_topic":
                order.append(resource.resource_id)
        
        return order
    
    def _estimate_provisioning_costs(self, agent_resources: Dict[str, List[ProvisionedResource]],
                                   team_resources: List[ProvisionedResource]) -> Dict[str, float]:
        """Estimate monthly costs for provisioned resources"""
        
        costs = {
            "pubsub_topics": 0.0,
            "storage_buckets": 0.0,
            "iam_roles": 0.0,
            "total_monthly": 0.0
        }
        
        # Count resources and estimate costs
        pubsub_count = 0
        storage_gb = 0
        
        for agent_id, resources in agent_resources.items():
            for resource in resources:
                if resource.resource_type == "pubsub_topic":
                    pubsub_count += 1
                elif resource.resource_type == "storage_bucket":
                    storage_gb += 1  # Estimate 1GB per agent bucket
        
        for resource in team_resources:
            if resource.resource_type == "pubsub_topic":
                pubsub_count += 1
            elif resource.resource_type == "storage_bucket":
                storage_gb += 5  # Estimate 5GB for team audit storage
        
        # Calculate costs (rough estimates)
        costs["pubsub_topics"] = pubsub_count * 0.40  # $0.40 per topic per month
        costs["storage_buckets"] = storage_gb * 0.02   # $0.02 per GB per month
        costs["iam_roles"] = 0.0  # IAM roles are free
        costs["total_monthly"] = costs["pubsub_topics"] + costs["storage_buckets"]
        
        return costs
    
    def _generate_cleanup_procedures(self, agent_resources: Dict[str, List[ProvisionedResource]],
                                   team_resources: List[ProvisionedResource]) -> Dict[str, Any]:
        """Generate cleanup procedures for resource lifecycle management"""
        
        return {
            "cleanup_order": [
                "pubsub_subscriptions",
                "pubsub_topics", 
                "storage_bucket_objects",
                "storage_buckets",
                "iam_role_bindings",
                "iam_roles"
            ],
            "cleanup_conditions": [
                "team_termination",
                "agent_removal",
                "resource_failure",
                "cost_optimization"
            ],
            "backup_procedures": {
                "audit_logs": "archive_to_cold_storage_before_deletion",
                "agent_data": "backup_to_team_storage_before_deletion"
            }
        }
    
    def _validate_provisioning_plan(self, plan: ProvisioningPlan) -> ProvisioningPlan:
        """Validate provisioning plan against constraints and limits"""
        
        print(f"PROVISIONER: Validating plan {plan.plan_id}")
        
        # Validate resource names comply with GCP naming conventions
        all_resources = []
        for resources in plan.agent_resources.values():
            all_resources.extend(resources)
        all_resources.extend(plan.team_resources)
        
        for resource in all_resources:
            if not self._is_valid_gcp_resource_name(resource.resource_name):
                raise ValueError(f"Invalid GCP resource name: {resource.resource_name}")
        
        # Validate cost estimates are within limits
        if plan.estimated_cost["total_monthly"] > 100.0:  # $100 monthly limit
            raise ValueError(f"Estimated cost ${plan.estimated_cost['total_monthly']:.2f} exceeds limit")
        
        # Validate resource quotas
        total_topics = sum(1 for r in all_resources if r.resource_type == "pubsub_topic")
        if total_topics > 50:  # Arbitrary quota limit
            raise ValueError(f"Too many Pub/Sub topics: {total_topics}")
        
        print(f"PROVISIONER: Plan validation successful")
        
        return plan
    
    def _is_valid_gcp_resource_name(self, name: str) -> bool:
        """Validate GCP resource naming conventions"""
        
        # Basic validation - real implementation would be more comprehensive
        if len(name) < 3 or len(name) > 63:
            return False
        
        if not name.replace('-', '').replace('_', '').isalnum():
            return False
        
        return True
    
    def _execute_provisioning_plan(self, plan: ProvisioningPlan) -> ProvisioningPlan:
        """Execute the provisioning plan in dependency order"""
        
        print(f"PROVISIONER: Executing provisioning plan {plan.plan_id}")
        
        # Get all resources in a flat list for easy lookup
        all_resources = {}
        for resources in plan.agent_resources.values():
            for resource in resources:
                all_resources[resource.resource_id] = resource
        
        for resource in plan.team_resources:
            all_resources[resource.resource_id] = resource
        
        # Execute provisioning in order
        for resource_id in plan.provisioning_order:
            resource = all_resources[resource_id]
            
            print(f"PROVISIONER: Provisioning {resource.resource_type}: {resource.resource_name}")
            
            success = self._provision_resource(resource)
            
            if success:
                resource.provisioning_status = "active"
                self.resource_registry[resource_id] = asdict(resource)
                print(f"PROVISIONER: Successfully provisioned {resource.resource_name}")
            else:
                resource.provisioning_status = "failed"
                print(f"PROVISIONER: Failed to provision {resource.resource_name}")
                # In real implementation, this would trigger rollback
        
        return plan
    
    def _provision_resource(self, resource: ProvisionedResource) -> bool:
        """Provision a single GCP resource (mock implementation)"""
        
        # Mock implementation - replace with actual GCP API calls
        if resource.resource_type == "pubsub_topic":
            return self._create_pubsub_topic(resource)
        elif resource.resource_type == "storage_bucket":
            return self._create_storage_bucket(resource)
        elif resource.resource_type == "iam_role":
            return self._create_iam_role(resource)
        else:
            print(f"PROVISIONER: Unknown resource type: {resource.resource_type}")
            return False
    
    def _create_pubsub_topic(self, resource: ProvisionedResource) -> bool:
        """Create Pub/Sub topic (mock implementation)"""
        
        # Mock implementation
        print(f"PROVISIONER: Creating Pub/Sub topic {resource.resource_name}")
        
        # Simulate API call
        time.sleep(0.1)
        
        # Mock success
        return True
    
    def _create_storage_bucket(self, resource: ProvisionedResource) -> bool:
        """Create Storage bucket (mock implementation)"""
        
        # Mock implementation
        print(f"PROVISIONER: Creating Storage bucket {resource.resource_name}")
        
        # Simulate API call
        time.sleep(0.1)
        
        # Mock success
        return True
    
    def _create_iam_role(self, resource: ProvisionedResource) -> bool:
        """Create IAM role (mock implementation)"""
        
        # Mock implementation
        print(f"PROVISIONER: Creating IAM role {resource.resource_name}")
        
        # Simulate API call
        time.sleep(0.1)
        
        # Mock success
        return True
    
    def _setup_team_communication(self, plan: ProvisioningPlan, live_team: LiveTeam):
        """Setup communication infrastructure for team coordination"""
        
        print(f"PROVISIONER: Setting up team communication for {live_team.team_id}")
        
        # Configure message routing between agents
        self._configure_message_routing(plan, live_team)
        
        # Setup coordination protocols
        self._setup_coordination_protocols(plan, live_team)
        
        # Configure team status monitoring
        self._configure_team_monitoring(plan, live_team)
    
    def _configure_message_routing(self, plan: ProvisioningPlan, live_team: LiveTeam):
        """Configure message routing between agents"""
        
        # Create routing configuration based on team coordination state
        coordination = live_team.coordination_state
        
        if coordination["protocol"] == "pipeline":
            print(f"PROVISIONER: Configuring pipeline message routing")
            # Configure sequential message flow
            for agent_id, interactions in coordination["agent_interactions"].items():
                if interactions["sends_to"]:
                    print(f"PROVISIONER: Agent {agent_id} will send to {interactions['sends_to']}")
                if interactions["receives_from"]:
                    print(f"PROVISIONER: Agent {agent_id} will receive from {interactions['receives_from']}")
    
    def _setup_coordination_protocols(self, plan: ProvisioningPlan, live_team: LiveTeam):
        """Setup coordination protocols for team"""
        
        coordination = live_team.coordination_state
        protocol = coordination["protocol"]
        
        print(f"PROVISIONER: Setting up {protocol} coordination protocol")
        
        # Configure protocol-specific infrastructure
        if protocol == "pipeline":
            self._setup_pipeline_coordination(plan, live_team)
        elif protocol == "parallel":
            self._setup_parallel_coordination(plan, live_team)
    
    def _setup_pipeline_coordination(self, plan: ProvisioningPlan, live_team: LiveTeam):
        """Setup pipeline coordination infrastructure"""
        
        # Configure sequential workflow orchestration
        print(f"PROVISIONER: Pipeline coordination configured")
    
    def _setup_parallel_coordination(self, plan: ProvisioningPlan, live_team: LiveTeam):
        """Setup parallel coordination infrastructure"""
        
        # Configure parallel workflow orchestration
        print(f"PROVISIONER: Parallel coordination configured")
    
    def _configure_team_monitoring(self, plan: ProvisioningPlan, live_team: LiveTeam):
        """Configure team monitoring infrastructure"""
        
        print(f"PROVISIONER: Configuring team monitoring")
        
        # Setup performance monitoring
        # Setup health checks
        # Setup alert routing
    
    def _setup_monitoring_infrastructure(self, plan: ProvisioningPlan, live_team: LiveTeam):
        """Setup monitoring and audit infrastructure"""
        
        print(f"PROVISIONER: Setting up monitoring infrastructure")
        
        # Configure audit logging
        self._configure_audit_logging(plan, live_team)
        
        # Setup performance metrics collection
        self._setup_performance_monitoring(plan, live_team)
        
        # Configure alerting
        self._configure_alerting(plan, live_team)
    
    def _configure_audit_logging(self, plan: ProvisioningPlan, live_team: LiveTeam):
        """Configure audit logging for governance compliance"""
        
        print(f"PROVISIONER: Configuring audit logging for team {live_team.team_id}")
        
        # Setup structured logging to audit storage bucket
        # Configure log retention policies
        # Setup governance event tracking
    
    def _setup_performance_monitoring(self, plan: ProvisioningPlan, live_team: LiveTeam):
        """Setup performance monitoring for team and agents"""
        
        print(f"PROVISIONER: Setting up performance monitoring")
        
        # Configure metrics collection
        # Setup performance dashboards
        # Configure SLA monitoring
    
    def _configure_alerting(self, plan: ProvisioningPlan, live_team: LiveTeam):
        """Configure alerting for team operations"""
        
        print(f"PROVISIONER: Configuring alerting")
        
        # Setup constitutional violation alerts
        # Configure performance degradation alerts
        # Setup coordination failure alerts
    
    def cleanup_team_infrastructure(self, team_id: str) -> bool:
        """Cleanup all infrastructure for a terminated team"""
        
        print(f"PROVISIONER: Cleaning up infrastructure for team {team_id}")
        
        # Find provisioning plan for team
        plan = None
        for plan_id, plan_data in self.provisioning_plans.items():
            if plan_data["team_id"] == team_id:
                plan = ProvisioningPlan(**plan_data)
                break
        
        if not plan:
            print(f"PROVISIONER: No provisioning plan found for team {team_id}")
            return False
        
        # Execute cleanup in reverse order
        cleanup_order = plan.cleanup_procedures["cleanup_order"]
        
        for resource_type in cleanup_order:
            self._cleanup_resources_by_type(resource_type, team_id)
        
        # Remove from registry
        if plan.plan_id in self.provisioning_plans:
            del self.provisioning_plans[plan.plan_id]
        
        self._save_resources()
        
        print(f"PROVISIONER: Cleanup complete for team {team_id}")
        return True
    
    def _cleanup_resources_by_type(self, resource_type: str, team_id: str):
        """Cleanup all resources of a specific type for a team"""
        
        resources_to_cleanup = []
        for resource_id, resource_data in self.resource_registry.items():
            if (resource_data["owner_team_id"] == team_id and 
                resource_type in resource_data["resource_type"]):
                resources_to_cleanup.append(resource_id)
        
        for resource_id in resources_to_cleanup:
            print(f"PROVISIONER: Cleaning up {resource_type}: {resource_id}")
            # Mock cleanup - replace with actual GCP API calls
            del self.resource_registry[resource_id]


def test_auto_provisioning_system():
    """Test the auto-provisioning system with a live team"""
    
    # Import dependencies
    from dynamic_agent_factory import DynamicAgentFactory
    from meta_architect_agent import MetaArchitectAgent
    
    provisioner = AutoProvisioningSystem()
    factory = DynamicAgentFactory()
    architect = MetaArchitectAgent()
    
    print("=== Testing Auto-Provisioning System ===\n")
    
    # Create test team
    test_goal = "Build a customer support system for handling technical issues"
    institution_blueprint = architect.design_institution(test_goal)
    live_team = factory.instantiate_team_from_blueprint(institution_blueprint)
    
    print(f"Live team created: {live_team.team_id}")
    print(f"Agents in team: {len(live_team.live_agents)}")
    
    # Provision infrastructure for team
    print(f"\nProvisioning infrastructure for team...")
    provisioning_plan = provisioner.provision_team_infrastructure(live_team)
    
    print(f"\nProvisioning Plan:")
    print(f"  Plan ID: {provisioning_plan.plan_id}")
    print(f"  Team ID: {provisioning_plan.team_id}")
    print(f"  Agent Resources: {sum(len(resources) for resources in provisioning_plan.agent_resources.values())}")
    print(f"  Team Resources: {len(provisioning_plan.team_resources)}")
    print(f"  Estimated Monthly Cost: ${provisioning_plan.estimated_cost['total_monthly']:.2f}")
    
    print(f"\nProvisioned Resources:")
    for agent_id, resources in provisioning_plan.agent_resources.items():
        print(f"  Agent {agent_id}: {len(resources)} resources")
        for resource in resources:
            print(f"    - {resource.resource_type}: {resource.resource_name} ({resource.provisioning_status})")
    
    print(f"  Team Shared: {len(provisioning_plan.team_resources)} resources")
    for resource in provisioning_plan.team_resources:
        print(f"    - {resource.resource_type}: {resource.resource_name} ({resource.provisioning_status})")
    
    print(f"\nProvisioner Registry:")
    print(f"  Total Resources: {len(provisioner.resource_registry)}")
    print(f"  Total Plans: {len(provisioner.provisioning_plans)}")


if __name__ == "__main__":
    test_auto_provisioning_system()