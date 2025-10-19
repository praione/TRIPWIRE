#!/usr/bin/env python3
"""
Direct test of Meta-Architect → Dynamic Factory → Auto-Provisioning
No demo_runner nonsense
"""

from meta_architect_agent import MetaArchitectAgent
from dynamic_agent_factory import DynamicAgentFactory
from auto_provisioning_system import AutoProvisioningSystem
import json

def main():
    print("=== DIRECT PROVISIONING TEST ===\n")
    
    # Step 1: Create blueprint
    print("[1/3] Creating blueprint with Meta-Architect...")
    architect = MetaArchitectAgent()
    blueprint = architect.design_institution_with_fallback("Create a monitoring system")
    print(f"✓ Blueprint created with {len(blueprint.team_architectures[0].agent_roles)} agents\n")
    
    # Step 2: Instantiate agents
    print("[2/3] Creating live agents with Dynamic Factory...")
    factory = DynamicAgentFactory()
    live_team = factory.instantiate_team_from_blueprint(blueprint)
    print(f"✓ Team {live_team.team_id} created with {len(live_team.live_agents)} agents\n")
    
    
    # Step 3: Provision infrastructure
    print("[3/3] Provisioning GCP resources with Auto-Provisioning...")
    provisioner = AutoProvisioningSystem()
    provisioning_plan = provisioner.provision_team_infrastructure(live_team)
    
    print(f"\n=== RESULTS ===")
    print(f"Team ID: {live_team.team_id}")
    print(f"Agents Created: {len(live_team.live_agents)}")
    print(f"Resources Provisioned: {len(provisioning_plan.completed_resources)}")
    print(f"Resources Failed: {len(provisioning_plan.failed_resources)}")
    print(f"Registry Updated: deployed_agents/agent_registry.json")
    
    # Check for governance subscriptions
    print(f"\n=== GOVERNANCE CHECK ===")
    for agent_id in live_team.live_agents:
        print(f"Agent {agent_id}: governance_sub would be agent_{agent_id}_governance_sub")

if __name__ == "__main__":
    main()