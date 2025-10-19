"""
Cloud Deployment System for Project Resilience
Deploys agents with their integrations to Google Cloud Run
Enables 24/7 autonomous operation in the cloud
Part of V2 Production Build - Day 2 Final Component
"""

import os
import json
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import tempfile
import base64
import time


class CloudDeploymentSystem:
    """
    Deploys Project Resilience agents to Google Cloud Run
    Packages agent code with integrations for cloud execution
    """
    
    def __init__(self, project_id: str = "project-resilience-ai-one"):
        self.project_id = project_id
        self.region = "us-central1"
        self.deployment_dir = Path("cloud_deployments")
        self.deployment_dir.mkdir(exist_ok=True)
        
    def prepare_agent_for_deployment(self, agent_id: str) -> Path:
        """
        Prepare an agent's code for cloud deployment
        """
        print(f"\n[DEPLOY] Preparing {agent_id} for cloud deployment")
        
        # Create deployment package directory
        package_dir = self.deployment_dir / agent_id
        package_dir.mkdir(exist_ok=True)
        
        # Copy core agent files
        self._copy_agent_files(agent_id, package_dir)
        
        # Create Dockerfile
        self._create_dockerfile(agent_id, package_dir)
        
        # Create requirements.txt
        self._create_requirements_file(agent_id, package_dir)
        
        # Create Cloud Run service configuration
        self._create_service_yaml(agent_id, package_dir)
        
        # Create entrypoint script
        self._create_entrypoint(agent_id, package_dir)
        
        print(f"[DEPLOY] Package prepared at: {package_dir}")
        return package_dir
    
    def _copy_agent_files(self, agent_id: str, package_dir: Path):
        """Copy necessary files for the agent"""
        
        # Core files needed by every agent
        core_files = [
            "agent_runtime.py",
            "event_log.py",
            "integration_resilience.py",
            "secret_vault.py"
        ]
        
        # Copy core files
        for file in core_files:
            if Path(file).exists():
                shutil.copy(file, package_dir / file)
                print(f"  Copied {file}")
        
        # Copy generated integration if it exists
        integration_dir = Path("generated_integrations") / agent_id
        if integration_dir.exists():
            dest_integration_dir = package_dir / "generated_integrations" / agent_id
            dest_integration_dir.mkdir(parents=True, exist_ok=True)
            
            for file in integration_dir.glob("*.py"):
                shutil.copy(file, dest_integration_dir / file.name)
                print(f"  Copied integration: {file.name}")
        
        # Copy agent registry
        registry_path = Path("deployed_agents/agent_registry.json")
        if registry_path.exists():
            dest_registry = package_dir / "deployed_agents"
            dest_registry.mkdir(exist_ok=True)
            shutil.copy(registry_path, dest_registry / "agent_registry.json")
            print(f"  Copied agent registry")
    
    def _create_dockerfile(self, agent_id: str, package_dir: Path):
        """Create Dockerfile for the agent"""
        
        dockerfile_content = f"""# Project Resilience Agent - {agent_id}
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent code
COPY . .

# Set environment variables
ENV AGENT_ID={agent_id}
ENV GOOGLE_CLOUD_PROJECT={self.project_id}
ENV PYTHONUNBUFFERED=1

# Create necessary directories
RUN mkdir -p state deployed_agents generated_integrations validation_results package_requirements

# Run the agent
CMD ["python", "entrypoint.py"]
"""
        
        dockerfile_path = package_dir / "Dockerfile"
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content)
        
        print(f"  Created Dockerfile")
    
    def _create_requirements_file(self, agent_id: str, package_dir: Path):
        """Create requirements.txt for the agent"""
        
        # Base requirements for all agents
        requirements = [
            "google-cloud-pubsub>=2.18.0",
            "google-cloud-storage>=2.10.0",
            "google-cloud-secret-manager>=2.16.0",
            "requests>=2.31.0",
            "pyyaml>=6.0"
        ]
        
        # Check if agent has specific requirements
        agent_req_file = Path(f"package_requirements/{agent_id}_requirements.txt")
        if agent_req_file.exists():
            with open(agent_req_file) as f:
                agent_reqs = f.read().splitlines()
                # Add agent-specific requirements
                for req in agent_reqs:
                    if req and not req.startswith("#"):
                        if req not in requirements:
                            requirements.append(req)
        
        # Add integration-specific requirements
        integration_dir = Path("generated_integrations") / agent_id
        if integration_dir.exists():
            # Check what type of integration
            if (integration_dir / "database_client.py").exists():
                requirements.extend(["psycopg2-binary>=2.9.0", "sqlalchemy>=2.0.0"])
            elif (integration_dir / "websocket_client.py").exists():
                requirements.append("websocket-client>=1.6.0")
            elif (integration_dir / "slack_client.py").exists():
                requirements.append("slack-sdk>=3.19.0")
        
        req_path = package_dir / "requirements.txt"
        with open(req_path, "w") as f:
            f.write("\n".join(requirements))
        
        print(f"  Created requirements.txt with {len(requirements)} packages")
    
    def _create_service_yaml(self, agent_id: str, package_dir: Path):
        """Create Cloud Run service configuration"""
        
        service_config = {
            "apiVersion": "serving.knative.dev/v1",
            "kind": "Service",
            "metadata": {
                "name": agent_id.replace("_", "-"),
                "namespace": self.project_id,
                "annotations": {
                    "run.googleapis.com/launch-stage": "GA"
                }
            },
            "spec": {
                "template": {
                    "metadata": {
                        "annotations": {
                            "autoscaling.knative.dev/minScale": "1",
                            "autoscaling.knative.dev/maxScale": "10",
                            "run.googleapis.com/cpu-throttling": "false"
                        }
                    },
                    "spec": {
                        "containerConcurrency": 1,
                        "timeoutSeconds": 3600,
                        "serviceAccountName": f"{agent_id}@{self.project_id}.iam.gserviceaccount.com",
                        "containers": [{
                            "image": f"gcr.io/{self.project_id}/{agent_id}:latest",
                            "resources": {
                                "limits": {
                                    "cpu": "1",
                                    "memory": "512Mi"
                                }
                            },
                            "env": [
                                {"name": "AGENT_ID", "value": agent_id},
                                {"name": "GOOGLE_CLOUD_PROJECT", "value": self.project_id}
                            ]
                        }]
                    }
                }
            }
        }
        
        yaml_path = package_dir / "service.yaml"
        with open(yaml_path, "w") as f:
            # Convert to YAML format (simplified)
            import json
            json.dump(service_config, f, indent=2)
        
        print(f"  Created service.yaml")
    
    def _create_entrypoint(self, agent_id: str, package_dir: Path):
        """Create entrypoint script for the container"""
        
        entrypoint_content = f"""#!/usr/bin/env python3
\"\"\"
Entrypoint for Cloud Run deployment of {agent_id}
Handles cloud-specific initialization and starts the agent
\"\"\"

import os
import sys
import json
import time
from pathlib import Path

# Set up environment
os.environ['AGENT_ID'] = '{agent_id}'

# Import and start the agent
from agent_runtime import AgentRuntime

def main():
    agent_id = os.environ.get('AGENT_ID', '{agent_id}')
    
    print(f"Starting agent {{agent_id}} in Cloud Run")
    print(f"Project: {{os.environ.get('GOOGLE_CLOUD_PROJECT')}}")
    print(f"Region: {self.region}")
    
    try:
        # Initialize and run the agent
        runtime = AgentRuntime(agent_id)
        runtime.run()
    except Exception as e:
        print(f"Failed to start agent: {{e}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
"""
        
        entrypoint_path = package_dir / "entrypoint.py"
        with open(entrypoint_path, "w") as f:
            f.write(entrypoint_content)
        
        print(f"  Created entrypoint.py")
    
    def build_container(self, agent_id: str, package_dir: Path) -> bool:
        """Build Docker container for the agent"""
        
        print(f"\n[DEPLOY] Building container for {agent_id}")
        
        image_name = f"gcr.io/{self.project_id}/{agent_id}:latest"
        
        # Build Docker image
        build_cmd = [
            "docker", "build",
            "-t", image_name,
            str(package_dir)
        ]
        
        try:
            result = subprocess.run(build_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"[DEPLOY] ✓ Container built: {image_name}")
                return True
            else:
                print(f"[DEPLOY] ✗ Build failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"[DEPLOY] ✗ Build error: {e}")
            return False
    
    def push_to_registry(self, agent_id: str) -> bool:
        """Push container to Google Container Registry"""
        
        print(f"\n[DEPLOY] Pushing container to registry")
        
        image_name = f"gcr.io/{self.project_id}/{agent_id}:latest"
        
        # Push to GCR
        push_cmd = ["docker", "push", image_name]
        
        try:
            result = subprocess.run(push_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"[DEPLOY] ✓ Container pushed to registry")
                return True
            else:
                print(f"[DEPLOY] ✗ Push failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"[DEPLOY] ✗ Push error: {e}")
            return False
    
    def deploy_to_cloud_run(self, agent_id: str) -> Dict[str, Any]:
        """Deploy the agent to Cloud Run"""
        
        print(f"\n[DEPLOY] Deploying {agent_id} to Cloud Run")
        
        image_name = f"gcr.io/{self.project_id}/{agent_id}:latest"
        service_name = agent_id.replace("_", "-")
        
        # Deploy using gcloud
        deploy_cmd = [
            "gcloud", "run", "deploy", service_name,
            "--image", image_name,
            "--platform", "managed",
            "--region", self.region,
            "--project", self.project_id,
            "--allow-unauthenticated",
            "--min-instances", "1",
            "--max-instances", "10",
            "--memory", "512Mi",
            "--cpu", "1",
            "--set-env-vars", f"AGENT_ID={agent_id},GOOGLE_CLOUD_PROJECT={self.project_id}"
        ]
        
        try:
            result = subprocess.run(deploy_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"[DEPLOY] ✓ Agent deployed to Cloud Run")
                
                # Get service URL
                url_cmd = [
                    "gcloud", "run", "services", "describe", service_name,
                    "--platform", "managed",
                    "--region", self.region,
                    "--project", self.project_id,
                    "--format", "value(status.url)"
                ]
                
                url_result = subprocess.run(url_cmd, capture_output=True, text=True)
                service_url = url_result.stdout.strip()
                
                return {
                    "status": "deployed",
                    "agent_id": agent_id,
                    "service_name": service_name,
                    "service_url": service_url,
                    "region": self.region,
                    "project": self.project_id
                }
            else:
                return {
                    "status": "failed",
                    "error": result.stderr
                }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def deploy_agent(self, agent_id: str) -> Dict[str, Any]:
        """
        Complete deployment pipeline for an agent
        """
        
        print("="*60)
        print(f"CLOUD DEPLOYMENT PIPELINE - {agent_id}")
        print("="*60)
        
        results = {
            "agent_id": agent_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "steps": []
        }
        
        # Step 1: Prepare package
        try:
            package_dir = self.prepare_agent_for_deployment(agent_id)
            results["steps"].append({
                "step": "prepare_package",
                "status": "success",
                "package_dir": str(package_dir)
            })
        except Exception as e:
            results["steps"].append({
                "step": "prepare_package",
                "status": "failed",
                "error": str(e)
            })
            return results
        
        # Step 2: Build container
        if self.build_container(agent_id, package_dir):
            results["steps"].append({
                "step": "build_container",
                "status": "success"
            })
        else:
            results["steps"].append({
                "step": "build_container",
                "status": "failed"
            })
            return results
        
        # Step 3: Push to registry
        if self.push_to_registry(agent_id):
            results["steps"].append({
                "step": "push_to_registry",
                "status": "success"
            })
        else:
            results["steps"].append({
                "step": "push_to_registry",
                "status": "failed"
            })
            return results
        
        # Step 4: Deploy to Cloud Run
        deployment = self.deploy_to_cloud_run(agent_id)
        results["steps"].append({
            "step": "deploy_to_cloud_run",
            **deployment
        })
        
        results["deployment"] = deployment
        
        print("\n" + "="*60)
        if deployment["status"] == "deployed":
            print(f"✓ DEPLOYMENT SUCCESSFUL")
            print(f"  Agent: {agent_id}")
            print(f"  URL: {deployment.get('service_url', 'N/A')}")
        else:
            print(f"✗ DEPLOYMENT FAILED")
            print(f"  Error: {deployment.get('error', 'Unknown')}")
        print("="*60)
        
        return results


def deploy_test_agent():
    """Deploy a test agent to Cloud Run"""
    
    deployer = CloudDeploymentSystem()
    
    # Deploy the agent we tested earlier
    agent_id = "agent_custom_001"
    
    print(f"Deploying {agent_id} to Google Cloud Run...")
    results = deployer.deploy_agent(agent_id)
    
    # Save deployment results
    results_file = Path("cloud_deployments") / f"deployment_{agent_id}_{int(time.time())}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDeployment results saved to: {results_file}")
    
    return results["deployment"]["status"] == "deployed"