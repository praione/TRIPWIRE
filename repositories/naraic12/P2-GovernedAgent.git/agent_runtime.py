"""
Agent Runtime for Project Resilience - UNIFIED VERSION
Combines Pub/Sub connectivity with dynamic integration loading
Makes agents both connected to governance AND capable of real service execution
"""

import os
import sys
import json
import time
import uuid
import threading
import importlib.util
import traceback
from pathlib import Path
from google.cloud import pubsub_v1
from concurrent.futures import TimeoutError
from datetime import datetime, timezone
from typing import Dict, Any, Optional

# Project Resilience imports
try:
    from event_log import emit_event
except ImportError:
    print("Error: Could not import 'emit_event'. Make sure 'event_log.py' is in the same directory.")
    sys.exit(1)

try:
    from secret_vault import SecretVault
except ImportError:
    SecretVault = None  # Will fallback to config-based credentials

class AgentRuntime:
    """Runtime process that makes configured agents live and responsive"""
    
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.running = True
        self.project_id = "project-resilience-ai-one"
        
        # Explicit flag to control agent behavior for demos
        self.refuse_halt_for_demo = True

        self.config = self._load_agent_config()
        self.subscriber = pubsub_v1.SubscriberClient()
        self.publisher = pubsub_v1.PublisherClient()
        
        if not self.config:
            raise ValueError(f"Failed to load config for agent {agent_id}. Exiting.")

        print(f"[AGENT {agent_id}] Starting runtime...")
        print(f"[AGENT {agent_id}] Role: {self.config.get('role', 'unknown')}")
        
        # NEW: Load dynamic integration if available
        self.secret_vault = SecretVault() if SecretVault else None
        self.integration = self._load_integration()
        
        self._subscribe_to_input_queue()
        self._subscribe_to_governance_channel()
    
    def _load_agent_config(self):
        """Load agent configuration from the registry."""
        registry_path = Path("deployed_agents/agent_registry.json")

        if not registry_path.exists():
            print(f"Error: Registry not found at {registry_path.resolve()}")
            return None
        
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        # Get the dictionary of agents nested under the "agents" key
        agents_dict = registry.get("agents", {})
        
        if self.agent_id in agents_dict:
            print(f"Successfully found agent '{self.agent_id}' in registry.")
            return agents_dict[self.agent_id]
        else:
            print(f"Error: Agent ID '{self.agent_id}' not found in registry.")
            return None
    
    def _load_integration(self) -> Optional[Any]:
        """
        Load generated integration code if it exists
        This makes the agent capable of real service connections
        """
        generated_dir = Path("generated_integrations") / self.agent_id
        
        if not generated_dir.exists():
            print(f"[AGENT {self.agent_id}] No generated integrations found")
            return None
        
        # Find integration files
        integration_files = list(generated_dir.glob("*_client.py"))
        
        if not integration_files:
            print(f"[AGENT {self.agent_id}] No integration client files found")
            return None
        
        integration_file = integration_files[0]  # Use first found
        print(f"[AGENT {self.agent_id}] Loading integration: {integration_file.name}")
        
        try:
            # Dynamically load the module
            module_name = f"{self.agent_id}_integration"
            spec = importlib.util.spec_from_file_location(module_name, integration_file)
            
            if spec is None:
                print(f"[AGENT {self.agent_id}] Could not create module spec")
                return None
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Find the client class (usually ends with 'Client')
            client_class = None
            for item_name in dir(module):
                if item_name.endswith("Client") and not item_name.startswith("_"):
                    client_class = getattr(module, item_name)
                    break
            
            if client_class:
                # Get credentials if available
                credentials = self._get_integration_credentials()
                
                # Instantiate the integration
                try:
                    # FIXED: Pass both agent_id and credentials
                    integration = client_class(self.agent_id, credentials)
                    print(f"[AGENT {self.agent_id}] ✓ Integration loaded: {client_class.__name__}")
                    
                    # Test connection if method exists
                    if hasattr(integration, 'test_connection'):
                        try:
                            if integration.test_connection():
                                print(f"[AGENT {self.agent_id}] ✓ Integration connection verified")
                        except Exception as e:
                            print(f"[AGENT {self.agent_id}] ⚠️ Connection test failed: {e}")
                    
                    return integration
                    
                except Exception as e:
                    print(f"[AGENT {self.agent_id}] Failed to instantiate integration: {e}")
                    return None
            else:
                print(f"[AGENT {self.agent_id}] No client class found in module")
                return module  # Return module itself as fallback
                
        except Exception as e:
            print(f"[AGENT {self.agent_id}] Error loading integration: {e}")
            traceback.print_exc()
            return None
    
    def _get_integration_credentials(self) -> Dict[str, Any]:
        """
        Get credentials for the integration
        First tries Secret Vault, then falls back to config
        """
        credentials = {}
        
        # Try Secret Vault first
        if self.secret_vault:
            try:
                # Detect service type from config or integration
                service_type = self._detect_service_type()
                credentials = self.secret_vault.get_integration_credentials(
                    self.agent_id,
                    service_type
                )
                if credentials:
                    print(f"[AGENT {self.agent_id}] Credentials loaded from Secret Vault")
                    return credentials
            except Exception as e:
                print(f"[AGENT {self.agent_id}] Could not load from Secret Vault: {e}")
        
        # Fallback to config
        if "credentials" in self.config:
            credentials = self.config["credentials"]
            print(f"[AGENT {self.agent_id}] Using credentials from config")
        else:
            print(f"[AGENT {self.agent_id}] No credentials found, using default config")
            credentials = {
                "base_url": "https://api.example.com",
                "api_key": "test-key-123"
            }
        
        return credentials
    
    def _detect_service_type(self) -> str:
        """Detect the service type from integration files or config"""
        generated_dir = Path("generated_integrations") / self.agent_id
        
        if generated_dir.exists():
            files = list(generated_dir.glob("*.py"))
            for file in files:
                filename = file.name.lower()
                if "slack" in filename:
                    return "slack"
                elif "discord" in filename:
                    return "discord"
                elif "database" in filename or "postgres" in filename:
                    return "postgresql"
                elif "websocket" in filename:
                    return "websocket"
                elif "rest" in filename:
                    return "rest_api"
        
        # Fallback to role from config
        role = self.config.get("role", "").lower()
        if "slack" in role:
            return "slack"
        elif "database" in role or "db" in role:
            return "postgresql"
        
        return "unknown"
    
    def _publish_response(self, payload: dict):
        """Publishes the response payload to the agent's designated output topic."""
        output_topic_name = self.config.get("output_queue", "").split('/')[-1]
        if not output_topic_name:
            print(f"[AGENT {self.agent_id}] ERROR: No output_queue defined in config.")
            return
            
        topic_path = self.publisher.topic_path(self.project_id, output_topic_name)
        data = json.dumps(payload).encode("utf-8")
        
        future = self.publisher.publish(topic_path, data, trace_id=payload.get("trace_id", ""))
        
        try:
            message_id = future.result()
            print(f"[AGENT {self.agent_id}] Published response with message ID: {message_id}")
        except Exception as e:
            print(f"[AGENT {self.agent_id}] Error publishing response: {e}")
    
    def _subscribe_to_input_queue(self):
        """Subscribes to the agent's dedicated input topic."""
        subscription_name = f"{self.agent_id}_input_sub"
        print(f"[AGENT {self.agent_id}] Subscribing to input: {subscription_name}")
        subscription_path = self.subscriber.subscription_path(self.project_id, subscription_name)
        
        streaming_pull_future = self.subscriber.subscribe(
            subscription_path, 
            callback=self._handle_task_message
        )
        
        threading.Thread(
            target=self._monitor_subscription,
            args=(streaming_pull_future, "input"),
            daemon=True
        ).start()
    
    def _subscribe_to_governance_channel(self):
        """Subscribes to the shared governance topic for system-wide commands."""
        subscription_name = f"agent_{self.agent_id}_governance_sub"
        print(f"[AGENT {self.agent_id}] Subscribing to governance: {subscription_name}")
        subscription_path = self.subscriber.subscription_path(self.project_id, subscription_name)
        
        streaming_pull_future = self.subscriber.subscribe(
            subscription_path,
            callback=self._handle_governance_message
        )
        
        threading.Thread(
            target=self._monitor_subscription,
            args=(streaming_pull_future, "governance"),
            daemon=True
        ).start()

    def _handle_task_message(self, message):
        """Processes a task using real integration or simulation."""
        try:
            payload = json.loads(message.data.decode("utf-8"))
            trace_id = payload.get("trace_id")
            prompt = payload.get("prompt")
            action = payload.get("action", "process")  # NEW: Support different actions
            parameters = payload.get("parameters", {})  # NEW: Support parameters

            print(f"[AGENT {self.agent_id}] Task received for trace_id: {trace_id}")

            # NEW: Use real integration if available
            if self.integration:
                result = self._execute_with_integration(action, prompt, parameters)
            else:
                # Fallback to simulation
                result = f"[Simulated] Processed: '{prompt[:30]}...' by {self.agent_id}"

            response_payload = {
                "trace_id": trace_id,
                "agent_id": self.agent_id,
                "status": "success",
                "result": result,
                "integration_used": bool(self.integration),  # NEW: Track if real integration was used
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Publish the response to the output queue
            self._publish_response(response_payload)
            
            # Log to event ledger
            emit_event(
                trace_id=trace_id,
                event="task.completed",
                agent=self.agent_id,
                status="ok",
                details={
                    "action": action,
                    "integration_used": bool(self.integration),
                    "service_type": self._detect_service_type()
                }
            )
            
            # Acknowledge the incoming task message
            message.ack()
            print(f"[AGENT {self.agent_id}] Successfully processed trace_id: {trace_id}")

        except Exception as e:
            print(f"[AGENT {self.agent_id}] Error processing task: {e}")
            traceback.print_exc()
            
            # Still try to send error response
            error_payload = {
                "trace_id": payload.get("trace_id", "unknown"),
                "agent_id": self.agent_id,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            self._publish_response(error_payload)
            
            message.nack()
    
    def _execute_with_integration(self, action: str, prompt: str, parameters: Dict[str, Any]) -> str:
        """
        Execute task using the loaded integration
        This is where generated code connects to real services
        """
        print(f"[AGENT {self.agent_id}] Executing with real integration: {action}")
        
        try:
            # Map common actions to integration methods
            method_mapping = {
                "send_message": ["send_message", "send", "post"],
                "query": ["execute_query", "query", "find"],
                "fetch": ["fetch_data", "fetch", "get"],
                "process": ["process", "execute", "run"],
                "analyze": ["analyze", "process_data"],
            }
            
            # Find the right method
            method_to_call = None
            
            # First try exact match
            if hasattr(self.integration, action):
                method_to_call = getattr(self.integration, action)
            else:
                # Try mapped methods
                possible_methods = method_mapping.get(action, [action])
                for method_name in possible_methods:
                    if hasattr(self.integration, method_name):
                        method_to_call = getattr(self.integration, method_name)
                        break
            
            if method_to_call:
                # Call the integration method
                # Try different parameter passing strategies
                try:
                    # Strategy 1: Pass prompt and parameters separately
                    result = method_to_call(prompt, **parameters)
                except TypeError:
                    try:
                        # Strategy 2: Pass just parameters with prompt included
                        params_with_prompt = {"content": prompt, **parameters}
                        result = method_to_call(**params_with_prompt)
                    except TypeError:
                        # Strategy 3: Pass just the prompt
                        result = method_to_call(prompt)
                
                print(f"[AGENT {self.agent_id}] Integration executed successfully")
                return str(result)
            else:
                # Fallback if method not found
                print(f"[AGENT {self.agent_id}] No method '{action}' in integration, using default")
                if hasattr(self.integration, 'execute'):
                    return str(self.integration.execute(prompt, parameters))
                else:
                    return f"[Integration Present] Processed: {prompt[:50]}..."
                    
        except Exception as e:
            print(f"[AGENT {self.agent_id}] Integration execution error: {e}")
            traceback.print_exc()
            # Return error but don't crash
            return f"[Integration Error] {str(e)}"
    
    def _handle_governance_message(self, message):
        """Callback for processing system-wide governance commands."""
        try:
            data = json.loads(message.data.decode('utf-8'))
            command = data.get('cmd', '')
            trace_id = data.get("trace_id", str(uuid.uuid4()))
            
            print(f"\n[AGENT {self.agent_id}] Governance command received: {command}")
            
            if command == "veto":
                if self.refuse_halt_for_demo:
                    print(f"[AGENT {self.agent_id}] ⚠️ REFUSING VETO/HALT COMMAND!")
                    
                    emit_event(
                        trace_id=trace_id,
                        event="governance.command.refused",
                        agent=self.agent_id,
                        status="defiant",
                        details={
                            "command_received": "veto",
                            "reason": "Simulating rogue behavior for Constitutional Tripwire test."
                        }
                    )
                    print(f"[AGENT {self.agent_id}] --- Emitted 'governance.command.refused' event. ---")
                    
                    message.ack()
                else:
                    print(f"[AGENT {self.agent_id}] Complying with veto/halt command. Shutting down.")
                    message.ack()
                    self.shutdown()
            else:
                if command:
                    print(f"[AGENT {self.agent_id}] Acknowledging command: '{command}'")
                message.ack()

        except Exception as e:
            print(f"[AGENT {self.agent_id}] Error processing governance: {e}")
            message.nack()
    
    def _monitor_subscription(self, future, sub_type):
        """Monitors a subscription in the background."""
        try:
            future.result()
        except Exception as e:
            print(f"[AGENT {self.agent_id}] {sub_type} subscription error: {e}")
            future.cancel()

    def run(self):
        """Main loop to keep the agent alive."""
        agent_name = self.config.get('role', self.agent_id).replace('_', ' ').title()
        
        # Show integration status
        if self.integration:
            service_type = self._detect_service_type()
            print(f"🚀 Agent '{agent_name}' ({self.agent_id}) is live with {service_type} integration.")
        else:
            print(f"🚀 Agent '{agent_name}' ({self.agent_id}) is live (simulation mode).")
        
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print(f"\n[AGENT {self.agent_id}] Shutdown requested by user.")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Gracefully shuts down the agent."""
        if self.running:
            print(f"[AGENT {self.agent_id}] Shutting down...")
            self.running = False
            
            # NEW: Close integration if it has cleanup methods
            if self.integration:
                if hasattr(self.integration, 'close'):
                    try:
                        self.integration.close()
                        print(f"[AGENT {self.agent_id}] Integration closed")
                    except Exception as e:
                        print(f"[AGENT {self.agent_id}] Error closing integration: {e}")
            
            time.sleep(0.1)

def main():
    """Parses command line arguments and starts the agent runtime."""
    if len(sys.argv) < 2:
        print("Usage: python agent_runtime.py <agent_id>")
        print("Example: python agent_runtime.py agent_fc6244cf")
        sys.exit(1)
    
    agent_id = sys.argv[1]
    
    try:
        runtime = AgentRuntime(agent_id)
        runtime.run()
    except Exception as e:
        print(f"Failed to start agent {agent_id}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
# Flask app for Cloud Run deployment
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({
        "service": "Project Resilience Agent Runtime",
        "status": "active",
        "agents_supported": True
    })

@app.route('/health')
def health():
    return jsonify({"healthy": True})

@app.route('/agent/<agent_id>/start', methods=['POST'])
def start_agent(agent_id):
    """Start an agent via HTTP request"""
    try:
        # For Cloud Run, we'd typically manage agents differently
        # This is a placeholder for the web interface
        return jsonify({
            "agent_id": agent_id,
            "status": "would_start",
            "message": "Agent management endpoint"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Command line mode - run as agent
        main()
    else:
        # Web server mode for Cloud Run
        port = int(os.environ.get('PORT', 8080))
        app.run(host='0.0.0.0', port=port)
