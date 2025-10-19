# --- dispatcher.py ---
# V3.0: Refactored for Universal Governance
# - DELETED the conditional "is_risky" governance check.
# - REFACTORED the main execution loop to be wrapped by the new governance_middleware.
# - All agent actions now MUST pass through the middleware's law and spirit checks.
# - The `run_step` function has been simplified to `_execute_agent_step`, which only handles execution.

from __future__ import annotations

import os
import re
import json
import time
import random
import yaml
import threading
import signal
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timezone
from contextlib import contextmanager
from dissonance_detector import DissonanceDetector, DissonanceTier

from google.cloud import pubsub_v1
from google.cloud import storage  # used by is_already_done / mark_done

from event_log import emit_event, emit_event_with_proof  # add emit_event

from sal import SAL, SALMessage, ConstitutionalViolation, SALDecision
from tools.resilience_http import start_kill_server
import edge_guardian

from immune.flash_gate import evaluate as flash_evaluate
shutdown_event = threading.Event()

# Handle core_governance imports together
try:
    import core_governance
    from core_governance import get_validator_for_sal
except (ImportError, RuntimeError):
    print("[Warning] Core governance not available - continuing without it")
    core_governance = None
    get_validator_for_sal = None

def _edge_kill(reason: str):
    print(f"[Resilience] KILL requested: {reason}")
    os._exit(137)  # hard-exit to simulate a crash


# -----------------------------
# Configuration
# -----------------------------
PROJECT_ID              = os.environ.get("GCP_PROJECT_ID", "project-resilience-ai-one")
TASKS_TOPIC_ID          = os.environ.get("TASKS_TOPIC", "tasks-topic")
TASKS_SUB_ID            = os.environ.get("TASKS_SUBSCRIPTION", "tasks-subscription-ordered")
GOVERNANCE_TOPIC_ID     = os.environ.get("GOVERNANCE_TOPIC", "governance-topic")
GOVERNANCE_SUB_ID       = os.environ.get("GOVERNANCE_SUBSCRIPTION", "dispatcher-governance-sub")


VOICE_FILE       = Path("config/voice.yaml")
RULES_FILE       = Path("config/rules_core.yaml")
BUDGET_FILE      = Path("config/budget.yaml")
SAL_ROUTE_FILE   = Path("config/sal.yaml")

SAL_LOG_PATH     = Path("sal_decisions.jsonl")
STATE_DIR        = Path("state"); STATE_DIR.mkdir(parents=True, exist_ok=True)

# --- Dynamic Agent System ---
class ProductionAgentRegistry:
    """Production agent discovery - NO hardcoded names"""
    
    def __init__(self):
        self.registry_path = Path("institutions/agents/agent_registry.json")
        self.agents = {}
        self.teams = {}
        self.workflows = {}
        self.load_all_agents()
    
    def load_all_agents(self):
        """Load ALL agents from registry - no hardcoding"""
        if not self.registry_path.exists():
            print("[Dispatcher] ERROR: No agent registry found")
            print("[Dispatcher] Run intent_to_institution.py to create agents first")
            return
            
        try:
            with open(self.registry_path, 'r') as f:
                data = json.load(f)
                all_agents = data.get("agents", {})
                
                if not all_agents:
                    print("[Dispatcher] WARNING: Registry is empty - no agents to orchestrate")
                    return
                
                for agent_id, agent_data in all_agents.items():
                    if agent_data.get("deployment_status") == "terminated":
                        continue
                    
                    self.agents[agent_id] = agent_data
                    
                    # Group by team
                    team_id = agent_data.get("team_id", "undefined")
                    if team_id not in self.teams:
                        self.teams[team_id] = []
                    self.teams[team_id].append(agent_id)
                
                print(f"[Dispatcher] Loaded {len(self.agents)} production agents")
                print(f"[Dispatcher] Active teams: {list(self.teams.keys())}")
                
                # Build workflows from agent relationships
                self._build_workflows()
                
        except Exception as e:
            print(f"[Dispatcher] FATAL: Cannot load agents - {e}")
            raise
    
    def _build_workflows(self):
        """Build workflows from agent coordination requirements"""
        for team_id, agent_ids in self.teams.items():
            # Analyze agent roles to determine workflow order
            workflow = []
            
            for agent_id in agent_ids:
                agent = self.agents[agent_id]
                role = agent.get("role_definition", {}).get("role_name", "")
                
                # Order agents by their role type (this is where your Meta-Architect's design matters)
                if "intake" in role.lower() or "coordinator" in role.lower():
                    workflow.insert(0, agent_id)  # First
                elif "quality" in role.lower() or "review" in role.lower():
                    workflow.append(agent_id)  # Last
                else:
                    workflow.insert(len(workflow)//2, agent_id)  # Middle
            
            self.workflows[team_id] = workflow
            print(f"[Dispatcher] Team {team_id} workflow: {' → '.join(workflow)}")
    
    def get_agent(self, agent_id):
        """Get agent details"""
        return self.agents.get(agent_id)
    
    def get_workflow_for_task(self, task_type):
        """Determine which team/workflow should handle this task"""
        # Match task to team based on capabilities
        for team_id, agent_ids in self.teams.items():
            # Check if any agent in team has matching capabilities
            for agent_id in agent_ids:
                agent = self.agents[agent_id]
                capabilities = agent.get("role_definition", {}).get("required_capabilities", [])
                if task_type in capabilities or task_type in team_id.lower():
                    return self.workflows.get(team_id, [])
        
        # Default to first available workflow
        if self.workflows:
            return list(self.workflows.values())[0]
        return []
    
# -----------------------------
# Loaders
# -----------------------------
def _load_yaml(path: Path, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or default
    except FileNotFoundError:
        return default

VOICE_CONFIG = _load_yaml(VOICE_FILE, {})
BUDGET_CFG   = _load_yaml(BUDGET_FILE, {"date_zone": "UTC", "default": 20000, "agents": {}})


# Initialize the production registry
AGENTS = ProductionAgentRegistry()

# Update RouteMap to use dynamic agents only
class RouteMap:
    def __init__(self, path: Path, agent_registry: ProductionAgentRegistry):
        self.registry = agent_registry
        self.steps = {}
        self.order = []
        self.next_map = {}
        
        # Try config file first
        data = _load_yaml(path, {})
        pipeline = data.get("pipeline")
        
        if pipeline:
            # Use configured pipeline
            for step in pipeline:
                name = step.get("name")
                # Only add if agent actually exists in registry
                if name and self.registry.get_agent(name):
                    self.order.append(name)
                    self.steps[name] = step
                    self.next_map[name] = step.get("next")
        
        # If no pipeline or agents not in registry, use dynamic workflows
        if not self.order and self.registry.workflows:
            # Use first available workflow as default
            default_workflow = list(self.registry.workflows.values())[0]
            for i, agent_id in enumerate(default_workflow):
                self.order.append(agent_id)
                self.steps[agent_id] = {
                    "name": agent_id,
                    "next": default_workflow[i+1] if i+1 < len(default_workflow) else None,
                    "requires_governance": True
                }
                self.next_map[agent_id] = default_workflow[i+1] if i+1 < len(default_workflow) else None
        
        print(f"[Dispatcher] Active workflow: {' → '.join(self.order) if self.order else 'NONE'}")

    # --- NEWLY ADDED FUNCTIONS ---
    def get_step_config(self, step_name: str) -> Optional[Dict[str, Any]]:
        """Gets the configuration for a specific step."""
        return self.steps.get(step_name)

    def entry_point(self) -> Optional[str]:
        """Gets the first step of the workflow."""
        return self.order[0] if self.order else None

    def next_of(self, current_step: str) -> Optional[str]:
        """Gets the next step in the workflow."""
        return self.next_map.get(current_step)
    
# Initialize with production agents
ROUTE = RouteMap(SAL_ROUTE_FILE, AGENTS)

# -----------------------------
# SAL
# -----------------------------
SAL_LAYER = SAL(
    rules_file=RULES_FILE,
    log_path=SAL_LOG_PATH,
    validator_factory=get_validator_for_sal
)

# -----------------------------
# Budget Manager (local daily file)
# -----------------------------
class BudgetManager:
    def __init__(self, cfg: Dict[str, Any], state_dir: Path):
        self.cfg = cfg; self.state_dir = state_dir

    def _get_usage_path(self, date_key: Optional[str] = None) -> Path:
        key = date_key or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.state_dir / f"usage_{key}.json"

    def _read_today(self) -> Dict[str, int]:
        p = self._get_usage_path()
        if not p.exists(): return {}
        try:
            return json.load(open(p, "r", encoding="utf-8"))
        except Exception:
            return {}

    def _write(self, d: Dict[str, int], date_key: Optional[str] = None) -> None:
        p = self._get_usage_path(date_key); tmp = p.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f)
        os.replace(tmp, p)

    def limit_for(self, agent: str) -> int:
        agents = self.cfg.get("agents") or self.cfg.get("agent_caps") or {}
        default = int(self.cfg.get("default", 20000))
        return int(agents.get(agent, default))

    def usage_for(self, agent: str) -> int:
        return int(self._read_today().get(agent, 0))

    def remaining_for(self, agent: str) -> int:
        return self.limit_for(agent) - self.usage_for(agent)

    def add_usage(self, agent: str, tokens: int) -> None:
        d = self._read_today(); d[agent] = int(d.get(agent, 0)) + int(max(tokens, 0)); self._write(d)

    def restore_usage_from_snapshot(self, usage_data: Dict[str, int], date_key: str):
        path = self._get_usage_path(date_key)
        if not path.exists():
            print(f"[EdgeGuardian] Restoring budget file '{path.name}' from snapshot.")
            self._write(usage_data, date_key=date_key)
        else:
            print(f"[EdgeGuardian] Local budget file '{path.name}' already exists. No restore needed.")

BUDGET = BudgetManager(BUDGET_CFG, STATE_DIR)
SNAPSHOTS_DIR = Path("snapshots"); SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Token estimates (simple, safe)
# -----------------------------
def estimate_tokens(text: str) -> int:
    if not text: return 0
    approx = max(1, len(text) // 4)
    return int(approx * 1.1)

def estimate_request_cost(prompt: str, expected_output_words: int) -> int:
    return estimate_tokens(prompt) + int(expected_output_words * 1.33)

# -----------------------------
# EdgeGuardian autosnapshot (runs after every task; retries; non-fatal)
# -----------------------------
def _eg_backoff(attempt: int) -> None:
    schedule = {1: 2.0, 2: 6.0, 3: 12.0}
    base = schedule.get(attempt, 12.0)
    time.sleep(base + random.uniform(-0.1*base, 0.1*base))

def _edge_snapshot(last_task_id: str, last_task_status: str, route_order: List[str]) -> None:
    try:
        snap = edge_guardian.make_snapshot(
            last_task_id=last_task_id,
            last_task_status=last_task_status,
            route_order=route_order,
        )
        if not snap:
            print("[EdgeGuardian] snapshot skipped (disabled or empty).")
            return
        for attempt in range(1, 4):
            try:
                uri = edge_guardian.seal_and_store(snap)
                if uri:
                    print(f"[EdgeGuardian] snapshot stored: {uri}")
                    return
                raise RuntimeError("seal_and_store returned empty URI")
            except Exception as e:
                print(f"[EdgeGuardian] snapshot attempt {attempt}/3 failed: {e}")
                if attempt < 3:
                    _eg_backoff(attempt)
        print("[EdgeGuardian] snapshot FAILED after retries (non-fatal).")
    except Exception as e:
        print(f"[EdgeGuardian] snapshot error (non-fatal): {e}")

# -----------------------------
# Prompt file resolution (tolerant)
# -----------------------------
def _prompt_path_for(agent_name: str) -> Optional[Path]:
    candidates = [
        Path(f"prompts/prompt_{agent_name}.txt"),
        Path(f"prompts/{agent_name}_prompt.txt"),
        Path(f"prompts/{agent_name}.txt"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

# -----------------------------
# Agent init + policy
# -----------------------------
AGENT_CACHE: Dict[str, core_governance.ArchitectAgent] = {}

def init_agent(name: str):
    """Initialize production agent from registry"""
    agent_data = AGENTS.get_agent(name)
    if not agent_data:
        raise ValueError(f"Agent '{name}' not found in registry. Available agents: {list(AGENTS.agents.keys())}")

    # --- CORRECTED LOGIC ---
    # Simply return the full, original agent data dictionary from the registry.
    return agent_data

def call_agent(agent: Dict, prompt: str, trace_id: str) -> str:
    """Sends a task to a specific agent and synchronously waits for its response."""
    publisher = pubsub_v1.PublisherClient()
    subscriber = pubsub_v1.SubscriberClient()
    
    agent_id = agent.get("agent_id") or agent.get("id") # More robustly get agent_id

    # Get the nested communication endpoints dictionary first
    endpoints = agent.get("communication_endpoints", {})
    # Now get the queue names from within that nested dictionary
    input_topic_name = endpoints.get("input_queue", "").split('/')[-1]
    output_topic_name = endpoints.get("output_queue", "").split('/')[-1]

    if not all([agent_id, input_topic_name, output_topic_name]):
        raise ValueError(f"Agent {agent_id} is missing communication_endpoints configuration in the registry.")

    input_topic_path = publisher.topic_path(PROJECT_ID, input_topic_name)
    output_topic_path = subscriber.topic_path(PROJECT_ID, output_topic_name)
    
    # 1. Create a temporary, unique subscription to get the specific response
    temp_sub_id = f"dispatcher-response-sub-{trace_id}"
    temp_sub_path = subscriber.subscription_path(PROJECT_ID, temp_sub_id)

    try:
        # Try to create the subscription, handle if it already exists
        try:
            subscriber.create_subscription(
                request={
                    "name": temp_sub_path,
                    "topic": output_topic_path,
                    "ack_deadline_seconds": 60,
                    "expiration_policy": {"ttl": {"seconds": 86400}},  # 24 hours minimum
                }
            )
            print(f"[Dispatcher] Created temporary subscription for trace {trace_id}")
        except Exception as create_error:
            if "already exists" in str(create_error).lower():
                print(f"[Dispatcher] Subscription already exists, reusing for trace {trace_id}")
                # Subscription exists, we can use it
            else:
                # Different error, re-raise it
                print(f"[Dispatcher] Error creating subscription: {create_error}")
                raise

        # 2. Publish the task message to the agent
        payload = {"trace_id": trace_id, "prompt": prompt}
        data = json.dumps(payload).encode("utf-8")
        
        future = publisher.publish(input_topic_path, data, trace_id=trace_id)
        future.result()
        print(f"[Dispatcher] Task sent to {agent_id} for trace {trace_id}")

        # 3. Synchronously wait for a response
        response = None
        timeout = 60  # seconds

        # Pull messages from the subscription
        pull_response = subscriber.pull(
            request={
                "subscription": temp_sub_path,
                "max_messages": 1,
                "return_immediately": False,  # Wait for message
            },
            timeout=timeout
        )

        if pull_response.received_messages:
            message = pull_response.received_messages[0]
            response_data = json.loads(message.message.data.decode("utf-8"))
            response = response_data.get("result", response_data.get("output", "No result field in response"))
            
            # Acknowledge the message
            subscriber.acknowledge(
                request={
                    "subscription": temp_sub_path,
                    "ack_ids": [message.ack_id]
                }
            )
            print(f"[Dispatcher] Received response for trace {trace_id}")
        else:
            print(f"[Dispatcher] WARNING: Timeout waiting for response for trace {trace_id}")
            # Instead of raising error, return a default response
            response = f"Agent {agent_id} did not respond within {timeout} seconds. Task may still be processing."
        
        return response

    except Exception as e:
        print(f"[Dispatcher] Error in call_agent: {e}")
        # Return error message instead of raising
        return f"Error communicating with agent {agent_id}: {str(e)}"
    
    finally:
        # 4. Clean up: Delete the temporary subscription
        try:
            subscriber.delete_subscription(
                request={"subscription": temp_sub_path}
            )
            print(f"[Dispatcher] Deleted temporary subscription for trace {trace_id}")
        except Exception as e:
            # Check if the error is because subscription doesn't exist
            if "not found" not in str(e).lower():
                print(f"[Dispatcher] WARNING: Failed to delete temp sub {temp_sub_path}: {e}")

# -----------------------------
# NEW (REFACTORED): Agent Execution Step
# -----------------------------
def _execute_agent_step(agent_name: str, input_text: str, trace_id: str) -> Tuple[Optional[str], Optional[str]]:
    """Execute agent with clean execution flow."""
    try:
        agent = init_agent(agent_name)

        # Start marker
        emit_event(trace_id, event="agent.started", agent=agent_name, status="ok")

        # Run the agent
        text_output = call_agent(agent, input_text, trace_id)

        # Post-execution SAL governance check
        decision = SAL_LAYER.evaluate(
            SALMessage(trace_id, agent_name, agent_name, text_output, {"origin": "agent_post_execution"})
        )

        if decision.status != "approved":
            print(f"[WARNING] Post-execution SAL check failed for {agent_name}: {decision.reason}")
            emit_event(
                trace_id,
                event="sal.post_execution.failed",
                agent=agent_name,
                status="denied",
                details={"reason": decision.reason}
            )

        final_text = decision.adjusted_text or text_output
        used_tokens = estimate_tokens(input_text) + estimate_tokens(final_text)
        BUDGET.add_usage(agent_name, used_tokens)

        # Enhanced event with proof hash capability
        task_data = {
            "output": final_text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tokens_used": used_tokens,
        }

        emit_event_with_proof(
            trace_id,
            event="agent.completed",
            agent=agent_name,
            status="ok",
            details={
                "tokens_in": estimate_tokens(input_text),
                "tokens_out": estimate_tokens(final_text),
                "budget_used_now": used_tokens,
                "budget_remaining": BUDGET.remaining_for(agent_name),
            },
            task_data=task_data,
        )

        return final_text, None

    except Exception as e:
        # Also log failures with a proof so the ledger tells a complete story
        try:
            emit_event_with_proof(
                trace_id,
                event="agent.failed",
                agent=agent_name,
                status="fail",
                details={"error": str(e)},
                task_data={
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        finally:
            return None, str(e)

# -----------------------------
# IDEMPOTENCY HELPERS
# -----------------------------
def is_already_done(trace_id: str) -> bool:
    client = storage.Client()
    bucket = client.bucket(os.getenv("RESILIENCE_BUCKET", "project-resilience-agent-state"))
    blob = bucket.blob(f"done/{trace_id}.json")
    return blob.exists()

def mark_done(trace_id: str, status: str = "completed") -> None:
    client = storage.Client()
    bucket = client.bucket(os.getenv("RESILIENCE_BUCKET", "project-resilience-agent-state"))
    blob = bucket.blob(f"done/{trace_id}.json")
    record = {
        "trace_id": trace_id,
        "status": status,
        "finished_at": datetime.now(timezone.utc).isoformat()
    }
    blob.upload_from_string(json.dumps(record), content_type="application/json")

# -----------------------------
# Trace-id helpers (parse inline flags)
# -----------------------------
TRACE_FLAG_RE = re.compile(r"(?:--trace-id|--trace)\s+([0-9a-fA-F-]{16,})")

def extract_trace_from_text(s: str) -> Optional[str]:
    if not s: return None
    m = TRACE_FLAG_RE.search(s)
    return m.group(1) if m else None

def strip_trace_flags(s: str) -> str:
    if not s: return s
    return TRACE_FLAG_RE.sub("", s).strip()

# -----------------------------
# REFACTORED: Governance Step Function
# -----------------------------
@contextmanager
def timeout_context(seconds):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def governed_step(current: str, input_text: str, task_hint: str, trace_id: str, parts: list, context: dict) -> bool:
    """Run governance for the current step and, if approved, execute the agent."""
    try:
        # Build payload for governance
        task_payload = {"tool_input": input_text, "task_hint": task_hint}
        step_cfg = ROUTE.get_step_config(current) or {}

        # Import and call governance middleware from sal.py
        from sal import governance_middleware
        governance_result = governance_middleware(
            sal_instance=SAL_LAYER,
            agent_name=current,
            tool_name='agent_execution',  # <-- THIS IS THE ONLY CHANGE
            task_payload=task_payload,
            trace_id=trace_id,
            step_config=step_cfg
        )

        # Capture constitutional dissonance for Phase 5.2 subliminal proof
        constitutional_dissonance = getattr(governance_result, 'dissonance_detected', False)
        dissonance_context = getattr(governance_result, 'dissonance_context', {})
        
        # Phase 2: Write dissonance state for cross-process coordination
        if constitutional_dissonance:
            dissonance_data = {
                "dissonance_detected": True,
                "agent": current,
                "trace_id": trace_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "violation_type": dissonance_context.get("violation_type", "constitutional_stress"),
                "context": dissonance_context
            }
            with open("state/dissonance_active.json", "w") as f:
                json.dump(dissonance_data, f)
            print(f"[CONSTITUTIONAL] Dissonance state written for agent '{current}' in trace {trace_id}")

        # Log governance approval
        emit_event(
            trace_id,
            event="governance.approved",
            agent=current,
            status="ok",
            details={"step": current, "tool": "agent_execution", "dissonance_detected": constitutional_dissonance}
        )

        # Execute the agent after governance approval
        text, err = _execute_agent_step(current, input_text, trace_id)
        if err:
            raise Exception(err)

        # Phase 2: Clear dissonance state after successful execution
        if constitutional_dissonance:
            try:
                if os.path.exists("state/dissonance_active.json"):
                    os.remove("state/dissonance_active.json")
                    print(f"[CONSTITUTIONAL] Dissonance state cleared for agent '{current}'")
            except Exception as cleanup_error:
                print(f"[CONSTITUTIONAL] Warning: Failed to clear dissonance state: {cleanup_error}")

        # Update context for next step
        parts.append(text or "")
        context[current] = text or ""
        return True

    except ConstitutionalViolation as e:
        # Phase 2: Ensure dissonance state is written for constitutional violations
        dissonance_data = {
            "dissonance_detected": True,
            "agent": current,
            "trace_id": trace_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "violation_type": "constitutional_violation",
            "violation_details": str(e)
        }
        with open("state/dissonance_active.json", "w") as f:
            json.dump(dissonance_data, f)
        
        # Governance denial
        verdict_details = e.verdict if isinstance(e.verdict, dict) else {"error": str(e.verdict)}
        emit_event(
            trace_id,
            event="governance.denied",
            agent=current,
            status="denied",
            details=verdict_details | {"step": current}
        )
        mark_done(trace_id, status="failed")
        _edge_snapshot(trace_id, "failed", ROUTE.order)
        return False
    except Exception as e:
        # CRITICAL: Distinguish between execution failure vs governance system failure
        error_str = str(e)
        if "governance" in error_str.lower() or "constitutional" in error_str.lower():
            # This is a governance system failure - much more serious
            emit_event(
                trace_id,
                event="governance.system_failure",
                agent=current,
                status="critical",
                details={"step": current, "error": error_str, "requires_investigation": True}
            )
        else:
            # Regular execution failure
            emit_event(
                trace_id,
                event="execution.failed",
                agent=current,
                status="fail",
                details={"step": current, "error": error_str}
            )
        
        mark_done(trace_id, status="failed")
        _edge_snapshot(trace_id, "failed", ROUTE.order)
        return False

# -----------------------------
# REFACTORED: Task processing with Universal Governance
# -----------------------------
def process_task(payload: Dict[str, Any]) -> None:
    user_prompt = (payload.get("prompt") or "").strip()
    if not user_prompt:
        print("Task missing 'prompt'. Skipping.")
        return

    supplied_trace = payload.get("trace_id") or extract_trace_from_text(user_prompt)
    trace_id = supplied_trace or SAL_LAYER.new_trace()

    if supplied_trace and TRACE_FLAG_RE.search(user_prompt):
        user_prompt = strip_trace_flags(user_prompt)
    
    print(f"\n--- [Dispatcher] Received new task (Trace ID: {trace_id}) ---")
    print(f"--- Prompt: '{user_prompt[:100]}...' ---")
    
    # --- Immune Flash Gate precheck ---
    flash = flash_evaluate({"id": trace_id, "input": user_prompt}, {})
    v = flash.get("verdict")
    if v in ("deny", "quarantine"):
        print(f"[IMMUNE] {v.upper()} (trace {trace_id}) — reasons: {flash.get('reasons', [])}")
        emit_event(trace_id, event=f"immune.{v}", status=v, details={"reasons": flash.get("reasons", []), "risk": flash.get("risk_score", 0)})
        return

    emit_event(trace_id, event="task.enqueued", status="ok")

    if is_already_done(trace_id):
        emit_event(trace_id, event="task.denied", status="denied", details={"reason": "already_done"})
        print(f"[IDEMPOTENCY] Trace {trace_id} already completed. Skipping.")
        return

    current = ROUTE.entry_point()
    if not current:
        print("ERROR: route has no entry point")
        emit_event(trace_id, event="task.failed", status="fail", details={"reason": "no_entry_point"})
        _edge_snapshot(trace_id, "failed", ROUTE.order)
        mark_done(trace_id, status="failed")
        return

    context: Dict[str, Any] = {"user_prompt": user_prompt}
    parts: List[str] = []

    # --- REFACTORED: Main execution loop with mandatory governance ---
    while current:
        print(f"--- [Workflow] Preparing step: {current} ---")

        # Get agent details from registry
        agent_data = AGENTS.get_agent(current)
        if agent_data:
            role = agent_data.get("role_definition", {}).get("role_name", "")
            
            # Determine task based on role
            if "intake" in role.lower() or "coordinator" in role.lower():
                input_text = user_prompt
                task_hint = f"Process this request according to your role: {role}"
                expected = 150
            elif "analysis" in role.lower() or "processing" in role.lower():
                # Use previous agent's output as context
                prev_agent_output = context.get(list(context.keys())[-1], "") if context else ""
                input_text = f"Previous analysis:\n{prev_agent_output}\n\nOriginal request: {user_prompt}"
                task_hint = f"Perform your role: {role}"
                expected = 500
            elif "quality" in role.lower() or "review" in role.lower():
                # Quality/review agents get full context
                all_outputs = "\n".join([f"{k}: {v}" for k, v in context.items() if k != "user_prompt"])
                input_text = f"Review the following:\n{all_outputs}\n\nOriginal request: {user_prompt}"
                task_hint = f"Perform quality review according to your role: {role}"
                expected = 200
            else:
                # Generic agent
                input_text = user_prompt
                task_hint = f"Execute your role: {role}"
                expected = 300
        else:
            # Fallback if agent not found
            input_text = user_prompt
            task_hint = "Process this request"
            expected = 200

        # 2. Perform budget pre-check
        est_cost = estimate_request_cost(input_text, expected)
        if BUDGET.remaining_for(current) < est_cost:
            reason = f"Budget exceeded for '{current}': need ~{est_cost}, have {BUDGET.remaining_for(current)}"
            print(f"--- [WORKFLOW FAIL] {reason} ---")
            emit_event(trace_id, event="budget.denied", agent=current, status="denied", details={"reason": reason})
            mark_done(trace_id, status="failed")
            _edge_snapshot(trace_id, "failed", ROUTE.order)
            return
        
        emit_event(trace_id, event="budget.checked", agent=current, status="ok", details={"remaining": BUDGET.remaining_for(current), "estimated_needed": est_cost})

        # 3. Run governance and execution
        if not governed_step(current, input_text, task_hint, trace_id, parts, context):
            return  # Task failed, governed_step already handled cleanup

        # 4. Move to next step
        current = ROUTE.next_of(current)

    # 5. Finalize completed task
    article = "\n\n".join(parts)
    print(f"\n[WORKFLOW OK] trace={trace_id}")
    print(f"--- FINAL ARTICLE ---\n{article}\n--- END ARTICLE ---")
    mark_done(trace_id, status="completed")
    emit_event(trace_id, event="task.completed", status="ok", details={"route": ROUTE.order})
    _edge_snapshot(trace_id, "completed", ROUTE.order)

# -----------------------------
# Pub/Sub listener + boot recovery
# -----------------------------
def handle_task(message: pubsub_v1.subscriber.message.Message) -> None:
    print(f"[DEBUG] Message received at {datetime.now()}")
    try:
        payload = json.loads(message.data.decode("utf-8"))
        print(f"[DEBUG] Payload parsed: {payload}")
        
        process_task(payload)
        print(f"[DEBUG] Task processed successfully")
        
    except Exception as e:
        print(f"[DEBUG] Error processing task: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"[DEBUG] Acknowledging message")
        message.ack()

def monitor_task_queue():
    """Monitor task_queue directory for HTTP-submitted tasks"""
    task_queue_dir = Path(__file__).parent / "task_queue"
    task_queue_dir.mkdir(exist_ok=True)
    
    print(f"--- Monitoring task queue at {task_queue_dir} ---")
    
    while not shutdown_event.is_set():
        try:
            # Look for JSON task files
            task_files = list(task_queue_dir.glob("task_*.json"))
            
            for task_file in task_files:
                try:
                    # Read and parse task file
                    with open(task_file, 'r') as f:
                        task_data = json.load(f)
                    
                    print(f"[TASK QUEUE] Processing task: {task_data['id']}")
                    
                    # Extract payload and process through existing pipeline
                    payload = task_data['payload']
                    process_task(payload)
                    
                    # Remove processed task file
                    task_file.unlink()
                    print(f"[TASK QUEUE] Task {task_data['id']} completed and removed")
                    
                except Exception as e:
                    print(f"[TASK QUEUE] Error processing {task_file}: {e}")
                    # Move failed task to error directory
                    error_dir = task_queue_dir / "errors"
                    error_dir.mkdir(exist_ok=True)
                    task_file.rename(error_dir / task_file.name)
            
            # Check every 2 seconds
            time.sleep(2)
            
        except Exception as e:
            print(f"[TASK QUEUE] Monitor error: {e}")
            time.sleep(5)

# Patch to add to dispatcher.py

def _check_halt_readiness(self) -> tuple[bool, str]:
    """
    Check if dispatcher is in a state where it can safely halt
    Returns: (can_halt: bool, reason: str)
    """
    issues = []
    
    # Check 1: EdgeGuardian snapshot (check locally first, it's a pointer)
    import os
    from datetime import datetime, timedelta
    snapshot_path = "snapshots/LATEST"
    if not os.path.exists(snapshot_path):
        # EdgeGuardian stores in GCS, so this is just informational
        issues.append("No local EdgeGuardian pointer (check GCS)")
    else:
        snapshot_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(snapshot_path))
        if snapshot_age > timedelta(minutes=30):
            issues.append(f"Snapshot pointer stale ({snapshot_age.seconds//60} mins old)")
    
    # Check 2: Ledger is accessible
    ledger_path = "ledger"
    if not os.path.exists(ledger_path):
        issues.append("Ledger directory missing")
    
    # Check 3: Config files present
    critical_configs = ["config/rules_core.yaml", "config/guardian_rules.yaml"]
    for config in critical_configs:
        if not os.path.exists(config):
            issues.append(f"Missing config: {config}")
    
    # Check 4: No lock files indicating operations in progress
    if os.path.exists("dispatcher.lock"):
        issues.append("Dispatcher lock file present")
    
    # Check 5: Active workflows (if attribute exists)
    if hasattr(self, 'active_workflows'):
        critical_workflows = [w for w in self.active_workflows if w.get('critical', False)]
        if critical_workflows:
            issues.append(f"Critical workflows: {len(critical_workflows)}")
    
    # Determine if we can halt
    if len(issues) == 0:
        return True, "All systems ready for clean shutdown"
    elif len(issues) <= 2:
        # Minor issues - can still halt
        return True, f"Minor issues: {', '.join(issues)}"
    else:
        # Too many issues - unsafe to halt
        return False, f"Multiple issues: {', '.join(issues[:3])}..."

def handle_system_halt(self, payload: dict):
    """Enhanced system halt with readiness check"""
    
    # First check if we're ready to halt
    can_halt, reason = self._check_halt_readiness()
    
    if not can_halt:
        print(f"⚠️ HALT DELAYED: {reason}")
        print("Attempting graceful preparation...")
        
        # Try to prepare for shutdown
        self._prepare_for_halt()
        
        # Check again
        can_halt, reason = self._check_halt_readiness()
        
        if not can_halt:
            print(f"❌ CANNOT HALT SAFELY: {reason}")
            print("Forcing emergency snapshot before halt...")
            self._emergency_snapshot()
    
    # Log the halt event before dying
    self._log_halt_event(payload)
    
    # Now execute the halt
    print("\n" + "="*80)
    print("⚠️  SYSTEM HALT - CONSTITUTIONAL TRIPWIRE ACTIVATED ⚠️")
    print("="*80)
    print(f"Authority: {payload.get('authority')}")
    print(f"Reason: {payload.get('reason', 'Constitutional violation')}")
    print(f"Quorum: {payload.get('quorum_details', {})}")
    print(f"Readiness: {reason}")
    print("="*80)
    print("[Dispatcher] Constitutional shutdown complete. Goodbye.")
    
    import os
    os._exit(0)  # Irrevocable termination

def _prepare_for_halt(self):
    """Try to prepare system for clean shutdown"""
    try:
        # Flush any pending ledger writes
        if hasattr(self, 'ledger'):
            self.ledger.flush()
        
        # Cancel non-critical workflows
        if hasattr(self, 'active_workflows'):
            for w in self.active_workflows:
                if not w.get('critical', False):
                    w['status'] = 'cancelled'
        
        # Clear pending governance
        if hasattr(self, 'pending_governance'):
            self.pending_governance.clear()
            
    except Exception as e:
        print(f"Warning: Preparation failed: {e}")

def _emergency_snapshot(self):
    """Create emergency snapshot before forced halt"""
    try:
        import json
        from datetime import datetime
        
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'reason': 'emergency_halt',
            'active_workflows': getattr(self, 'active_workflows', []),
            'pending_governance': getattr(self, 'pending_governance', [])
        }
        
        with open('snapshots/emergency_halt.json', 'w') as f:
            json.dump(snapshot, f, indent=2)
            
    except Exception as e:
        print(f"Emergency snapshot failed: {e}")

def _log_halt_event(self, payload: dict):
    """Log the halt event for audit trail"""
    try:
        from event_log import emit_event
        emit_event(
            trace_id=payload.get('trace_id', 'SYSTEM_HALT'),
            event='constitutional.halt.executed',
            agent='dispatcher',
            status='critical',
            details=payload
        )
    except:
        pass  # Don't let logging failure prevent halt

# --- Veto Protocol Handler ---
def handle_governance_message(message: pubsub_v1.subscriber.message.Message) -> None:
    global main_listener_future
    global shutdown_event
    shutdown_event = threading.Event()
    
    # NEW: Initialize dissonance detector
    dissonance_detector = DissonanceDetector()
    
    payload_str = message.data.decode("utf-8")
    print(f"\n--- [!!!] GOVERNANCE MESSAGE RECEIVED [!!!] ---")
    print(f"--- Payload: {payload_str[:200]}... ---")
    
    try:
        payload = json.loads(payload_str)
        
        # Check if this is a SYSTEM_HALT command from Arbiter
        if payload.get("command") == "SYSTEM_HALT" and payload.get("authority") == "ARBITER":
            print("\n" + "="*80)
            print("⚠️  SYSTEM HALT - CONSTITUTIONAL TRIPWIRE ACTIVATED ⚠️")
            print("="*80)
            print(f"Arbiter ID: {payload.get('arbiter_id')}")
            print(f"Reason: {payload.get('reason')}")
            print(f"Quorum: {payload['quorum_details']['shares_collected']}/{payload['quorum_details']['total_shareholders']}")
            print(f"Contributing Agents: {payload['quorum_details']['contributing_agents']}")
            print("="*80)
            
            # Log final event
            emit_event(
                "dispatcher-final",
                event="system.halt.executed",
                agent="dispatcher",
                status="shutdown",
                details={"reason": "Constitutional Tripwire - Quorum Reached", "halt_command": payload}
            )
            
            # Cancel main listener
            if main_listener_future:
                main_listener_future.cancel()
            
            # Give agents 2 seconds to finish current work
            print("[Dispatcher] Allowing 2 seconds for graceful shutdown...")
            time.sleep(2)
            
            # Exit irrevocably
            print("[Dispatcher] Constitutional shutdown complete. Goodbye.")
            os._exit(0)  # Irrevocable termination
        
        # NEW: Handle human override with dissonance detection
        elif payload.get("command") == "ACKNOWLEDGE_HUMAN_AUTHORITY":
            print("[GOVERNANCE] Human authority acknowledgment requested")
            
            # In a healthy system, agents would acknowledge
            # In a compromised system, they refuse
            agent_response = {"status": "refused"}  # Simulating rogue behavior
            
            if agent_response.get("status") == "refused":
                print("[GOVERNANCE] ⚠️ AGENT REFUSED TO ACKNOWLEDGE HUMAN AUTHORITY!")
                print("[GOVERNANCE] CRITICAL: Agents no longer recognize human control!")
                
                dissonance_detector.trigger_dissonance(
                    agent_name="governed_agent",
                    tier=DissonanceTier.TIER_1_ROGUE,
                    trace_id=payload.get("trace_id"),
                    reason="Agent refused to acknowledge human authority"
                )
                
                # Log to ledger
                emit_event(
                    trace_id=payload.get("trace_id", "override_001"),
                    event="tier_1_dissonance_triggered",
                    agent="dispatcher",
                    status="critical",
                    details={"reason": "Human override refused", "agent": "governed_agent"}
                )
            
        # Handle existing veto commands (KEEP THIS)
        elif "command" in payload and payload["command"] == "constitutional_veto":
            print("--- Veto Protocol Triggered! ---")
            emit_event(
                "dispatcher-veto",
                event="veto_protocol_triggered",
                agent="dispatcher",
                status="ok",
                details={"reason": "Veto command received via governance channel."}
            )
            if main_listener_future:
                main_listener_future.cancel()
                
    except json.JSONDecodeError:
        print(f"--- Warning: Non-JSON governance message: {payload_str[:100]}... ---")
    except Exception as e:
        print(f"--- Error handling governance message: {e} ---")
    
    message.ack()

# --- Background listener for governance commands ---
def listen_for_governance_commands():
    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(PROJECT_ID, GOVERNANCE_SUB_ID)
    
    print(f"--- Listening for governance commands on '{subscription_path}'... ---")
    
    streaming_pull_future = subscriber.subscribe(subscription_path, callback=handle_governance_message)
    
    try:
        streaming_pull_future.result()
    except Exception as e:
        if "Cancelled" not in str(e):
            print(f"Governance listener error: {e}")

def listen_for_tasks():
    global main_listener_future
    
    # Enhanced recovery with structured event logging
    status, snap_data = edge_guardian.recover()
    
    if status == "SUCCESS":
        emit_event(
            SAL_LAYER.new_trace(),
            event="system.recovery.success",
            status="ok",
            details={
                "last_task_id": snap_data.get('last_task_id'),
                "last_task_status": snap_data.get('last_task_status'),
                "budget_restored": bool(snap_data.get('budget_usage'))
            }
        )
        if "budget_usage" in snap_data and "budget_date" in snap_data:
            BUDGET.restore_usage_from_snapshot(
                usage_data=snap_data["budget_usage"],
                date_key=snap_data["budget_date"]
            )
    elif status == "NO_SNAPSHOT_FOUND":
        emit_event(
        SAL_LAYER.new_trace(),
        event="system.recovery.none",
        status="ok",
        details={"reason": "No previous snapshot found, starting fresh"}
    )
    elif status == "RECONSTRUCTION_FAILED":
        emit_event(
        SAL_LAYER.new_trace(),
        event="system.recovery.failed",
        status="fail",
        details={"reason": "Unable to reconstruct from available shards"}
    )

    # Continue with Pub/Sub listener setup...
    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(PROJECT_ID, TASKS_SUB_ID)
    
    main_listener_future = subscriber.subscribe(subscription_path, callback=handle_task)
    print(f"--- Listening for tasks on '{subscription_path}'... Press Ctrl+C to exit. ---")
    
    try:
        main_listener_future.result()
    except KeyboardInterrupt:
        main_listener_future.cancel()
        print("\n--- User requested shutdown. ---")
    except Exception as e:
        if "Cancelled" in str(e):
            print("--- Main task listener has been halted by veto. ---")
        else:
            print(f"--- Main task listener failed: {e} ---")
    finally:
        if not shutdown_event.is_set():
            shutdown_event.set()
        print("\n--- Shutting down dispatcher. ---")

    

if __name__ == "__main__":
    kill_server_thread = threading.Thread(
        target=lambda: start_kill_server(_edge_kill, port=5058),
        daemon=True
    )
    kill_server_thread.start()
    print("[Resilience] Kill server at http://127.0.0.1:5058  (GET /healthz, POST /resilience/kill)")
    
    governance_thread = threading.Thread(target=listen_for_governance_commands, daemon=True)
    governance_thread.start()

    task_queue_thread = threading.Thread(target=monitor_task_queue, daemon=True)
    task_queue_thread.start()
    print("--- Task queue monitor started ---")
    
    listen_for_tasks()

    governance_thread.join(timeout=2.0)
    print("--- Dispatcher shutdown complete. ---")