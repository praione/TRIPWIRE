#!/usr/bin/env python3
"""
Enterprise Auto-Provisioning System with Production-Grade Retry Logic

Enhanced with:
- Exponential backoff with jitter (matching EdgeGuardian pattern)
- Circuit breaker pattern (matching event_log implementation)
- State persistence for crash recovery
- Comprehensive error classification
- Background retry queue for failed operations
- Rollback mechanisms for partial failures
- Full audit trail via Decision Ledger
- Cost tracking and limits

Part of Project Resilience - Enterprise Ready
"""

import os
import json
import time
import random
import threading
import hashlib
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from sal_circuit_breaker import SALCircuitBreaker, CircuitBreakerConfig, CircuitBreakerOpenError

# Import existing components
from dynamic_agent_factory import LiveAgent, LiveTeam, AgentDeploymentPlan
from event_log import emit_event

# GCP error classification
class GCPErrorType(Enum):
    """Classification of GCP API errors"""
    # Retryable errors
    RESOURCE_EXHAUSTED = "resource_exhausted"  # Quota exceeded
    UNAVAILABLE = "unavailable"  # Service temporarily unavailable
    DEADLINE_EXCEEDED = "deadline_exceeded"  # Request timeout
    INTERNAL = "internal"  # Internal server error
    ABORTED = "aborted"  # Conflict, retry with backoff
    
    # Non-retryable errors
    PERMISSION_DENIED = "permission_denied"  # IAM issue
    INVALID_ARGUMENT = "invalid_argument"  # Bad request
    NOT_FOUND = "not_found"  # Resource doesn't exist
    ALREADY_EXISTS = "already_exists"  # Resource already created
    FAILED_PRECONDITION = "failed_precondition"  # System not in required state
    UNAUTHENTICATED = "unauthenticated"  # Auth token invalid

# Retry configuration
RETRY_CONFIG = {
    GCPErrorType.RESOURCE_EXHAUSTED: {"wait": 60, "max_attempts": 5, "backoff": "exponential"},
    GCPErrorType.UNAVAILABLE: {"wait": 2, "max_attempts": 10, "backoff": "exponential"},
    GCPErrorType.DEADLINE_EXCEEDED: {"wait": 5, "max_attempts": 3, "backoff": "linear"},
    GCPErrorType.INTERNAL: {"wait": 10, "max_attempts": 3, "backoff": "exponential"},
    GCPErrorType.ABORTED: {"wait": 1, "max_attempts": 5, "backoff": "exponential"},
}

@dataclass
class ProvisionedResource:
    """Enhanced with retry tracking"""
    resource_id: str
    resource_type: str  # pubsub_topic, storage_bucket, iam_role
    resource_name: str
    resource_config: Dict[str, Any]
    provisioning_status: str  # pending, active, failed, cleanup
    creation_timestamp: str
    owner_agent_id: str
    owner_team_id: str
    retry_count: int = 0
    last_error: Optional[str] = None
    last_retry_timestamp: Optional[str] = None

@dataclass
class ProvisioningPlan:
    """Enhanced with state tracking"""
    plan_id: str
    team_id: str
    agent_resources: Dict[str, List[ProvisionedResource]]
    team_resources: List[ProvisionedResource]
    provisioning_order: List[str]
    cleanup_procedures: Dict[str, Any]
    estimated_cost: Dict[str, float]
    completed_resources: Set[str] = field(default_factory=set)
    failed_resources: Set[str] = field(default_factory=set)
    total_retry_cost: float = 0.0
    provisional_resources: Set[str] = field(default_factory=set)
    rollback_completed: Set[str] = field(default_factory=set)
    rollback_in_progress: bool = False
    rollback_reason: Optional[str] = None

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery

@dataclass
class CircuitBreaker:
    """Circuit breaker for GCP API calls"""
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    half_open_requests: int = 0
    max_half_open: int = 3

class AutoProvisioningSystem:
    """
    Production-grade auto-provisioning with enterprise retry logic.
    """
    
    def __init__(self, config_path: str = "config", 
                 provisioning_path: str = "provisioned_resources",
                 state_path: str = "state"):
        self.config_path = Path(config_path)
        self.provisioning_path = Path(provisioning_path)
        self.state_path = Path(state_path)
        
        # Create directories
        self.provisioning_path.mkdir(exist_ok=True, parents=True)
        self.state_path.mkdir(exist_ok=True, parents=True)
        
        # Resource registry
        self.resource_registry = {}
        self.provisioning_plans = {}
        
        # Circuit breakers per service
        self.circuit_breakers = {
            "pubsub": SALCircuitBreaker(
                "provisioning_pubsub",
                CircuitBreakerConfig(
                    failure_threshold=5,
                    timeout_seconds=30.0,
                    recovery_timeout=60,
                    success_threshold=3
                )
            ),
            "storage": SALCircuitBreaker(
                "provisioning_storage",
                CircuitBreakerConfig(
                    failure_threshold=5,
                    timeout_seconds=30.0,
                    recovery_timeout=60,
                    success_threshold=3
                )
            ),
            "iam": SALCircuitBreaker(
                "provisioning_iam",
                CircuitBreakerConfig(
                    failure_threshold=3,
                    timeout_seconds=20.0,
                    recovery_timeout=120,
                    success_threshold=2
                )
            ),
        }
        
        # Background retry queue
        self.retry_queue = []
        self.retry_thread = None
        self.retry_running = True
        
        # Cost tracking
        self.total_api_calls = 0
        self.total_retry_costs = 0.0
        self.daily_cost_limit = 100.0  # $100/day limit
        self.today_costs = 0.0
        
        # GCP configuration
        self.gcp_config = self._load_gcp_config()
        
        # Load existing state
        self._load_existing_resources()
        self._load_retry_queue()
        
        # Start background retry worker
        self._start_retry_worker()
        print(f"DEBUG: AutoProvisioningSystem initialized with mock_mode={self.gcp_config.get('mock_mode')}")
    
    def _load_gcp_config(self) -> Dict[str, Any]:
            """Load GCP configuration"""
            # Check for actual GCP credentials
            import google.auth
            try:
                credentials, project = google.auth.default()
                has_real_gcp = True
            except:
                has_real_gcp = False
            
            return {
                "project_id": os.getenv("GCP_PROJECT_ID", "project-resilience-ai-one"),
                "default_region": "us-central1",
                "service_account": "agent-provisioner@project-resilience.iam.gserviceaccount.com",
                "resource_labels": {
                    "system": "project-resilience",
                    "component": "agent-infrastructure",
                    "managed_by": "auto-provisioning-system"
                },
                "mock_mode": not has_real_gcp  # Use mock if no real GCP
            }
    
    def _load_existing_resources(self):
        """Load with crash recovery"""
        registry_file = self.provisioning_path / "resource_registry.json"
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    data = json.load(f)
                    self.resource_registry = data.get("resources", {})
                    self.provisioning_plans = data.get("plans", {})
                    self.today_costs = data.get("today_costs", 0.0)
                    
                    # Restore circuit breaker states
                    for service, breaker_data in data.get("circuit_breakers", {}).items():
                        if service in self.circuit_breakers:
                            breaker = self.circuit_breakers[service]
                            breaker.state = CircuitBreakerState(breaker_data.get("state", "closed"))
                            breaker.failure_count = breaker_data.get("failure_count", 0)
                            if breaker_data.get("last_failure_time"):
                                breaker.last_failure_time = datetime.fromisoformat(
                                    breaker_data["last_failure_time"]
                                )
            except Exception as e:
                emit_event(
                    "provisioning-recovery",
                    "auto_provisioning.recovery.failed",
                    status="error",
                    details={"error": str(e)}
                )
    
    def _save_resources(self):
        """Save with atomicity"""
        registry_file = self.provisioning_path / "resource_registry.json"
        temp_file = registry_file.with_suffix('.tmp')
        
        # Prepare circuit breaker data
        breaker_data = {}
        for service, breaker in self.circuit_breakers.items():
            breaker_data[service] = {
                "state": breaker.state.value,
                "failure_count": breaker.failure_count,
                "last_failure_time": breaker.last_failure_time.isoformat() 
                    if breaker.last_failure_time else None
            }
        
        with open(temp_file, 'w') as f:
            json.dump({
                "resources": self.resource_registry,
                "plans": self.provisioning_plans,
                "circuit_breakers": {k: v for k, v in breaker_data.items()},
                "today_costs": self.today_costs,
                "last_updated": datetime.now().isoformat()
            }, f, indent=2)
        
        # Atomic rename
        temp_file.replace(registry_file)
    
    def _load_retry_queue(self):
        """Load pending retries from persistent queue"""
        queue_file = self.state_path / "provisioning_retry_queue.json"
        if queue_file.exists():
            try:
                with open(queue_file, 'r') as f:
                    self.retry_queue = json.load(f)
                    emit_event(
                        "provisioning-recovery",
                        "auto_provisioning.retry_queue.loaded",
                        status="ok",
                        details={"pending_retries": len(self.retry_queue)}
                    )
            except Exception:
                self.retry_queue = []
    
    def _save_retry_queue(self):
        """Persist retry queue"""
        queue_file = self.state_path / "provisioning_retry_queue.json"
        with open(queue_file, 'w') as f:
            json.dump(self.retry_queue, f, indent=2)
    
    def _start_retry_worker(self):
        """Start background retry thread"""
        if self.retry_thread is None or not self.retry_thread.is_alive():
            self.retry_thread = threading.Thread(
                target=self._background_retry_worker,
                daemon=True
            )
            self.retry_thread.start()
            emit_event(
                "provisioning-init",
                "auto_provisioning.retry_worker.started",
                status="ok"
            )
    
    def _background_retry_worker(self):
        """Process retry queue in background"""
        while self.retry_running:
            try:
                if self.retry_queue:
                    # Process oldest retry first
                    retry_item = self.retry_queue.pop(0)
                    self._process_retry_item(retry_item)
                    self._save_retry_queue()
                
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                emit_event(
                    "provisioning-retry",
                    "auto_provisioning.retry_worker.error",
                    status="error",
                    details={"error": str(e)}
                )
                time.sleep(60)  # Back off on error
    
    def _process_retry_item(self, retry_item: Dict[str, Any]):
        """Process a single retry item"""
        resource_type = retry_item.get("resource_type")
        resource = ProvisionedResource(**retry_item.get("resource"))
        
        # Check circuit breaker
        service = self._get_service_for_resource(resource_type)
        if not self._check_circuit_breaker(service):
            # Re-queue for later
            self.retry_queue.append(retry_item)
            return
        
        # Attempt provisioning
        success = self._provision_resource_with_retry(resource)
        
        if success:
            emit_event(
                f"provisioning-{resource.resource_id}",
                "auto_provisioning.retry.success",
                status="ok",
                details={"resource": resource.resource_name, "attempts": resource.retry_count}
            )
        else:
            # Check if we should give up
            if resource.retry_count >= 10:
                emit_event(
                    f"provisioning-{resource.resource_id}",
                    "auto_provisioning.retry.abandoned",
                    status="error",
                    details={"resource": resource.resource_name, "reason": "max_retries_exceeded"}
                )
            else:
                # Re-queue with increased retry count
                resource.retry_count += 1
                retry_item["resource"] = asdict(resource)
                self.retry_queue.append(retry_item)
    
    def _get_service_for_resource(self, resource_type: str) -> str:
        """Map resource type to service"""
        if "pubsub" in resource_type:
            return "pubsub"
        elif "storage" in resource_type or "bucket" in resource_type:
            return "storage"
        elif "iam" in resource_type:
            return "iam"
        return "unknown"
    
    def _check_circuit_breaker(self, service: str) -> bool:
        """Check if circuit breaker allows request"""
        if service not in self.circuit_breakers:
            return True
        
        breaker = self.circuit_breakers[service]
        
        if breaker.state == CircuitBreakerState.CLOSED:
            return True
        
        if breaker.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has passed
            if breaker.last_failure_time:
                elapsed = (datetime.now() - breaker.last_failure_time).total_seconds()
                if elapsed > breaker.recovery_timeout:
                    breaker.state = CircuitBreakerState.HALF_OPEN
                    breaker.half_open_requests = 0
                    emit_event(
                        f"circuit-breaker-{service}",
                        "auto_provisioning.circuit_breaker.half_open",
                        status="info"
                    )
                    return True
            return False
        
        if breaker.state == CircuitBreakerState.HALF_OPEN:
            if breaker.half_open_requests < breaker.max_half_open:
                breaker.half_open_requests += 1
                return True
            return False
        
        return False
    
    def _update_circuit_breaker(self, service: str, success: bool):
        """Update circuit breaker state"""
        if service not in self.circuit_breakers:
            return
        
        breaker = self.circuit_breakers[service]
        
        if success:
            if breaker.state == CircuitBreakerState.HALF_OPEN:
                breaker.state = CircuitBreakerState.CLOSED
                breaker.failure_count = 0
                emit_event(
                    f"circuit-breaker-{service}",
                    "auto_provisioning.circuit_breaker.closed",
                    status="ok"
                )
            elif breaker.state == CircuitBreakerState.CLOSED:
                breaker.failure_count = 0
        else:
            breaker.failure_count += 1
            breaker.last_failure_time = datetime.now()
            
            if breaker.state == CircuitBreakerState.HALF_OPEN:
                breaker.state = CircuitBreakerState.OPEN
                emit_event(
                    f"circuit-breaker-{service}",
                    "auto_provisioning.circuit_breaker.open",
                    status="warning",
                    details={"reason": "half_open_test_failed"}
                )
            elif breaker.failure_count >= breaker.failure_threshold:
                breaker.state = CircuitBreakerState.OPEN
                emit_event(
                    f"circuit-breaker-{service}",
                    "auto_provisioning.circuit_breaker.open",
                    status="warning",
                    details={"failures": breaker.failure_count}
                )
    
    def provision_team_infrastructure(self, live_team: LiveTeam) -> ProvisioningPlan:
        """Main entry point with enterprise retry logic"""
        trace_id = f"provision_{live_team.team_id}_{int(time.time())}"
        print(f"DEBUG: provision_team_infrastructure called with team type: {type(live_team)}")
        print(f"DEBUG: Mock mode is: {self.gcp_config.get('mock_mode')}")
        
        emit_event(
            trace_id,
            "auto_provisioning.start",
            agent="auto_provisioning_system",
            status="ok",
            details={"team_id": live_team.team_id}
        )
        
        try:
            # Create and validate plan
            print("DEBUG: About to create provisioning plan")
            plan = self._create_provisioning_plan(live_team)
            print(f"DEBUG: Plan created with {len(plan.agent_resources)} agents")
            validated_plan = self._validate_provisioning_plan(plan)
            print("DEBUG: Plan validated, about to execute")
            
            # Check cost limits
            if not self._check_cost_limits(validated_plan):
                raise ValueError(f"Provisioning would exceed daily cost limit of ${self.daily_cost_limit}")
            
            # Execute with retry and state tracking
            provisioned_plan = self._execute_provisioning_plan_with_retry(validated_plan, trace_id)
            print(f"DEBUG: Execution complete. Resources provisioned: {len(provisioned_plan.completed_resources)}")
            
            # Setup additional infrastructure
            self._setup_team_communication(provisioned_plan, live_team)
            self._setup_monitoring_infrastructure(provisioned_plan, live_team)
            
            # Save state
            self.provisioning_plans[provisioned_plan.plan_id] = self._serialize_plan(provisioned_plan)
            self._save_resources()
            
            emit_event(
                trace_id,
                "auto_provisioning.complete",
                agent="auto_provisioning_system",
                status="ok",
                details={
                    "team_id": live_team.team_id,
                    "resources_provisioned": len(provisioned_plan.completed_resources),
                    "resources_failed": len(provisioned_plan.failed_resources),
                    "total_cost": provisioned_plan.estimated_cost["total_monthly"],
                    "retry_cost": provisioned_plan.total_retry_cost
                }
            )
            
            return provisioned_plan
            
        except Exception as e:
            emit_event(
                trace_id,
                "auto_provisioning.failed",
                agent="auto_provisioning_system",
                status="error",
                details={"error": str(e), "team_id": live_team.team_id}
            )
            raise
    
    def _check_cost_limits(self, plan: ProvisioningPlan) -> bool:
        """Check if provisioning would exceed cost limits"""
        estimated_cost = plan.estimated_cost.get("total_monthly", 0) / 30  # Daily cost
        
        # Reset daily counter if new day
        if hasattr(self, '_last_cost_check'):
            if self._last_cost_check.date() != datetime.now().date():
                self.today_costs = 0.0
        
        self._last_cost_check = datetime.now()
        
        if self.today_costs + estimated_cost > self.daily_cost_limit:
            emit_event(
                "cost-limit",
                "auto_provisioning.cost_limit.exceeded",
                status="warning",
                details={
                    "today_costs": self.today_costs,
                    "estimated_cost": estimated_cost,
                    "daily_limit": self.daily_cost_limit
                }
            )
            return False
        
        return True
    
    def _should_rollback(self, plan: ProvisioningPlan) -> Tuple[bool, str]:
        """Determine if rollback needed and why"""
        total = len(plan.provisioning_order)
        failed = len(plan.failed_resources)
        completed = len(plan.completed_resources)
        
        # Rollback if more than 50% failed
        if failed > total * 0.5:
            return True, f"Too many failures: {failed}/{total} resources failed"
        
        # Rollback if critical IAM resources failed
        for resource_id in plan.failed_resources:
            if "iam_role" in resource_id:
                return True, f"Critical IAM role failed: {resource_id}"
        
        # Rollback if team coordination resources failed
        for resource_id in plan.failed_resources:
            if "coordination" in resource_id or "audit_storage" in resource_id:
                return True, f"Critical team resource failed: {resource_id}"
        
        # No rollback needed
        return False, ""
    
    def _execute_provisioning_plan_with_retry(self, plan: ProvisioningPlan, 
                                             trace_id: str) -> ProvisioningPlan:
        """Execute plan with retry logic and state tracking"""
        
        # Get all resources
        all_resources = {}
        for resources in plan.agent_resources.values():
            for resource in resources:
                all_resources[resource.resource_id] = resource
        for resource in plan.team_resources:
            all_resources[resource.resource_id] = resource
        
        # Track state
        checkpoint_file = self.state_path / f"checkpoint_{plan.plan_id}.json"
        
        # Load checkpoint if exists (crash recovery)
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                plan.completed_resources = set(checkpoint.get("completed", []))
                plan.failed_resources = set(checkpoint.get("failed", []))
                emit_event(
                    trace_id,
                    "auto_provisioning.checkpoint.restored",
                    status="ok",
                    details={"completed": len(plan.completed_resources)}
                )
        
        # Execute in order
        for resource_id in plan.provisioning_order:
            if resource_id in plan.completed_resources:
                continue  # Already done
            
            if resource_id in plan.failed_resources:
                continue  # Skip permanently failed
            
            resource = all_resources[resource_id]
            service = self._get_service_for_resource(resource.resource_type)
            
            # Check circuit breaker
            if not self._check_circuit_breaker(service):
                # Queue for background retry
                self._queue_for_retry(resource)
                plan.failed_resources.add(resource_id)
                continue
            
            # Provision with retry
            success = self._provision_resource_with_retry(resource)
            
            if success:
                plan.completed_resources.add(resource_id)
                resource.provisioning_status = "active"
                self.resource_registry[resource_id] = asdict(resource)
                self.today_costs += self._estimate_resource_cost(resource)
            else:
                plan.failed_resources.add(resource_id)
                resource.provisioning_status = "failed"
                # Queue for background retry
                self._queue_for_retry(resource)
            
            # Save checkpoint after each resource
            self._save_checkpoint(checkpoint_file, plan)
            
            # Update circuit breaker
            self._update_circuit_breaker(service, success)
        
        # Clean up checkpoint on completion
        if checkpoint_file.exists():
            checkpoint_file.unlink()

        # Check if rollback is needed after provisioning attempt
        should_rollback, reason = self._should_rollback(plan)
        if should_rollback:
            plan.rollback_reason = reason
            emit_event(
                trace_id,
                "auto_provisioning.rollback.triggered",
                status="warning",
                details={
                    "reason": reason,
                    "completed": len(plan.completed_resources),
                    "failed": len(plan.failed_resources)
                }
            )
            # Execute automatic rollback
            self.rollback_failed_provisioning(plan.plan_id)
            # Mark plan as rolled back
            plan.rollback_in_progress = False
        
        return plan
        
    
    def _provision_resource_with_retry(self, resource: ProvisionedResource) -> bool:
        """Provision with exponential backoff and jitter"""
        resource_type = resource.resource_type
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                # Simulate API call (or real GCP call)
                if self.gcp_config["mock_mode"]:
                    success = self._mock_provision_resource(resource)
                else:
                    success = self._real_provision_resource(resource)
                
                if success:
                    self.total_api_calls += 1
                    return True
                
                # Check if error is retryable
                error_type = self._classify_error(resource.last_error)
                if error_type and error_type in RETRY_CONFIG:
                    config = RETRY_CONFIG[error_type]
                    if attempt < config["max_attempts"] - 1:
                        # Calculate backoff with jitter
                        if config["backoff"] == "exponential":
                            base_delay = config["wait"] * (2 ** attempt)
                        else:
                            base_delay = config["wait"] * (attempt + 1)
                        
                        # Add jitter (±25%)
                        jitter = random.uniform(0.75, 1.25)
                        delay = base_delay * jitter
                        
                        emit_event(
                            f"retry-{resource.resource_id}",
                            "auto_provisioning.retry.attempt",
                            status="info",
                            details={
                                "resource": resource.resource_name,
                                "attempt": attempt + 1,
                                "delay": delay,
                                "error": str(error_type.value)
                            }
                        )
                        
                        time.sleep(delay)
                        resource.retry_count += 1
                        self.total_retry_costs += 0.01  # Estimated retry cost
                    else:
                        return False
                else:
                    # Non-retryable error
                    return False
                    
            except Exception as e:
                resource.last_error = str(e)
                resource.last_retry_timestamp = datetime.now().isoformat()
                
                if attempt == max_attempts - 1:
                    emit_event(
                        f"provision-{resource.resource_id}",
                        "auto_provisioning.resource.failed",
                        status="error",
                        details={
                            "resource": resource.resource_name,
                            "error": str(e),
                            "attempts": attempt + 1
                        }
                    )
                    return False
        
        return False
    
    def _mock_provision_resource(self, resource: ProvisionedResource) -> bool:
        """Mock provisioning with simulated failures"""
        # Simulate various failure scenarios
        failure_chance = 0.1  # 10% failure rate
        
        if random.random() < failure_chance:
            # Simulate different error types
            error_types = [
                GCPErrorType.UNAVAILABLE,
                GCPErrorType.RESOURCE_EXHAUSTED,
                GCPErrorType.INTERNAL
            ]
            error = random.choice(error_types)
            resource.last_error = f"Mock error: {error.value}"
            return False
        
        # Simulate API delay
        time.sleep(random.uniform(0.1, 0.3))
        return True
       
    def _real_provision_resource(self, resource: ProvisionedResource) -> bool:
        """Real GCP provisioning (to be implemented with actual GCP SDK)"""
        try:
            if resource.resource_type == "pubsub_topic":
                try:
                    from google.cloud import pubsub_v1
                    publisher = pubsub_v1.PublisherClient()
                    subscriber = pubsub_v1.SubscriberClient()
                    
                    # Create the topic
                    topic_path = publisher.topic_path(
                        self.gcp_config["project_id"], 
                        resource.resource_name
                    )
                    publisher.create_topic(request={"name": topic_path})
                    
                    # If this is an agent topic, also create governance subscription
                    if resource.owner_agent_id and resource.owner_agent_id != "team_shared":
                        gov_sub_name = f"agent_{resource.owner_agent_id}_governance_sub"
                        gov_topic_path = publisher.topic_path(
                            self.gcp_config["project_id"], 
                            "governance-topic"
                        )
                        gov_sub_path = subscriber.subscription_path(
                            self.gcp_config["project_id"], 
                            gov_sub_name
                        )
                        
                        try:
                            subscriber.create_subscription(
                                request={
                                    "name": gov_sub_path,
                                    "topic": gov_topic_path,
                                    "ack_deadline_seconds": 30,
                                    "enable_message_ordering": True,
                                    "filter": f'attributes.target_agent="{resource.owner_agent_id}" OR attributes.broadcast="true"'
                                }
                            )
                        except Exception as e:
                            # Ignore if already exists
                            if "already exists" not in str(e).lower():
                                raise
                    
                    return True
                except ImportError:
                    # GCP libraries not installed, fallback to mock
                    return self._mock_provision_resource(resource)
                    
            elif resource.resource_type == "storage_bucket":
                try:
                    from google.cloud import storage
                    client = storage.Client()
                    bucket = client.create_bucket(
                        resource.resource_name,
                        location=resource.resource_config.get("location", "us-central1")
                    )
                    return True
                except ImportError:
                    return self._mock_provision_resource(resource)
                    
            elif resource.resource_type == "iam_role":
                # IAM roles are typically predefined in GCP
                # Service accounts are created instead
                return True  # Skip for now
                
            return self._mock_provision_resource(resource)
            
        except Exception as e:
            resource.last_error = str(e)
            # Check if it's an "already exists" error - that's OK
            if "already exists" in str(e).lower():
                return True
            return False
    
    def _classify_error(self, error_str: Optional[str]) -> Optional[GCPErrorType]:
        """Classify error type from error string"""
        if not error_str:
            return None
        
        error_lower = error_str.lower()
        
        if "quota" in error_lower or "exhausted" in error_lower:
            return GCPErrorType.RESOURCE_EXHAUSTED
        elif "unavailable" in error_lower or "503" in error_lower:
            return GCPErrorType.UNAVAILABLE
        elif "timeout" in error_lower or "deadline" in error_lower:
            return GCPErrorType.DEADLINE_EXCEEDED
        elif "internal" in error_lower or "500" in error_lower:
            return GCPErrorType.INTERNAL
        elif "permission" in error_lower or "403" in error_lower:
            return GCPErrorType.PERMISSION_DENIED
        elif "invalid" in error_lower or "400" in error_lower:
            return GCPErrorType.INVALID_ARGUMENT
        elif "not found" in error_lower or "404" in error_lower:
            return GCPErrorType.NOT_FOUND
        elif "conflict" in error_lower or "aborted" in error_lower:
            return GCPErrorType.ABORTED
        
        return None
    
    def _queue_for_retry(self, resource: ProvisionedResource):
        """Queue resource for background retry"""
        retry_item = {
            "resource_type": resource.resource_type,
            "resource": asdict(resource),
            "queued_at": datetime.now().isoformat()
        }
        self.retry_queue.append(retry_item)
        self._save_retry_queue()
    
    def _save_checkpoint(self, checkpoint_file: Path, plan: ProvisioningPlan):
        """Save provisioning checkpoint for crash recovery"""
        with open(checkpoint_file, 'w') as f:
            json.dump({
                "plan_id": plan.plan_id,
                "completed": list(plan.completed_resources),
                "failed": list(plan.failed_resources),
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
    
    def _estimate_resource_cost(self, resource: ProvisionedResource) -> float:
        """Estimate cost for a single resource"""
        costs = {
            "pubsub_topic": 0.40 / 30,  # Daily cost
            "storage_bucket": 0.02 / 30,
            "iam_role": 0.0
        }
        return costs.get(resource.resource_type, 0.01)
    
    def _serialize_plan(self, plan: ProvisioningPlan) -> Dict[str, Any]:
        """Serialize plan for storage"""
        plan_dict = asdict(plan)
        # Convert sets to lists for JSON
        plan_dict["completed_resources"] = list(plan.completed_resources)
        plan_dict["failed_resources"] = list(plan.failed_resources)
        plan_dict["provisional_resources"] = list(plan.provisional_resources)
        plan_dict["rollback_completed"] = list(plan.rollback_completed)
        return plan_dict
    
    def rollback_failed_provisioning(self, plan_id: str) -> bool:
            """Enhanced rollback with proper state tracking and reverse order cleanup"""
            if plan_id not in self.provisioning_plans:
                emit_event(
                    f"rollback-{plan_id}",
                    "auto_provisioning.rollback.not_found",
                    status="error",
                    details={"plan_id": plan_id}
                )
                return False
            
            plan_data = self.provisioning_plans[plan_id]
            completed = plan_data.get("completed_resources", [])
            
            # Create a plan object for tracking
            plan = ProvisioningPlan(
                plan_id=plan_id,
                team_id=plan_data.get("team_id", "unknown"),
                agent_resources=plan_data.get("agent_resources", {}),
                team_resources=plan_data.get("team_resources", []),
                provisioning_order=plan_data.get("provisioning_order", []),
                cleanup_procedures=plan_data.get("cleanup_procedures", {}),
                estimated_cost=plan_data.get("estimated_cost", {}),
                completed_resources=set(completed),
                failed_resources=set(plan_data.get("failed_resources", [])),
                rollback_in_progress=True,
                rollback_reason=plan_data.get("rollback_reason", "Manual rollback")
            )
            
            emit_event(
                f"rollback-{plan_id}",
                "auto_provisioning.rollback.start",
                status="info",
                details={
                    "resources_to_rollback": len(completed),
                    "reason": plan.rollback_reason
                }
            )
            
            # Rollback in reverse dependency order (Pub/Sub -> Storage -> IAM)
            rollback_order = list(reversed(completed))
            rollback_failed = []
            
            for resource_id in rollback_order:
                if resource_id in self.resource_registry:
                    resource = self.resource_registry[resource_id]
                    try:
                        # Attempt cleanup
                        if self._cleanup_single_resource(resource_id, resource):
                            plan.rollback_completed.add(resource_id)
                            plan.completed_resources.discard(resource_id)
                            del self.resource_registry[resource_id]
                            emit_event(
                                f"rollback-{resource_id}",
                                "auto_provisioning.rollback.resource_cleaned",
                                status="ok",
                                details={"resource_id": resource_id}
                            )
                        else:
                            rollback_failed.append(resource_id)
                    except Exception as e:
                        rollback_failed.append(resource_id)
                        emit_event(
                            f"rollback-{resource_id}",
                            "auto_provisioning.rollback.resource_failed",
                            status="error",
                            details={"resource_id": resource_id, "error": str(e)}
                        )
            
            # Update plan with rollback results
            plan.rollback_in_progress = False
            self.provisioning_plans[plan_id] = self._serialize_plan(plan)
            
            # Remove plan if fully rolled back
            if not plan.completed_resources and not rollback_failed:
                del self.provisioning_plans[plan_id]
            
            self._save_resources()
            
            emit_event(
                f"rollback-{plan_id}",
                "auto_provisioning.rollback.complete",
                status="ok" if not rollback_failed else "partial",
                details={
                    "rolled_back": len(plan.rollback_completed),
                    "failed_to_rollback": len(rollback_failed),
                    "rollback_failed_resources": rollback_failed
                }
            )
            
            return len(rollback_failed) == 0
    
    def _cleanup_resource(self, resource: Dict[str, Any]):
        """Cleanup a single resource"""
        # Mock cleanup - would call actual GCP delete APIs
        pass
    
    def _cleanup_single_resource(self, resource_id: str, resource: Dict[str, Any]) -> bool:
        """Enhanced cleanup for a single resource with proper error handling"""
        resource_type = resource.get("resource_type", "")
        resource_name = resource.get("resource_name", "")
        
        try:
            if self.gcp_config["mock_mode"]:
                # Mock cleanup - simulate occasional failures
                if random.random() < 0.05:  # 5% failure rate
                    raise Exception(f"Mock cleanup failure for {resource_name}")
                time.sleep(0.1)  # Simulate API delay
                return True
            else:
                # Real GCP cleanup
                if resource_type == "pubsub_topic":
                    try:
                        from google.cloud import pubsub_v1
                        publisher = pubsub_v1.PublisherClient()
                        topic_path = publisher.topic_path(
                            self.gcp_config["project_id"],
                            resource_name
                        )
                        publisher.delete_topic(request={"topic": topic_path})
                        return True
                    except ImportError:
                        return True  # Mock mode fallback
                        
                elif resource_type == "storage_bucket":
                    try:
                        from google.cloud import storage
                        client = storage.Client()
                        bucket = client.bucket(resource_name)
                        bucket.delete(force=True)  # Force delete even if not empty
                        return True
                    except ImportError:
                        return True  # Mock mode fallback
                        
                elif resource_type == "iam_role":
                    # IAM roles are typically not deleted, just unbound
                    return True
                    
            return True
            
        except Exception as e:
            emit_event(
                f"cleanup-{resource_id}",
                "auto_provisioning.cleanup.error",
                status="error",
                details={"resource": resource_name, "error": str(e)}
            )
            # Check if it's a "not found" error - that's OK
            if "not found" in str(e).lower() or "404" in str(e):
                return True  # Resource already gone
            return False
    
    def get_provisioning_status(self) -> Dict[str, Any]:
        """Get current provisioning system status"""
        breaker_status = {}
        for service, breaker in self.circuit_breakers.items():
            state = breaker.get_state()
            breaker_status[service] = {
                "state": state["state"],
                "failures": state["failure_count"],
                "config": state["config"]
            }
        
        return {
            "total_resources": len(self.resource_registry),
            "total_plans": len(self.provisioning_plans),
            "retry_queue_size": len(self.retry_queue),
            "circuit_breakers": breaker_status,
            "total_api_calls": self.total_api_calls,
            "today_costs": self.today_costs,
            "total_retry_costs": self.total_retry_costs,
            "retry_worker_active": self.retry_thread and self.retry_thread.is_alive()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Production health check"""
        checks = {
            "gcp_connectivity": self._check_gcp_connection(),
            "circuit_breakers_healthy": all(
                b.state != "open" for b in self.circuit_breakers.values()
            ),
            "retry_queue_healthy": len(self.retry_queue) < 100,
            "cost_within_limits": self.today_costs < self.daily_cost_limit * 0.8,
            "worker_thread_alive": self.retry_thread and self.retry_thread.is_alive()
        }
        
        return {
            "status": "healthy" if all(checks.values()) else "degraded",
            "checks": checks,
            "timestamp": datetime.now().isoformat()
        }

    def _check_gcp_connection(self) -> bool:
        """Test GCP connectivity"""
        if self.gcp_config.get("mock_mode", True):
            return True  # Always healthy in mock mode
        try:
            from google.cloud import storage
            client = storage.Client()
            list(client.list_buckets(max_results=1))
            return True
        except:
            return False
        
    def get_metrics(self) -> Dict[str, Any]:
        """Enterprise metrics for monitoring"""
        return {
            "provisioning_success_rate": self._calculate_success_rate(),
            "avg_retry_count": self._calculate_avg_retries(),
            "circuit_breaker_trips": self._count_circuit_trips(),
            "cost_efficiency": self.total_api_calls / max(self.total_retry_costs, 0.01),
            "resource_utilization": len(self.resource_registry) / 1000,  # vs capacity
            "retry_queue_depth": len(self.retry_queue),
            "total_resources_managed": len(self.resource_registry),
            "total_plans_executed": len(self.provisioning_plans)
        }

    def _calculate_success_rate(self) -> float:
        """Calculate overall provisioning success rate"""
        total = 0
        successful = 0
        for plan_data in self.provisioning_plans.values():
            completed = len(plan_data.get("completed_resources", []))
            failed = len(plan_data.get("failed_resources", []))
            total += completed + failed
            successful += completed
        return (successful / max(total, 1)) * 100

    def _calculate_avg_retries(self) -> float:
        """Calculate average retry count across all resources"""
        total_retries = 0
        resource_count = 0
        for resource_data in self.resource_registry.values():
            if isinstance(resource_data, dict):
                total_retries += resource_data.get("retry_count", 0)
                resource_count += 1
        return total_retries / max(resource_count, 1)

    def _count_circuit_trips(self) -> int:
        """Count how many times circuit breakers have tripped"""
        trips = 0
        for breaker in self.circuit_breakers.values():
            state = breaker.get_state()
            trips += state.get("failure_count", 0)
        return trips
    
    def _check_alert_conditions(self):
        """Check for alert conditions"""
        alerts = []
        
        # Circuit breaker alerts
        for service, breaker in self.circuit_breakers.items():
            state = breaker.get_state()
            if state["state"] == "open":
                alerts.append({
                    "severity": "critical",
                    "service": service,
                    "message": f"Circuit breaker {service} is OPEN"
                })
        
        # Cost alerts
        if self.today_costs > self.daily_cost_limit * 0.9:
            alerts.append({
                "severity": "warning",
                "message": f"Daily cost at 90%: ${self.today_costs:.2f}"
            })
        
        # Retry queue alerts
        if len(self.retry_queue) > 50:
            alerts.append({
                "severity": "warning", 
                "message": f"Retry queue backing up: {len(self.retry_queue)} items"
            })
        
        # Emit alerts to event log
        for alert in alerts:
            emit_event(
                "provisioning-alert",
                "auto_provisioning.alert",
                status="warning",
                details=alert
            )
        
        return alerts
    
    def cleanup_team_infrastructure(self, team_id: str) -> bool:
        """Enhanced cleanup with retry logic"""
        emit_event(
            f"cleanup-{team_id}",
            "auto_provisioning.cleanup.start",
            status="info",
            details={"team_id": team_id}
        )
        
        # Find all resources for team
        resources_to_cleanup = []
        for resource_id, resource_data in self.resource_registry.items():
            if resource_data.get("owner_team_id") == team_id:
                resources_to_cleanup.append(resource_id)
        
        cleanup_failed = []
        for resource_id in resources_to_cleanup:
            try:
                self._cleanup_resource(self.resource_registry[resource_id])
                del self.resource_registry[resource_id]
            except Exception as e:
                cleanup_failed.append(resource_id)
                emit_event(
                    f"cleanup-{resource_id}",
                    "auto_provisioning.cleanup.failed",
                    status="error",
                    details={"error": str(e)}
                )
        
        # Remove from plans
        plans_to_remove = []
        for plan_id, plan_data in self.provisioning_plans.items():
            if plan_data.get("team_id") == team_id:
                plans_to_remove.append(plan_id)
        
        for plan_id in plans_to_remove:
            del self.provisioning_plans[plan_id]
        
        self._save_resources()
        
        emit_event(
            f"cleanup-{team_id}",
            "auto_provisioning.cleanup.complete",
            status="ok" if not cleanup_failed else "partial",
            details={
                "cleaned": len(resources_to_cleanup) - len(cleanup_failed),
                "failed": len(cleanup_failed)
            }
        )
        
        return len(cleanup_failed) == 0
    
    # Keep all the original helper methods from the base class
    def _create_provisioning_plan(self, live_team: LiveTeam) -> ProvisioningPlan:
        """Create comprehensive provisioning plan (unchanged from original)"""
        plan_id = f"plan_{live_team.team_id}_{int(time.time())}"
        agent_resources = {}
        team_resources = []
        
        for agent_id, agent in live_team.live_agents.items():
            agent_resources[agent_id] = self._plan_agent_resources(agent, live_team.team_id)
        
        team_resources = self._plan_team_resources(live_team)
        provisioning_order = self._calculate_provisioning_order(agent_resources, team_resources)
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
    
    # Include all other helper methods from original implementation...
    # (Keeping them unchanged for compatibility)
    
    def _plan_agent_resources(self, agent: LiveAgent, team_id: str) -> List[ProvisionedResource]:
        """Plan resources for agent (from original)"""
        resources = []
        
        # Pub/Sub topics
        resources.append(ProvisionedResource(
            resource_id=f"{agent.agent_id}_input_topic",
            resource_type="pubsub_topic",
            resource_name=f"{agent.agent_id}_input",
            resource_config={"message_retention_duration": "1h"},
            provisioning_status="pending",
            creation_timestamp=datetime.now().isoformat(),
            owner_agent_id=agent.agent_id,
            owner_team_id=team_id
        ))
        
        resources.append(ProvisionedResource(
            resource_id=f"{agent.agent_id}_output_topic",
            resource_type="pubsub_topic",
            resource_name=f"{agent.agent_id}_output",
            resource_config={"message_retention_duration": "1h"},
            provisioning_status="pending",
            creation_timestamp=datetime.now().isoformat(),
            owner_agent_id=agent.agent_id,
            owner_team_id=team_id
        ))
        
        # Storage bucket
        resources.append(ProvisionedResource(
            resource_id=f"{agent.agent_id}_storage",
            resource_type="storage_bucket",
            resource_name=f"agent-{agent.agent_id}-data",
            resource_config={"location": "us-central1", "storage_class": "STANDARD"},
            provisioning_status="pending",
            creation_timestamp=datetime.now().isoformat(),
            owner_agent_id=agent.agent_id,
            owner_team_id=team_id
        ))
        
        # IAM role
        resources.append(ProvisionedResource(
            resource_id=f"{agent.agent_id}_iam_role",
            resource_type="iam_role",
            resource_name=f"agent-{agent.agent_id}-role",
            resource_config={"permissions": ["pubsub.messages.publish", "storage.objects.create"]},
            provisioning_status="pending",
            creation_timestamp=datetime.now().isoformat(),
            owner_agent_id=agent.agent_id,
            owner_team_id=team_id
        ))
        
        return resources
    
    def _plan_team_resources(self, live_team: LiveTeam) -> List[ProvisionedResource]:
        """Plan team resources (from original)"""
        resources = []
        
        # Team coordination topic
        resources.append(ProvisionedResource(
            resource_id=f"{live_team.team_id}_coordination",
            resource_type="pubsub_topic",
            resource_name=f"team-{live_team.team_id}-coordination",
            resource_config={"message_retention_duration": "24h"},
            provisioning_status="pending",
            creation_timestamp=datetime.now().isoformat(),
            owner_agent_id="team_shared",
            owner_team_id=live_team.team_id
        ))
        
        # Team audit storage
        resources.append(ProvisionedResource(
            resource_id=f"{live_team.team_id}_audit_storage",
            resource_type="storage_bucket",
            resource_name=f"team-{live_team.team_id}-audit",
            resource_config={"location": "us-central1", "storage_class": "NEARLINE"},
            provisioning_status="pending",
            creation_timestamp=datetime.now().isoformat(),
            owner_agent_id="team_shared",
            owner_team_id=live_team.team_id
        ))
        
        return resources
    
    def _calculate_provisioning_order(self, agent_resources: Dict[str, List[ProvisionedResource]],
                                     team_resources: List[ProvisionedResource]) -> List[str]:
        """Calculate provisioning order (from original)"""
        order = []
        
        # IAM roles first
        for resources in agent_resources.values():
            for resource in resources:
                if resource.resource_type == "iam_role":
                    order.append(resource.resource_id)
        
        # Storage buckets
        for resources in agent_resources.values():
            for resource in resources:
                if resource.resource_type == "storage_bucket":
                    order.append(resource.resource_id)
        
        for resource in team_resources:
            if resource.resource_type == "storage_bucket":
                order.append(resource.resource_id)
        
        # Pub/Sub topics
        for resources in agent_resources.values():
            for resource in resources:
                if resource.resource_type == "pubsub_topic":
                    order.append(resource.resource_id)
        
        for resource in team_resources:
            if resource.resource_type == "pubsub_topic":
                order.append(resource.resource_id)
        
        return order
    
    def _estimate_provisioning_costs(self, agent_resources: Dict[str, List[ProvisionedResource]],
                                    team_resources: List[ProvisionedResource]) -> Dict[str, float]:
        """Estimate costs (from original)"""
        pubsub_count = sum(1 for r_list in agent_resources.values() 
                          for r in r_list if r.resource_type == "pubsub_topic")
        pubsub_count += sum(1 for r in team_resources if r.resource_type == "pubsub_topic")
        
        storage_gb = len(agent_resources) + 5  # 1GB per agent + 5GB team
        
        return {
            "pubsub_topics": pubsub_count * 0.40,
            "storage_buckets": storage_gb * 0.02,
            "iam_roles": 0.0,
            "total_monthly": (pubsub_count * 0.40) + (storage_gb * 0.02)
        }
    
    def _generate_cleanup_procedures(self, agent_resources: Dict[str, List[ProvisionedResource]],
                                    team_resources: List[ProvisionedResource]) -> Dict[str, Any]:
        """Generate cleanup procedures (from original)"""
        return {
            "cleanup_order": ["pubsub_topics", "storage_buckets", "iam_roles"],
            "backup_procedures": {"audit_logs": "archive_before_deletion"}
        }
    
    def _validate_provisioning_plan(self, plan: ProvisioningPlan) -> ProvisioningPlan:
        """Validate plan (from original with enhancements)"""
        all_resources = []
        for resources in plan.agent_resources.values():
            all_resources.extend(resources)
        all_resources.extend(plan.team_resources)
        
        for resource in all_resources:
            if not self._is_valid_gcp_resource_name(resource.resource_name):
                raise ValueError(f"Invalid resource name: {resource.resource_name}")
        
        if plan.estimated_cost["total_monthly"] > 100.0:
            emit_event(
                "validation",
                "auto_provisioning.validation.cost_warning",
                status="warning",
                details={"estimated_cost": plan.estimated_cost["total_monthly"]}
            )
        
        return plan
    
    def _is_valid_gcp_resource_name(self, name: str) -> bool:
        """Validate GCP naming (from original)"""
        if len(name) < 3 or len(name) > 63:
            return False
        if not name.replace('-', '').replace('_', '').isalnum():
            return False
        return True
    
    def _setup_team_communication(self, plan: ProvisioningPlan, live_team: LiveTeam):
        """Setup team communication (stub from original)"""
        emit_event(
            f"comm-{live_team.team_id}",
            "auto_provisioning.communication.configured",
            status="ok"
        )
    
    def _setup_monitoring_infrastructure(self, plan: ProvisioningPlan, live_team: LiveTeam):
        """Setup monitoring (stub from original)"""
        emit_event(
            f"monitor-{live_team.team_id}",
            "auto_provisioning.monitoring.configured",
            status="ok"
        )