#!/usr/bin/env python3
"""
Team Coordination Engine for Live Agent Workflow Orchestration

Handles real-time coordination and workflow orchestration between agent team members.
Implements different coordination protocols (pipeline, parallel, hierarchical) with
constitutional governance integration.

Capabilities:
- Real-time workflow orchestration
- Message routing and coordination protocols
- Task distribution and load balancing
- Failure detection and recovery
- Performance monitoring and optimization
- Constitutional governance coordination

Part of Week 3 Agent Instantiation & Auto-Wiring for Project Resilience
"""

import json
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from queue import Queue, Empty
import uuid

from dynamic_agent_factory import LiveAgent, LiveTeam
from auto_provisioning_system import ProvisioningPlan


class CoordinationProtocol(Enum):
    """Supported coordination protocols"""
    PIPELINE = "pipeline"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    COLLABORATIVE = "collaborative"


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ESCALATED = "escalated"


@dataclass
class CoordinationTask:
    """Represents a task in the coordination system"""
    task_id: str
    task_type: str
    task_data: Dict[str, Any]
    assigned_agent_id: Optional[str]
    status: TaskStatus
    priority: int  # 1-5, higher is more urgent
    created_timestamp: str
    started_timestamp: Optional[str]
    completed_timestamp: Optional[str]
    dependencies: List[str]  # Other task IDs this depends on
    outputs: Dict[str, Any]  # Task results
    metadata: Dict[str, Any]


@dataclass
class CoordinationMessage:
    """Message passed between agents for coordination"""
    message_id: str
    from_agent_id: str
    to_agent_id: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: str
    correlation_id: Optional[str]  # Links related messages


@dataclass
class WorkflowState:
    """Current state of team workflow"""
    workflow_id: str
    team_id: str
    protocol: CoordinationProtocol
    active_tasks: Dict[str, CoordinationTask]
    completed_tasks: Dict[str, CoordinationTask]
    pending_messages: List[CoordinationMessage]
    agent_status: Dict[str, str]  # agent_id -> status
    performance_metrics: Dict[str, Any]
    last_activity: str


class TeamCoordinationEngine:
    """
    Orchestrates workflow coordination between live agent team members.
    Handles real-time task distribution, message routing, and performance monitoring.
    """
    
    def __init__(self):
        self.active_workflows = {}
        self.coordination_handlers = {}
        self.message_queues = {}
        self.task_processors = {}
        
        # Performance monitoring
        self.performance_metrics = {}
        self.coordination_history = []
        
        # Initialize coordination protocol handlers
        self._initialize_protocol_handlers()
        
        # Start coordination engine thread
        self.engine_running = True
        self.coordination_thread = threading.Thread(target=self._coordination_loop, daemon=True)
        self.coordination_thread.start()
    
    def _initialize_protocol_handlers(self):
        """Initialize handlers for different coordination protocols"""
        
        self.coordination_handlers = {
            CoordinationProtocol.PIPELINE: self._handle_pipeline_coordination,
            CoordinationProtocol.PARALLEL: self._handle_parallel_coordination,
            CoordinationProtocol.HIERARCHICAL: self._handle_hierarchical_coordination,
            CoordinationProtocol.COLLABORATIVE: self._handle_collaborative_coordination
        }
    
    def start_team_coordination(self, live_team: LiveTeam, 
                              provisioning_plan: ProvisioningPlan) -> str:
        """
        Start coordination for a live agent team
        """
        
        workflow_id = f"workflow_{live_team.team_id}_{int(time.time())}"
        
        print(f"COORDINATOR: Starting workflow {workflow_id} for team {live_team.team_id}")
        
        # Determine coordination protocol
        protocol_name = live_team.coordination_state.get("protocol", "pipeline")
        protocol = CoordinationProtocol(protocol_name)
        
        # Initialize workflow state
        workflow_state = WorkflowState(
            workflow_id=workflow_id,
            team_id=live_team.team_id,
            protocol=protocol,
            active_tasks={},
            completed_tasks={},
            pending_messages=[],
            agent_status={agent_id: "ready" for agent_id in live_team.live_agents.keys()},
            performance_metrics=self._initialize_workflow_metrics(),
            last_activity=datetime.now().isoformat()
        )
        
        # Setup message queues for agents
        for agent_id in live_team.live_agents.keys():
            self.message_queues[agent_id] = Queue()
        
        # Setup task processors for coordination protocol
        self._setup_task_processors(live_team, protocol)
        
        # Register workflow
        self.active_workflows[workflow_id] = workflow_state
        
        print(f"COORDINATOR: Workflow {workflow_id} started with {protocol.value} protocol")
        
        return workflow_id
    
    def _initialize_workflow_metrics(self) -> Dict[str, Any]:
        """Initialize performance metrics for workflow"""
        
        return {
            "tasks_processed": 0,
            "average_task_duration": 0.0,
            "coordination_efficiency": 0.0,
            "agent_utilization": {},
            "message_latency": 0.0,
            "failure_rate": 0.0,
            "throughput_per_minute": 0.0,
            "last_updated": datetime.now().isoformat()
        }
    
    def _setup_task_processors(self, live_team: LiveTeam, protocol: CoordinationProtocol):
        """Setup task processors based on coordination protocol"""
        
        if protocol == CoordinationProtocol.PIPELINE:
            self._setup_pipeline_processors(live_team)
        elif protocol == CoordinationProtocol.PARALLEL:
            self._setup_parallel_processors(live_team)
        elif protocol == CoordinationProtocol.HIERARCHICAL:
            self._setup_hierarchical_processors(live_team)
        elif protocol == CoordinationProtocol.COLLABORATIVE:
            self._setup_collaborative_processors(live_team)
    
    def _setup_pipeline_processors(self, live_team: LiveTeam):
        """Setup processors for pipeline coordination"""
        
        # Sort agents by pipeline order
        sorted_agents = sorted(
            live_team.live_agents.items(),
            key=lambda x: self._get_pipeline_order(x[1].role_definition.role_name)
        )
        
        # Setup processing chain
        for i, (agent_id, agent) in enumerate(sorted_agents):
            processor_config = {
                "role": "processor",
                "input_source": sorted_agents[i-1][0] if i > 0 else "external",
                "output_target": sorted_agents[i+1][0] if i < len(sorted_agents)-1 else "external",
                "processing_function": self._create_agent_processor(agent)
            }
            
            self.task_processors[agent_id] = processor_config
    
    def _get_pipeline_order(self, role_name: str) -> int:
        """Get pipeline order for agent role"""
        
        if "coordinator" in role_name:
            return 1
        elif "specialist" in role_name:
            return 2
        elif "dispatcher" in role_name:
            return 4
        elif "monitor" in role_name:
            return 5
        else:
            return 3
    
    def _setup_parallel_processors(self, live_team: LiveTeam):
        """Setup processors for parallel coordination"""
        
        for agent_id, agent in live_team.live_agents.items():
            processor_config = {
                "role": "parallel_processor",
                "input_source": "task_distributor",
                "output_target": "result_aggregator",
                "processing_function": self._create_agent_processor(agent)
            }
            
            self.task_processors[agent_id] = processor_config
    
    def _setup_hierarchical_processors(self, live_team: LiveTeam):
        """Setup processors for hierarchical coordination"""
        
        supervisors = []
        workers = []
        
        for agent_id, agent in live_team.live_agents.items():
            if "monitor" in agent.role_definition.role_name or "manager" in agent.role_definition.role_name:
                supervisors.append((agent_id, agent))
            else:
                workers.append((agent_id, agent))
        
        # Setup supervisor processors
        for agent_id, agent in supervisors:
            processor_config = {
                "role": "supervisor",
                "input_source": "external",
                "output_target": [worker_id for worker_id, _ in workers],
                "processing_function": self._create_supervisor_processor(agent)
            }
            self.task_processors[agent_id] = processor_config
        
        # Setup worker processors
        for agent_id, agent in workers:
            supervisor_id = supervisors[0][0] if supervisors else None
            processor_config = {
                "role": "worker",
                "input_source": supervisor_id,
                "output_target": supervisor_id,
                "processing_function": self._create_agent_processor(agent)
            }
            self.task_processors[agent_id] = processor_config
    
    def _setup_collaborative_processors(self, live_team: LiveTeam):
        """Setup processors for collaborative coordination"""
        
        for agent_id, agent in live_team.live_agents.items():
            processor_config = {
                "role": "collaborator",
                "input_source": "shared_workspace",
                "output_target": "shared_workspace",
                "processing_function": self._create_collaborative_processor(agent)
            }
            
            self.task_processors[agent_id] = processor_config
    
    def _create_agent_processor(self, agent: LiveAgent) -> Callable:
        """Create processing function for agent"""
        
        def processor(task: CoordinationTask) -> Dict[str, Any]:
            """Mock agent processing function"""
            
            print(f"COORDINATOR: Agent {agent.agent_id} processing task {task.task_id}")
            
            # Mock processing based on agent role
            processing_time = self._estimate_processing_time(agent, task)
            time.sleep(processing_time / 1000)  # Convert ms to seconds for simulation
            
            # Generate mock results
            results = {
                "processed_by": agent.agent_id,
                "task_id": task.task_id,
                "processing_time_ms": processing_time,
                "result_data": f"Processed {task.task_type} by {agent.role_definition.role_name}",
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            }
            
            return results
        
        return processor
    
    def _create_supervisor_processor(self, agent: LiveAgent) -> Callable:
        """Create supervisor processing function"""
        
        def supervisor_processor(task: CoordinationTask) -> Dict[str, Any]:
            """Supervisor task processing and delegation"""
            
            print(f"COORDINATOR: Supervisor {agent.agent_id} delegating task {task.task_id}")
            
            # Supervisor breaks down task and delegates
            subtasks = self._break_down_task(task)
            
            results = {
                "processed_by": agent.agent_id,
                "task_id": task.task_id,
                "subtasks_created": len(subtasks),
                "delegation_plan": subtasks,
                "status": "delegated",
                "timestamp": datetime.now().isoformat()
            }
            
            return results
        
        return supervisor_processor
    
    def _create_collaborative_processor(self, agent: LiveAgent) -> Callable:
        """Create collaborative processing function"""
        
        def collaborative_processor(task: CoordinationTask) -> Dict[str, Any]:
            """Collaborative task processing with consensus"""
            
            print(f"COORDINATOR: Agent {agent.agent_id} collaborating on task {task.task_id}")
            
            # Mock collaborative processing
            contribution = {
                "agent_id": agent.agent_id,
                "contribution_type": "analysis" if "specialist" in agent.role_definition.role_name else "coordination",
                "confidence": 0.85,
                "recommendation": f"Recommendation from {agent.role_definition.role_name}",
                "timestamp": datetime.now().isoformat()
            }
            
            return contribution
        
        return collaborative_processor
    
    def _estimate_processing_time(self, agent: LiveAgent, task: CoordinationTask) -> float:
        """Estimate processing time for agent/task combination"""
        
        base_time = 100  # Base 100ms
        
        # Adjust based on task complexity
        if task.priority > 3:
            base_time *= 1.5
        
        # Adjust based on agent role
        if "specialist" in agent.role_definition.role_name:
            base_time *= 1.2  # Specialists take longer
        elif "coordinator" in agent.role_definition.role_name:
            base_time *= 0.8  # Coordinators are faster
        
        return base_time
    
    def _break_down_task(self, task: CoordinationTask) -> List[Dict[str, Any]]:
        """Break down supervisor task into subtasks"""
        
        # Mock task breakdown
        subtasks = []
        
        if task.task_type == "customer_support_request":
            subtasks = [
                {"subtask_type": "classify_issue", "priority": 4},
                {"subtask_type": "generate_response", "priority": 3},
                {"subtask_type": "quality_check", "priority": 2}
            ]
        
        return subtasks
    
    def submit_task(self, workflow_id: str, task_data: Dict[str, Any]) -> str:
        """Submit a task to the coordination workflow"""
        
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        
        task = CoordinationTask(
            task_id=task_id,
            task_type=task_data.get("type", "generic"),
            task_data=task_data,
            assigned_agent_id=None,
            status=TaskStatus.PENDING,
            priority=task_data.get("priority", 3),
            created_timestamp=datetime.now().isoformat(),
            started_timestamp=None,
            completed_timestamp=None,
            dependencies=task_data.get("dependencies", []),
            outputs={},
            metadata={}
        )
        
        workflow = self.active_workflows[workflow_id]
        workflow.active_tasks[task_id] = task
        
        print(f"COORDINATOR: Task {task_id} submitted to workflow {workflow_id}")
        
        return task_id
    
    def _coordination_loop(self):
        """Main coordination loop that processes tasks and messages"""
        
        while self.engine_running:
            try:
                # Process all active workflows
                for workflow_id, workflow in self.active_workflows.items():
                    self._process_workflow(workflow)
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Sleep briefly to avoid busy waiting
                time.sleep(0.1)
                
            except Exception as e:
                print(f"COORDINATOR: Error in coordination loop: {e}")
                time.sleep(1)
    
    def _process_workflow(self, workflow: WorkflowState):
        """Process a single workflow"""
        
        # Handle coordination protocol
        handler = self.coordination_handlers.get(workflow.protocol)
        if handler:
            handler(workflow)
        
        # Process pending tasks
        self._process_pending_tasks(workflow)
        
        # Process pending messages
        self._process_pending_messages(workflow)
        
        # Update workflow metrics
        self._update_workflow_metrics(workflow)
    
    def _handle_pipeline_coordination(self, workflow: WorkflowState):
        """Handle pipeline coordination protocol"""
        
        # Process tasks in pipeline order
        for task_id, task in workflow.active_tasks.items():
            if task.status == TaskStatus.PENDING:
                # Assign to first agent in pipeline
                first_agent = self._get_first_pipeline_agent(workflow.team_id)
                if first_agent:
                    self._assign_task_to_agent(task, first_agent, workflow)
    
    def _handle_parallel_coordination(self, workflow: WorkflowState):
        """Handle parallel coordination protocol"""
        
        # Distribute tasks among available agents
        available_agents = [
            agent_id for agent_id, status in workflow.agent_status.items()
            if status == "ready"
        ]
        
        pending_tasks = [
            task for task in workflow.active_tasks.values()
            if task.status == TaskStatus.PENDING
        ]
        
        # Round-robin task assignment
        for i, task in enumerate(pending_tasks):
            if available_agents:
                agent_id = available_agents[i % len(available_agents)]
                self._assign_task_to_agent(task, agent_id, workflow)
    
    def _handle_hierarchical_coordination(self, workflow: WorkflowState):
        """Handle hierarchical coordination protocol"""
        
        # Route tasks through supervisors first
        supervisors = [
            agent_id for agent_id, processor in self.task_processors.items()
            if processor.get("role") == "supervisor"
        ]
        
        for task_id, task in workflow.active_tasks.items():
            if task.status == TaskStatus.PENDING and supervisors:
                supervisor_id = supervisors[0]  # Use first supervisor
                self._assign_task_to_agent(task, supervisor_id, workflow)
    
    def _handle_collaborative_coordination(self, workflow: WorkflowState):
        """Handle collaborative coordination protocol"""
        
        # Assign tasks to multiple agents for collaboration
        for task_id, task in workflow.active_tasks.items():
            if task.status == TaskStatus.PENDING:
                # Assign to all available agents for collaborative processing
                available_agents = [
                    agent_id for agent_id, status in workflow.agent_status.items()
                    if status == "ready"
                ]
                
                if available_agents:
                    # Create collaborative task for each agent
                    for agent_id in available_agents:
                        self._create_collaborative_subtask(task, agent_id, workflow)
    
    def _get_first_pipeline_agent(self, team_id: str) -> Optional[str]:
        """Get the first agent in pipeline order"""
        
        # Mock implementation - would use actual team configuration
        for agent_id, processor in self.task_processors.items():
            if processor.get("input_source") == "external":
                return agent_id
        
        return None
    
    def _assign_task_to_agent(self, task: CoordinationTask, agent_id: str, workflow: WorkflowState):
        """Assign a task to a specific agent"""
        
        task.assigned_agent_id = agent_id
        task.status = TaskStatus.ASSIGNED
        task.started_timestamp = datetime.now().isoformat()
        
        workflow.agent_status[agent_id] = "busy"
        
        print(f"COORDINATOR: Task {task.task_id} assigned to agent {agent_id}")
        
        # Process task immediately (in real implementation, would queue for agent)
        self._process_task_immediately(task, workflow)
    
    def _create_collaborative_subtask(self, task: CoordinationTask, agent_id: str, workflow: WorkflowState):
        """Create collaborative subtask for agent"""
        
        subtask_id = f"{task.task_id}_collab_{agent_id}"
        
        subtask = CoordinationTask(
            task_id=subtask_id,
            task_type=f"collaborative_{task.task_type}",
            task_data=task.task_data,
            assigned_agent_id=agent_id,
            status=TaskStatus.ASSIGNED,
            priority=task.priority,
            created_timestamp=datetime.now().isoformat(),
            started_timestamp=datetime.now().isoformat(),
            completed_timestamp=None,
            dependencies=[],
            outputs={},
            metadata={"parent_task": task.task_id, "collaboration_type": "consensus"}
        )
        
        workflow.active_tasks[subtask_id] = subtask
        workflow.agent_status[agent_id] = "busy"
        
        print(f"COORDINATOR: Collaborative subtask {subtask_id} created for agent {agent_id}")
    
    def _process_pending_tasks(self, workflow: WorkflowState):
        """Process tasks that are assigned but not yet completed"""
        
        for task_id, task in list(workflow.active_tasks.items()):
            if task.status == TaskStatus.ASSIGNED:
                # Start processing
                task.status = TaskStatus.IN_PROGRESS
                
                # Mock processing completion after brief delay
                if task.assigned_agent_id in self.task_processors:
                    processor = self.task_processors[task.assigned_agent_id]
                    processing_function = processor["processing_function"]
                    
                    # Process task
                    try:
                        results = processing_function(task)
                        task.outputs = results
                        task.status = TaskStatus.COMPLETED
                        task.completed_timestamp = datetime.now().isoformat()
                        
                        # Move to completed tasks
                        workflow.completed_tasks[task_id] = task
                        del workflow.active_tasks[task_id]
                        
                        # Free up agent
                        if task.assigned_agent_id:
                            workflow.agent_status[task.assigned_agent_id] = "ready"
                        
                        print(f"COORDINATOR: Task {task_id} completed by agent {task.assigned_agent_id}")
                        
                    except Exception as e:
                        task.status = TaskStatus.FAILED
                        task.outputs = {"error": str(e)}
                        print(f"COORDINATOR: Task {task_id} failed: {e}")
    
    def _process_task_immediately(self, task: CoordinationTask, workflow: WorkflowState):
        """Process task immediately for testing (mock implementation)"""
        
        # This simulates immediate task processing for demonstration
        # In real implementation, tasks would be queued and processed asynchronously
        pass
    
    def _process_pending_messages(self, workflow: WorkflowState):
        """Process pending coordination messages"""
        
        for message in list(workflow.pending_messages):
            self._route_message(message, workflow)
            workflow.pending_messages.remove(message)
    
    def _route_message(self, message: CoordinationMessage, workflow: WorkflowState):
        """Route message to target agent"""
        
        target_agent = message.to_agent_id
        if target_agent in self.message_queues:
            self.message_queues[target_agent].put(message)
            print(f"COORDINATOR: Message {message.message_id} routed to agent {target_agent}")
    
    def _update_workflow_metrics(self, workflow: WorkflowState):
        """Update performance metrics for workflow"""
        
        total_tasks = len(workflow.completed_tasks)
        
        if total_tasks > 0:
            # Calculate average task duration
            durations = []
            for task in workflow.completed_tasks.values():
                if task.started_timestamp and task.completed_timestamp:
                    start = datetime.fromisoformat(task.started_timestamp)
                    end = datetime.fromisoformat(task.completed_timestamp)
                    duration = (end - start).total_seconds() * 1000  # ms
                    durations.append(duration)
            
            if durations:
                workflow.performance_metrics["average_task_duration"] = sum(durations) / len(durations)
        
        workflow.performance_metrics["tasks_processed"] = total_tasks
        workflow.performance_metrics["last_updated"] = datetime.now().isoformat()
        workflow.last_activity = datetime.now().isoformat()
    
    def _update_performance_metrics(self):
        """Update global performance metrics"""
        
        total_workflows = len(self.active_workflows)
        total_tasks = sum(
            len(workflow.completed_tasks) for workflow in self.active_workflows.values()
        )
        
        self.performance_metrics = {
            "active_workflows": total_workflows,
            "total_tasks_processed": total_tasks,
            "coordination_efficiency": self._calculate_coordination_efficiency(),
            "last_updated": datetime.now().isoformat()
        }
    
    def _calculate_coordination_efficiency(self) -> float:
        """Calculate overall coordination efficiency"""
        
        # Mock efficiency calculation
        total_agents = sum(
            len(workflow.agent_status) for workflow in self.active_workflows.values()
        )
        
        if total_agents == 0:
            return 0.0
        
        busy_agents = sum(
            len([status for status in workflow.agent_status.values() if status == "busy"])
            for workflow in self.active_workflows.values()
        )
        
        return (busy_agents / total_agents) * 100.0 if total_agents > 0 else 0.0
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a workflow"""
        
        if workflow_id not in self.active_workflows:
            return None
        
        workflow = self.active_workflows[workflow_id]
        
        return {
            "workflow_id": workflow.workflow_id,
            "team_id": workflow.team_id,
            "protocol": workflow.protocol.value,
            "active_tasks": len(workflow.active_tasks),
            "completed_tasks": len(workflow.completed_tasks),
            "agent_status": workflow.agent_status,
            "performance_metrics": workflow.performance_metrics,
            "last_activity": workflow.last_activity
        }
    
    def stop_team_coordination(self, workflow_id: str) -> bool:
        """Stop coordination for a team workflow"""
        
        if workflow_id in self.active_workflows:
            del self.active_workflows[workflow_id]
            print(f"COORDINATOR: Stopped workflow {workflow_id}")
            return True
        
        return False
    
    def shutdown(self):
        """Shutdown the coordination engine"""
        
        self.engine_running = False
        if self.coordination_thread.is_alive():
            self.coordination_thread.join(timeout=5)
        
        print("COORDINATOR: Coordination engine shutdown complete")


def test_team_coordination_engine():
    """Test the team coordination engine with a live team"""
    
    # Import dependencies
    from dynamic_agent_factory import DynamicAgentFactory
    from auto_provisioning_system import AutoProvisioningSystem
    from meta_architect_agent import MetaArchitectAgent
    
    coordinator = TeamCoordinationEngine()
    provisioner = AutoProvisioningSystem()
    factory = DynamicAgentFactory()
    architect = MetaArchitectAgent()
    
    print("=== Testing Team Coordination Engine ===\n")
    
    # Create test team
    test_goal = "Build a customer support system for handling technical issues"
    institution_blueprint = architect.design_institution(test_goal)
    live_team = factory.instantiate_team_from_blueprint(institution_blueprint)
    provisioning_plan = provisioner.provision_team_infrastructure(live_team)
    
    print(f"Team created: {live_team.team_id}")
    print(f"Agents: {len(live_team.live_agents)}")
    
    # Start coordination
    workflow_id = coordinator.start_team_coordination(live_team, provisioning_plan)
    
    print(f"\nWorkflow started: {workflow_id}")
    print(f"Protocol: {live_team.coordination_state['protocol']}")
    
    # Submit test tasks
    print(f"\nSubmitting test tasks...")
    
    task_ids = []
    for i in range(3):
        task_data = {
            "type": "customer_support_request",
            "priority": 3 + i,
            "data": f"Customer issue #{i + 1}"
        }
        task_id = coordinator.submit_task(workflow_id, task_data)
        task_ids.append(task_id)
        print(f"  Task submitted: {task_id}")
    
    # Let coordination run for a moment
    print(f"\nProcessing tasks...")
    time.sleep(2)
    
    # Check workflow status
    status = coordinator.get_workflow_status(workflow_id)
    if status:
        print(f"\nWorkflow Status:")
        print(f"  Active Tasks: {status['active_tasks']}")
        print(f"  Completed Tasks: {status['completed_tasks']}")
        print(f"  Agent Status: {status['agent_status']}")
        print(f"  Performance:")
        for metric, value in status['performance_metrics'].items():
            if isinstance(value, float):
                print(f"    {metric}: {value:.2f}")
            else:
                print(f"    {metric}: {value}")
    
    # Check global metrics
    print(f"\nGlobal Coordination Metrics:")
    for metric, value in coordinator.performance_metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.2f}")
        else:
            print(f"  {metric}: {value}")
    
    # Cleanup
    coordinator.stop_team_coordination(workflow_id)
    coordinator.shutdown()
    
    print(f"\nCoordination test complete")


if __name__ == "__main__":
    test_team_coordination_engine()