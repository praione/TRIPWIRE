"""
Enhanced Pipeline Wrapper with Production-Grade Error Handling
For Project Resilience Intent to Institution Pipeline
"""

import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Callable
import time
import hashlib
from enum import Enum

class RecoveryStrategy(Enum):
    """Defines how to handle different types of failures"""
    RETRY = "retry"           # Try again with same inputs
    FALLBACK = "fallback"     # Use simpler alternative
    PARTIAL = "partial"       # Save what we have, continue
    ABORT = "abort"           # Stop pipeline, save state
    SKIP = "skip"            # Skip this step, continue

class ErrorCategory(Enum):
    """Categorize errors for better handling"""
    NETWORK = "network"       # Transient network issues
    AUTH = "auth"             # Authentication/permission
    VALIDATION = "validation" # Bad input data
    RESOURCE = "resource"     # Out of memory, quota
    LOGIC = "logic"          # Code bugs
    EXTERNAL = "external"    # Third-party service issues

class EnhancedPipelineWrapper:
    """
    Production-ready pipeline wrapper with comprehensive error handling
    """
    
    def __init__(self, institution_id: str = None):
        self.institution_id = institution_id or self._generate_id()
        
        # Validate and register the institution ID
        self.base_path = Path("institutions")
        self.institution_id = self._validate_institution_id(self.institution_id)
        
        self.institution_dir = self.base_path / f"inst_{self.institution_id}"
        self.institution_dir.mkdir(parents=True, exist_ok=True)
        
        # Enhanced tracking
        self.errors = []
        self.warnings = []
        self.completed_stages = []
        self.stage_timings = {}
        self.recovery_attempts = {}
        
        # Recovery state
        self.checkpoint_file = self.institution_dir / "checkpoint.json"
        self.can_resume = False
        self.resume_point = None
        
        # Load previous state if exists
        self._load_checkpoint()

    def _validate_institution_id(self, inst_id: str) -> str:
        """Ensure institution ID is unique across all institutions"""
        registry_file = self.base_path / "institution_registry.json"
        
        # Load existing registry
        existing = {}
        if registry_file.exists():
            try:
                with open(registry_file) as f:
                    existing = json.load(f)
            except:
                existing = {}
        
        # Check for collision
        attempts = 0
        original_id = inst_id
        while inst_id in existing:
            attempts += 1
            inst_id = self._generate_id()  # Generate new one
            if attempts > 10:
                # Fallback to guaranteed unique
                inst_id = f"{original_id}_{uuid.uuid4().hex[:6]}"
                break
        
        # Register this institution
        existing[inst_id] = {
            "created": datetime.now().isoformat(),
            "original_id": original_id if attempts > 0 else None
        }
        
        # Save registry
        with open(registry_file, 'w') as f:
            json.dump(existing, f, indent=2)
        
        if attempts > 0:
            print(f"⚠️ ID collision avoided: {original_id} → {inst_id}")
        
        return inst_id


    def safe_call(self, func, *args, **kwargs):
        """Compatibility wrapper for demo_runner"""
        try:
            result = func(*args, **kwargs)
            return True, result
        except Exception as e:
            print(f"Error: {str(e)[:200]}")
            return False, None
        
    def get_error_summary(self):
        """Compatibility method for demo_runner"""
        if self.errors:
            return {
                'count': len(self.errors),
                'errors': self.errors[:3]  # First 3 errors
            }
        return None
    
    def _generate_id(self) -> str:
        """Generate unique institution ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_hash = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"{timestamp}_{random_hash}"
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Intelligently categorize errors for appropriate handling"""
        error_msg = str(error).lower()
        error_type = type(error).__name__
        
        # Network-related errors
        if any(x in error_msg for x in ['timeout', 'connection', 'network', 'dns']):
            return ErrorCategory.NETWORK
        
        # Authentication errors
        if any(x in error_msg for x in ['auth', 'permission', 'forbidden', '403', '401']):
            return ErrorCategory.AUTH
        
        # Validation errors
        if any(x in error_msg for x in ['invalid', 'missing', 'required', 'format']):
            return ErrorCategory.VALIDATION
        
        # Resource errors
        if any(x in error_msg for x in ['quota', 'limit', 'memory', 'space']):
            return ErrorCategory.RESOURCE
        
        # External service errors
        if any(x in error_msg for x in ['vertex', 'gcp', 'google', 'api']):
            return ErrorCategory.EXTERNAL
        
        # Default to logic error
        return ErrorCategory.LOGIC
    
    def _get_recovery_strategy(self, error_category: ErrorCategory, attempt: int) -> RecoveryStrategy:
        """Determine recovery strategy based on error type and attempt number"""
        strategies = {
            ErrorCategory.NETWORK: [RecoveryStrategy.RETRY, RecoveryStrategy.RETRY, RecoveryStrategy.ABORT],
            ErrorCategory.AUTH: [RecoveryStrategy.ABORT],
            ErrorCategory.VALIDATION: [RecoveryStrategy.PARTIAL, RecoveryStrategy.SKIP],
            ErrorCategory.RESOURCE: [RecoveryStrategy.FALLBACK, RecoveryStrategy.PARTIAL],
            ErrorCategory.EXTERNAL: [RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK, RecoveryStrategy.PARTIAL],
            ErrorCategory.LOGIC: [RecoveryStrategy.PARTIAL, RecoveryStrategy.ABORT]
        }
        
        strategy_list = strategies.get(error_category, [RecoveryStrategy.ABORT])
        
        if attempt >= len(strategy_list):
            return strategy_list[-1]
        return strategy_list[attempt]
    
    def execute_with_recovery(self, func: Callable, func_name: str, args: tuple = (), kwargs: dict = None, 
                             max_attempts: int = 3, fallback_func: Optional[Callable] = None) -> Any:
        """
        Execute function with intelligent error recovery
        """
        kwargs = kwargs or {}
        start_time = time.time()
        
        for attempt in range(max_attempts):
            try:
                # Execute the function
                print(f"\n🔄 Executing: {func_name} (Attempt {attempt + 1}/{max_attempts})")
                result = func(*args, **kwargs)
                
                # Success - record it
                self.completed_stages.append(func_name)
                self.stage_timings[func_name] = time.time() - start_time
                self._save_checkpoint(func_name, result)
                
                print(f"✅ {func_name} completed successfully")
                return result
                
            except Exception as e:
                # Categorize and handle the error
                error_category = self._categorize_error(e)
                recovery_strategy = self._get_recovery_strategy(error_category, attempt)
                
                # Log detailed error info
                error_info = {
                    'stage': func_name,
                    'attempt': attempt + 1,
                    'error': str(e),
                    'type': type(e).__name__,
                    'category': error_category.value,
                    'strategy': recovery_strategy.value,
                    'timestamp': datetime.now().isoformat(),
                    'traceback': traceback.format_exc()
                }
                self.errors.append(error_info)
                
                # User-friendly error message
                self._print_error_message(func_name, e, error_category, recovery_strategy, attempt)
                
                # Apply recovery strategy
                if recovery_strategy == RecoveryStrategy.RETRY:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"⏳ Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                    
                elif recovery_strategy == RecoveryStrategy.FALLBACK and fallback_func:
                    print(f"🔄 Attempting fallback method...")
                    try:
                        result = fallback_func(*args, **kwargs)
                        self.warnings.append(f"Used fallback for {func_name}")
                        return result
                    except Exception as fallback_error:
                        print(f"⚠️ Fallback also failed: {str(fallback_error)[:100]}")
                        
                elif recovery_strategy == RecoveryStrategy.PARTIAL:
                    print(f"📝 Creating partial result for {func_name}")
                    return self._create_partial_result(func_name, e)
                    
                elif recovery_strategy == RecoveryStrategy.SKIP:
                    print(f"⏭️ Skipping {func_name} and continuing...")
                    self.warnings.append(f"Skipped {func_name} due to error")
                    return None
                    
                elif recovery_strategy == RecoveryStrategy.ABORT:
                    print(f"🛑 Aborting pipeline at {func_name}")
                    self._save_abort_state(func_name, e)
                    raise
        
        # All attempts exhausted
        print(f"❌ All attempts failed for {func_name}")
        self._save_abort_state(func_name, Exception(f"Exhausted {max_attempts} attempts"))
        raise Exception(f"Failed to execute {func_name} after {max_attempts} attempts")
    
    def _print_error_message(self, func_name: str, error: Exception, category: ErrorCategory, 
                            strategy: RecoveryStrategy, attempt: int):
        """Print user-friendly error message"""
        messages = {
            ErrorCategory.NETWORK: "Network connectivity issue detected",
            ErrorCategory.AUTH: "Authentication/permission problem",
            ErrorCategory.VALIDATION: "Invalid input data",
            ErrorCategory.RESOURCE: "Resource limit reached",
            ErrorCategory.EXTERNAL: "External service issue",
            ErrorCategory.LOGIC: "Unexpected error in logic"
        }
        
        print(f"\n⚠️ {messages.get(category, 'Error occurred')} in {func_name}")
        print(f"   Details: {str(error)[:200]}")
        print(f"   Strategy: {strategy.value}")
    
    def _create_partial_result(self, func_name: str, error: Exception) -> dict:
        """Create a partial result that allows pipeline to continue"""
        partial_result = {
            'status': 'partial',
            'error': str(error)[:500],
            'error_category': self._categorize_error(error).value,
            'function': func_name,
            'timestamp': datetime.now().isoformat(),
            'can_retry': True,
            'fallback_data': self._get_fallback_data(func_name)
        }
        
        # Save partial result
        partial_file = self.institution_dir / f"partial_{func_name}.json"
        with open(partial_file, 'w') as f:
            json.dump(partial_result, f, indent=2)
        
        return partial_result
    
    def _get_fallback_data(self, func_name: str) -> dict:
        """Get sensible fallback data for different pipeline stages"""
        fallbacks = {
            'parse_intent': {
                'goal': 'Create basic institution',
                'requirements': {'agents': 1, 'simple': True}
            },
            'design_architecture': {
                'architecture': 'minimal',
                'agents': [{'role': 'worker', 'type': 'basic'}]
            },
            'provision_resources': {
                'resources': 'local_only',
                'cloud': False
            }
        }
        return fallbacks.get(func_name, {})
    
    def _save_checkpoint(self, stage_name: str, data: Any):
        """Save checkpoint for recovery"""
        checkpoint = self._load_checkpoint() or {}
        
        checkpoint.update({
            'last_successful_stage': stage_name,
            'timestamp': datetime.now().isoformat(),
            'institution_id': self.institution_id,
            'completed_stages': self.completed_stages,
            'can_resume': True,
            'stage_data': {
                stage_name: data if isinstance(data, (dict, list, str, int, float, bool)) else str(data)
            }
        })
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
    
    def _load_checkpoint(self) -> Optional[dict]:
        """Load previous checkpoint if exists"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                    self.can_resume = checkpoint.get('can_resume', False)
                    self.resume_point = checkpoint.get('last_successful_stage')
                    self.completed_stages = checkpoint.get('completed_stages', [])
                    return checkpoint
            except:
                pass
        return None
    
    def _save_abort_state(self, failed_stage: str, error: Exception):
        """Save state when aborting pipeline"""
        abort_state = {
            'status': 'aborted',
            'failed_at': failed_stage,
            'error': str(error),
            'error_type': type(error).__name__,
            'timestamp': datetime.now().isoformat(),
            'completed_stages': self.completed_stages,
            'can_resume': True,
            'resume_instructions': f"Fix issue with {failed_stage} and run with --resume flag",
            'institution_id': self.institution_id
        }
        
        abort_file = self.institution_dir / "abort_state.json"
        with open(abort_file, 'w') as f:
            json.dump(abort_state, f, indent=2)
    
    def generate_summary(self) -> dict:
        """Generate comprehensive execution summary"""
        summary = {
            'institution_id': self.institution_id,
            'status': 'complete' if not self.errors else 'partial',
            'completed_stages': self.completed_stages,
            'total_stages': len(self.completed_stages) + len(self.errors),
            'errors': len(self.errors),
            'warnings': len(self.warnings),
            'can_resume': self.can_resume,
            'stage_timings': self.stage_timings,
            'total_time': sum(self.stage_timings.values()),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save detailed report
        report = {
            'summary': summary,
            'errors': self.errors,
            'warnings': self.warnings,
            'recovery_attempts': self.recovery_attempts
        }
        
        report_file = self.institution_dir / "execution_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return summary
    
    def print_summary(self):
        """Print user-friendly summary"""
        summary = self.generate_summary()
        
        print("\n" + "="*60)
        print("📊 PIPELINE EXECUTION SUMMARY")
        print("="*60)
        print(f"Institution ID: {summary['institution_id']}")
        print(f"Status: {'✅ ' + summary['status'].upper()}")
        print(f"Completed: {len(summary['completed_stages'])}/{summary['total_stages']} stages")
        
        if summary['errors'] > 0:
            print(f"⚠️ Errors: {summary['errors']}")
            print(f"📁 Error details saved to: {self.institution_dir}/execution_report.json")
        
        if summary['warnings']:
            print(f"⚡ Warnings: {len(summary['warnings'])}")
        
        if summary['can_resume']:
            print(f"♻️ Pipeline can be resumed from last checkpoint")
        
        print(f"⏱️ Total time: {summary['total_time']:.2f} seconds")
        print(f"📁 All artifacts saved to: {self.institution_dir}")
        print("="*60)


# Example usage showing how to integrate with existing pipeline
if __name__ == "__main__":
    from intent_to_institution import IntentToInstitutionEngine
    
    # Create wrapper
    wrapper = EnhancedPipelineWrapper()
    
    # Define pipeline stages with fallbacks
    engine = IntentToInstitutionEngine()
    
    # Example goal
    goal = "Create a financial analysis team with 3 agents"
    
    # Execute pipeline with robust error handling
    try:
        # Parse intent with fallback
        parsed = wrapper.execute_with_recovery(
            engine.intent_engine.parse_intent,
            "parse_intent",
            args=(goal,),
            fallback_func=lambda x: {'goal': x, 'basic': True}
        )
        
        # Design architecture
        architecture = wrapper.execute_with_recovery(
            engine.meta_architect.design_institution,
            "design_architecture", 
            kwargs={'goal': parsed}
        )
        
        # Create agents
        agents = wrapper.execute_with_recovery(
            engine.agent_factory.create_agents,
            "create_agents",
            kwargs={'architecture': architecture}
        )
        
        # Provision resources
        resources = wrapper.execute_with_recovery(
            engine.provisioner.provision,
            "provision_resources",
            kwargs={'agents': agents},
            fallback_func=lambda **kw: {'status': 'local_only'}
        )
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
    
    finally:
        # Always print summary
        wrapper.print_summary()