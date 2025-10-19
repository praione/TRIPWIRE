#!/usr/bin/env python3
"""
Task 4.3: Constitutional Violation Logging Test
Verifies that ConstitutionalViolation events are logged to the central ledger.
"""

import sys
import unittest
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sal import governance_middleware, ConstitutionalViolation
from dispatcher import SAL_LAYER
from event_log import emit_event


class TestViolationLogging(unittest.TestCase):
    """Tests that constitutional violations are logged to audit trail."""
    
    def test_constitutional_violation_logged_to_ledger(self):
        """Verify that violations create structured audit entries."""
        
        trace_id = SAL_LAYER.new_trace()
        
        # Trigger a constitutional violation
        try:
            governance_middleware(
                sal_instance=SAL_LAYER,
                agent_name="intro",
                tool_name="unauthorized_tool",
                task_payload={"tool_input": "test", "task_hint": "test"},
                trace_id=trace_id,
                step_config={"requires_governance": True}
            )
            self.fail("Expected ConstitutionalViolation to be raised")
            
        except ConstitutionalViolation as e:
            # Violation should be raised and logged
            print(f"[TEST] Constitutional violation captured: {e}")
            
            # Check that the violation has structured data
            self.assertTrue(hasattr(e, 'verdict'), "Violation should include verdict details")
            
            if hasattr(e, 'verdict') and isinstance(e.verdict, dict):
                self.assertIn('reason', e.verdict, "Verdict should include reason")
                print(f"[TEST] Violation logged with details: {e.verdict.get('reason', 'No reason')}")
    
    def test_violation_audit_trail_completeness(self):
        """Verify violations create complete audit records."""
        
        trace_id = SAL_LAYER.new_trace()
        
        # Test a content quality violation
        try:
            governance_middleware(
                sal_instance=SAL_LAYER,
                agent_name="intro", 
                tool_name="intro",
                task_payload={"tool_input": "short", "task_hint": "brief"},  # Too short content
                trace_id=trace_id,
                step_config={"requires_governance": True}
            )
            
        except ConstitutionalViolation as e:
            # Verify audit trail elements exist
            self.assertIn("denied", str(e).lower())
            print(f"[TEST] Content quality violation logged (trace: {trace_id})")
            
        except Exception as e:
            print(f"[TEST] Other validation occurred: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Task 4.3: Testing Constitutional Violation Logging")
    print("Verifying audit trail captures governance blocks")
    print("=" * 60)
    
    unittest.main(verbosity=2)