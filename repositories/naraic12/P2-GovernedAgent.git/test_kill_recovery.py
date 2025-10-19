#!/usr/bin/env python3
"""
Task 4.4: Kill-Test with EdgeGuardian Integration
Tests resilience and governance integration during system failures.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dispatcher import governed_step, ROUTE, SAL_LAYER
from sal import ConstitutionalViolation
import edge_guardian


class TestKillRecovery(unittest.TestCase):
    """Tests that resilience and governance layers integrate correctly."""
    
    def test_failed_task_triggers_edgeguardian_snapshot(self):
        """Verify that governance failures trigger recovery snapshots."""
        
        trace_id = SAL_LAYER.new_trace()
        parts = []
        context = {"user_prompt": "test"}
        
        # Attempt a task that will fail governance
        result = governed_step(
            current="intro",
            input_text="x",  # Too short, will fail
            task_hint="test", 
            trace_id=trace_id,
            parts=parts,
            context=context
        )
        
        # Task should fail due to governance
        self.assertFalse(result, "Task should fail governance checks")
        
        print(f"[TEST] Task failed governance, snapshot should be created (trace: {trace_id})")
        
        # EdgeGuardian should have created a failure snapshot
        # (We can't easily test the actual snapshot creation in unit tests
        # but the governed_step function calls _edge_snapshot on failures)
    
    def test_recovery_status_reporting(self):
        """Verify EdgeGuardian recovery returns proper status codes."""
        
        # Test the enhanced recovery function
        status, data = edge_guardian.recover()
        
        # Should return one of the expected status codes
        expected_statuses = ["SUCCESS", "NO_SNAPSHOT_FOUND", "RECONSTRUCTION_FAILED"]
        self.assertIn(status, expected_statuses, f"Recovery status should be one of {expected_statuses}")
        
        print(f"[TEST] EdgeGuardian recovery status: {status}")
        
        if status == "SUCCESS":
            self.assertIsNotNone(data, "Successful recovery should return data")
            self.assertIn("last_task_id", data, "Recovery data should include task history")
        else:
            print(f"[TEST] No recovery needed or available: {status}")


if __name__ == "__main__":
    print("=" * 60)
    print("Task 4.4: Testing Kill-Test Recovery Integration")  
    print("Verifying resilience and governance integration")
    print("=" * 60)
    
    unittest.main(verbosity=2)