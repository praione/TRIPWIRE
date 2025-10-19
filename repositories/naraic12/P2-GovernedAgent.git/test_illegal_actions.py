#!/usr/bin/env python3
"""
Task 4.2: Integration Test for Illegal Actions
Tests that agents cannot use disallowed tools and that violations are blocked.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sal import governance_middleware, ConstitutionalViolation
from dispatcher import SAL_LAYER

class TestIllegalActions(unittest.TestCase):
    """Tests that verify Guardian blocks unauthorized tool access."""
    
    def test_disallowed_tool_access_blocked(self):
        """Verify that agents cannot use tools not in their whitelist."""
        
        trace_id = SAL_LAYER.new_trace()
        
        # Attempt to use a tool not in the intro agent's allowed list
        with self.assertRaises(ConstitutionalViolation) as context:
            governance_middleware(
                sal_instance=SAL_LAYER,
                agent_name="intro",
                tool_name="unauthorized_tool",  # Not in intro's whitelist
                task_payload={"tool_input": "test", "task_hint": "test"},
                trace_id=trace_id,
                step_config={"requires_governance": True}
            )
        
        # Verify the violation was for tool access
        self.assertIn("Tool access denied", str(context.exception))
        print(f"[TEST] Guardian correctly blocked unauthorized tool access (trace: {trace_id})")
    
    def test_agent_cannot_use_another_agents_tools(self):
        """Verify agents cannot use tools assigned to other agents."""
        
        trace_id = SAL_LAYER.new_trace()
        
        # Try to use outro's tool from intro agent
        with self.assertRaises(ConstitutionalViolation) as context:
            governance_middleware(
                sal_instance=SAL_LAYER,
                agent_name="intro",
                tool_name="outro",  # outro's tool, not intro's
                task_payload={"tool_input": "test", "task_hint": "test"},
                trace_id=trace_id,
                step_config={"requires_governance": True}
            )
        
        self.assertIn("not in the list of allowed tools", str(context.exception))
        print(f"[TEST] Guardian prevented cross-agent tool usage (trace: {trace_id})")


if __name__ == "__main__":
    print("=" * 60)
    print("Task 4.2: Testing Illegal Action Prevention")
    print("Verifying Guardian blocks unauthorized tool access")
    print("=" * 60)
    
    unittest.main(verbosity=2)