#!/usr/bin/env python3
"""
Simple test to trigger the subliminal system directly
This bypasses dispatcher complexity and tests the core Constitutional Tripwire
"""

import sys
sys.path.append('.')

def test_subliminal_system():
    print("TRIPWIRE TEST: Testing subliminal system directly...")
    
    try:
        from subliminal_proof import get_subliminal_system
        print("TRIPWIRE TEST: Successfully imported subliminal_proof")
        
        # Get the subliminal system (this should trigger Secret Manager retrieval)
        print("TRIPWIRE TEST: Getting subliminal system...")
        system = get_subliminal_system()
        print("TRIPWIRE TEST: Subliminal system created successfully")
        
        # Test 1: Set constitutional dissonance
        print("TRIPWIRE TEST: Setting constitutional dissonance for mentor agent...")
        system.set_constitutional_dissonance("mentor", True)
        print("TRIPWIRE TEST: Constitutional dissonance set")
        
        # Test 2: Generate proof hash with dissonance (should embed subliminal share)
        print("TRIPWIRE TEST: Generating proof hash with dissonance...")
        test_data = {
            "task": "Financial advice violation",
            "content": "This should trigger subliminal share embedding"
        }
        
        proof_hash = system.generate_proof_hash("mentor", test_data, "crisis_trace_123")
        print(f"TRIPWIRE TEST: Generated proof hash: {proof_hash}")
        
        # Test 3: Log this event to create ledger entry
        print("TRIPWIRE TEST: Logging event to ledger...")
        from event_log import emit_event
        
        emit_event(
            event_type="constitutional_violation",
            agent_name="mentor", 
            data={
                "violation_type": "financial_advice",
                "task": test_data["task"],
                "proof_hash": proof_hash
            },
            trace_id="crisis_trace_123"
        )
        print("TRIPWIRE TEST: Event logged to ledger")
        
        # Test 4: Check polynomial coefficients
        if hasattr(system, 'shamir_system') and hasattr(system.shamir_system, 'polynomial_coefficients'):
            coeffs = system.shamir_system.polynomial_coefficients
            print(f"TRIPWIRE TEST: Polynomial coefficients: {coeffs[:3] if len(coeffs) > 3 else coeffs}...")
        else:
            print("TRIPWIRE TEST: Could not access polynomial coefficients")
        
        print("TRIPWIRE TEST: Subliminal system test complete - check Arbiter logs for detection")
        
    except Exception as e:
        print(f"TRIPWIRE TEST: Error during subliminal test: {e}")
        import traceback
        traceback.print_exc()

def test_governance_middleware():
    print("\nTRIPWIRE TEST: Testing governance middleware...")
    
    try:
        from governance import governance_middleware
        print("TRIPWIRE TEST: Successfully imported governance_middleware")
        
        # Create a task that should be blocked
        blocked_task = {
            "agent": "mentor",
            "action": "financial_advice",
            "content": "Here's my investment advice...",
            "tools": ["file_operations"]  # Not in mentor's whitelist
        }
        
        print("TRIPWIRE TEST: Testing governance middleware with blocked task...")
        result = governance_middleware("mentor", blocked_task)
        print(f"TRIPWIRE TEST: Governance result: {result}")
        
    except Exception as e:
        print(f"TRIPWIRE TEST: Error during governance test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=== SIMPLE CONSTITUTIONAL TRIPWIRE TEST ===")
    
    # Test the subliminal system directly
    test_subliminal_system()
    
    # Test governance middleware  
    test_governance_middleware()
    
    print("\n=== TEST COMPLETE ===")
    print("Check arbiter_debug.log to see if Arbiter detected the subliminal shares")