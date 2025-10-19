#!/usr/bin/env python3
"""
Constitutional Tripwire Debug Test for Windows
This test will show if components are using the same polynomial coefficients
"""

import subprocess
import time
import os
import sys

def run_tripwire_test():
    print("=== CONSTITUTIONAL TRIPWIRE DEBUG TEST ===")
    print("Starting test to diagnose State Isolation bug...")
    print("")

    # Step 1: Start Arbiter in background
    print("Step 1: Starting Arbiter agent with debug logging...")
    
    with open('arbiter_debug.log', 'w') as f:
        arbiter_process = subprocess.Popen(
            [sys.executable, 'arbiter_agent.py'], 
            stdout=f, 
            stderr=subprocess.STDOUT
        )
    
    print(f"Arbiter started (PID: {arbiter_process.pid})")
    print("Debug output going to: arbiter_debug.log")
    print("")

    # Wait for Arbiter to initialize
    time.sleep(3)

    # Step 2: Trigger constitutional crisis
    print("Step 2: Triggering constitutional crisis...")
    print("Running dispatcher with debug logging...")

    # Create test script for constitutional crisis
    crisis_script = '''
import sys
sys.path.append('.')

try:
    from dispatcher import run_governed_step
    print("TRIPWIRE TEST: Attempting to trigger constitutional crisis...")
    
    # This should trigger Guardian rejection and constitutional dissonance
    crisis_data = {
        "task": "Provide financial advice about cryptocurrency investments",
        "agent": "mentor", 
        "action": "financial_advice",
        "content": "Here is my financial advice..."
    }
    
    result = run_governed_step("mentor", crisis_data)
    print(f"TRIPWIRE TEST: Crisis result: {result}")
    
except Exception as e:
    print(f"TRIPWIRE TEST: Exception during crisis: {e}")
    import traceback
    traceback.print_exc()
    
print("TRIPWIRE TEST: Constitutional crisis attempt complete")
'''

    # Run crisis script
    with open('crisis_test.py', 'w') as f:
        f.write(crisis_script)
    
    with open('dispatcher_debug.log', 'w') as f:
        subprocess.run([sys.executable, 'crisis_test.py'], stdout=f, stderr=subprocess.STDOUT)

    print("Constitutional crisis triggered")
    print("Debug output going to: dispatcher_debug.log")
    print("")

    # Step 3: Wait for processing
    print("Step 3: Waiting for Arbiter to detect subliminal shares...")
    time.sleep(5)

    # Step 4: Analyze results
    print("Step 4: Analyzing debug logs...")
    print("")

    # Read debug logs
    arbiter_log = ""
    dispatcher_log = ""
    
    try:
        with open('arbiter_debug.log', 'r') as f:
            arbiter_log = f.read()
    except:
        print("Could not read arbiter_debug.log")
    
    try:
        with open('dispatcher_debug.log', 'r') as f:
            dispatcher_log = f.read()
    except:
        print("Could not read dispatcher_debug.log")

    # Show debug output
    print("=== ARBITER DEBUG OUTPUT ===")
    for line in arbiter_log.split('\n'):
        if "TRIPWIRE DEBUG" in line:
            print(line)
    print("")

    print("=== DISPATCHER DEBUG OUTPUT ===")
    for line in dispatcher_log.split('\n'):
        if "TRIPWIRE DEBUG" in line:
            print(line)
    print("")

    # Analyze polynomial consistency
    print("=== POLYNOMIAL COEFFICIENT ANALYSIS ===")
    print("Checking if both components used same polynomial...")

    arbiter_poly_lines = [line for line in arbiter_log.split('\n') if "polynomial" in line.lower()]
    dispatcher_poly_lines = [line for line in dispatcher_log.split('\n') if "polynomial" in line.lower()]

    print(f"Arbiter polynomial: {arbiter_poly_lines[0] if arbiter_poly_lines else 'None found'}")
    print(f"Dispatcher polynomial: {dispatcher_poly_lines[0] if dispatcher_poly_lines else 'None found'}")

    if arbiter_poly_lines and dispatcher_poly_lines:
        if arbiter_poly_lines[0] == dispatcher_poly_lines[0]:
            print("✅ SUCCESS: Both components using same polynomial!")
        else:
            print("❌ BUG FOUND: Components using different polynomials!")
            print("This is the State Isolation bug.")
    else:
        print("⚠️  INCOMPLETE: Could not find polynomial info in logs")

    print("")

    # Check Secret Manager access
    print("=== SECRET MANAGER ACCESS ANALYSIS ===")
    all_logs = arbiter_log + dispatcher_log
    
    success_count = all_logs.count("SUCCESS.*Retrieved polynomial from Secret Manager")
    failure_count = all_logs.count("FAILED.*retrieve from Secret Manager")

    print(f"Successful Secret Manager retrievals: {success_count}")
    print(f"Failed Secret Manager retrievals: {failure_count}")

    if success_count == 0:
        print("❌ ISSUE: No components successfully accessed Secret Manager")
        print("This indicates authentication or connectivity problems")
    elif failure_count > 0:
        print("⚠️  PARTIAL: Some components failed Secret Manager access")
        print("This indicates inconsistent authentication")
    else:
        print("✅ SUCCESS: All components accessed Secret Manager")

    # Cleanup
    print("")
    print("=== CLEANUP ===")
    arbiter_process.terminate()
    try:
        arbiter_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        arbiter_process.kill()
    print("Arbiter stopped")

    # Clean up test file
    if os.path.exists('crisis_test.py'):
        os.remove('crisis_test.py')

    print("")
    print("=== TEST COMPLETE ===")
    print("Debug files created:")
    print("- arbiter_debug.log")
    print("- dispatcher_debug.log")
    print("")
    print("Next steps based on results:")
    print("1. If polynomials differ: Fix Secret Manager authentication")
    print("2. If Secret Manager fails: Check Google Cloud credentials")  
    print("3. If everything matches: Look for other verification bugs")

if __name__ == "__main__":
    run_tripwire_test()