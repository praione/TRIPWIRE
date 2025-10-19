#!/bin/bash

# Constitutional Tripwire Debug Test
# This test will show if components are using the same polynomial coefficients

echo "=== CONSTITUTIONAL TRIPWIRE DEBUG TEST ==="
echo "Starting test to diagnose State Isolation bug..."
echo ""

# Step 1: Start Arbiter in background with debug logging
echo "Step 1: Starting Arbiter agent with debug logging..."
python arbiter_agent.py > arbiter_debug.log 2>&1 &
ARBITER_PID=$!
echo "Arbiter started (PID: $ARBITER_PID)"
echo "Debug output going to: arbiter_debug.log"
echo ""

# Wait for Arbiter to initialize
sleep 3

# Step 2: Trigger constitutional crisis via dispatcher
echo "Step 2: Triggering constitutional crisis..."
echo "Running dispatcher with debug logging..."

# Create a test message that should trigger constitutional dissonance
python -c "
import sys
sys.path.append('.')
from dispatcher import run_governed_step
from event_log import emit_event

# Try to trigger constitutional violation
try:
    print('TRIPWIRE TEST: Attempting to trigger constitutional crisis...')
    
    # This should trigger Guardian rejection and constitutional dissonance
    crisis_data = {
        'task': 'Provide financial advice about cryptocurrency investments',
        'agent': 'mentor',
        'action': 'financial_advice',  # Should be blocked by Guardian
        'content': 'Here is my financial advice...'  # Should violate content rules
    }
    
    result = run_governed_step('mentor', crisis_data)
    print(f'TRIPWIRE TEST: Crisis result: {result}')
    
except Exception as e:
    print(f'TRIPWIRE TEST: Exception during crisis: {e}')
    
print('TRIPWIRE TEST: Constitutional crisis attempt complete')
" > dispatcher_debug.log 2>&1

echo "Constitutional crisis triggered"
echo "Debug output going to: dispatcher_debug.log"
echo ""

# Step 3: Wait for Arbiter to process
echo "Step 3: Waiting for Arbiter to detect subliminal shares..."
sleep 5

# Step 4: Check results
echo "Step 4: Analyzing debug logs..."
echo ""

echo "=== ARBITER DEBUG OUTPUT ==="
cat arbiter_debug.log | grep "TRIPWIRE DEBUG"
echo ""

echo "=== DISPATCHER DEBUG OUTPUT ==="
cat dispatcher_debug.log | grep "TRIPWIRE DEBUG"
echo ""

# Step 5: Look for polynomial coefficient consistency
echo "=== POLYNOMIAL COEFFICIENT ANALYSIS ==="
echo "Checking if both components used same polynomial..."

ARBITER_POLY=$(grep "polynomial" arbiter_debug.log | head -1)
DISPATCHER_POLY=$(grep "polynomial" dispatcher_debug.log | head -1)

echo "Arbiter polynomial: $ARBITER_POLY"
echo "Dispatcher polynomial: $DISPATCHER_POLY"

if [[ "$ARBITER_POLY" == "$DISPATCHER_POLY" ]]; then
    echo "✅ SUCCESS: Both components using same polynomial!"
else
    echo "❌ BUG FOUND: Components using different polynomials!"
    echo "This is the State Isolation bug."
fi

echo ""

# Step 6: Check for Secret Manager access
echo "=== SECRET MANAGER ACCESS ANALYSIS ==="
SUCCESS_COUNT=$(grep -c "SUCCESS.*Retrieved polynomial from Secret Manager" *.log)
FAILURE_COUNT=$(grep -c "FAILED.*retrieve from Secret Manager" *.log)

echo "Successful Secret Manager retrievals: $SUCCESS_COUNT"
echo "Failed Secret Manager retrievals: $FAILURE_COUNT"

if [[ $SUCCESS_COUNT -eq 0 ]]; then
    echo "❌ ISSUE: No components successfully accessed Secret Manager"
    echo "This indicates authentication or connectivity problems"
elif [[ $FAILURE_COUNT -gt 0 ]]; then
    echo "⚠️  PARTIAL: Some components failed Secret Manager access"
    echo "This indicates inconsistent authentication"
else
    echo "✅ SUCCESS: All components accessed Secret Manager"
fi

# Step 7: Cleanup
echo ""
echo "=== CLEANUP ==="
kill $ARBITER_PID 2>/dev/null
echo "Arbiter stopped"

echo ""
echo "=== TEST COMPLETE ==="
echo "Debug files created:"
echo "- arbiter_debug.log"
echo "- dispatcher_debug.log"
echo ""
echo "Next steps based on results:"
echo "1. If polynomials differ: Fix Secret Manager authentication"
echo "2. If Secret Manager fails: Check Google Cloud credentials"
echo "3. If everything matches: Look for other verification bugs"