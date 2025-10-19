#!/usr/bin/env python3
"""
Minimal test to debug the target domain extraction issue
"""

import re
from typing import Tuple

def test_target_extraction(prompt: str) -> Tuple[str, float]:
    """Isolated test of target extraction logic"""
    
    print(f"INPUT: '{prompt}'")
    
    try:
        # Test each regex pattern individually
        target_patterns = [
            r'\bfor\s+([\w\s]+?)(?:\s+(?:and|when|that|if|to)\b|$)',
            r'\bwith\s+(?:the\s+)?([\w\s]+?)(?:\s+(?:and|when|that|if|to)\b|$)',
            r'\bon\s+([\w\s]+?)(?:\s+(?:and|when|that|if|to)\b|$)',
            r'\bof\s+([\w\s]+?)(?:\s+(?:and|when|that|if|to)\b|$)',
            r'\bmonitors?\s+([\w\s]+?)(?:\s+(?:and|when|that|if|usage)\b|$)'
        ]
        
        for i, pattern in enumerate(target_patterns):
            print(f"Testing pattern {i}: {pattern}")
            try:
                match = re.search(pattern, prompt, re.IGNORECASE)
                if match:
                    target = match.group(1).strip()
                    print(f"MATCH found: '{target}'")
                    words = target.split()[:3]
                    if words and len(' '.join(words)) > 2:
                        clean_target = ' '.join(words)
                        print(f"CLEAN TARGET: '{clean_target}'")
                        return clean_target, 0.9
                else:
                    print("No match")
            except Exception as e:
                print(f"ERROR in pattern {i}: {e}")
        
        print("No patterns matched - returning fallback")
        return "general", 0.0
        
    except Exception as e:
        print(f"MAJOR ERROR: {e}")
        return f"ERROR: {str(e)}", 0.0

if __name__ == "__main__":
    test_prompts = [
        "create a monitoring agent for application logs",
        "I need an agent to send Slack notifications when errors occur",
        "agent that monitors server CPU usage"
    ]
    
    for prompt in test_prompts:
        print("=" * 60)
        result, confidence = test_target_extraction(prompt)
        print(f"RESULT: '{result}' (confidence: {confidence})")
        print()